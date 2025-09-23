"""Fine-grained intervention for length 3 legal move conditions"""
from __future__ import annotations

import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight.models import NNsightModel

import neel_utils as neel_utils
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from ground_truth_dt.dtypes import DecisionTreeResults
from ground_truth_dt.regression.gt_dt_viz import find_neurons_for_query

import json
import sys
from pathlib import Path
from jaxtyping import Bool, Float, Int
from typing import Tuple
from tqdm import tqdm
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from pprint import pprint
from dataclasses import dataclass, asdict

from ground_truth_dt.fine_grained_intervention.utils import (
    load_model,
    load_data,
    get_filtered_positions,
    sanity_check,
    find_neurons_for_query_DLA,
    merge_dicts,
    get_legal_moves_batch,
    right_pad,
    no_ablation,
)


@dataclass(frozen=True)
class InterventionMetrics:
    logit_diff: float = 0.0
    prob_diff: float = 0.0
    clean_accuracy: float = 0.0
    corrupted_accuracy: float = 0.0
    accuracy_diff: float = 0.0
    below_1_percent: float = 0.0
    below_5_percent: float = 0.0
    below_10_percent: float = 0.0

    def __add__(self, other: InterventionMetrics) -> InterventionMetrics:
        return InterventionMetrics(
            logit_diff=self.logit_diff + other.logit_diff,
            prob_diff=self.prob_diff + other.prob_diff,
            clean_accuracy=self.clean_accuracy + other.clean_accuracy,
            corrupted_accuracy=self.corrupted_accuracy + other.corrupted_accuracy,
            accuracy_diff=self.accuracy_diff + other.accuracy_diff,
            below_1_percent=self.below_1_percent + other.below_1_percent,
            below_5_percent=self.below_5_percent + other.below_5_percent,
            below_10_percent=self.below_10_percent + other.below_10_percent,
        )

    def __truediv__(self, divisor: float) -> InterventionMetrics:
        return InterventionMetrics(
            logit_diff=self.logit_diff / divisor,
            prob_diff=self.prob_diff / divisor,
            clean_accuracy=self.clean_accuracy / divisor,
            corrupted_accuracy=self.corrupted_accuracy / divisor,
            accuracy_diff=self.accuracy_diff / divisor,
            below_1_percent=self.below_1_percent / divisor,
            below_5_percent=self.below_5_percent / divisor,
            below_10_percent=self.below_10_percent / divisor
        )

    def save(self, file_path: Path):
        """Saves the metrics to a JSON file."""
        # asdict converts the dataclass instance to a dictionary
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)


@dataclass
class ExperimentSetup:
    legal_square: str
    intervention_query: set[str]
    control_query: set[str]
    dt_queries: list[set[str]]


def zero_ablation(
    model: NNsightModel,
    batch_tensor: Int[Tensor, "batch seq"],
    batch_indices: Int[Tensor, "batch"],
    last_token_indices: Int[Tensor, "batch"],
    legal_square_id: int,
    neurons: dict[int, list[int]],
) -> Tuple[Float[Tensor, "batch d_vocab"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    with model.trace(batch_tensor):
        for layer in range(0, model.cfg.n_layers - 1):
            if neurons.get(layer):
                neuron_indices = t.tensor(neurons[layer], device=device)
                n_neurons = len(neurons[layer])
                batch_indices_repeated = einops.repeat(
                    batch_indices,
                    'batch -> batch neurons',
                    neurons=n_neurons,
                )
                last_token_indices_repeated = einops.repeat(
                    last_token_indices,
                    'batch -> batch neurons',
                    neurons=n_neurons,
                )
                neuron_indices_repeated = einops.repeat(
                    neuron_indices,
                    'neurons -> batch neurons',
                    batch=len(batch_tensor),
                )
                model.blocks[layer].mlp.hook_post.output[
                    batch_indices_repeated, 
                    last_token_indices_repeated, 
                    neuron_indices_repeated
                ] = 0
        
        logits = model.unembed.output[batch_indices, last_token_indices].save()
        probs = t.nn.functional.softmax(logits, dim=-1)
        
        logits_square = logits[:, legal_square_id].save()
        probs_square = probs[:, legal_square_id].save()

        return logits, logits_square, probs_square


def is_accurate_batch(
    logits: Float[Tensor, "batch d_vocab"],
    legal_moves_batch: list[list[int]],
    legal_square_id,
) -> list[bool]:
    """If there are K legal moves, we say accurate if the legal square is in the
    top K logits"""
    accurate = []

    for j, legal_moves in enumerate(legal_moves_batch):
        k = len(legal_moves)

        top_k_tokens = logits[j].topk(k=k).indices.tolist()
        accurate.append(legal_square_id in top_k_tokens)
        
    return accurate


def below_threshold(
    probs_square: Float[Tensor, "batch"],
    legal_moves_batch: list[list[int]],
    alpha: float = 0.01,
) -> list[bool]:
    return [(prob < alpha * 1 / len(legal_moves)).item() for prob, legal_moves in zip(probs_square, legal_moves_batch)]


def intervene(
    model: NNsightModel, 
    positions: list[Int[Tensor, "n_moves"]], 
    legal_square: str = "C0",
    dt_queries: list[set[str]] | None = None, 
    dla_positions: list[Int[Tensor, "n_moves"]] | None = None,
    dla: bool = False,
    k: int | None = None,
    batch_size: int = 1024,
    device = "cuda",
) -> InterventionMetrics:
    if dla:
        neurons = find_neurons_for_query_DLA(model, dla_positions, legal_square, k=k, device=device)
    else:
        neurons = merge_dicts([find_neurons_for_query(query) for query in dt_queries])

    print(f"Ablating {sum(len(neurons) for neurons in neurons.values())} neurons")

    legal_square_id = neel_utils.to_id(legal_square)

    total_logit_diff = 0
    total_prob_diff = 0
    total_clean_accuracy = 0
    total_corrupted_accuracy = 0
    total_below_1_percent = 0
    total_below_5_percent = 0
    total_below_10_percent = 0

    for i in tqdm(range(0, len(positions), batch_size), desc="Batches"):
        batch = positions[i:i + batch_size]

        legal_moves_batch = get_legal_moves_batch(batch)
        batch_tensor, batch_indices, last_token_indices = right_pad(batch, device=device) 

        clean_logits, clean_logits_square, clean_probs_square = no_ablation(
            model, 
            batch_tensor, 
            batch_indices,
            last_token_indices,
            legal_square_id,
        )

        corrupted_logits, corrupted_logits_square, corrupted_probs_square = zero_ablation(
            model, 
            batch_tensor, 
            batch_indices,
            last_token_indices,
            legal_square_id,
            neurons,
        )

        is_accurate_clean = is_accurate_batch(clean_logits, legal_moves_batch, legal_square_id)
        total_clean_accuracy += sum(is_accurate_clean)

        is_accurate_corrupted = is_accurate_batch(corrupted_logits, legal_moves_batch, legal_square_id)
        total_corrupted_accuracy += sum(is_accurate_corrupted)

        below_1_percent_corrupted = below_threshold(corrupted_probs_square, legal_moves_batch, alpha=0.01)
        total_below_1_percent += sum(below_1_percent_corrupted)

        below_5_percent_corrupted = below_threshold(corrupted_probs_square, legal_moves_batch, alpha=0.05)
        total_below_5_percent += sum(below_5_percent_corrupted)

        below_10_percent_corrupted = below_threshold(corrupted_probs_square, legal_moves_batch, alpha=0.1)
        total_below_10_percent += sum(below_10_percent_corrupted)
        
        total_logit_diff += (clean_logits_square - corrupted_logits_square).sum().item()
        total_prob_diff += (clean_probs_square - corrupted_probs_square).sum().item()
    
    avg_logit_diff = total_logit_diff / len(positions)
    avg_prob_diff = total_prob_diff / len(positions)
    avg_clean_accuracy = total_clean_accuracy / len(positions)
    avg_corrupted_accuracy = total_corrupted_accuracy / len(positions)
    avg_num_below_1_percent = total_below_1_percent / len(positions)
    avg_num_below_5_percent = total_below_5_percent / len(positions)
    avg_num_below_10_percent = total_below_10_percent / len(positions)

    return InterventionMetrics(
        logit_diff=avg_logit_diff,
        prob_diff=avg_prob_diff,
        clean_accuracy=avg_clean_accuracy,
        corrupted_accuracy=avg_corrupted_accuracy,
        accuracy_diff=avg_clean_accuracy - avg_corrupted_accuracy,
        below_1_percent=avg_num_below_1_percent,
        below_5_percent=avg_num_below_5_percent,
        below_10_percent=avg_num_below_10_percent,
    )
        

def print_table(intervened_metrics: InterventionMetrics, control_metrics: InterventionMetrics) -> None:
    table = Table(title="Intervention Results")
    
    table.add_column("Metric", style="", no_wrap=True)
    table.add_column("Intervention", justify="right", style="")
    table.add_column("Control", justify="right", style="")
    
    table.add_row(
        "Logit Diff",
        f"{intervened_metrics.logit_diff:.4f}",
        f"{control_metrics.logit_diff:.4f}"
    )
    table.add_row(
        "Prob Diff",
        f"{intervened_metrics.prob_diff:.4f}",
        f"{control_metrics.prob_diff:.4f}"
    )
    table.add_row(
        "Clean Accuracy",
        f"{intervened_metrics.clean_accuracy:.2%}",
        f"{control_metrics.clean_accuracy:.2%}"
    )
    table.add_row(
        "Corrupted Accuracy",
        f"{intervened_metrics.corrupted_accuracy:.2%}",
        f"{control_metrics.corrupted_accuracy:.2%}"
    )
    table.add_row(
        "Accuracy Diff",
        f"{intervened_metrics.accuracy_diff:.2%}",
        f"{control_metrics.accuracy_diff:.2%}"
    )
    table.add_row(
        "Below 1 Percent Original",
        f"{intervened_metrics.below_1_percent:.2%}",
        f"{control_metrics.below_1_percent:.2%}"
    )
    table.add_row(
        "Below 5 Percent",
        f"{intervened_metrics.below_5_percent:.2%}",
        f"{control_metrics.below_5_percent:.2%}"
    )
    table.add_row(
        "Below 10 Percent",
        f"{intervened_metrics.below_10_percent:.2%}",
        f"{control_metrics.below_10_percent:.2%}"
    )
    rprint(table)


def coords_to_square(coords: tuple[int, int]) -> str:
    row, col = coords
    row_letter = chr(row + ord("A"))
    return f"{row_letter}{col}"


def square_to_coords(name: str) -> tuple[int, int]:
    row_letter, col_letter = tuple(name)
    return (ord(row_letter) - ord("A"), int(col_letter))


def generate_experiment_setup(legal_square: str, rng: np.random.default_rng) -> ExperimentSetup:
    row, col = square_to_coords(legal_square)
    sx, sy = int(np.sign(3.5 - row)), int(np.sign(3.5 - col))
    dx, dy = abs(3.5 - row), abs(3.5 - col)

    diagonal = (
        legal_square,
        coords_to_square((row + sx, col + sy)),
        coords_to_square((row + 2 * sx, col + 2 * sy)),
    )

    if dx >= dy:
        axial_dir = (sx, 0)

    else:
        axial_dir = (0, sy)

    ax, ay = axial_dir

    axial = (
        legal_square,
        coords_to_square((row + ax, col + ay)),
        coords_to_square((row + 2*ax, col + 2*ay)),
    )
    
    if rng.random() < 0.5:
        intervention_query = {f'{diagonal[0]}_empty', f'{diagonal[1]}_theirs', f'{diagonal[2]}_mine'}
        control_query = {f'{axial[0]}_empty', f'{axial[1]}_theirs', f'{axial[2]}_mine'}
        dt_queries = [{f'{diagonal[0]}_empty', f'{diagonal[1]}_theirs'}]

    else:
        intervention_query = {f'{axial[0]}_empty', f'{axial[1]}_theirs', f'{axial[2]}_mine'}
        control_query = {f'{diagonal[0]}_empty', f'{diagonal[1]}_theirs', f'{diagonal[2]}_mine'}
        dt_queries = [{f'{axial[0]}_empty', f'{axial[1]}_theirs'}]

    return ExperimentSetup(
        legal_square=legal_square,
        intervention_query=intervention_query,
        control_query=control_query,
        dt_queries=dt_queries,
    )


def display_experiment_summary(
    setup: ExperimentSetup, 
    n_intervention: int, 
    n_control: int
):
    """Prints a single panel summarizing the setup and data counts."""
    order_map = {'empty': 0, 'theirs': 1, 'mine': 2}

    def sort_by_suffix(query_string: str) -> int:
        suffix = query_string.split('_')[1]
        return order_map.get(suffix, 99)

    intervention_list = sorted(list(setup.intervention_query), key=sort_by_suffix)
    control_list = sorted(list(setup.control_query), key=sort_by_suffix)

    intervention_str = " AND ".join(intervention_list)
    control_str = " AND ".join(control_list)

    # This is the new, combined output string
    output = (
        f"[bold]Legal Square:[/] {setup.legal_square}\n"
        f"[bold blue]Intervention:[/] {intervention_str} [dim](n={n_intervention})[/]\n"
        f"[bold green]Control:[/]      {control_str} [dim](n={n_control})[/]"
    )
    
    rprint(Panel(output, title="Experiment Setup", expand=False))


def run_single_experiment(
    experiment_setup: ExperimentSetup,
    model: NNsightModel,
    data: Tensor,
    device: str,
) -> tuple[InterventionMetrics | None, InterventionMetrics | None]:
    intervention_positions_encoded, intervention_positions_decoded = get_filtered_positions(data, experiment_setup.intervention_query, experiment_setup.control_query, intervention=True)

    control_positions_encoded, control_positions_decoded = get_filtered_positions(data, experiment_setup.intervention_query, experiment_setup.control_query, intervention=False)

    if len(intervention_positions_encoded) == 0 or len(control_positions_encoded) == 0:
        rprint(
            f"[yellow]Skipping {experiment_setup.legal_square}: "
            f"Not enough positions found "
            f"(|interv|={len(intervention_positions_encoded)}, |ctrl|={len(control_positions_encoded)}).[/yellow]"
        )
        return None, None

    display_experiment_summary(
        experiment_setup,
        len(intervention_positions_encoded),
        len(control_positions_encoded)
    )

    intervened_metrics = intervene(
        model=model,
        positions=intervention_positions_encoded,
        legal_square=experiment_setup.legal_square,
        dt_queries=experiment_setup.dt_queries,
        # dla_positions=intervention_positions_encoded,
        # dla=True,
        # k=k,
        device=device,
    )
    control_metrics = intervene(
        model=model,
        positions=control_positions_encoded,
        legal_square=experiment_setup.legal_square,
        dt_queries=experiment_setup.dt_queries,
        # dla_positions=intervention_positions_encoded,
        # dla=True,
        # k=k,
        device=device,
    )

    print_table(intervened_metrics, control_metrics)

    return intervened_metrics, control_metrics

if __name__ == "__main__":
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = load_model(device=device)
    data = load_data(device=device)

    rng = np.random.default_rng(0)

    agg_intervened = InterventionMetrics()
    agg_control = InterventionMetrics()

    total_squares = 0

    for row in range(8):
        for col in range(8):
            if 2 <= row <= 5 and 2 <= col <= 5:
                continue

            coords = (row, col)
            legal_square = coords_to_square(coords)

            experiment_setup = generate_experiment_setup(legal_square, rng)

            intervened_metrics, control_metrics = run_single_experiment(
                experiment_setup,
                model, 
                data,
                device,
            )

            if intervened_metrics and control_metrics:
                agg_intervened += intervened_metrics
                agg_control += control_metrics
                total_squares += 1

            else:
                continue

    avg_intervened = agg_intervened / total_squares
    avg_control = agg_control / total_squares

    print_table(avg_intervened, avg_control)

    current_dir = Path(__file__).resolve().parent

    intervention_save_path = current_dir / "intervention_metrics_48_squares.json"
    control_save_path = current_dir / "control_metrics_48_squares.json"

    avg_intervened.save(file_path=intervention_save_path)
    avg_control.save(file_path=control_save_path)