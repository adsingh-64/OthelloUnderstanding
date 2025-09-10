import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight.models import NNsightModel

import neel_utils as neel_utils
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from cont_dt.cont_dt_viz import Condition, DecisionTreeResults, find_neurons_for_query

import json
import sys
from jaxtyping import Bool, Float, Int
from typing import Tuple
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table
from pprint import pprint
from dataclasses import dataclass

from fine_grained_intervention.utils import (
    load_model,
    load_data,
    get_feature_indices,
    find_neurons_for_query_DLA,
    merge_dicts,
    get_legal_moves_batch,
    right_pad,
    no_ablation,
)


@dataclass(frozen=True)
class RestoreInterventionMetrics:
    logit_diff: float
    prob_diff: float


def get_filtered_positions(
    data: dict[str, Tensor], 
    query: list[Condition], 
) -> Tuple[list[Int[Tensor, "n_moves"]], list[Int[Tensor, "n_moves"]]]:
    encoded_inputs = data["encoded_inputs"]
    decoded_inputs = data["decoded_inputs"]
    features = data["features"]

    indices = get_feature_indices(query)

    individual_masks = [features[..., *idx] == 1 for idx in indices]
    mask = t.stack(individual_masks).all(dim=0)

    filtered_positions_encoded = []
    filtered_positions_decoded = []
    for game_idx in range(mask.shape[0]):
        indices = t.where(mask[game_idx])[0]
        for idx in indices:
            if 5 <= idx <= 30:
                filtered_positions_encoded.append(encoded_inputs[game_idx, : idx + 1])
                filtered_positions_decoded.append(decoded_inputs[game_idx, : idx + 1])

    return filtered_positions_encoded, filtered_positions_decoded


def sanity_check(
    positions_decoded: list[Int[Tensor, "n_moves"]],
) -> None:
    for position in positions_decoded[:10]:
        board_states, _, _ = neel_utils.get_board_states_and_legal_moves(position)
        neel_utils.plot_board_values(
            board_states[-1],
        )


def cache_mean_acts(
    positions_encoded: list[Int[Tensor, "n_moves"]],
    neurons: list[int, list[int]],
    batch_size: int = 1024,
    device: str = "cuda",
) -> list[int, list[float]]:
    batch_tensor, batch_indices, legal_moves = right_pad(positions_encoded, device=device)
    pass


def mean_ablation(
    model: NNsightModel,
    batch_tensor: Int[Tensor, "batch seq"],
    batch_indices: Int[Tensor, "batch"],
    last_token_indices: Int[Tensor, "batch"],
    legal_square_id: int,
    neurons: dict[int, list[int]],
) -> Tuple[Float[Tensor, "batch d_vocab"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    pass


def intervene(
    model: NNsightModel, 
    base_positions: list[Int[Tensor, "n_moves"]],
    intervention_positions: list[Int[Tensor, "n_moves"]], 
    query: list[Condition],
    dt_queries: list[list[Condition]] | None = None, 
    batch_size: int = 1024,
    device = "cuda",
) -> RestoreInterventionMetrics:
    legal_square_id = neel_utils.to_id(query[0].feature_name.split()[0])

    neurons = merge_dicts([find_neurons_for_query(query) for query in dt_queries])
    print(f"Ablating {sum(len(neurons) for neurons in neurons.values())} neurons")

    neuron_acts = cache_mean_acts(
        base_positions,
        neurons,
        batch_size=batch_size,
        device=device,
    )

    total_logit_diff = 0
    total_prob_diff = 0
    
    for i in tqdm(range(0, len(intervention_positions), batch_size), desc="Batches"):
        batch = intervention_positions[i:i + batch_size]

        batch_tensor, batch_indices, last_token_indices = right_pad(batch, device=device) 

        clean_logits, clean_logits_square, clean_probs_square = no_ablation(
            model, 
            batch_tensor, 
            batch_indices,
            last_token_indices,
            legal_square_id,
        )

        restored_logits, restored_logits_square, restored_probs_square = mean_ablation(
            model, 
            batch_tensor, 
            batch_indices,
            last_token_indices,
            legal_square_id,
            neurons,
            neuron_acts,
        )

        total_logit_diff += (restored_logits_square - clean_logits_square).sum().item()
        total_prob_diff += (restored_probs_square - clean_probs_square).sum().item()

    avg_logit_diff = total_logit_diff / len(intervention_positions)
    avg_prob_diff = total_prob_diff / len(intervention_positions)

    return RestoreInterventionMetrics(
        logit_diff=avg_logit_diff,
        prob_diff=avg_prob_diff,
    )


def print_table(restore_metrics: RestoreInterventionMetrics) -> None:
    table = Table(title="Intervention Results")
    
    table.add_column("Metric", style="", no_wrap=True)
    table.add_column("Result", justify="right", style="")
    table.add_row(
        "Logit Diff",
        f"{restore_metrics.logit_diff:.4f}",
    )
    table.add_row(
        "Prob Diff",
        f"{restore_metrics.prob_diff:.4f}",
    )

    rprint(table)


if __name__ == "__main__":
    # Hack: messed up .pkl b/c I put dataclass definition in same pickling script
    sys.modules['__main__'].DecisionTreeResults = DecisionTreeResults

    device = "cuda" if t.cuda.is_available() else "cpu"

    model = load_model(device=device)
    data = load_data(device=device)

    # Games over which to cache mean acts
    base_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1),
        Condition(feature_name='E2 mine-theirs', operator='>', threshold=-1)
    ]

    # Games over which to intervene
    intervention_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='D1 mine-theirs', operator='>', threshold=-1),
    ]

    # Neurons to mean ablate
    dt_queries = [
        [Condition(feature_name='C0 blank', operator='>', threshold=-1), Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1)],
    ]

    base_positions_encoded, base_positions_decoded = get_filtered_positions(data, base_query)

    rprint(f"\n[bold]Number of base (cache) positions:[/bold] {len(base_positions_encoded)}")

    #sanity_check(base_positions_decoded)

    intervention_positions_encoded, intervention_positions_decoded = get_filtered_positions(data, intervention_query)

    rprint(f"\n[bold]Number of intervention positions:[/bold] {len(intervention_positions_encoded)}")

    restore_metrics = intervene(
        model=model,
        base_positions=base_positions_encoded,
        intervention_positions=intervention_positions_encoded,
        query=base_query,
        dt_queries=dt_queries,
        device=device,
    )

    print_table(restore_metrics)