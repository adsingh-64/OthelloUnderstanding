"""Fine-grained intervention for length 3 legal move conditions"""

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


@dataclass(frozen=True)
class InterventionMetrics:
    logit_diff: float
    prob_diff: float
    clean_accuracy: float
    corrupted_accuracy: float
    accuracy_diff: float
    below_1_percent: float
    below_5_percent: float
    below_10_percent: float


def load_model(
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens",
    device="cuda",
) -> NNsightModel:
    return utils.get_model(model_name, device)


def load_data(
    dataset_size: int = 500,
    custom_function: str = "games_batch_to_state_stack_mine_yours_BLRRC",
) -> dict[str, Tensor]:
    if custom_function == "games_batch_to_state_stack_mine_yours_BLRRC":
        custom_functions = [
            othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
        ]

    data = construct_othello_dataset(
        custom_functions=custom_functions,
        n_inputs=dataset_size,
        split="test",
        device=device,
    )

    encoded_inputs = t.tensor(data["encoded_inputs"]).long()
    decoded_inputs = t.tensor(data["decoded_inputs"]).long()

    return {
        "encoded_inputs": encoded_inputs,
        "decoded_inputs": decoded_inputs,
        "features": data[custom_function]
    }


def get_feature_indices(query: list[Condition]) -> list[Tuple[int, int, int]]:
    indices = []
    for condition in query:
        feature_name = condition.feature_name
        operator = condition.operator

        square, feature = feature_name.split()
        row, col = list(square)
        row = ord(row) - ord('A')
        col = int(col)
        
        mode = None
        if feature == 'blank':
            mode = 1
        else:
            if operator == '>':
                mode = 0
            else:
                mode = 2

        indices.append((row, col, mode))

    return indices


def get_filtered_positions(
    data: dict[str, Tensor], 
    intervention_query: list[Condition], 
    control_query: list[Condition], 
    intervention: bool
) -> Tuple[list[Int[Tensor, "n_moves"]], list[Int[Tensor, "n_moves"]]]:
    encoded_inputs = data["encoded_inputs"]
    decoded_inputs = data["decoded_inputs"]
    features = data["features"]

    intervention_indices = get_feature_indices(intervention_query)
    control_indices = get_feature_indices(control_query)

    intervention_mask = (features[..., *(intervention_indices[0])] == 1) & (features[..., *(intervention_indices[1])] == 1) & (features[..., *(intervention_indices[2])] == 1)
    control_mask = (features[..., *(control_indices[0])] == 1) & (features[..., *(control_indices[1])] == 1) & (features[..., *(control_indices[2])] == 1)

    if intervention:
        #mask = intervention_mask
        mask = intervention_mask & ~control_mask
    else:
        #mask = control_mask
        mask = control_mask & ~intervention_mask

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
    intervention_positions_decoded: list[Int[Tensor, "n_moves"]],
    control_positions_decoded: list[Int[Tensor, "n_moves"]],
) -> None:
    print("Intervention positions")
    for position in intervention_positions_decoded[:10]:
        board_states, _, _ = neel_utils.get_board_states_and_legal_moves(position)
        neel_utils.plot_board_values(
            board_states[-1],
        )
    
    print("Control positions")
    for position in control_positions_decoded[:10]:
        board_states, _, _ = neel_utils.get_board_states_and_legal_moves(position)
        neel_utils.plot_board_values(
            board_states[-1],
        )


def find_neurons_for_query_DLA(
    model: NNsightModel,
    positions: list[Int[Tensor, "n_moves"]], 
    query: list[Condition], 
    batch_size: int = 1024,
    k: int = 25,
) -> dict[int, list[int]]:
    legal_square_id = neel_utils.to_id(query[0].feature_name.split()[0])

    W_out = model.W_out[1:, :].detach().clone()
    unembed = model.W_U[:, legal_square_id].detach().clone()
    weights = einops.einsum(W_out, unembed, "n_layers d_mlp d_model, d_model -> n_layers d_mlp")

    neuron_acts = {layer: t.zeros((model.cfg.d_mlp), device=device) for layer in range(1, model.cfg.n_layers)}

    for i in tqdm(range(0, len(positions), batch_size), desc="Batches"):
        batch = positions[i:i + batch_size]

        batch_tensor, batch_indices, last_token_indices = right_pad(batch)

        batch_acts = {}
        with model.trace(batch_tensor):
            for layer in range(1, model.cfg.n_layers):
                layer_acts = model.blocks[layer].mlp.hook_post.output[batch_indices, last_token_indices].sum(dim = 0).save()
                batch_acts[layer] = layer_acts
                
        for layer in range(1, model.cfg.n_layers):
            neuron_acts[layer] += batch_acts[layer]

    neuron_acts = t.stack(list(neuron_acts.values()))
    neuron_attrs = neuron_acts * weights
    neuron_attrs_flattened = einops.rearrange(neuron_attrs, "n_layers d_mlp -> (n_layers d_mlp)")

    flattened_indices = neuron_attrs_flattened.topk(k=k).indices
    results = {layer: [] for layer in range(1, model.cfg.n_layers)}
    for idx in flattened_indices:
        idx = idx.item()
        layer = idx // model.cfg.d_mlp + 1
        neuron = idx % model.cfg.d_mlp
        results[layer].append(neuron)

    return results


def merge_dicts(dicts: list[dict]):
    """Merge any number of dicts with identical keys, combining list values and removing duplicates."""
    merged = {}
    
    for key in dicts[0].keys():
        combined = []
        for d in dicts:
            combined.extend(d.get(key, []))
        merged[key] = list(set(combined))
    
    return merged


def get_legal_moves_batch(
    batch: list[Int[Tensor, "seq"]],
) -> list[list[int]]:
    """Return a list of list of ints in token id format, where each inner list corresponds to the legal moves at the final position of its batch example"""
    legal_moves_per_position = []

    for pos in batch:
        squares = neel_utils.to_square(pos.cpu())
        squares_tensor = t.tensor(squares)
        _, legal_moves_tensor, _ = neel_utils.get_board_states_and_legal_moves(squares_tensor)
        legal_squares = t.where(legal_moves_tensor[-1].flatten())[0].tolist()
        legal_token_ids = [neel_utils.to_id(sq) for sq in legal_squares]
        legal_moves_per_position.append(legal_token_ids)

    return legal_moves_per_position


def right_pad(
    batch: list[Int[Tensor, "..."]],
) -> Tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch"], Int[Tensor, "batch"]]:
    batch_indices = t.arange(len(batch), device=device)
    seq_lengths = [pos.shape[0] for pos in batch]
    last_token_indices = t.tensor([length - 1 for length in seq_lengths], device=device)
    max_len = max(seq_lengths)
    
    padded_batch = []
    for pos, length in zip(batch, seq_lengths):
        if length < max_len:
            padding = t.zeros(max_len - length, dtype=pos.dtype, device=pos.device)
            padded = t.cat([pos, padding])
        else:
            padded = pos
        padded_batch.append(padded)
    
    batch_tensor = t.stack(padded_batch).to(device) 
    return batch_tensor, batch_indices, last_token_indices


def no_ablation(
    model: NNsightModel,
    batch_tensor: Int[Tensor, "batch seq"],
    batch_indices: Int[Tensor, "batch"],
    last_token_indices: Int[Tensor, "batch"],
    legal_square_id: int,
) -> Tuple[Float[Tensor, "batch d_vocab"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    with model.trace(batch_tensor):
        logits = model.unembed.output[batch_indices, last_token_indices].save()
        probs = t.nn.functional.softmax(logits, dim=-1)
        
        logits_square = logits[:, legal_square_id].save()
        probs_square = probs[:, legal_square_id].save()

    return logits, logits_square, probs_square


def zero_ablation(
    model: NNsightModel,
    batch_tensor: Int[Tensor, "batch seq"],
    batch_indices: Int[Tensor, "batch"],
    last_token_indices: Int[Tensor, "batch"],
    legal_square_id: int,
    neurons: dict[int, list[int]],
) -> Tuple[Float[Tensor, "batch d_vocab"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    with model.trace(batch_tensor):
        for layer in range(1, model.cfg.n_layers):
            if neurons[layer]:
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
    query: list[Condition],
    dt_queries: list[list[Condition]] | None = None, 
    dla_positions: list[Int[Tensor, "n_moves"]] | None = None,
    dla: bool = False,
    k: int | None = None,
    batch_size: int = 1024
) -> InterventionMetrics:
    if dla:
        neurons = find_neurons_for_query_DLA(model, dla_positions, query, k=k)
    else:
        neurons = merge_dicts([find_neurons_for_query(query) for query in dt_queries])

    print(f"Ablating {sum(len(neurons) for neurons in neurons.values())} neurons")
    legal_square_id = neel_utils.to_id(query[0].feature_name.split()[0])

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
        batch_tensor, batch_indices, last_token_indices = right_pad(batch) 

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

if __name__ == "__main__":
    # Hack: messed up .pkl b/c I put dataclass definition in same pickling script
    sys.modules['__main__'].DecisionTreeResults = DecisionTreeResults

    device = "cuda" if t.cuda.is_available() else "cpu"

    model = load_model()
    data = load_data()

    intervention_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1),
        Condition(feature_name='E2 mine-theirs', operator='>', threshold=-1),
    ]

    dt_queries = [
        [Condition(feature_name='C0 blank', operator='>', threshold=-1), Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1)],
    ]

    control_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='C1 mine-theirs', operator='<=', threshold=1),
        Condition(feature_name='C2 mine-theirs', operator='>', threshold=-1),
    ] 

    intervention_positions_encoded, intervention_positions_decoded = get_filtered_positions(data, intervention_query, control_query, intervention=True)
    control_positions_encoded, control_positions_decoded = get_filtered_positions(data, intervention_query, control_query, intervention=False)
    #sanity_check(intervention_positions_decoded, control_positions_decoded)

    rprint(f"\n[bold]Number of intervention positions:[/bold] {len(intervention_positions_encoded)}")
    rprint(f"[bold]Number of control positions:[/bold] {len(control_positions_encoded)}")

    intervened_metrics = intervene(
        model=model,
        positions=intervention_positions_encoded,
        dt_queries=dt_queries,
        query=intervention_query,
        # dla_positions=intervention_positions_encoded,
        # dla=True,
        # k=k,
    )
    control_metrics = intervene(
        model=model,
        positions=control_positions_encoded,
        dt_queries=dt_queries,
        query=intervention_query,
        # dla_positions=intervention_positions_encoded,
        # dla=True,
        # k=k,
    )

    print_table(intervened_metrics, control_metrics)