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
        seq_lengths = [pos.shape[0] for pos in batch]
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
        batch_indices = t.arange(len(batch), device=device)
        last_token_indices = t.tensor([length - 1 for length in seq_lengths], device=device)

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


def intervene(
    model: NNsightModel, 
    positions: list[Int[Tensor, "n_moves"]], 
    query: list[Condition],
    dt_queries: list[list[Condition]] | None = None, 
    dla_positions: list[Int[Tensor, "n_moves"]] | None = None,
    dla: bool = False,
    k: int | None = None,
    batch_size: int = 1024
) -> dict[str, float]:
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

    for i in tqdm(range(0, len(positions), batch_size), desc="Batches"):
        batch = positions[i:i + batch_size]
        
        seq_lengths = [pos.shape[0] for pos in batch]
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
        batch_indices = t.arange(len(batch), device=device)
        last_token_indices = t.tensor([length - 1 for length in seq_lengths], device=device)

        # Get legal moves for each position in the batch
        legal_moves_per_position = []
        for pos in batch:
            squares = neel_utils.to_square(pos.cpu())
            squares_tensor = t.tensor(squares)
            _, legal_moves_tensor, _ = neel_utils.get_board_states_and_legal_moves(squares_tensor)
            legal_squares = t.where(legal_moves_tensor[-1].flatten())[0].tolist()
            legal_token_ids = [neel_utils.to_id(sq) for sq in legal_squares]
            legal_moves_per_position.append(legal_token_ids)

        with model.trace(batch_tensor):
            clean_logits = model.unembed.output[batch_indices, last_token_indices].save()
            clean_probs = t.nn.functional.softmax(clean_logits, dim=-1)
            
            clean_logits_square = clean_logits[:, legal_square_id].save()
            clean_probs_square = clean_probs[:, legal_square_id].save()

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
                        batch=len(batch),
                    )
                    model.blocks[layer].mlp.hook_post.output[
                        batch_indices_repeated, 
                        last_token_indices_repeated, 
                        neuron_indices_repeated
                    ] = 0
            
            corrupted_logits = model.unembed.output[batch_indices, last_token_indices].save()
            corrupted_probs = t.nn.functional.softmax(corrupted_logits, dim=-1)
            
            corrupted_logits_square = corrupted_logits[:, legal_square_id].save()
            corrupted_probs_square = corrupted_probs[:, legal_square_id].save()

        # Calculate accuracy for this batch
        for j, legal_moves in enumerate(legal_moves_per_position):
            k = len(legal_moves)

            clean_top_k = t.topk(clean_logits[j], k=k).indices
            if legal_square_id in clean_top_k.tolist():
                total_clean_accuracy += 1
            
            corrupted_top_k = t.topk(corrupted_logits[j], k=k).indices
            if legal_square_id in corrupted_top_k.tolist():
                total_corrupted_accuracy += 1

            # Check if dropped below 1 percent of original accuracy
            if corrupted_probs_square[j] < 0.01 * 1/k:
                total_below_1_percent += 1
        
        total_logit_diff += (clean_logits_square - corrupted_logits_square).sum().item()
        total_prob_diff += (clean_probs_square - corrupted_probs_square).sum().item()
    
    avg_logit_diff = total_logit_diff / len(positions)
    avg_prob_diff = total_prob_diff / len(positions)
    avg_clean_accuracy = total_clean_accuracy / len(positions)
    avg_corrupted_accuracy = total_corrupted_accuracy / len(positions)
    avg_num_below_1_percent = total_below_1_percent / len(positions)
    
    return {
        "logit_diff": avg_logit_diff, 
        "prob_diff": avg_prob_diff,
        "clean_accuracy": avg_clean_accuracy,
        "corrupted_accuracy": avg_corrupted_accuracy,
        "accuracy_diff": avg_clean_accuracy - avg_corrupted_accuracy,
        "below_1_percent": avg_num_below_1_percent,
    }
        

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

    # Create a rich table
    rprint(f"\n[bold]Number of intervention positions:[/bold] {len(intervention_positions_encoded)}")
    rprint(f"[bold]Number of control positions:[/bold] {len(control_positions_encoded)}")

    table = Table(title="Intervention Results")
    
    table.add_column("Metric", style="", no_wrap=True)
    table.add_column("Intervention", justify="right", style="")
    table.add_column("Control", justify="right", style="")
    
    table.add_row(
        "Logit Diff",
        f"{intervened_metrics['logit_diff']:.4f}",
        f"{control_metrics['logit_diff']:.4f}"
    )
    table.add_row(
        "Prob Diff",
        f"{intervened_metrics['prob_diff']:.4f}",
        f"{control_metrics['prob_diff']:.4f}"
    )
    table.add_row(
        "Clean Accuracy",
        f"{intervened_metrics['clean_accuracy']:.2%}",
        f"{control_metrics['clean_accuracy']:.2%}"
    )
    table.add_row(
        "Corrupted Accuracy",
        f"{intervened_metrics['corrupted_accuracy']:.2%}",
        f"{control_metrics['corrupted_accuracy']:.2%}"
    )
    table.add_row(
        "Accuracy Diff",
        f"{intervened_metrics['accuracy_diff']:.2%}",
        f"{control_metrics['accuracy_diff']:.2%}"
    )
    table.add_row(
        "Below 1 Percent",
        f"{intervened_metrics['below_1_percent']:.2%}",
        f"{control_metrics['below_1_percent']:.2%}"
    )
    rprint(table)

    # dt_neurons = find_neurons_for_query(intervention_query[:2])
    # dla_neurons = find_neurons_for_query_DLA(model, intervention_positions_encoded, intervention_query, k=25)
    # dla_unique_neurons = {layer: set(dla_neurons[layer]) - set(dt_neurons[layer]) for layer in dt_neurons.keys()}
    # pprint(dt_neurons)
    # print("="*80)
    # pprint(dla_neurons)
    # print("="*80)
    # pprint(dla_unique_neurons)