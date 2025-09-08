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


def intervene(positions: list[Int[Tensor, "n_moves"]], query: list[Condition]) -> dict[str, float]:
    neurons = find_neurons_for_query(query)
    legal_square_id = neel_utils.to_id(query[0].feature_name.split()[0])
    logit_diff = 0
    prob_diff = 0
    for position in tqdm(positions, total = len(positions)):
        position = position.to(device)
        with model.trace(position):
            clean_logits = model.unembed.output[0, -1]
            clean_probs = t.nn.functional.softmax(clean_logits, dim=0)

            clean_logits_square =clean_logits[legal_square_id].save()
            clean_probs_square = clean_probs[legal_square_id].save()

        with model.trace(position):
            for layer in range(1, model.cfg.n_layers - 1):
                if neurons[layer]:
                    model.blocks[layer].mlp.hook_post.output[0, -1, neurons[layer]] = 0

            corrupted_logits = model.unembed.output[0, -1]
            corrupted_probs = t.nn.functional.softmax(corrupted_logits, dim=0)

            corrupted_logits_square = corrupted_logits[legal_square_id].save()
            corrupted_probs_square = corrupted_probs[legal_square_id].save()

        logit_diff += (clean_logits_square - corrupted_logits_square).item()
        prob_diff += (clean_probs_square - corrupted_probs_square).item()

    logit_diff /= len(positions)
    prob_diff /= len(positions)

    return {"logit_diff": logit_diff, "prob_diff": prob_diff}
        

if __name__ == "__main__":
    # Hack: messed up .pkl b/c I put dataclass definition in same pickling script
    sys.modules['__main__'].DecisionTreeResults = DecisionTreeResults

    device = "cuda" if t.cuda.is_available() else "cpu"

    model = load_model()
    data = load_data()

    intervention_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=0),
        Condition(feature_name='D1 mine-theirs', operator='<=', threshold=0),
        Condition(feature_name='E2 mine-theirs', operator='>', threshold=0),
    ] 

    control_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=0),
        Condition(feature_name='C1 mine-theirs', operator='<=', threshold=0),
        Condition(feature_name='C2 mine-theirs', operator='>', threshold=0),
    ] 

    intervention_positions_encoded, intervention_positions_decoded = get_filtered_positions(data, intervention_query, control_query, intervention=True)
    control_positions_encoded, control_positions_decoded = get_filtered_positions(data, intervention_query, control_query, intervention=False)
    # sanity_check(intervention_positions_decoded, control_positions_decoded)

    intervened_metrics = intervene(intervention_positions_encoded, intervention_query[:1])
    print(intervened_metrics)

