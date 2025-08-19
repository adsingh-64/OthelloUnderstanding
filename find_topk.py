# %%
import torch as t
import numpy as np
import einops
import nnsight
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy
from torch import Tensor
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from tqdm.notebook import tqdm
import neel_utils as neel_utils

device = "cuda" if t.cuda.is_available() else "cpu"

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_size = 500
custom_functions = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
]
model = utils.get_model(model_name, device)
data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size,
    split="test",
    device=device,
)

# %%
def get_board_states_and_legal_moves(
    games_square: Int[Tensor, "n_games n_moves"],
) -> tuple[
    Int[Tensor, "n_games n_moves rows cols"],
    Int[Tensor, "n_games n_moves rows cols"],
    list,
]:
    """
    Returns the following:
        states:                 (n_games, n_moves, 8, 8): tensor of board states after each move
        legal_moves:            (n_games, n_moves, 8, 8): tensor of 1s for legal moves, 0s for illegal moves
        legal_moves_annotation: (n_games, n_moves, 8, 8): list containing strings of "o" for legal moves (for plotting)
    """
    # Create tensors to store the board state & legal moves
    n_games, n_moves = games_square.shape
    states = t.zeros((n_games, 60, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((n_games, 60, 8, 8), dtype=t.int32)

    # Loop over each game, populating state & legal moves tensors after each move
    for n in range(n_games):
        board = neel_utils.OthelloBoardState()
        for i in range(n_moves):
            board.umpire(games_square[n, i].item())
            states[n, i] = t.from_numpy(board.state)
            legal_moves[n, i].flatten()[board.get_valid_moves()] = 1

    # Convert legal moves to annotation
    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

    return states, legal_moves, legal_moves_annotation

# %%
encoded_inputs = t.tensor(data["encoded_inputs"]).long()
decoded_inputs = t.tensor(data["decoded_inputs"]).long()
focus_states, focus_legal_moves, focus_legal_moves_annotation = (
    get_board_states_and_legal_moves(decoded_inputs)
)

# %%
neel_utils.plot_board_values(
    focus_states[0, 0:10],
    title="Board states",
    width=1000,
    boards_per_row=5,
    board_titles=[
        f"Move {i}, {'black' if i % 2 == 1 else 'white'} to play" for i in range(0, 10)
    ],
    text=np.where(to_numpy(focus_legal_moves[0, 0:10]), "o", "").tolist(),
)

# %%
features = data["games_batch_to_input_tokens_flipped_bs_classifier_input_BLC"]
mask = features[..., -14] == 1 # G2 was flipped
filtered_games_encoded = []
filtered_games_decoded = []
for game_idx in range(mask.shape[0]):
    # Find indices where mask is 1 for this game
    indices = t.where(mask[game_idx])[0]
    for idx in indices:
        filtered_game_encoded = encoded_inputs[game_idx, : idx + 1]
        filtered_games_encoded.append(filtered_game_encoded)
        filtered_game_decoded = decoded_inputs[game_idx, :idx+1]
        filtered_games_decoded.append(filtered_game_decoded)

# %%
# Check G2 was flipped
for sequence in filtered_games_decoded[:10]:
    focus_states, focus_legal_moves, focus_legal_moves_annotation = (
        get_board_states_and_legal_moves(sequence.unsqueeze(dim = 0))
    )
    n_moves = sequence.shape[0]

    # Only plot the board states for actual moves (last 2 moves of the sequence)
    neel_utils.plot_board_values(
        focus_states[0, max(0, n_moves - 2) : n_moves],  # Use actual move indices
        title="Board states",
        width=1000,
        boards_per_row=5,
    )

# %%
probe_dict = {
    i: t.load(
        f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth",
        map_location=str(device),
        weights_only="True",
    )["linear_probe"].squeeze()
    for i in range(model.cfg.n_layers)
}
# probe_dict_G2 = {
#     k: v[:, 6, 2, :] for k, v in probe_dict.items()
# }
# G2_theirs_mine = {
#     k: v[:, -1] - v[:, 0] for k, v in probe_dict_G2.items()
# }
layer = 5
probe_layer = probe_dict[layer]
probe_G2 = probe_layer[:, 6, 2, :]
G2_theirs_mine = probe_G2[..., 2] - probe_G2[..., 0]

# %%
attrs = t.zeros((layer + 1, model.cfg.d_model))
for sequence in tqdm(filtered_games_encoded, total = len(filtered_games_encoded)):
    with model.trace(sequence):
        for i in range(layer + 1):
            acts = model.blocks[i].output[0, -1].save()
            grads = model.blocks[i].output.grad[0, -1].save()
            attrs[i] += acts * grads
        resid = model.blocks[5].output[0, -1].save()
        metric = einops.einsum(resid, G2_theirs_mine, "d_model, d_model ->")
        metric.backward()
    break
#attrs /= len(filtered_games_encoded)

# %%
attrs_flattened = attrs.flatten()
values, neuron_idx = attrs_flattened.topk(10)
neuron_idx_by_layer = [(idx // model.cfg.d_model, idx % model.cfg.d_model) for idx in neuron_idx]

# %%
neuron_idx_by_layer

# %%
t.save(attrs, "G2_flipped_attributions.pt")
# %%
