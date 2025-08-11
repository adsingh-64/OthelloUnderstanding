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
import neel_utils as neel_utils

device = "cuda" if t.cuda.is_available() else "cpu"
MAIN = __name__ == "__main__"

# %%
if MAIN:
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    model = utils.get_model(model_name, device)
    probe_dict = {
        i: t.load(
            f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth",
            map_location=str(device),
            weights_only="True",
        )["linear_probe"].squeeze()
        for i in range(model.cfg.n_layers)
    }

# %%
def get_board_states_and_legal_moves(
    games_square: Int[Tensor, "n_moves"],
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
    n_moves = games_square.shape[0]
    states = t.zeros((n_moves, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((n_moves, 8, 8), dtype=t.int32)
    board = neel_utils.OthelloBoardState()
    for i in range(n_moves):
        board.umpire(games_square[i].item())
        states[i] = t.from_numpy(board.state)
        legal_moves[i].flatten()[board.get_valid_moves()] = 1

    # Convert legal moves to annotation
    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

    return states, legal_moves, legal_moves_annotation

def plot_probe_outputs(
    game: Int[Tensor, "n_moves"], probe_dict: dict[int, Tensor], layer: int, title: str
):
    _, cache = model.run_with_cache(game.to(device))
    linear_probe = probe_dict[layer]
    residual_stream = cache["resid_post", layer][0, -1]
    probe_out = einops.einsum(
        residual_stream,
        linear_probe,
        "d_model, d_model row col options -> options row col",
    )
    neel_utils.plot_board_values(
        probe_out.softmax(dim=0),
        title=title,
        width=900,
        height=400,
        board_titles=["P(Mine)", "P(Empty)", "P(Their's)"],
        # text=BOARD_LABELS_2D,
    )

# %%
if MAIN:
    game_encoded = [20, 19, 41, 21, 27, 34, 13, 33, 29, 12, 26, 43, 38, 14, 10, 48, 42, 18, 28, 32, 49, 22, 4, 15, 44, 50, 37, 31, 39, 2, 55, 57, 51, 6, 17, 24, 40, 47, 45, 46, 54, 52, 23]
    game_decoded = neel_utils.id_to_square(game_encoded)
    probe = probe_dict[4]
    theirs_direction = probe[..., 2] - probe[..., 0]
    g2_theirs = theirs_direction[:, 6, 2]
    with model.trace(t.tensor(game_encoded)):
        hidden_states = model.blocks[4].mlp.hook_post.output[0, -1].save()
        hidden_grads = model.blocks[4].mlp.hook_post.output.grad[0, -1].save()
        resid_post = model.blocks[4].hook_resid_post.output[0, -1]
        metric = einops.einsum(resid_post, g2_theirs, "d_model, d_model ->") 
        metric.backward()
    hidden_attrs = hidden_states * hidden_grads
    top_attrs, top_indices = hidden_attrs.abs().topk(10)
    for neuron_idx in top_indices:
        print(f"Attribution score for L{4}N{neuron_idx}: {hidden_attrs[neuron_idx]}")

# %%
if MAIN:
    with model.trace(t.tensor(game_encoded)):
        hidden_states = model.blocks[0].mlp.hook_post.output[0, -1].save()
        hidden_grads = model.blocks[0].mlp.hook_post.output.grad[0, -1].save()
        metric = model.blocks[4].mlp.hook_post.output[0, -1, 2046]
        metric.backward()
    hidden_attrs = hidden_states * hidden_grads
    top_attrs, top_indices = hidden_attrs.abs().topk(10)
    for neuron_idx in top_indices:
        print(f"Attribution score for L{0}N{neuron_idx}: {hidden_attrs[neuron_idx]}")

# %%
if MAIN:
    with model.trace(t.tensor(game_encoded)):
        hidden_states = model.blocks[0].mlp.hook_post.output[0, -1].save()
        hidden_grads = model.blocks[0].mlp.hook_post.output.grad[0, -1].save()
        metric = model.blocks[1].mlp.hook_post.output[0, -1, 748]
        metric.backward()
    hidden_attrs = hidden_states * hidden_grads
    top_attrs, top_indices = hidden_attrs.abs().topk(10)
    for neuron_idx in top_indices:
        print(f"Attribution score for L{0}N{neuron_idx}: {hidden_attrs[neuron_idx]}")

# %%
if MAIN:
    focus_states, focus_legal_moves, focus_legal_moves_annotation = (
        get_board_states_and_legal_moves(t.tensor(game_decoded))
    )
    neel_utils.plot_board_values(
        focus_states.squeeze(),
        title="Board states",
        width=1000,
        height = 2500,
        boards_per_row=5,
        board_titles=[
            f"Move {i}, {'black' if i % 2 == 1 else 'white'} to play" for i in range(t.tensor(game_decoded).shape[0])
        ],
        text=np.where(to_numpy(focus_legal_moves.squeeze()), "o", "").tolist(),
    )

# %%
if MAIN:
    for layer in range(model.cfg.n_layers):
        plot_probe_outputs(
            t.tensor(game_encoded),
            probe_dict,
            layer,
            title=f"Layer {layer} probe outputs",
        )

# %%
