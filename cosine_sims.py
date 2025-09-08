"""
Computes a neuron's cosine sims with various probe directions
"""

import torch as t
from torch import Tensor
import einops
from nnsight.models import NNsightModel

import circuits.utils as utils
import neel_utils as neel_utils

from typing import Tuple
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from functools import partial


jaxtyped = partial(jaxtyped, typechecker=typechecked)
device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)


MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]


def load_model(
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens", 
    device = device,
) -> NNsightModel:
    return utils.get_model(model_name, device)


@jaxtyped
def load_board_state_probes(
    model: NNsightModel, 
    path: str = "linear_probes/resid_{layer}_board_state.pth", 
    device = device,
) -> dict[int, Float[Tensor, "d_model row col mode"]]:
    return {
        layer: t.load(path.format(layer=layer), map_location=device)["linear_probe"].squeeze()
        for layer in range(model.cfg.n_layers)
    }


@jaxtyped
def load_flipped_probes(
    model: NNsightModel, 
    path: str = "flipped_probes/resid_{layer}_flipped.pth",
) -> dict[int, Float[Tensor, "d_model row col mode"]]:
    return {
        layer: t.load(path.format(layer=layer), map_location=device).squeeze()
        for layer in range(model.cfg.n_layers)
    }


@jaxtyped
def load_played_probes(
    model,
    path: str = "played_probes/resid_{layer}_played.pth",
) -> dict[int, Float[Tensor, "d_model row col"]]:
    return {
        layer: t.load(path.format(layer=layer), map_location=device)
        for layer in range(model.cfg.n_layers)
    }


@jaxtyped
def get_mine_theirs_normed(
    board_state_probes: dict[int, Float[Tensor, "d_model row col mode"]],
    normalize: bool = True,
) -> dict[int, Float[Tensor, "d_model row col"]]:
    mine_theirs = {
        layer: probe[..., 0] - probe[..., 2]
        for layer, probe in board_state_probes.items()
    }

    if normalize:
        mine_theirs = {
            layer: probe / probe.norm(dim=0, keepdim=True)
            for layer, probe in mine_theirs.items()
        }

    return mine_theirs


@jaxtyped
def get_blank_normed(
    board_state_probes: dict[int, Float[Tensor, "d_model row col mode"]],
    normalize: bool = True,
) -> dict[int, Float[Tensor, "d_model row col"]]:
    blank = {
        layer: probe[..., 1] - (probe[..., 0] + probe[..., 2])/2
        for layer, probe in board_state_probes.items()
    }

    if normalize:
        blank = {
            layer: probe / probe.norm(dim=0, keepdim=True)
            for layer, probe in blank.items()
        }

    return blank


@jaxtyped
def get_flipped_normed(
    flipped_probes: dict[int, Float[Tensor, "d_model row col mode"]],
    normalize: bool = True,
) -> dict[int, Float[Tensor, "d_model row col"]]:
    flipped = {
        layer: probe[..., 0] - probe[..., 1]
        for layer, probe in flipped_probes.items()
    }

    if normalize:
        flipped = {
            layer: probe / probe.norm(dim=0, keepdim=True)
            for layer, probe in flipped.items()
        }

    return flipped


@jaxtyped
def get_played_normed(
    played_probes: dict[int, Float[Tensor, "d_model row col"]],
    normalize: bool = True,
) -> dict[int, Float[Tensor, "d_model row col"]]:
    played = {
        layer: probe
        for layer, probe in played_probes.items()
    }

    if normalize:
        played = {
            layer: probe / probe.norm(dim=0, keepdim=True)
            for layer, probe in played_probes.items()
        }

    return played


@jaxtyped
def get_w_in(
    model: NNsightModel,
    layer: int,
    neuron: int,
) -> Float[Tensor, "d_model"]:
    """
    Returns the input weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_in = model.W_in[layer, :, neuron].detach().clone()
    w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in


@jaxtyped
def get_w_out(
    model: NNsightModel,
    layer: int,
    neuron: int,
) -> Float[Tensor, "d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out[layer, neuron, :].detach().clone()
    w_out /= w_out.norm(dim=0, keepdim=True)
    return w_out


@jaxtyped
def calculate_neuron_input_weights(
    model: NNsightModel,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
    mask_center: bool = False,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the input weights for the given neuron, at each square on the board, projected
    along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_in = get_w_in(model, layer, neuron)
    out = einops.einsum(w_in, probe, "d_model, d_model row col -> row col")
    if mask_center:
        out.flatten()[MIDDLE_SQUARES] = 0
    return out


@jaxtyped
def calculate_neuron_output_weights(
    model: NNsightModel,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
    mask_center: bool = False,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_out = get_w_out(model, layer, neuron)
    out = einops.einsum(w_out, probe, "d_model, d_model row col -> row col")
    if mask_center:
        out.flatten()[MIDDLE_SQUARES] = 0
    return out


@jaxtyped
def calculate_neuron_inputs(
    model: NNsightModel,
    mine_theirs_normed: dict[int, Float[Tensor, "d_model row col"]],
    blank_normed: dict[int, Float[Tensor, "d_model row col"]],
    flipped_normed: dict[int, Float[Tensor, "d_model row col"]],
    played_normed: dict[int, Float[Tensor, "d_model row col"]],
    layer: int,
    neuron: int,
) -> Tuple[Float[Tensor, "rows cols"], Float[Tensor, "rows cols"], Float[Tensor, "rows cols"], Float[Tensor, "rows cols"]]:
    # layer - 1 because reading
    mine_theirs_probe = mine_theirs_normed[layer - 1]
    blank_probe = blank_normed[layer - 1]
    flipped_probe = flipped_normed[layer - 1]
    played_probe = played_normed[layer - 1]

    mine_theirs = calculate_neuron_input_weights(model, mine_theirs_probe, layer, neuron)
    blank = calculate_neuron_input_weights(model, blank_probe, layer, neuron, mask_center=True)
    flipped = calculate_neuron_input_weights(model, flipped_probe, layer, neuron)
    played = calculate_neuron_input_weights(model, played_probe, layer, neuron, mask_center=True)

    return mine_theirs, blank, flipped, played


@jaxtyped
def calculate_neuron_outputs(
    model: NNsightModel,
    mine_theirs_normed: dict[int, Float[Tensor, "d_model row col"]],
    blank_normed: dict[int, Float[Tensor, "d_model row col"]],
    flipped_normed: dict[int, Float[Tensor, "d_model row col"]],
    played_normed: dict[int, Float[Tensor, "d_model row col"]],
    layer: int,
    neuron: int,
) -> Tuple[Float[Tensor, "rows cols"], Float[Tensor, "rows cols"], Float[Tensor, "rows cols"], Float[Tensor, "rows cols"]]:
    # layer - 1 because reading
    mine_theirs_probe = mine_theirs_normed[layer]
    blank_probe = blank_normed[layer]
    flipped_probe = flipped_normed[layer]
    played_probe = played_normed[layer]

    mine_theirs = calculate_neuron_output_weights(model, mine_theirs_probe, layer, neuron)
    blank = calculate_neuron_output_weights(model, blank_probe, layer, neuron, mask_center=True)
    flipped = calculate_neuron_output_weights(model, flipped_probe, layer, neuron)
    played = calculate_neuron_output_weights(model, played_probe, layer, neuron, mask_center=True)

    return mine_theirs, blank, flipped, played


@jaxtyped
def calculate_neuron_unembedding(
    model: NNsightModel,
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    W_U = model.W_U.detach().clone()
    w_out = get_w_out(model, layer, neuron)
    W_U_normalized = W_U[:, 1:] / W_U[:, 1:].norm(dim=0, keepdim=True)
    cos_sim = einops.einsum(
        w_out, W_U_normalized, "d_model, d_model d_vocab -> d_vocab"
    )
    cos_sim_rearranged = t.zeros((8, 8), device=device)
    cos_sim_rearranged.flatten()[ALL_SQUARES] = cos_sim
    return cos_sim_rearranged


if __name__ == "__main__":
    model = load_model()
    board_state_probes = load_board_state_probes(model)
    flipped_probes = load_flipped_probes(model)
    played_probes = load_played_probes(model)

    mine_theirs_normed = get_mine_theirs_normed(board_state_probes)
    blank_normed = get_blank_normed(board_state_probes)
    flipped_normed = get_flipped_normed(flipped_probes)
    played_normed = get_played_normed(played_probes)

    layer = 1
    neuron = 98

    mine_theirs_in, blank_in, flipped_in, played_in = calculate_neuron_inputs(
        model, 
        mine_theirs_normed, 
        blank_normed, 
        flipped_normed, 
        played_normed, 
        layer, 
        neuron
    )

    mine_theirs_out, blank_out, flipped_out, played_out = calculate_neuron_outputs(
        model, 
        mine_theirs_normed, 
        blank_normed, 
        flipped_normed, 
        played_normed, 
        layer, 
        neuron
    )

    unembedding = calculate_neuron_unembedding(
        model,
        layer,
        neuron
    )

    neel_utils.plot_board_values(
        t.stack([mine_theirs_in, blank_in, flipped_in, played_in]),
        title=f"L{layer}N{neuron} reading",
        board_titles=["Mine/Theirs In", "Blank In", "Flipped In", "Played In"],
        boards_per_row=2,
        width=600,
        height=800,
    )

    neel_utils.plot_board_values(
        t.stack([mine_theirs_out, blank_out, flipped_out, played_out]),
        title=f"L{layer}N{neuron} writing",
        board_titles=["Mine/Theirs Out", "Blank Out", "Flipped Out", "Played Out"],
        boards_per_row=2,
        width=600,
        height=800,
    )

    neel_utils.plot_board_values(
        unembedding.unsqueeze(dim = 0),
        title = f"L{layer}N{neuron} logit lens",
        board_titles=[""],
        boards_per_row=1,
        width=400,
        height=400,
    )