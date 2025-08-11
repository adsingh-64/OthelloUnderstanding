# %%
# Plots the three canonical plots for a neuron
# 1) Cosine sim of neuron encoder weights with mine direction of prev residual stream probe
# 2) Cosine sim of neuron encoder weights with blank direction of prev residual stream probe
# 3) Cosine sim of neuron decoder weights with unembedding matrix
import torch as t
import numpy as np
import einops
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
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)

# %%
LAYER = 2
NEURON = 595

# %%
MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]
probe_dict = {
    i: t.load(
        f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth",
        map_location=str(device),
        weights_only="True",
    )["linear_probe"].squeeze()
    for i in range(model.cfg.n_layers)
}
probe = probe_dict[LAYER - 1]
blank_probe = probe[..., 1] - (probe[..., 0] + probe[..., 2]) / 2
my_probe = probe[..., 0] - probe[..., 2]
blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the input weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_in = model.W_in[layer, :, neuron].detach().clone()
    if normalize:
        w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in

def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out[layer, neuron, :].detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=0, keepdim=True)
    return w_out

def calculate_neuron_input_weights(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the input weights for the given neuron, at each square on the board, projected
    along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)

    return einops.einsum(w_in, probe, "d_model, d_model row col -> row col")

def calculate_neuron_output_weights(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_out = get_w_out(model, layer, neuron, normalize=True)

    return einops.einsum(w_out, probe, "d_model, d_model row col -> row col")

def calculate_neuron_unembedding(
    model: HookedTransformer,
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    W_U = model.W_U.detach().clone()
    w_out = get_w_out(model, layer, neuron, normalize=True)
    W_U_normalized = W_U[:, 1:] / W_U[:, 1:].norm(dim=0, keepdim=True)
    cos_sim = einops.einsum(w_out, W_U_normalized, "d_model, d_model d_vocab -> d_vocab")
    cos_sim_rearranged = t.zeros((8, 8), device=device)
    cos_sim_rearranged.flatten()[ALL_SQUARES] = cos_sim
    return cos_sim_rearranged

w_in_blank = calculate_neuron_input_weights(
    model, blank_probe_normalised, LAYER, NEURON
)
w_in_my = calculate_neuron_input_weights(
    model, my_probe_normalised, LAYER, NEURON
)
w_out_unembed = calculate_neuron_unembedding(
    model, LAYER, NEURON
)

neel_utils.plot_board_values(
    t.stack([w_in_blank, w_in_my]),
    title=f"Input weights in terms of the probe for neuron L{LAYER}N{NEURON}",
    board_titles=["Blank In", "My In"],
    width=650,
    height=380,
)

neel_utils.plot_board_values(
    w_out_unembed,
    title=f"Cosine sim of neuron L{LAYER}N{NEURON} with W<sub>U</sub> directions",
    width=450,
    height=380,
)

# %%
probe_writing = probe_dict[LAYER]
my_probe = probe_writing[..., 0] - probe_writing[..., 2]
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
w_out_my = calculate_neuron_output_weights(
    model, my_probe_normalised, LAYER, NEURON
)
neel_utils.plot_board_values(
    w_out_my,
    title=f"Cosine sim of neuron L{LAYER}N{NEURON} with mine - theirs L{LAYER} resid post direction",
    width=450,
    height=380,
)

# %%
