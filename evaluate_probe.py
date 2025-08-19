# %%
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
from ablate_probe import directional_ablation_single_square, plot_cosine_sim

device = "cuda" if t.cuda.is_available() else "cpu"


# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_size = 10000
custom_functions = [
    othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
]
model = utils.get_model(model_name, device)
train_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size,
    split="test",
    device=device,
)

# %%
# probe_dict = {
#     i: t.load(
#         f"20250802_165058/probe_{i}.pt",
#         map_location=str(device),
#         weights_only="True",
#     )
#     for i in range(model.cfg.n_layers)
# }

# %%
batch_size = 64
keys = [f"blocks.{6}.hook_resid_post"]
focus_cache_tensor = t.empty((10000, 59, model.cfg.d_model), device=device)

for i in range(0, len(train_data["encoded_inputs"]), batch_size):
    batch_inputs = train_data["encoded_inputs"][i : i + batch_size]
    _, cache = model.run_with_cache(
        t.tensor(batch_inputs).to(device),
        names_filter=lambda name: name in keys,
    )
    focus_cache_tensor[i : i + batch_size] = cache[keys[0]]


# %%
probe = t.load(
    f"20250805_135718/probe_{6}.pt",
    map_location=str(device),
    weights_only="True",
)

subspace_sims = plot_cosine_sim(probe)

# %%
neel_utils.plot_board_values(
    subspace_sims[:8],
    width=1000,
    boards_per_row=4,
    title="Angles Between Subspaces",
    board_titles=[f"" for i in range(1, 9)],
)

# %%
focus_cache_tensor = directional_ablation_single_square(
    focus_cache_tensor.double(), probe[:, 0, 3, :]
).float()

# %%
probe_out = einops.einsum(
    focus_cache_tensor,
    probe,
    "game move d_model, d_model row col options -> game move row col options",
)
probe_predictions = probe_out.argmax(dim=-1)
print(probe_predictions.shape)

# %%
probe_board = (probe_out == probe_out.max(dim=-1, keepdim=True).values).long().cpu()

# %%
accuracies = t.all(
    probe_board == train_data["games_batch_to_state_stack_mine_yours_BLRRC"], dim=-1
).float()
square_accuracies = einops.reduce(
    accuracies.float(), "game move row col -> row col", "mean"
)
final_accuracy = square_accuracies.mean(dim=(0, 1))
print(final_accuracy)

# %%
neel_utils.plot_board_values(
    square_accuracies,
    title="Per Square Linear Probe Accuracy (A3 Probe Directions Ablated)",
    zmax=1,
    zmin=0,
)

# %%
import json

with open("20250805_150845/accuracy.json", "w") as f:
    json.dump(final_accuracy.tolist(), f)
# %%
