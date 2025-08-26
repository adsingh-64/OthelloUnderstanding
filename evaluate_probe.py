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
import matplotlib.pyplot as plt
import seaborn as sns
import json

# from ablate_probe import directional_ablation_single_square, plot_cosine_sim

device = "cuda" if t.cuda.is_available() else "cpu"

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_size = 50
custom_functions = [
    othello_utils.games_batch_to_flipped_classifier_input_BLC,
]
model = utils.get_model(model_name, device)
train_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size,
    split="test",
    device=device,
)

# %%
# FLIPPED probe evaluation
flipped_probe = t.load("flipped_probes/resid_5_flipped.pth", map_location=t.device("cpu")).squeeze()

batch_size = 64
keys = [f"blocks.{5}.hook_resid_post"]
focus_cache_tensor = t.empty((dataset_size, 59, model.cfg.d_model), device=device)

for i in range(0, len(train_data["encoded_inputs"]), batch_size):
    batch_inputs = train_data["encoded_inputs"][i : i + batch_size]
    _, cache = model.run_with_cache(
        t.tensor(batch_inputs).to(device),
        names_filter=lambda name: name in keys,
    )
    focus_cache_tensor[i : i + batch_size] = cache[keys[0]]

probe_out = einops.einsum(
    focus_cache_tensor,
    flipped_probe,
    "game move d_model, d_model row col options -> game move row col options",
)

predictions = (probe_out * -1).argmax(dim=-1) # original probe labels were [flipped, not flipped], but data labels are 0 for not_flipped and 1 for flipped

labels = einops.rearrange(train_data["games_batch_to_flipped_classifier_input_BLC"], "batch seq (row col) -> batch seq row col", row = 8)

predictions = predictions.flatten()
labels = labels.flatten()
TP = (predictions & labels).sum()
FP = (predictions & ~labels).sum()
FN = (~predictions & labels).sum()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)
print("Precision", precision), print("Recall", recall), print("F1", F1)

# %%
# BOARD STATE probe evaluation
probe = t.load(
    f"20250805_135718/probe_{5}.pt",
    map_location=str(device),
    weights_only="True",
)

probe_out = einops.einsum(
    focus_cache_tensor,
    probe,
    "game move d_model, d_model row col options -> game move row col options",
)

probe_predictions = probe_out.argmax(dim=-1)
probe_board = (probe_out == probe_out.max(dim=-1, keepdim=True).values).long().cpu()
accuracies = t.all(
    probe_board == train_data["games_batch_to_state_stack_mine_yours_BLRRC"], dim=-1
).float()
square_accuracies = einops.reduce(
    accuracies.float(), "game move row col -> row col", "mean"
)
final_accuracy = square_accuracies.mean(dim=(0, 1))
print(final_accuracy)

neel_utils.plot_board_values(
    square_accuracies,
    title="Per Square Linear Probe Accuracy (A3 Probe Directions Ablated)",
    zmax=1,
    zmin=0,
)

# %%
###
# BOARD STATE + FLIPPED Comparison
###
flipped_probe_dict = {
    i: t.load(f"flippedprobes/resid{i}_flipped.pth", map_location=device).squeeze()
    for i in range(model.cfg.n_layers)
}

board_state_probe_dict = {
    i: t.load(
        f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probelayer{i}.pth",
        map_location=device,
    )["linear_probe"].squeeze()
    for i in range(model.cfg.n_layers)
}

flipped = t.stack(list(flipped_probe_dict.values()))
flipped = flipped[..., 0] - flipped[..., 1]  # empty - not empty

mine_yours = t.stack(list(board_state_probe_dict.values()))
mine_yours = mine_yours[..., 0] - mine_yours[..., 2]  # mine - theirs

print(flipped.shape), print(mine_yours.shape)

# %%
mine_yours_normed = mine_yours / mine_yours.norm(dim=1, keepdim=True)
flipped_normed = flipped / flipped.norm(dim=1, keepdim=True)
mine_yours_cosine_sim = einops.reduce(
    einops.einsum(
        mine_yours_normed,
        mine_yours_normed,
        "layer_1 d_model row col, layer_2 d_model row col -> layer_1 layer_2 row col",
    ),
    "layer_1 layer_2 row col -> layer_1 layer_2",
    "mean",
)
flipped_cosine_sim = einops.reduce(
    einops.einsum(
        flipped_normed,
        flipped_normed,
        "layer_1 d_model row col, layer_2 d_model row col -> layer_1 layer_2 row col",
    ),
    "layer_1 layer_2 row col -> layer_1 layer_2",
    "mean",
)
mine_yours_flipped_cosine_sim = einops.reduce(
    einops.einsum(
        mine_yours_normed,
        flipped_normed,
        "layer_1 d_model row col, layer_2 d_model row col -> layer_1 layer_2 row col",
    ),
    "layer_1 layer_2 row col -> layer_1 layer_2",
    "mean",
)
mine_yours_flipped_cosine_sim_square = einops.reduce(
    einops.rearrange(
        einops.einsum(
            mine_yours_normed,
            flipped_normed,
            "layer d_model row_1 col_1, layer d_model row_2 col_2 -> layer row_1 col_1 row_2 col_2",
        ),
        "layer row_1 col_1 row_2 col_2 -> layer (row_1 col_1) (row_2 col_2)",
    ),
    "layer squares_1 squares_2 -> squares_1 squares_2",
    "mean",
)

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(
    mine_yours_cosine_sim.detach().cpu().numpy(),
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    fmt=".2f",
)
plt.title("Mine - Yours Cosine Similarity Across Layers (Averaged over squares)")
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    flipped_cosine_sim.detach().cpu().numpy(),
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    fmt=".2f",
)
plt.title("Flipped Cosine Similarity Across Layers (Averaged over squares)")
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    mine_yours_flipped_cosine_sim.detach().cpu().numpy(),
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    fmt=".2f",
)
plt.title(
    "Mine - Yours vs Flipped Cosine Similarity Across Layers (Averaged over squares)"
)
plt.xlabel("Layer (mine - yours)")
plt.ylabel("Layer (flipped)")
plt.show()

# %%
# %%
plt.figure(figsize=(12, 10))

# Create chess notation labels A0-H7
chess_labels = []
for row in range(8):
    for col in range(8):
        chess_labels.append(f"{chr(65+col)}{row}")  # A-H for columns, 0-7 for rows

ax = sns.heatmap(
    mine_yours_flipped_cosine_sim_square.detach().cpu().numpy(),
    annot=False,
    cmap="RdBu_r",
    center=0,
    square=True,
    cbar_kws={"label": "Cosine Similarity"},
)

# Set custom tick labels
ax.set_xticks(np.arange(64) + 0.5)
ax.set_yticks(np.arange(64) + 0.5)
ax.set_xticklabels(chess_labels, rotation=90, fontsize=8)
ax.set_yticklabels(chess_labels, rotation=0, fontsize=8)

plt.title("Mine-Yours vs Flipped Cosine Similarity By Square (Averaged over layers)")
plt.xlabel("Square (flipped probe)")
plt.ylabel("Square (mine-yours probe)")
plt.tight_layout()
plt.show()

# %%
W_in = model.W_in.clone().detach().to(device)

with open("flipped_square_ablation_results/G2_dla_rankings.json", "r") as f:
    G2_dla_rankings = json.load(f)

# Compute all cosine similarities, skipping layer 0
all_cosine_sims = []

for item in G2_dla_rankings[:2**10]:  # Get up to 512 neurons (2^9)
    layer, neuron = item["layer"], item["neuron"]
    if layer == 0:
        continue

    encoder = W_in[layer, :, neuron]
    # does this neuron READ from is_flipped
    flipped_probe = flipped_probe_dict[layer - 1]
    flipped_probe = flipped_probe[..., 0] - flipped_probe[..., 1]
    encoder_normed = encoder / encoder.norm()
    flipped_probe_normed = flipped_probe / flipped_probe.norm(dim=0, keepdim=True)
    # max_cosine_sim = einops.reduce(
    #     einops.einsum(
    #         encoder_normed, flipped_probe_normed, "d_model, d_model row col -> row col"
    #     ),
    #     "row col ->",
    #     "max",
    # )
    max_cosine_sim = einops.einsum(
        encoder_normed, flipped_probe_normed[:, 6, 2], "d_model, d_model ->"
    )
    all_cosine_sims.append(max_cosine_sim.item())


# %%
# Create histogram collage - 3x3 grid for 2^2 to 2^10
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(
    "Top G2 is THEIRS neuron encoder + flipped Probe Cosine Similarities", fontsize=16
)

powers = list(range(2, 11))  # 2^2 to 2^10
axes_flat = axes.flatten()

for i, power in enumerate(powers):
    ax = axes_flat[i]
    n_neurons = 2**power

    # Get cosine sims for first n_neurons
    cosine_sims_subset = all_cosine_sims[:n_neurons]

    if cosine_sims_subset:
        ax.hist(
            cosine_sims_subset, bins=20, alpha=0.7, edgecolor="black", color="steelblue"
        )
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"First {n_neurons} neurons (2^{power})", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_sim = np.mean(cosine_sims_subset)
        std_sim = np.std(cosine_sims_subset)
        ax.text(
            0.05,
            0.95,
            f"μ={mean_sim:.3f}\nσ={std_sim:.3f}\nn={len(cosine_sims_subset)}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        # Set consistent x-axis limits for comparison
        ax.set_xlim(-0.2, 0.4)
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

plt.tight_layout()
plt.show()
# %%
