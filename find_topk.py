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
import json

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
mask = features[..., -14] == 1  # G2 was flipped
filtered_games_encoded = []
filtered_games_decoded = []
for game_idx in range(mask.shape[0]):
    # Find indices where mask is 1 for this game
    indices = t.where(mask[game_idx])[0]
    for idx in indices:
        filtered_game_encoded = encoded_inputs[game_idx, : idx + 1]
        filtered_games_encoded.append(filtered_game_encoded)
        filtered_game_decoded = decoded_inputs[game_idx, : idx + 1]
        filtered_games_decoded.append(filtered_game_decoded)

# %%
# Check G2 was flipped
for sequence in filtered_games_decoded[:10]:
    n_moves = sequence.shape[0]
    focus_states, focus_legal_moves, focus_legal_moves_annotation = (
        get_board_states_and_legal_moves(sequence.unsqueeze(dim=0))
    )

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
probe_dict_G2 = {k: v[:, 6, 2, :] for k, v in probe_dict.items()}
G2_theirs_mine = {k: v[:, 2] - v[:, 0] for k, v in probe_dict_G2.items()}
layer = 5

# %%
acts = t.zeros((layer + 1, model.cfg.d_mlp)).to(device)
for sequence in tqdm(filtered_games_encoded, total=len(filtered_games_encoded)):
    with model.trace(sequence):
        for i in range(layer + 1):
            neuron_acts = model.blocks[i].mlp.hook_post.output[0, -1].save()
            acts[i] += neuron_acts

# %%
W_out = model.W_out[: layer + 1].clone().detach()  # [6, 2048, 512]
all_G2 = t.stack([G2_theirs_mine[i] for i in range(layer + 1)], dim=0)  # [6, 512]
neuron_decoders = acts.unsqueeze(dim=-1) * W_out
dla = einops.einsum(
    neuron_decoders, all_G2, "layer neuron d_model, layer d_model -> layer neuron"
)
print(dla.shape)

# %%
for layer_idx in range(layer + 1):
    top_neurons = dla[layer_idx].topk(k=5)  # Top 20 neurons
    print(f"Layer {layer_idx} top neurons: {top_neurons.indices.tolist()}")

# %%
import json

# Get sorted indices by DLA (descending order)
dla_flat = dla.flatten()
sorted_indices = dla_flat.argsort(descending=True)

# Convert to (layer, neuron_idx) tuples with DLA values
sorted_neurons = []
for flat_idx in sorted_indices:
    layer = flat_idx.item() // model.cfg.d_mlp
    neuron = flat_idx.item() % model.cfg.d_mlp
    dla_value = dla_flat[flat_idx].item()

    sorted_neurons.append({"layer": layer, "neuron": neuron, "dla": dla_value})

# Save to JSON
with open("G2_dla_rankings.json", "w") as f:
    json.dump(sorted_neurons, f, indent=2)


# %%
def find_intersection_topk(file1, file2, k=50):
    """
    Find intersection of top k neurons from two ranking files.

    Args:
        file1: Path to first ranking JSON file
        file2: Path to second ranking JSON file
        k: Number of top elements to consider

    Returns:
        int: Number of neurons in intersection
    """
    # Load the ranking files
    with open(file1, "r") as f:
        rankings1 = json.load(f)

    with open(file2, "r") as f:
        rankings2 = json.load(f)

    # Extract top k (layer, neuron) tuples
    top_k1 = [(item["layer"], item["neuron"]) for item in rankings1[:k]]
    top_k2 = [(item["layer"], item["neuron_idx"]) for item in rankings2[:k]]

    # Find intersection
    intersection = set(top_k1) & set(top_k2)

    return len(intersection)


# Example usage
k = 2000
intersection_count = find_intersection_topk(
    "G2_dla_rankings.json", "G2_dt_rankings.json", k=k
)
print(f"Intersection of top {k} neurons: {intersection_count}")

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Calculate intersections for k = 2^0, 2^1, ..., 2^14
k_values = [2**i for i in range(15)]  # 2^0 to 2^14
intersections = []

for k in k_values:
    intersection_count = find_intersection_topk(
        "G2_dla_rankings.json", "G2_dt_rankings.json", k=k
    )
    intersections.append(intersection_count)
    print(f"k={k}: {intersection_count} intersections")

# Plot the results
plt.figure(figsize=(10, 6))
plt.loglog(k_values, intersections, "bo-", linewidth=2, markersize=8)
plt.xlabel("k (top k neurons)")
plt.ylabel("Number of intersections")
plt.title("Intersection of Top k Neurons Between DLA and DT Rankings")
plt.grid(True, alpha=0.3)

# Format x-axis to show powers of 2
ax = plt.gca()
ax.set_xticks(k_values)
ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(15)])

plt.show()

# Also plot the intersection ratio
plt.figure(figsize=(10, 6))
intersection_ratios = [intersections[i] / k_values[i] for i in range(len(k_values))]
plt.semilogx(k_values, intersection_ratios, "ro-", linewidth=2, markersize=8)
plt.xlabel("k (top k neurons)")
plt.ylabel("Intersection ratio (intersections / k)")
plt.title("Intersection Ratio of Top k Neurons Between DLA and DT Rankings")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Format x-axis to show powers of 2
ax = plt.gca()
ax.set_xticks(k_values)
ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(15)])

plt.show()
# %%
