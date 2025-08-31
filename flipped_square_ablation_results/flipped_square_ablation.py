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
import json

device = "cuda" if t.cuda.is_available() else "cpu"


# %%
def load_data_and_model():
    """Load model and construct filtered dataset"""
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

    # Filter for G2-flipped examples
    encoded_inputs = t.tensor(data["encoded_inputs"]).long()
    decoded_inputs = t.tensor(data["decoded_inputs"]).long()
    features = data["games_batch_to_input_tokens_flipped_bs_classifier_input_BLC"]
    mask = features[..., -14] == 1  # G2 was flipped

    filtered_games_encoded = []
    filtered_games_decoded = []
    for game_idx in range(mask.shape[0]):
        indices = t.where(mask[game_idx])[0]
        for idx in indices:
            filtered_game_encoded = encoded_inputs[game_idx, : idx + 1]
            filtered_games_encoded.append(filtered_game_encoded)
            filtered_game_decoded = decoded_inputs[game_idx, : idx + 1]
            filtered_games_decoded.append(filtered_game_decoded)

    return model, data, filtered_games_encoded, filtered_games_decoded


# %%
def get_activations_with_ablation(
    model, filtered_games_encoded, ablation_dict=None, batch_size=32
):
    """Extract layer 5 activations with optional neuron ablations"""
    filtered_activations = []

    for batch_start in range(0, len(filtered_games_encoded), batch_size):
        batch_sequences = filtered_games_encoded[batch_start : batch_start + batch_size]

        # Pad sequences to same length for batching
        max_len = max(seq.shape[0] for seq in batch_sequences)
        padded_batch = []
        final_token_indices = []

        for seq in batch_sequences:
            final_token_indices.append(seq.shape[0] - 1)
            if seq.shape[0] < max_len:
                padded_seq = t.cat(
                    [seq, t.zeros(max_len - seq.shape[0], dtype=seq.dtype)]
                )
            else:
                padded_seq = seq
            padded_batch.append(padded_seq)

        batch_tensor = t.stack(padded_batch).to(device)

        with model.trace(batch_tensor):
            # Apply ablations if provided
            if ablation_dict:
                for layer, neuron_indices in ablation_dict.items():
                    if layer < 6:  # Only layers 0-5 exist
                        # Zero out specific neurons at final token positions for each sequence
                        for batch_idx, final_token_idx in enumerate(
                            final_token_indices
                        ):
                            model.blocks[layer].mlp.hook_post.output[
                                batch_idx, final_token_idx, neuron_indices
                            ] = 0

            # Save layer 5 output
            layer5_output = model.blocks[5].output.save()

        # Extract final token activation for each sequence
        for j, (seq, final_token_idx) in enumerate(
            zip(batch_sequences, final_token_indices)
        ):
            activation = layer5_output[j, final_token_idx]  # Shape: (d_model,)
            filtered_activations.append(activation)

    return t.stack(filtered_activations)


# %%
def evaluate_g2_probe(activations, g2_probe, true_g2_states):
    """Apply G2 probe and calculate accuracy"""
    g2_probe_out = einops.einsum(
        activations,
        g2_probe,
        "n_examples d_model, d_model options -> n_examples options",
    )
    g2_predictions = g2_probe_out.argmax(dim=-1)
    accuracy = (g2_predictions == true_g2_states).float().mean()
    return accuracy, g2_predictions


# %%
def ablate_top_k_neurons(
    k, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
):
    """Ablate top k neurons and return G2 probe accuracy"""
    print(f"  Ablating top {k} neurons...")

    # Get neurons to ablate
    neurons_to_ablate = ranked_neurons[:k]

    # Group by layer for efficient ablation
    ablation_dict = {}
    for layer, neuron_idx in neurons_to_ablate:
        if layer not in ablation_dict:
            ablation_dict[layer] = []
        ablation_dict[layer].append(neuron_idx)

    # Get activations with ablations
    ablated_activations = get_activations_with_ablation(
        model, filtered_games_encoded, ablation_dict
    )

    # Evaluate probe accuracy
    accuracy, _ = evaluate_g2_probe(ablated_activations, g2_probe, true_g2_states)

    print(f"    Accuracy: {accuracy:.4f}")
    return accuracy.item()


# %%
# Load data and model
model, data, filtered_games_encoded, filtered_games_decoded = load_data_and_model()
print(f"Loaded {len(filtered_games_encoded)} filtered sequences")

# %%
# Load G2 probe and set up ground truth
probe = t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{5}.pth",
    map_location=str(device),
    weights_only="True",
)["linear_probe"].squeeze()

g2_probe = probe[:, 6, 2, :]  # Shape: (d_model, 3) for [empty, mine, yours]
true_g2_states = t.tensor(
    [2] * len(filtered_games_encoded), device=device
)  # All should be "theirs"

print(f"G2 probe shape: {g2_probe.shape}")
print(f"True G2 states shape: {true_g2_states.shape}")

# %%
# Baseline accuracy (no ablations)
print("Computing baseline accuracy...")
baseline_activations = get_activations_with_ablation(model, filtered_games_encoded)
baseline_accuracy, baseline_predictions = evaluate_g2_probe(
    baseline_activations, g2_probe, true_g2_states
)
print(f"Baseline G2 Probe Accuracy: {baseline_accuracy:.4f}")

# Check prediction distribution
unique_preds, counts = t.unique(baseline_predictions, return_counts=True)
labels = ["empty", "mine", "theirs"]
print(f"\nBaseline prediction distribution:")
for pred, count in zip(unique_preds, counts):
    print(
        f"  {labels[pred.item()]}: {count.item()} ({100*count.item()/len(baseline_predictions):.1f}%)"
    )

# %%
# Load ranked neurons for ablation
with open("G2_dt_rankings.json", "r") as f:
    dt_rankings = json.load(f)

ranked_neurons = [(neuron["layer"], neuron["neuron_idx"]) for neuron in dt_rankings]
print(f"\nLoaded {len(ranked_neurons)} ranked neurons")


# %%
# Ablation experiment functions for each power of 2
def ablate_2_power_0():
    """Ablate 2^0 = 1 neuron"""
    return ablate_top_k_neurons(
        2**0, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_1():
    """Ablate 2^1 = 2 neurons"""
    return ablate_top_k_neurons(
        2**1, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_2():
    """Ablate 2^2 = 4 neurons"""
    return ablate_top_k_neurons(
        2**2, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_3():
    """Ablate 2^3 = 8 neurons"""
    return ablate_top_k_neurons(
        2**3, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_4():
    """Ablate 2^4 = 16 neurons"""
    return ablate_top_k_neurons(
        2**4, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_5():
    """Ablate 2^5 = 32 neurons"""
    return ablate_top_k_neurons(
        2**5, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_6():
    """Ablate 2^6 = 64 neurons"""
    return ablate_top_k_neurons(
        2**6, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_7():
    """Ablate 2^7 = 128 neurons"""
    return ablate_top_k_neurons(
        2**7, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_8():
    """Ablate 2^8 = 256 neurons"""
    return ablate_top_k_neurons(
        2**8, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_9():
    """Ablate 2^9 = 512 neurons"""
    return ablate_top_k_neurons(
        2**9, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_10():
    """Ablate 2^10 = 1024 neurons"""
    return ablate_top_k_neurons(
        2**10, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_11():
    """Ablate 2^11 = 2048 neurons"""
    return ablate_top_k_neurons(
        2**11, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_12():
    """Ablate 2^12 = 4096 neurons"""
    return ablate_top_k_neurons(
        2**12, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_13():
    """Ablate 2^13 = 8192 neurons"""
    return ablate_top_k_neurons(
        2**13, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_2_power_14():
    """Ablate 2^14 = 16384 neurons"""
    return ablate_top_k_neurons(
        2**14, ranked_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )


def ablate_random_k_neurons(
    k, all_neurons, model, filtered_games_encoded, g2_probe, true_g2_states, seed=42
):
    """Ablate k randomly selected neurons and return G2 probe accuracy"""
    print(f"  Ablating {k} random neurons (seed={seed})...")

    # Set seed for reproducibility
    np.random.seed(seed)

    # Randomly sample neurons
    random_indices = np.random.choice(
        len(all_neurons), size=min(k, len(all_neurons)), replace=False
    )
    neurons_to_ablate = [all_neurons[i] for i in random_indices]

    # Group by layer for efficient ablation
    ablation_dict = {}
    for layer, neuron_idx in neurons_to_ablate:
        if layer not in ablation_dict:
            ablation_dict[layer] = []
        ablation_dict[layer].append(neuron_idx)

    # Get activations with ablations
    ablated_activations = get_activations_with_ablation(
        model, filtered_games_encoded, ablation_dict
    )

    # Evaluate probe accuracy
    accuracy, _ = evaluate_g2_probe(ablated_activations, g2_probe, true_g2_states)

    print(f"    Random accuracy: {accuracy:.4f}")
    return accuracy.item()


# Generate all possible neurons (6 layers × 2048 neurons each)
def get_all_neurons():
    """Generate list of all (layer, neuron_idx) pairs"""
    all_neurons = []
    for layer in range(6):  # Layers 0-5
        for neuron_idx in range(2048):  # 2048 neurons per layer
            all_neurons.append((layer, neuron_idx))
    return all_neurons


# %%
# Run all ablation experiments
print("\nRunning ablation experiments...")

ablation_functions = [
    (0, ablate_2_power_0),
    (1, ablate_2_power_1),
    (2, ablate_2_power_2),
    (3, ablate_2_power_3),
    (4, ablate_2_power_4),
    (5, ablate_2_power_5),
    (6, ablate_2_power_6),
    (7, ablate_2_power_7),
    (8, ablate_2_power_8),
    (9, ablate_2_power_9),
    (10, ablate_2_power_10),
    (11, ablate_2_power_11),
    (12, ablate_2_power_12),
    (13, ablate_2_power_13),
    (14, ablate_2_power_14),
]

ablation_results = [
    {"n_ablated": 0, "power_of_2": -1, "accuracy": baseline_accuracy.item()}
]

for power, ablation_func in ablation_functions:
    n_ablated = 2**power
    if n_ablated > len(ranked_neurons):
        break

    print(f"\n2^{power} = {n_ablated} neurons:")
    accuracy = ablation_func()

    ablation_results.append(
        {"n_ablated": n_ablated, "power_of_2": power, "accuracy": accuracy}
    )

# %%
# Save results and display summary
with open("g2_ablation_results_dt.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

print(f"\nSaved ablation results to g2_ablation_results.json")

print("\nAblation Summary:")
print("=" * 50)
print(f"{'N Ablated':<12} {'Power of 2':<12} {'Accuracy':<12} {'Δ from baseline':<15}")
print("-" * 50)
for result in ablation_results:
    delta = result["accuracy"] - baseline_accuracy.item()
    power_str = "baseline" if result["power_of_2"] == -1 else str(result["power_of_2"])
    print(
        f"{result['n_ablated']:<12} {power_str:<12} {result['accuracy']:<12.4f} {delta:<+15.4f}"
    )

# %%
# Plot ablation results
import matplotlib.pyplot as plt

n_ablated = [r["n_ablated"] for r in ablation_results]
accuracies = [r["accuracy"] for r in ablation_results]

plt.figure(figsize=(10, 6))
plt.semilogx(n_ablated, accuracies, "bo-", linewidth=2, markersize=6)
plt.axhline(
    y=baseline_accuracy.item(),
    color="r",
    linestyle="--",
    alpha=0.7,
    label=f"Baseline: {baseline_accuracy:.4f}",
)
plt.xlabel("Number of Neurons Ablated (log scale)")
plt.ylabel("G2 Probe Accuracy")
plt.title(
    "G2 Probe Accuracy vs Number of Ablated Neurons\n(Neurons ranked by decision tree F1)"
)
plt.grid(True, alpha=0.3)
plt.legend()

# Add power of 2 annotations
for i, (n, acc) in enumerate(zip(n_ablated, accuracies)):
    if n > 0:  # Skip baseline
        power = int(np.log2(n))
        plt.annotate(
            f"2^{power}",
            (n, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            alpha=0.7,
        )

plt.tight_layout()
plt.savefig("g2_ablation_curve_dt.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Random ablation baseline experiment
print("\n" + "=" * 60)
print("Running random ablation baseline experiments...")
print("=" * 60)

all_neurons = get_all_neurons()
print(f"Total neurons in model: {len(all_neurons)}")

random_ablation_results = []

for power in range(15):  # 2^0 to 2^14
    n_ablated = 2**power
    if n_ablated > len(ranked_neurons):
        break

    print(f"\n2^{power} = {n_ablated} random neurons:")
    random_accuracy = ablate_random_k_neurons(
        n_ablated, all_neurons, model, filtered_games_encoded, g2_probe, true_g2_states
    )

    random_ablation_results.append(
        {"n_ablated": n_ablated, "power_of_2": power, "accuracy": random_accuracy}
    )

with open("g2_ablation_results_random.json", "w") as f:
    json.dump(random_ablation_results, f, indent=2)

# %%
# Plot ablation results
import matplotlib.pyplot as plt

n_ablated = [r["n_ablated"] for r in random_ablation_results]
accuracies = [r["accuracy"] for r in random_ablation_results]

plt.figure(figsize=(10, 6))
plt.semilogx(n_ablated, accuracies, "bo-", linewidth=2, markersize=6)
plt.axhline(
    y=baseline_accuracy.item(),
    color="r",
    linestyle="--",
    alpha=0.7,
    label=f"Baseline: {baseline_accuracy:.4f}",
)
plt.xlabel("Number of Neurons Ablated (log scale)")
plt.ylabel("G2 Probe Accuracy")
plt.title(
    "G2 Probe Accuracy vs Number of Ablated Neurons\n(Neurons ranked by direct probe attribution)"
)
plt.grid(True, alpha=0.3)
plt.legend()

# Add power of 2 annotations
for i, (n, acc) in enumerate(zip(n_ablated, accuracies)):
    if n > 0:  # Skip baseline
        power = int(np.log2(n))
        plt.annotate(
            f"2^{power}",
            (n, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            alpha=0.7,
        )

plt.tight_layout()
plt.savefig("g2_ablation_curve_random.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
