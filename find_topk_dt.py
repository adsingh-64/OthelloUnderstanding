# %%
import torch as t
import numpy as np
import pickle
import einops
from tqdm.notebook import tqdm
from IPython.display import HTML, display

device = "cuda" if t.cuda.is_available() else "cpu"

# %%
# Load the filtered games from find_topk.py results
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset

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
# Get the same filtered games where G2 is flipped
features = data["games_batch_to_input_tokens_flipped_bs_classifier_input_BLC"]
mask = features[..., -14] == 1  # G2 was flipped

# Collect feature vectors for filtered games (last position of each filtered sequence)
filtered_features = []
for game_idx in range(mask.shape[0]):
    indices = t.where(mask[game_idx])[0]
    for idx in indices:
        # Get the feature vector at the position where G2 was flipped
        feature_vec = features[game_idx, idx].cpu().numpy()
        filtered_features.append(feature_vec)

filtered_features = np.array(filtered_features)
print(f"Number of filtered sequences: {len(filtered_features)}")
print(f"Feature vector shape: {filtered_features[0].shape}")

# %%
# Load decision trees for all layers
decision_tree_files = [
    "decision_trees_binary/decision_trees_mlp_neuron_60.pkl",
    "decision_trees_binary/decision_trees_mlp_neuron_600.pkl", 
    "decision_trees_binary/decision_trees_mlp_neuron_6000.pkl"
]

# We'll use the one with most training data (6000)
dt_file = decision_tree_files[2]
print(f"Loading decision trees from: {dt_file}")

with open(dt_file, "rb") as f:
    decision_trees = pickle.load(f)

# %%
# Function to evaluate recall for each neuron across all layers
def evaluate_neuron_recall(decision_trees, filtered_features, layers=range(6)):
    """
    For each neuron, evaluate its decision tree's recall on filtered games.
    Recall = (# of times DT predicts active when G2 flipped) / (total # of G2 flipped examples)
    """
    neuron_recalls = {}
    
    for layer in layers:
        if layer not in decision_trees:
            continue
            
        func_name = 'games_batch_to_input_tokens_flipped_bs_classifier_input_BLC'
        if func_name not in decision_trees[layer]:
            continue
            
        if 'binary_decision_tree' not in decision_trees[layer][func_name]:
            continue
            
        binary_dt_data = decision_trees[layer][func_name]['binary_decision_tree']
        
        # The model is stored under the 'model' key
        if 'model' not in binary_dt_data:
            continue
            
        multi_output_model = binary_dt_data['model']
        
        # Get predictions for all neurons in this layer
        predictions = multi_output_model.predict(filtered_features)
        
        # Calculate recall for each neuron
        n_neurons = predictions.shape[1]
        for neuron_idx in range(n_neurons):
            # Recall = how often the DT is active (predicts 1) on G2-flipped examples
            neuron_predictions = predictions[:, neuron_idx]
            recall = np.mean(neuron_predictions)
            neuron_recalls[(layer, neuron_idx)] = recall
    
    return neuron_recalls

# %%
# Calculate recalls for all neurons
print("Evaluating decision tree recalls on filtered games...")
neuron_recalls = evaluate_neuron_recall(decision_trees, filtered_features)

# %%
# Sort neurons by recall (descending)
sorted_neurons = sorted(neuron_recalls.items(), key=lambda x: x[1], reverse=True)

# Display top neurons by recall
print("\nTop 20 neurons by recall (DT activation rate on G2-flipped examples):")
print("="*60)
print(f"{'Rank':<6} {'Layer':<7} {'Neuron':<8} {'Recall':<10}")
print("-"*60)

for rank, ((layer, neuron), recall) in enumerate(sorted_neurons[:20], 1):
    print(f"{rank:<6} {layer:<7} {neuron:<8} {recall:<10.4f}")

# %%
# Compare with attribution-based ranking from find_topk.py
print("\n\nComparison with attribution-based ranking:")
print("="*60)

# Load the attribution results (you computed these in find_topk.py)
layer = 5
probe_dict = {
    i: t.load(
        f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth",
        map_location=str(device),
        weights_only="True",
    )["linear_probe"].squeeze()
    for i in range(model.cfg.n_layers)
}
probe_layer = probe_dict[layer]
probe_G2 = probe_layer[:, 6, 2, :]
G2_theirs_mine = probe_G2[..., 2] - probe_G2[..., 0]

# Recompute attributions for comparison
encoded_inputs = t.tensor(data["encoded_inputs"]).long()
attrs = t.zeros((layer + 1, model.cfg.d_model))

filtered_games_encoded = []
for game_idx in range(mask.shape[0]):
    indices = t.where(mask[game_idx])[0]
    for idx in indices:
        filtered_game_encoded = encoded_inputs[game_idx, : idx + 1]
        filtered_games_encoded.append(filtered_game_encoded)

print(f"Computing attributions for {len(filtered_games_encoded)} sequences...")
for sequence in tqdm(filtered_games_encoded, total=len(filtered_games_encoded)):
    with model.trace(sequence):
        for i in range(layer + 1):
            acts = model.blocks[i].output[0, -1].save()
            grads = model.blocks[i].output.grad[0, -1].save()
            attrs[i] += acts * grads
        resid = model.blocks[5].output[0, -1].save()
        metric = einops.einsum(resid, G2_theirs_mine, "d_model, d_model ->")
        metric.backward()
attrs /= len(filtered_games_encoded)

# Get top neurons by attribution
attrs_flattened = attrs.flatten()
values, neuron_idx = attrs_flattened.topk(20)
neuron_idx_by_layer = [(idx // model.cfg.d_model, idx % model.cfg.d_model) for idx in neuron_idx]

# %%
# Display comparison
print("\nTop 20 neurons by Attribution vs Decision Tree Recall:")
print("="*80)
print(f"{'Rank':<6} {'Attribution-based':<25} {'DT Recall-based':<25} {'DT Recall Value':<15}")
print(f"{'':6} {'(Layer, Neuron)':<25} {'(Layer, Neuron)':<25}")
print("-"*80)

for rank in range(20):
    # Attribution-based
    attr_layer, attr_neuron = neuron_idx_by_layer[rank]
    attr_value = values[rank].item()
    
    # DT recall-based
    dt_layer, dt_neuron = sorted_neurons[rank][0]
    dt_recall = sorted_neurons[rank][1]
    
    print(f"{rank+1:<6} ({attr_layer}, {attr_neuron:<4})<25 ({dt_layer}, {dt_neuron:<4})<25 {dt_recall:<15.4f}")

# %%
# Check overlap between top neurons from both methods
top_k = 50
attr_top_neurons = set(neuron_idx_by_layer[:top_k])
dt_top_neurons = set([n[0] for n in sorted_neurons[:top_k]])

overlap = attr_top_neurons.intersection(dt_top_neurons)
print(f"\n\nOverlap Analysis (Top {top_k} neurons):")
print(f"Attribution-based top {top_k}: {len(attr_top_neurons)} neurons")
print(f"DT recall-based top {top_k}: {len(dt_top_neurons)} neurons")
print(f"Overlap: {len(overlap)} neurons ({100*len(overlap)/top_k:.1f}%)")

if overlap:
    print(f"\nNeurons in both top {top_k} lists:")
    for layer, neuron in sorted(overlap):
        attr_rank = neuron_idx_by_layer.index((layer, neuron)) + 1
        dt_rank = next(i for i, (n, _) in enumerate(sorted_neurons) if n == (layer, neuron)) + 1
        dt_recall = neuron_recalls[(layer, neuron)]
        print(f"  Layer {layer}, Neuron {neuron}: Attribution rank #{attr_rank}, DT rank #{dt_rank}, Recall={dt_recall:.4f}")

# %%