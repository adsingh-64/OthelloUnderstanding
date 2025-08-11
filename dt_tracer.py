# %%
import pickle
import numpy as np
import circuits.othello_utils as othello_utils
import neel_utils as neel_utils
from dt_viz import create_placeholder_feature_names

# %%
# Load your tree
with open(
    "neuron_simulation/decision_trees_new/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl",
    "rb",
) as f:
    data = pickle.load(f)


# %%
# Get the specific tree for layer 3, neuron 1199
layer = 3
neuron_idx = 1199
function_name = list(data[layer].keys())[0]
tree_model = data[layer][function_name]["decision_tree"]["model"].estimators_[
    neuron_idx
]

# %%
game_encoded = [20, 19, 41, 21, 27, 34, 13, 33, 29, 12, 26, 43, 38, 14, 10, 48, 42, 18, 28, 32, 49, 22, 4, 15, 44, 50, 37, 31, 39, 2, 55, 57, 51, 6, 17, 24, 40, 47, 45, 46, 54, 52, 23]
game_decoded = [neel_utils.id_to_square(game_encoded)]
game_data = othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC(game_decoded)

# %%
# Now let's trace a single game
# Assume you have one game's features (shape: 389,)
game_features = game_data[0, -1].numpy()  # Replace with your actual game data

# Method 1: Find which LEAF node this game ends up in
leaf_node_id = tree_model.apply(game_features.reshape(1, -1))[0]
print(f"This game ends in leaf node: {leaf_node_id}")

# Method 2: Get ALL nodes this game passes through
path = tree_model.decision_path(game_features.reshape(1, -1))
nodes_visited = path.indices  # These are the node IDs in order
print(f"This game passes through nodes: {nodes_visited}")

# Create feature names for better readability
n_features = tree_model.n_features_in_
feature_names = create_placeholder_feature_names(n_features)

# Method 3: See the actual decisions made at each step
for node_id in nodes_visited:
    if tree_model.tree_.feature[node_id] != -2:  # -2 means leaf node
        feature_idx = tree_model.tree_.feature[node_id]
        threshold = tree_model.tree_.threshold[node_id]
        game_value = game_features[feature_idx]
        feature_name = feature_names[feature_idx]

        if game_value <= threshold:
            direction = "LEFT"
        else:
            direction = "RIGHT"

        print(
            f"Node {node_id}: {feature_name} = {game_value:.3f} vs threshold {threshold:.3f} → go {direction}"
        )
    else:
        activation = tree_model.tree_.value[node_id][0][0]
        print(f"Node {node_id}: LEAF with activation = {activation:.4f}")


# %%
node_id = tree_model.apply(game_features.reshape(1, -1))
activation = tree_model.tree_.value[node_id][0][0]
print(activation)

# %%
