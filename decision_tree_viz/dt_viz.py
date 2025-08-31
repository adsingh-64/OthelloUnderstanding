# %%
# Decision Tree Visualization Tool
# This script allows you to visualize decision trees for specific neurons in specific layers
# Based on the approach shown in dt_tutorial.py but adapted for our neuron decision tree data

import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text, export_graphviz
import numpy as np
from typing import Optional, List
import graphviz 

# %%
# Configuration - modify these values to select neuron and layer
LAYER = 3
NEURON_IDX = 2018
DATA_PATH = 'neuron_simulation/decision_trees_new/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
MAX_DEPTH = None  # Set to an integer to limit tree depth in visualization
SAVE_PATH = None  # Set to a file path to save the visualization

# %%
# Load decision tree data
def load_decision_trees(filepath: str) -> dict:
    """Load decision tree data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

data = load_decision_trees(DATA_PATH)
print(f"Loaded decision tree data from {DATA_PATH}")
print(f"Available layers: {list(data.keys())}")

# Get function name (assuming only one function per layer for now)
function_name = list(data[LAYER].keys())[0]
print(f"Using function: {function_name}")

# %%
# Helper functions
def get_neuron_decision_tree(data: dict, layer: int, neuron_idx: int, function_name: str):
    """Extract the decision tree for a specific neuron."""
    if layer not in data:
        raise ValueError(f"Layer {layer} not found in data. Available layers: {list(data.keys())}")
    
    if function_name not in data[layer]:
        available_funcs = list(data[layer].keys())
        raise ValueError(f"Function {function_name} not found. Available: {available_funcs}")
    
    multi_output_model = data[layer][function_name]['decision_tree']['model']
    
    if neuron_idx >= len(multi_output_model.estimators_):
        raise ValueError(f"Neuron {neuron_idx} not found. Max neuron index: {len(multi_output_model.estimators_) - 1}")
    
    neuron_tree = multi_output_model.estimators_[neuron_idx]
    r2_scores = data[layer][function_name]['decision_tree']['r2']
    neuron_r2 = r2_scores[neuron_idx]
    
    return neuron_tree, neuron_r2

def create_placeholder_feature_names(n_features: int) -> List[str]:
    """Create feature names based on the actual feature structure:
    (192) + (64 + 64 + 5) + (64) = 389 dimensional vector
    - Board state: 192 one-hot (8x8x3 mine/empty/theirs)
    - Last move: 64 one-hot move + 64 pre-occupied + 5 coordinates
    - Flipped moves: 64 binary encoding of flipped squares

    Square notation: A0-H7 where A0 is top-left, H7 is bottom-right
    """
    feature_names = []
    idx = 0

    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8
        col = square_idx % 8
        square = chr(ord("A") + row) + str(col)

        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"{square}_mine")
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_empty")
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_theirs")
            idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord("A") + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1

    # Next 64: Pre-occupied squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord("A") + row) + str(col)
        feature_names.append(f"{square}_pre_occupied")
        idx += 1

    # Next 5: Move coordinates and player info
    coord_names = [
        "move_row",
        "move_col",
        "move_number_just_played",
        "white_played",
        "black_played",
    ]
    for i in range(min(5, n_features - idx)):
        feature_names.append(coord_names[i])
        idx += 1

    # Last 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord("A") + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1

    # Add any remaining features as generic (shouldn't happen with 389 total)
    while idx < n_features:
        feature_names.append(f"Feature_{idx}")
        idx += 1

    return feature_names

# %%
# Visualize the decision tree
def visualize_decision_tree(
    tree_model,
    neuron_idx: int,
    layer: int,
    r2_score: float,
    feature_names: List[str],
    max_depth: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Visualize a decision tree for a specific neuron."""
    plt.figure(figsize=(20, 12))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )

    plt.title(
        f"Decision Tree for Layer {layer}, Neuron {neuron_idx}\nR² Score: {r2_score:.4f}",
        fontsize=16,
        pad=20,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


tree_model, r2_score = get_neuron_decision_tree(data, LAYER, NEURON_IDX, function_name)
n_features = tree_model.n_features_in_
feature_names = create_placeholder_feature_names(n_features)
visualize_decision_tree(
    tree_model,
    NEURON_IDX,
    LAYER,
    r2_score,
    feature_names,
    max_depth=3,
    save_path=SAVE_PATH,
)

# %%
# APPROACH 2: Interactive HTML Visualization using Graphviz
def create_interactive_tree(
    tree_model,
    neuron_idx: int,
    layer: int,
    r2_score: float,
    feature_names: List[str],
    max_depth: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Create an interactive tree visualization using graphviz."""

    # Export tree to graphviz format
    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth,
        proportion=True,
        precision=2,
    )

    # Add title to the dot data string
    dot_data_with_title = dot_data.replace(
        "digraph Tree {",
        f'digraph Tree {{\nlabel="Decision Tree for Layer {layer}, Neuron {neuron_idx}\\nR² Score: {r2_score:.4f}";\nlabelloc="t";\nfontsize=20;\n',
    )

    # Create graph with modified dot data
    graph = graphviz.Source(dot_data_with_title)
    graph.format = "svg"  # SVG format allows zooming without quality loss

    if save_path:
        # Handle save_path properly (check if it's a string path or boolean)
        if isinstance(save_path, str):
            base_path = save_path.replace(".png", "")
        else:
            base_path = f"tree_layer{layer}_neuron{neuron_idx}"

        # Save as SVG (can be opened in browser for interactive viewing)
        graph.render(base_path, format="svg", cleanup=True)
        # Also save as PDF for high quality printing
        graph.render(base_path, format="pdf", cleanup=True)
        print(
            f"Saved interactive SVG and PDF versions as {base_path}.svg and {base_path}.pdf"
        )

    # Display in Jupyter notebook if available
    try:
        from IPython.display import display

        display(graph)
    except ImportError:
        pass

    return graph

# %%
graph = create_interactive_tree(
    tree_model,
    NEURON_IDX,
    LAYER,
    r2_score,
    feature_names,
    max_depth=8,
    save_path=True,
)

# %%
# Get layer statistics
def get_neuron_stats(data: dict, layer: int, function_name: str):
    """Get statistics about neurons in a layer."""
    if layer not in data or function_name not in data[layer]:
        return None

    r2_scores = np.array(data[layer][function_name]['decision_tree']['r2'])

    stats = {
        'total_neurons': len(r2_scores),
        'mean_r2': r2_scores.mean(),
        'median_r2': np.median(r2_scores),
        'max_r2': r2_scores.max(),
        'min_r2': r2_scores.min(),
        'well_explained': (r2_scores > 0.7).sum(),
        'top_10_neurons': np.argsort(r2_scores)[-25:][::-1],  # Top 10 by R²
        'top_10_r2s': r2_scores[np.argsort(r2_scores)[-25:][::-1]]
    }

    return stats

# Show layer statistics
stats = get_neuron_stats(data, LAYER, function_name)
if stats:
    print(f"\nLayer {LAYER} Statistics:")
    print(f"Total neurons: {stats['total_neurons']}")
    print(f"Mean R²: {stats['mean_r2']:.4f}")
    print(f"Median R²: {stats['median_r2']:.4f}")
    print(f"Well explained (R² > 0.7): {stats['well_explained']}")
    print(f"Top 10 neurons by R²:")
    for neuron, r2 in zip(stats['top_10_neurons'], stats['top_10_r2s']):
        print(f"  Neuron {neuron}: R² = {r2:.4f}")

# %%
r2_scores = np.array(data[LAYER][function_name]['decision_tree']['r2'])
mid = np.where((r2_scores < 0.9) & (r2_scores > 0.8))
mid[0][:10]

# %%

# # %%
# # Create histogram of R² scores
# def plot_r2_histogram(data: dict, layer: int, function_name: str, save_path: Optional[str] = None):
#     """Plot histogram of R² scores for all neurons in a layer."""
#     if layer not in data or function_name not in data[layer]:
#         print(f"No data found for layer {layer}, function {function_name}")
#         return

#     r2_scores = np.array(data[layer][function_name]['decision_tree']['r2'])

#     plt.figure(figsize=(10, 6))
#     plt.hist(r2_scores, bins=50, alpha=0.7, edgecolor='black')
#     plt.xlabel('R² Score')
#     plt.ylabel('Number of Neurons')
#     plt.title(f'Distribution of R² Scores for Layer {layer}\n(Mean: {r2_scores.mean():.3f}, Median: {np.median(r2_scores):.3f})')
#     plt.grid(True, alpha=0.3)

#     # Add vertical line for current neuron's R²
#     current_r2 = r2_scores[NEURON_IDX] if NEURON_IDX < len(r2_scores) else None
#     if current_r2 is not None:
#         plt.axvline(current_r2, color='red', linestyle='--', linewidth=2,
#                    label=f'Neuron {NEURON_IDX}: R² = {current_r2:.3f}')
#         plt.legend()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved R² histogram to {save_path}")

#     plt.show()

# # Plot the histogram
# plot_r2_histogram(data, LAYER, function_name)

# # %%
# # Extract the specific neuron's decision tree
# tree_model, r2_score = get_neuron_decision_tree(data, LAYER, NEURON_IDX, function_name)
# print(f"\nRetrieved decision tree for Layer {LAYER}, Neuron {NEURON_IDX}")
# print(f"R² Score: {r2_score:.4f}")

# # Debug: Check if trees are actually different
# multi_output_model = data[LAYER][function_name]['decision_tree']['model']
# print(f"\nDebugging - comparing with other neurons:")
# for test_neuron in [0, 1, 2]:
#     if test_neuron < len(multi_output_model.estimators_):
#         test_tree = multi_output_model.estimators_[test_neuron]
#         print(f"Neuron {test_neuron}: tree depth = {test_tree.get_depth()}, n_nodes = {test_tree.tree_.node_count}")

# # Create feature names specifically for this tree's features
# n_features = tree_model.n_features_in_
# feature_names = create_placeholder_feature_names(n_features)
# print(f"Number of input features for this neuron's tree: {n_features}")

# # Debug: Show tree-specific info
# print(f"This tree depth: {tree_model.get_depth()}")
# print(f"This tree node count: {tree_model.tree_.node_count}")
# print(f"Tree feature importances (top 5): {sorted(enumerate(tree_model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}")

# %%
# Print decision tree rules in text format
def print_tree_rules(tree_model, neuron_idx: int, layer: int, r2_score: float,
                    feature_names: List[str], max_depth: Optional[int] = None):
    """Print the decision tree rules in text format."""
    print(f"\n{'='*60}")
    print(f"Decision Tree Rules for Layer {layer}, Neuron {neuron_idx}")
    print(f"R² Score: {r2_score:.4f}")
    print(f"{'='*60}")

    # Handle None max_depth for export_text (scikit-learn bug workaround)
    export_max_depth = max_depth if max_depth is not None else tree_model.get_depth()

    tree_rules = export_text(
        tree_model,
        feature_names=feature_names,
        max_depth=export_max_depth
    )
    print(tree_rules)

print_tree_rules(tree_model, NEURON_IDX, LAYER, r2_score, feature_names, max_depth=MAX_DEPTH)


# %%
