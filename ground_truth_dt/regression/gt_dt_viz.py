"""
Visualize decision tree trained on Othello neuron activations
"""

import pickle
import json
import gzip
from functools import lru_cache
from pprint import pprint
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, TypeAlias
from matplotlib.figure import Figure
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from skimage.filters import threshold_otsu
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from ground_truth_dt.dtypes import DecisionTreeResults


FILE_PATH = Path(__file__).resolve()
PARENT_DIR = FILE_PATH.parent
RESULTS_DIR = PARENT_DIR / "results"


N_LAYERS = 8
D_MLP = 2048


def get_feature_names(n_features: int = 320) -> list[str]:
    feature_names = []
    idx = 0
    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + row) + str(col)
        
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
    
    # Next 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    return feature_names


def load_decision_tree_for_layer(
    layer : int, 
) -> list[DecisionTreeResults]:
    """Load decision trees for layer"""
    file_name = f"layer_{layer}_trees.pkl.gz"
    model_path = RESULTS_DIR / file_name

    with gzip.open(model_path, 'rb') as f:
        trees = pickle.load(f)
    
    return trees


def visualize_decision_tree(
    tree: DecisionTreeResults, 
    feature_names: list[str], 
    save_path: str | None = None, 
    figsize: tuple[float, float] = (20, 10),
 ) -> Figure:
    """
    Create a visualization of the decision tree with proper feature labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create title with metrics
    title = (f"Decision Tree: Layer {tree.layer}, Neuron {tree.neuron}\n"
             f"Test R² = {tree.test_R2:.3f}\n"
             f"Depth = {tree.tree.max_depth}, "
             f"Leaves = {tree.tree.tree_.n_leaves}")
    
    # Plot the tree
    model = tree.tree
    plot_tree(model, 
              feature_names=feature_names,
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax,
              impurity=False,  # Don't show impurity (MSE) values
              precision=2)  # Round values to 2 decimal places
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return fig


def traverse_tree(tree: DecisionTreeRegressor) -> list[tuple[set[str], float]]:
    tree_obj = tree.tree_

    children_left = tree_obj.children_left
    children_right = tree_obj.children_right
    values = tree_obj.value
    features = tree_obj.feature
    
    feature_names = get_feature_names()

    def _traverse(node_id: int, path: set[str]) -> list[tuple[set[str], float]]:
        # base case: if leaf node
        if children_left[node_id] == children_right[node_id]:
            return [(path, values[node_id][0][0])]

        # left implies NOT feature
        left_path = path | {f"NOT {feature_names[features[node_id]]}"}

        # right implies features
        right_path = path | {f"{feature_names[features[node_id]]}"}

        return _traverse(children_left[node_id], left_path) + _traverse(children_right[node_id], right_path)

    return _traverse(0, set())


def otsu(leaf_nodes: list[tuple[set[str], float]]) -> list[set[str]]:
    leaf_values = np.array([value for _, value in leaf_nodes])
    on_off_threshold = threshold_otsu(leaf_values)

    on_paths = [
        path for path, value in leaf_nodes
        if value > on_off_threshold
    ]

    return on_paths


def process_neuron(tree: DecisionTreeRegressor) -> list[set[str]]:
    """Takes in a neuron's tree, returns a list of on decision paths
    representing OR-of-ANDs structure"""
    leaf_nodes = traverse_tree(tree)
    on_paths = otsu(leaf_nodes)
    return on_paths


def check_neuron(tree: DecisionTreeResults, query: set[str]) -> bool:
    """Takes a neuron's tree and checks if any of its ON conditions
    guarantee the path condition"""
    decision_tree = tree.tree
    on_paths = process_neuron(decision_tree)

    # Is query contained in any of the on paths?
    return any(query <= path for path in on_paths)


def check_layer(trees: list[DecisionTreeResults], query: set[str]) -> list[int]:
    """Return neuron ids satisfying query"""
    return [neuron_id for neuron_id, neuron_tree in enumerate(trees) if check_neuron(neuron_tree, query)]


def check_model(trees: dict[int, list[set[str]]], query: set[str]) -> dict[int: list[int]]:
    """Returns all neurons satisfying query"""
    return {layer: check_layer(layer_trees, query) for layer, layer_trees in trees.items()}


@lru_cache(maxsize=1)
def load_all_trees(
    n_layers: int = 8,
) -> dict[int, list[DecisionTreeResults]]:
    """Loads and caches all decision trees from disk."""
    print("Loading all decision trees from disk... (this will happen only once)")
    return {
        layer: load_decision_tree_for_layer(layer=layer)
        for layer in range(n_layers)
    }


def find_neurons_for_query(query: set[str]) -> dict[int, list[int]]:
    """
    Finds neurons that satisfy the query using a cached tree loader.
    """
    all_trees = load_all_trees() 
    return check_model(all_trees, query)


if __name__ == "__main__":
    query = {"C0_empty", "NOT D1_empty"}
    result = find_neurons_for_query(query)
    pprint(result)