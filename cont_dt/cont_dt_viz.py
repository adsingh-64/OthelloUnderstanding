"""
Visualize decision tree trained on Othello neuron activations
"""

import pickle
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from matplotlib.figure import Figure
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from dataclasses import dataclass
from cont_feature_dt import DecisionTreeResults


def get_feature_names():
    """
    Generate feature names for all projections.
    
    Returns list of feature names in order:
    - 64 projections for mine - theirs (all squares)
    - 60 projections for blank (excluding middle 4: D3, D4, E3, E4)
    - 64 projections for flipped (all squares)
    - 60 projections for placed (excluding middle 4)
    """
    feature_names = []
    
    # Helper to get square name from row/col
    def get_square_name(row, col):
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col}"
    
    # 1. Mine - Theirs (all 64 squares)
    for row in range(8):
        for col in range(8):
            square = get_square_name(row, col)
            feature_names.append(f"{square} mine-theirs")
    
    # 2. Blank (60 squares, excluding D3, D4, E3, E4)
    middle_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}  # D3, D4, E3, E4
    for row in range(8):
        for col in range(8):
            if (row, col) not in middle_squares:
                square = get_square_name(row, col)
                feature_names.append(f"{square} blank")
    
    # 3. Flipped (all 64 squares)
    for row in range(8):
        for col in range(8):
            square = get_square_name(row, col)
            feature_names.append(f"{square} flipped")
    
    # 4. Placed (60 squares, excluding middle 4)
    for row in range(8):
        for col in range(8):
            if (row, col) not in middle_squares:
                square = get_square_name(row, col)
                feature_names.append(f"{square} placed")
    
    return feature_names


def load_decision_tree_for_layer(
    layer : int, 
    results_dir: str ="results",
) -> list[DecisionTreeResults]:
    """Load decision trees for layer"""
    results_dir = Path(results_dir)
    file_name = f"layer_{layer}_trees.pkl.gz"
    model_path = results_dir / file_name

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


@dataclass(frozen=True) 
class Condition:
    feature_name: str
    operator: Literal['<=', '>']
    threshold: float


def traverse_tree(tree: DecisionTreeRegressor) -> list[tuple[list[Condition], float]]:
    children_left = tree.children_left
    children_right = tree.children_right
    values = tree.values
    thresholds = tree.thresholsd
    features = tree.features

    feature_names = get_feature_names()

    all_leaf_info = []
    node_id = 0
    if tree.children_left[id] == tree.children_right[id]:
        all_leaf_info.append(node_id)
        return all_leaf_info
    else:
        all_leaf_info.append(
            Condition(
                feature_name=feature_names[features[node_id]],
                operator='<=',
                threshold=thresholds[node_id]
            )
        )
        all_leaf_info.append(
            Condition(
                feature_name=feature_names[features[node_id]],
                operator='>',
                threshold=thresholds[node_id]
            )
        )
        all_leaf_info += # call fucntion on node_id = children_left[node_id] and node_id = children_right[node_id]
    

def convert_to_binary_dt(
    tree: DecisionTreeResults,
):
    regressor = tree.tree



if __name__ == "__main__":
    # Configuration
    layer = 5
    neuron = 766
    
    # Load the decision tree
    layer_5 = load_decision_tree_for_layer(layer=layer)
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Verify we have the right number of features
    L5N766 = layer_5[neuron]
    visualize_decision_tree(L5N766, feature_names)