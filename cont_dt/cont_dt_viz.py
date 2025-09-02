"""
Visualize decision tree trained on Othello neuron activations
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path


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


def load_decision_tree(layer, neuron, depth, results_dir="results"):
    """Load saved decision tree model and metrics."""
    results_dir = Path(results_dir)
    filename_base = f"layer{layer}_neuron{neuron}_depth{depth}"
    
    # Load model
    model_path = results_dir / f"{filename_base}_model.pkl"
    with open(model_path, 'rb') as f:
        tree = pickle.load(f)
    
    # Load metrics
    metrics_path = results_dir / f"{filename_base}_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return tree, metrics


def visualize_decision_tree(tree, metrics, feature_names, save_path=None, figsize=(20, 10)):
    """
    Create a visualization of the decision tree with proper feature labels.
    
    Args:
        tree: Trained DecisionTreeRegressor
        metrics: Dictionary of tree metrics
        feature_names: List of feature names
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create title with metrics
    title = (f"Decision Tree: Layer {metrics['layer']}, Neuron {metrics['neuron']}\n"
             f"Test R² = {metrics['test_R2']:.3f}, Train R² = {metrics['train_R2']:.3f}\n"
             f"Depth = {metrics['tree_info']['max_depth']}, "
             f"Leaves = {metrics['tree_info']['n_leaves']}")
    
    # Plot the tree
    plot_tree(tree, 
              feature_names=feature_names,
              filled=True,
              rounded=True,
              fontsize=16,
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


def print_tree_rules(tree, feature_names, metrics):
    """Print human-readable rules from the decision tree."""
    
    print(f"\n{'='*60}")
    print(f"Decision Tree Rules for Layer {metrics['layer']}, Neuron {metrics['neuron']}")
    print(f"{'='*60}\n")
    
    def get_rules(tree, feature_names):
        """Extract rules from tree structure."""
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, depth, parent_rule="Root"):
            indent = "  " * depth
            
            if tree_.feature[node] != -2:  # Not a leaf
                name = feature_name[node]
                threshold = tree_.threshold[node]
                value = tree_.value[node][0][0]
                n_samples = tree_.n_node_samples[node]
                
                print(f"{indent}if {name} <= {threshold:.3f}:")
                print(f"{indent}  (samples: {n_samples}, value: {value:.3f})")
                recurse(tree_.children_left[node], depth + 1, f"{name} <= {threshold:.3f}")
                
                print(f"{indent}else:  # {name} > {threshold:.3f}")
                print(f"{indent}  (samples: {n_samples}, value: {value:.3f})")
                recurse(tree_.children_right[node], depth + 1, f"{name} > {threshold:.3f}")
            else:  # Leaf node
                value = tree_.value[node][0][0]
                n_samples = tree_.n_node_samples[node]
                print(f"{indent}-> Prediction: {value:.3f} (samples: {n_samples})")
        
        recurse(0, 0)
    
    get_rules(tree, feature_names)
    
    # Print feature importance
    print(f"\n{'='*60}")
    print("Feature Importance (top 10):")
    print(f"{'='*60}\n")
    
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices):
        if importances[idx] > 0:
            print(f"{i+1:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")


if __name__ == "__main__":
    # Configuration
    layer = 5
    neuron = 1393
    depth = 3
    results_dir = 'results'
    save_fig = True  # Set to True to save the figure
    print_rules = True  # Set to True to print text rules
    figsize = (20, 10)
    
    # Load the decision tree
    tree, metrics = load_decision_tree(layer, neuron, depth, results_dir)
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Verify we have the right number of features
    expected_features = 64 + 60 + 64 + 60  # 248 total
    assert len(feature_names) == expected_features, \
        f"Expected {expected_features} features, got {len(feature_names)}"
    
    print(f"\nLoaded decision tree for Layer {layer}, Neuron {neuron}")
    print(f"Tree depth: {metrics['tree_info']['max_depth']}")
    print(f"Number of leaves: {metrics['tree_info']['n_leaves']}")
    print(f"Test R²: {metrics['test_R2']:.4f}")
    print(f"Train R²: {metrics['train_R2']:.4f}")
    
    # Visualize the tree
    save_path = f"tree_layer{layer}_neuron{neuron}.png" if save_fig else None
    visualize_decision_tree(tree, metrics, feature_names, save_path, figsize)
    
    # Optionally print rules
    if print_rules:
        print_tree_rules(tree, feature_names, metrics)