"""
Train per-square is_placed probes for each layer of OthelloGPT.

For each layer and each square position (64 total), trains a separate 
logistic regression probe to detect if that square was just played.
Evaluates F1 score per layer by aggregating predictions across all squares.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple
import torch as t
from dataclasses import dataclass
from transformer_lens import HookedTransformer

# ============ Data Structures ============


@dataclass
class ProbeResults:
    """Results for a single square's probe."""

    layer: int
    square: int
    probe: LogisticRegression
    train_f1: float
    test_f1: float
    predictions: np.ndarray


# ============ Data Loading ============


def load_model_and_data(n_train: int = 100000, n_test: int = 10000, device: str = "cuda") -> Tuple[HookedTransformer, List, List, List, List]:
    """
    Returns:
        model: OthelloGPT model
        train_data: Training game data
        train_labels: Training is_placed labels (n_samples, seq_len, 64)
        test_data: Test game data
        test_labels: Test is_placed labels
    """
    pass


# ============ Feature Extraction ============


def extract_activations_by_layer(
    model: HookedTransformer, data : List
) -> Dict[int, np.ndarray]:
    """
    Extract activations from each layer of the model.

    Args:
        model: OthelloGPT model
        data_loader: Data to process

    Returns:
        Dict mapping layer_idx -> activations array (n_games, n_tokens, hidden_dim)
    """
    pass


def get_is_placed_labels(data: List) -> np.ndarray:
    """
    Get is_placed labels for each square at each position.

    Returns:
        Array of shape (n_games, n_tokens, 64) with binary labels
    """
    pass


# ============ Probe Training ============


def prepare_probe_data(
    activations: np.ndarray, labels: np.ndarray, square_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training a probe for a specific square.

    Args:
        activations: Layer activations (n_samples, n_tokens, hidden_dim)
        labels: Is_placed labels (n_samples, n_tokens, 64)
        square_idx: Which square (0-63) to train probe for

    Returns:
        X: Flattened activations (n_games * n_tokens, hidden_dim)
        y: Binary labels for this square (n_games * n_tokens,)
    """
    pass


def train_single_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> ProbeResults:
    """
    Train and evaluate a single probe.

    Returns:
        ProbeResults with trained probe and metrics
    """
    pass


def train_probes_for_layer(
    layer_idx: int,
    layer_activations: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> List[ProbeResults]:
    """
    Train probes for all 64 squares at a specific layer.

    Args:
        layer_idx: Which layer we're training probes for
        layer_activations: Activations from this layer
        train_labels: Training labels for all squares
        test_labels: Test labels for all squares

    Returns:
        List of 64 ProbeResults, one per square
    """
    pass


# ============ Evaluation ============


def aggregate_square_predictions(probe_results: List[ProbeResults]) -> np.ndarray:
    """
    Combine predictions from all 64 square probes.

    Args:
        probe_results: List of 64 ProbeResults for one layer

    Returns:
        Aggregated predictions array (n_samples * n_tokens, 64)
    """
    pass


def compute_layer_f1(
    probe_results: List[ProbeResults], true_labels: np.ndarray
) -> float:
    """
    Compute F1 score for a layer by aggregating across all squares.

    Args:
        probe_results: List of 64 ProbeResults for one layer
        true_labels: Ground truth labels (n_samples, n_tokens, 64)

    Returns:
        F1 score for this layer
    """
    pass


# ============ Main Pipeline ============


def train_all_probes(
    model: HookedTransformer,
    train_loader: List,
    test_loader: List,
    layers_to_probe: List[int] = None,
) -> Dict[int, List[ProbeResults]]:
    """
    Train probes for all layers and squares.

    Args:
        model: OthelloGPT model
        train_loader: Training data
        test_loader: Test data
        layers_to_probe: Which layers to probe (None = all layers)

    Returns:
        Dict mapping layer_idx -> list of 64 ProbeResults
    """
    pass


def evaluate_by_layer(
    all_results: Dict[int, List[ProbeResults]], test_labels: np.ndarray
) -> Dict[int, float]:
    """
    Compute F1 scores per layer.

    Returns:
        Dict mapping layer_idx -> F1 score
    """
    pass


# ============ Visualization ============


def plot_probe_performance(
    layer_f1_scores: Dict[int, float], square_f1_by_layer: Dict[int, np.ndarray]
) -> None:
    """
    Visualize probe performance across layers and squares.

    Args:
        layer_f1_scores: Overall F1 per layer
        square_f1_by_layer: Per-square F1 for each layer (layer -> 64 scores)
    """
    pass


def create_heatmap_by_layer(all_results: Dict[int, List[ProbeResults]]) -> None:
    """
    Create 8x8 heatmaps showing per-square F1 for each layer.
    """
    pass


# ============ Main ============


def main():
    """Run complete probe training pipeline."""

    # Load everything
    print("Loading model and data...")
    model, train_loader, test_loader = load_model_and_data()

    # Train probes for all layers and squares
    print("Training probes for all layers and squares...")
    all_results = train_all_probes(model, train_loader, test_loader)

    # Evaluate performance by layer
    print("Evaluating by layer...")
    test_labels = get_is_placed_labels(test_loader)
    layer_f1_scores = evaluate_by_layer(all_results, test_labels)

    # Display results
    print("\nF1 Scores by Layer:")
    for layer_idx, f1 in layer_f1_scores.items():
        print(f"  Layer {layer_idx}: {f1:.3f}")

    # Visualize
    plot_probe_performance(layer_f1_scores, square_f1_by_layer)

    return all_results


def get_is_placed_labels(data_loader: DataLoader) -> np.ndarray:
    """
    Get is_placed labels for each square at each position.

    Returns:
        Array of shape (n_samples, n_tokens, 64) with binary labels
    """
    # Will use the existing utils you mentioned
    pass


# ============ Probe Training ============


def prepare_probe_data(
    activations: np.ndarray, labels: np.ndarray, square_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training a probe for a specific square.

    Args:
        activations: Layer activations (n_samples, n_tokens, hidden_dim)
        labels: Is_placed labels (n_samples, n_tokens, 64)
        square_idx: Which square (0-63) to train probe for

    Returns:
        X: Flattened activations (n_samples * n_tokens, hidden_dim)
        y: Binary labels for this square (n_samples * n_tokens,)
    """
    pass


def train_single_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> ProbeResults:
    """
    Train and evaluate a single probe.

    Returns:
        ProbeResults with trained probe and metrics
    """
    pass


def train_probes_for_layer(
    layer_idx: int,
    layer_activations: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> List[ProbeResults]:
    """
    Train probes for all 64 squares at a specific layer.

    Args:
        layer_idx: Which layer we're training probes for
        layer_activations: Activations from this layer
        train_labels: Training labels for all squares
        test_labels: Test labels for all squares

    Returns:
        List of 64 ProbeResults, one per square
    """
    pass


# ============ Evaluation ============


def aggregate_square_predictions(probe_results: List[ProbeResults]) -> np.ndarray:
    """
    Combine predictions from all 64 square probes.

    Args:
        probe_results: List of 64 ProbeResults for one layer

    Returns:
        Aggregated predictions array (n_samples * n_tokens, 64)
    """
    pass


def compute_layer_f1(
    probe_results: List[ProbeResults], true_labels: np.ndarray
) -> float:
    """
    Compute F1 score for a layer by aggregating across all squares.

    Args:
        probe_results: List of 64 ProbeResults for one layer
        true_labels: Ground truth labels (n_samples, n_tokens, 64)

    Returns:
        F1 score for this layer
    """
    pass


# ============ Main Pipeline ============


def train_all_probes(
    model: OthelloGPT,
    train_loader: DataLoader,
    test_loader: DataLoader,
    layers_to_probe: List[int] = None,
) -> Dict[int, List[ProbeResults]]:
    """
    Train probes for all layers and squares.

    Args:
        model: OthelloGPT model
        train_loader: Training data
        test_loader: Test data
        layers_to_probe: Which layers to probe (None = all layers)

    Returns:
        Dict mapping layer_idx -> list of 64 ProbeResults
    """
    pass


def evaluate_by_layer(
    all_results: Dict[int, List[ProbeResults]], test_labels: np.ndarray
) -> Dict[int, float]:
    """
    Compute F1 scores per layer.

    Returns:
        Dict mapping layer_idx -> F1 score
    """
    pass


# ============ Visualization ============


def plot_probe_performance(
    layer_f1_scores: Dict[int, float], square_f1_by_layer: Dict[int, np.ndarray]
) -> None:
    """
    Visualize probe performance across layers and squares.

    Args:
        layer_f1_scores: Overall F1 per layer
        square_f1_by_layer: Per-square F1 for each layer (layer -> 64 scores)
    """
    pass


def create_heatmap_by_layer(all_results: Dict[int, List[ProbeResults]]) -> None:
    """
    Create 8x8 heatmaps showing per-square F1 for each layer.
    """
    pass


# ============ Main ============


def main():
    """Run complete probe training pipeline."""

    # Load everything
    print("Loading model and data...")
    model, train_loader, test_loader = load_model_and_data()

    # Train probes for all layers and squares
    print("Training probes for all layers and squares...")
    all_results = train_all_probes(model, train_loader, test_loader)

    # Evaluate performance by layer
    print("Evaluating by layer...")
    test_labels = get_is_placed_labels(test_loader)
    layer_f1_scores = evaluate_by_layer(all_results, test_labels)

    # Display results
    print("\nF1 Scores by Layer:")
    for layer_idx, f1 in layer_f1_scores.items():
        print(f"  Layer {layer_idx}: {f1:.3f}")

    # Visualize
    plot_probe_performance(layer_f1_scores, square_f1_by_layer)

    return all_results


if __name__ == "__main__":
    results = main()
