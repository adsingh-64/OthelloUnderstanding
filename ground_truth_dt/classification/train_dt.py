"""
Train decision tree on neuron using board state features
"""

import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight import NNsight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
import neel_utils as neel_utils

import pickle
import gzip
from typing import Tuple, Optional
from jaxtyping import Int, Float, jaxtyped
from typeguard import typechecked
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

from ground_truth_dt.dtypes import BinaryDecisionTreeResults


jaxtyped = partial(jaxtyped, typechecker=typechecked)
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
t.set_grad_enabled(False)
CURRENT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = CURRENT_DIR.parent


MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]


def load_model(
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens",
    device: t.device = device,
) -> NNsight:
    return utils.get_model(model_name, device)


@jaxtyped
def extract_activations_for_layer(
    model: NNsight,
    data: Int[Tensor, "n_games n_total_moves"],
    layer: int,
    batch_size: int = 256,
    device="cuda",
) -> Float[np.ndarray, "n_games n_moves d_mlp"]:
    """
    Extract activations from each layer of the model.

    Args:
        model: OthelloGPT model
        data: train or test encoded inputs

    Returns:
        Dict mapping layer_idx -> activations array (n_games, n_tokens, hidden_dim)
    """
    keys = [f"blocks.{layer}.mlp.hook_post"]
    mlp = np.empty((data.shape[0], 26, model.cfg.d_mlp))

    for i in range(0, len(data), batch_size):
        batch_inputs = data[i : i + batch_size].to(device)
        _, cache = model.run_with_cache(
            batch_inputs,
            names_filter=lambda name: name in keys,
        )

        mlp[i : i + batch_size] = cache[keys[0]][:, 5:31].detach().cpu()

    return mlp


@jaxtyped
def binarize_activations(
    acts: Float[np.ndarray, "n_games n_moves d_mlp"],
    threshold: float = 0.1,
    maxes: Float[np.ndarray, "d_mlp"] | None = None,
) -> tuple[Int[np.ndarray, "n_games n_moves d_mlp"], Float[np.ndarray, "d_mlp"]]:
    if maxes is None:
        maxes = einops.reduce(acts, 'n_games n_moves d_mlp -> d_mlp', 'max')
    thresholds = threshold * maxes
    acts_binarized = (acts > thresholds).astype(np.int64)
    return acts_binarized, maxes


@jaxtyped
def load_data(
    model: NNsight,
    n_train: int = 10000,
    n_test: int = 10000,
    layer: int = 0,
    threshold: float = 0.1,
) -> Tuple[
    Float[np.ndarray, "n_train_samples n_feats"], 
    Int[np.ndarray, "n_train_samples d_mlp"],
    Float[np.ndarray, "n_test_samples n_feats"],
    Int[np.ndarray, "n_test_samples d_mlp"]
]:
    """
    Returns:
        model: OthelloGPT model
        train_data: Training game data
        test_data: Test game data
    """
    train_data = construct_othello_dataset(
        custom_functions=[othello_utils.games_batch_to_board_state_flipped_played_BLC],
        n_inputs=n_train,
        split="train",
        device="cpu",
    )

    train_input_ids = t.tensor(
        train_data["encoded_inputs"], dtype=t.long, device="cpu"
    )

    X_train = train_data["games_batch_to_board_state_flipped_played_BLC"][:, 5:31].numpy()
    y_train = extract_activations_for_layer(model, train_input_ids, layer=layer)
    y_train, train_maxes = binarize_activations(y_train, threshold=threshold)

    
    test_data = construct_othello_dataset(
        custom_functions=[othello_utils.games_batch_to_board_state_flipped_played_BLC],
        n_inputs=n_test,
        split="test",
        device="cpu",
    )

    test_input_ids = t.tensor(
        test_data["encoded_inputs"], dtype=t.long, device="cpu"
    )
    
    X_test = test_data["games_batch_to_board_state_flipped_played_BLC"][:, 5:31].numpy()
    y_test = extract_activations_for_layer(model, test_input_ids, layer=layer)
    y_test, _ = binarize_activations(y_test, maxes=train_maxes, threshold=threshold)

    # Format for ML
    X_train = einops.rearrange(X_train, "n_train_games seq n_feats -> (n_train_games seq) n_feats")
    y_train = einops.rearrange(y_train, "n_train_games seq d_mlp -> (n_train_games seq) d_mlp")
    X_test = einops.rearrange(X_test, "n_test_games seq n_feats -> (n_test_games seq) n_feats")
    y_test = einops.rearrange(y_test, "n_test_games seq d_mlp -> (n_test_games seq) d_mlp")

    return X_train, y_train, X_test, y_test


def train_dt_for_layer(
    model: NNsight,
    X_train: Float[np.ndarray, "n_train_samples n_feats"],
    y_train: Int[np.ndarray, "n_train_samples d_mlp"],
    X_test: Float[np.ndarray, "n_test_samples n_feats"],
    y_test: Int[np.ndarray, "n_test_samples d_mlp"],
    layer: int,
    depth: int = 4,
    n_jobs: int = -1,
) -> list[BinaryDecisionTreeResults]:
    # Convert to plain numpy arrays (joblib no like jaxtyping)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    def worker(neuron):
        tree = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
            min_samples_leaf=50,
            min_samples_split=100,
            criterion="gini",
        )
        
        y_train_neuron = y_train[:, neuron]
        y_test_neuron = y_test[:, neuron]

        # Skip training if a neuron is always off (or always on)
        if len(np.unique(y_train_neuron)) < 2:
            return BinaryDecisionTreeResults(
                layer=layer, neuron=neuron, tree=tree,
                train_precision=0.0, train_recall=0.0, train_accuracy=0.0, train_F1=0.0,
                test_precision=0.0, test_recall=0.0, test_accuracy=0.0, test_F1=0.0
            )
        
        tree.fit(X_train, y_train_neuron)
        
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)
        
        train_precision = precision_score(y_train_neuron, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train_neuron, y_train_pred, zero_division=0)
        train_accuracy = accuracy_score(y_train_neuron, y_train_pred)
        train_f1 = f1_score(y_train_neuron, y_train_pred, zero_division=0)
        
        test_precision = precision_score(y_test_neuron, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test_neuron, y_test_pred, zero_division=0)
        test_accuracy = accuracy_score(y_test_neuron, y_test_pred)
        test_f1 = f1_score(y_test_neuron, y_test_pred, zero_division=0)
        
        return BinaryDecisionTreeResults(
            layer=layer,
            neuron=neuron,
            tree=tree,
            train_precision=train_precision,
            train_recall=train_recall,
            train_accuracy=train_accuracy,
            train_F1=train_f1,
            test_precision=test_precision,
            test_recall=test_recall,
            test_accuracy=test_accuracy,
            test_F1=test_f1,
        )
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(neuron)
        for neuron in tqdm(range(model.cfg.d_mlp), desc=f"Training layer {layer} trees")
    )
    return results


def save_layer_results(results, layer, save_dir):
    """Save all trees in one compressed pickle"""
    save_dir = CURRENT_DIR / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"layer_{layer}_trees.pkl.gz"
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(results, f)
    return save_path


if __name__ == "__main__":
    layer = 0
    depth = 4

    model = load_model()
    X_train, y_train, X_test, y_test = load_data(
        model, 
        n_train=6000, 
        n_test=500,
        layer=layer,
    )

    results = train_dt_for_layer(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        layer=layer,
        depth=depth
    )

    save_path = save_layer_results(results, layer, save_dir="results")
    print(f"Saved results to {save_path}")