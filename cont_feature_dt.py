"""
Train decision tree on neuron using continuous features
of board state projection onto probe directions
"""

import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight.models import NNsightModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
import neel_utils as neel_utils
from cosine_sims import (
    load_board_state_probes,
    load_flipped_probes,
    load_played_probes,
    get_mine_theirs_normed,
    get_blank_normed,
    get_flipped_normed,
    get_played_normed,
)

import json
from typing import Tuple
from jaxtyping import Int, Float, jaxtyped
from typeguard import typechecked
from functools import partial
from dataclasses import dataclass
from joblib import Parallel, delayed


jaxtyped = partial(jaxtyped, typechecker=typechecked)
device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)


@dataclass
class DecisionTreeResults:
    """Results for a single square's decision tree."""

    layer: int
    neuron: int
    tree: DecisionTreeRegressor
    train_R2: float
    train_MSE: float
    test_R2: float
    test_MSE: float


def load_model(
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens",
    device=device,
) -> NNsightModel:
    return utils.get_model(model_name, device)


@jaxtyped
def load_data(
    n_train: int = 10000,
    n_test: int = 10000,
) -> Tuple[Int[Tensor, "n_train_games n_moves"], Int[Tensor, "n_test_games n_moves"]]:
    """
    Returns:
        model: OthelloGPT model
        train_data: Training game data
        test_data: Test game data
    """
    train_data = construct_othello_dataset(
        custom_functions=[],
        n_inputs=n_train,
        split="train",
        device="cpu",
    )
    train_encoded_inputs = t.tensor(
        train_data["encoded_inputs"], dtype=t.long, device="cpu"
    )

    test_data = construct_othello_dataset(
        custom_functions=[],
        n_inputs=n_test,
        split="test",
        device="cpu",
    )
    test_encoded_inputs = t.tensor(
        test_data["encoded_inputs"], dtype=t.long, device="cpu"
    )

    return train_encoded_inputs, test_encoded_inputs


@jaxtyped
def extract_activations_for_layer(
    model: NNsightModel,
    data: Int[Tensor, "n_games n_moves"],
    layer: int,
    batch_size: int = 256,
    device="cuda",
) -> Tuple[
    Float[Tensor, "n_games n_moves d_model"], Float[np.ndarray, "n_games n_moves d_mlp"]
]:
    """
    Extract activations from each layer of the model.

    Args:
        model: OthelloGPT model
        data: train or test encoded inputs

    Returns:
        Dict mapping layer_idx -> activations array (n_games, n_tokens, hidden_dim)
    """
    keys = [f"blocks.{layer}.hook_resid_pre", f"blocks.{layer}.mlp.hook_post"]
    resid = t.empty((data.shape[0], data.shape[1], model.cfg.d_model))
    mlp = np.ndarray((data.shape[0], data.shape[1], model.cfg.d_mlp))

    for i in range(0, len(data), batch_size):
        batch_inputs = data[i : i + batch_size].to(device)
        _, cache = model.run_with_cache(
            batch_inputs,
            names_filter=lambda name: name in keys,
        )

        resid[i : i + batch_size] = cache[keys[0]].detach().cpu()
        mlp[i : i + batch_size] = cache[keys[1]].detach().cpu()

    return resid, mlp


@jaxtyped
def prepare_dt_train_data_for_layer(
    model: NNsightModel,
    train_resid_acts: Float[Tensor, "n_train_games n_moves d_model"],
    train_mlp_post_acts: Float[np.ndarray, "n_train_games n_moves d_mlp"],
    test_resid_acts: Float[Tensor, "n_test_games n_moves d_model"],
    test_mlp_post_acts: Float[np.ndarray, "n_test_games n_moves d_mlp"],
    layer: int,
    batch_size: int = 256,
) -> Tuple[
    Float[np.ndarray, "n_train_samples n_feats"],
    Float[np.ndarray, "n_train_samples d_mlp"],
    Float[np.ndarray, "n_test_samples n_feats"],
    Float[np.ndarray, "n_test_samples d_mlp"],
]:
    board_state_probes = load_board_state_probes(model)
    flipped_probes = load_flipped_probes(model)
    played_probes = load_played_probes(model)

    mine_theirs = get_mine_theirs_normed(board_state_probes, normalize=False)[layer - 1]
    blank = get_blank_normed(board_state_probes, normalize=False)[layer - 1]
    flipped = get_flipped_normed(flipped_probes, normalize=False)[layer - 1]
    played = get_played_normed(played_probes, normalize=False)[layer - 1]

    all_probe = t.stack([mine_theirs, blank, flipped, played], dim=1)
    all_probe = einops.rearrange(
        all_probe, "d_model type row col -> d_model (type row col)"
    )

    # project onto probes
    train_resid_acts = einops.rearrange(
        train_resid_acts, "n_games n_moves d_model -> (n_games n_moves) d_model"
    )
    X_train = np.ndarray((train_resid_acts.shape[0], all_probe.shape[1]))
    for i in range(0, len(train_resid_acts), batch_size):
        batch_inputs = train_resid_acts[i : i + batch_size].to(device)
        X_train[i : i + batch_size] = (
            einops.einsum(
                batch_inputs, all_probe, "batch d_model, d_model feats -> batch feats"
            )
            .detach()
            .cpu()
            .numpy()
        )

    test_resid_acts = einops.rearrange(
        test_resid_acts, "n_games n_moves d_model -> (n_games n_moves) d_model"
    )
    X_test = np.ndarray((test_resid_acts.shape[0], all_probe.shape[1]))
    for i in range(0, len(test_resid_acts), batch_size):
        batch_inputs = test_resid_acts[i : i + batch_size].to(device)
        X_test[i : i + batch_size] = (
            einops.einsum(
                batch_inputs, all_probe, "batch d_model, d_model feats -> batch feats"
            )
            .detach()
            .cpu()
            .numpy()
        )

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    epsilon = 1e-8
    X_train_scaled = (X_train - X_train_mean) / (X_train_std + epsilon)

    X_test_scaled = (X_test - X_train_mean) / (X_train_std + epsilon)

    y_train = einops.rearrange(
        train_mlp_post_acts, "n_games n_moves d_mlp -> (n_games n_moves) d_mlp"
    )
    y_test = einops.rearrange(
        test_mlp_post_acts, "n_games n_moves d_mlp -> (n_games n_moves) d_mlp"
    )

    return X_train_scaled, y_train, X_test_scaled, y_test


def train_dt_for_neuron(
    X_train: Float[np.ndarray, "n_train_samples n_feats"],
    y_train: Float[np.ndarray, "n_train_samples d_mlp"],
    X_test: Float[np.ndarray, "n_test_samples n_feats"],
    y_test: Float[np.ndarray, "n_test_samples d_mlp"],
    layer: int,
    neuron: int,
) -> DecisionTreeResults:
    """
    Takes in activations and labels for a single layer,
    trains a decision tree for a specified neuron in that layer,
    evaluates it, and returns the results.
    """
    # 1. Initialize the DecisionTreeRegressor with hyperparameters
    #    that prioritize interpretability and prevent overfitting.
    tree = DecisionTreeRegressor(
        max_depth=8,
        random_state=42,
        # min_samples_leaf=50,
        # min_samples_split=100,
        criterion="squared_error",
    )

    # 2. Select the activation data for the specific neuron we are analyzing.
    y_train_neuron = y_train[:, neuron]
    y_test_neuron = y_test[:, neuron]

    # 3. Fit the decision tree to the training data.
    tree.fit(X_train, y_train_neuron)

    # 4. Evaluate the tree's performance on both training and test sets.
    # The .score() method for a regressor returns the R^2 value.
    train_r2 = tree.score(X_train, y_train_neuron)
    test_r2 = tree.score(X_test, y_test_neuron)

    train_mse = mean_squared_error(y_train_neuron, tree.predict(X_train))
    test_mse = mean_squared_error(y_test_neuron, tree.predict(X_test))

    # 5. Return the results in a structured dataclass.
    # Note: The DecisionTreeResults dataclass has a 'square' field.
    # As we are analyzing a neuron's general function here, not its effect
    # on a specific square, we will use a placeholder value of -1.
    return DecisionTreeResults(
        layer=layer,
        neuron=neuron,  # Placeholder, as this is a neuron-level analysis
        tree=tree,
        train_R2=train_r2,
        train_MSE=train_mse,
        test_R2=test_r2,
        test_MSE=test_mse,
    )


if __name__ == "__main__":
    layer = 1
    neuron = 421

    model = load_model()

    train_data, test_data = load_data(n_train=60000, n_test=1000)

    train_resid_acts, train_mlp_acts = extract_activations_for_layer(
        model, train_data, layer=layer
    )
    test_resid_acts, test_mlp_acts = extract_activations_for_layer(
        model, test_data, layer=layer
    )

    X_train, y_train, X_test, y_test = prepare_dt_train_data_for_layer(
        model,
        train_resid_acts,
        train_mlp_acts,
        test_resid_acts,
        test_mlp_acts,
        layer,
    )

    dt_out = train_dt_for_neuron(X_train, y_train, X_test, y_test, layer, neuron)

    print(dt_out)


# train_dt_for_layer, parallelize train_dt_for_neuron
