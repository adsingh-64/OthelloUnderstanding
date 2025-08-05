import os
import torch
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from circuits import utils
from circuits import othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset


def make_dataset(n_games):
    dataset = construct_othello_dataset(
        [othello_utils.games_batch_to_state_stack_mine_yours_BLRRC],
        n_inputs=n_games,
        split="train",
        device="cuda",
    )
    boards = (
        dataset["games_batch_to_state_stack_mine_yours_BLRRC"]
        .reshape(-1, 59, 64, 3)
        .permute(3, 0, 1, 2)
    )
    games = torch.tensor(dataset["encoded_inputs"])
    return games, boards  # games : [n_games, 59], boards : [3, n_games, 59, 64]


def save_dataset(games, boards, n_games, cache_dir="dataset_cache"):
    """Save the dataset to disk."""
    # Make cache_dir relative to the script file location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"othello_dataset_{n_games}.pt")
    
    torch.save({
        'games': games,
        'boards': boards,
        'n_games': n_games
    }, cache_file)
    
    print(f"Dataset saved to {cache_file}")
    return cache_file


def load_dataset(n_games, cache_dir="dataset_cache"):
    """Load the dataset from disk if it exists."""
    # Make cache_dir relative to the script file location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, cache_dir)
    cache_file = os.path.join(cache_dir, f"othello_dataset_{n_games}.pt")
    
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        data = torch.load(cache_file)
        return data['games'], data['boards'], True
    else:
        print(f"No cached dataset found for {n_games} games")
        return None, None, False


def get_or_create_dataset(n_games, cache_dir="dataset_cache", force_recreate=False):
    """Get dataset from cache or create new one."""
    if not force_recreate:
        games, boards, found = load_dataset(n_games, cache_dir)
        if found:
            return games, boards
    
    print(f"Creating new dataset with {n_games} games...")
    games, boards = make_dataset(n_games)
    save_dataset(games, boards, n_games, cache_dir)
    return games, boards


def save_activations(layer_activations, n_games, layers, cache_dir="dataset_cache"):
    """Save layer activations to disk."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create filename that includes which layers are cached
    layers_str = "_".join(map(str, sorted(layers)))
    cache_file = os.path.join(cache_dir, f"activations_{n_games}_layers_{layers_str}.pt")
    
    torch.save({
        'layer_activations': layer_activations,
        'layers': layers
    }, cache_file)
    print(f"Activations saved to {cache_file}")
    return cache_file


def load_activations(n_games, layers, cache_dir="dataset_cache"):
    """Load activations from disk if they exist for the specified layers."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, cache_dir)
    
    # Check if we have cached activations for these specific layers
    layers_str = "_".join(map(str, sorted(layers)))
    cache_file = os.path.join(cache_dir, f"activations_{n_games}_layers_{layers_str}_A0_ablated.pt")
    
    if os.path.exists(cache_file):
        print(f"Loading cached activations from {cache_file}")
        data = torch.load(cache_file)
        return data['layer_activations'], True
    else:
        # Check if we have a cache file with all layers that we can use
        all_layers_file = os.path.join(cache_dir, f"activations_{n_games}_layers_0_1_2_3_4_5_6_7.pt")
        if os.path.exists(all_layers_file):
            print(f"Loading subset of layers from full cache: {all_layers_file}")
            data = torch.load(all_layers_file)
            # Extract only the layers we need
            subset_activations = {layer: data['layer_activations'][layer] for layer in layers}
            return subset_activations, True
        
        print(f"No cached activations found for {n_games} games with layers {layers}")
        return None, False


def get_or_create_activations(model, games, n_games, layers, cache_dir="dataset_cache", force_recreate=False):
    """Get activations from cache or compute new ones for specified layers."""
    if not force_recreate:
        layer_activations, found = load_activations(n_games, layers, cache_dir)
        if found:
            return layer_activations
    
    print(f"Computing layer activations for layers: {layers}...")
    layer_activations = get_all_activations(model, games, layers)
    save_activations(layer_activations, n_games, layers, cache_dir)
    return layer_activations


def get_all_activations(model, games, layers):
    """Cache activations for specified layers in a single forward pass."""
    minibatch_size = 64
    
    # Only prepare to collect activations for requested layers
    keys = [f"blocks.{i}.hook_resid_post" for i in layers]
    
    all_layer_acts = {i: [] for i in layers}
    
    for i in range(0, len(games), minibatch_size):
        minibatch = games[i : i + minibatch_size]
        _, cache = model.run_with_cache(
            minibatch, names_filter=lambda name: name in keys
        )
        
        # Collect activations only for requested layers
        for layer_idx in layers:
            key = f"blocks.{layer_idx}.hook_resid_post"
            all_layer_acts[layer_idx].append(cache[key])
    
    # Concatenate batches for each layer
    layer_activations = {}
    for layer_idx in layers:
        layer_activations[layer_idx] = torch.cat(all_layer_acts[layer_idx], dim=0)
    
    return layer_activations  # Dict[int, Tensor] where each tensor is [n_games, 59, d_model]


def train_least_squares(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    probe = torch.linalg.lstsq(X_train, y_train).solution

    y_pred = X_test @ probe
    y_mean = y_test.mean()
    ss_tot = torch.sum((y_test - y_mean) ** 2)
    ss_res = torch.sum((y_test - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    y_test = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    same_class = (y_test == y_test[0]).all(axis=0)
    roc_auc = roc_auc_score(y_test[:, ~same_class], y_pred[:, ~same_class])

    per_square_roc_auc = []
    for square in range(64):
        if same_class[square]:
            score = 1
        else:
            score = roc_auc_score(y_test[:, square], y_pred[:, square])
        per_square_roc_auc.append(score)

    return probe, r_squared.item(), roc_auc, per_square_roc_auc


def train_probe_for_layer(layer_acts, boards):
    """Train probe for a specific layer's activations."""
    # Move activations to GPU if they're on CPU
    if layer_acts.device.type == 'cpu':
        layer_acts = layer_acts.cuda()
    
    # Reshape activations
    X = layer_acts.reshape(-1, layer_acts.shape[-1])  # [n_games * 59, d_model]

    results = []
    probes = []

    for mode in range(3):
        print(f"  Training for mode {mode}...")
        y = boards[mode].reshape(-1, 64).float()  # [n_games * 59, 64]
        probe, r_squared, roc_auc, per_square_roc_auc = train_least_squares(X, y)
        print(f"  R-squared score: {r_squared:.4f} for mode {mode}")
        print(f"  ROC AUC score: {roc_auc:.4f} for mode {mode}")
        probes.append(probe)
        results.append(
            {
                "r_squared": r_squared,
                "roc_auc": roc_auc,
                "per_square_roc_auc": per_square_roc_auc,
                "mode": mode,
            }
        )

    probe = (
        torch.stack(probes).reshape(3, layer_acts.shape[-1], 8, 8).permute(1, 2, 3, 0)
    )
    return probe, results


def main(output_dir, n_games=100000, device="cuda", layers=None, cache_dir="dataset_cache", force_recreate_dataset=False, force_recreate_activations=False):
    model = utils.get_model("Baidicoot/Othello-GPT-Transformer-Lens", device)
    n_layers = model.cfg.n_layers

    if layers is None:
        layers = list(range(n_layers))
    
    print(f"Training probes for layers: {layers}")

    print(f"Model loaded. Getting dataset with {n_games} games...")
    games, boards = get_or_create_dataset(n_games, cache_dir, force_recreate_dataset)
    boards, games = boards.to(device), games.to(device)

    print("Getting layer activations...")
    layer_activations = get_or_create_activations(model, games, n_games, layers, cache_dir, force_recreate_activations)
    print("All activations ready!")

    os.makedirs(output_dir, exist_ok=True)

    # Train probes for each layer
    for layer in layers:
        print(f"\nTraining probe for layer {layer}...")
        layer_acts = layer_activations[layer]
        probe, results = train_probe_for_layer(layer_acts, boards)

        # Add layer info to results
        for result in results:
            result["layer"] = layer

        print(f"Saving results for layer {layer}...")
        torch.save(probe, os.path.join(output_dir, f"probe_{layer}.pt"))
        with open(os.path.join(output_dir, f"results_{layer}.json"), "w") as f:
            json.dump(results, f)

    print(f"\nAll probes trained and saved to {output_dir}")


if __name__ == "__main__":
    output_dir = datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
    
    # Examples of specifying layers:
    
    # Train on all layers (0-7):
    # main(output_dir)
    
    # Train only on layers 5, 6, and 7:
    # main(output_dir, layers=[5, 6, 7])
    
    # Train only on layer 7:
    # main(output_dir, layers=[7])
    
    # Train on layers 0, 2, 4, 6:
    # main(output_dir, layers=[0, 2, 4, 6])
    
    # Default: train on all layers
    main(output_dir, layers=[6], force_recreate_dataset=False, force_recreate_activations=False)