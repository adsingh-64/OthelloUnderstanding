# %%
from circuits.othello_utils import hf_othello_dataset_to_generator
from circuits.othello_utils import games_batch_to_input_tokens_flipped_bs_classifier_input_BLC
from collections import defaultdict
import pickle
from datasets import load_dataset
import neel_utils as neel_utils
import einops
from tqdm.notebook import tqdm
import math
import gzip
from single_game_analysis import get_board_states_and_legal_moves
import torch as t
from transformer_lens.utils import to_numpy
import numpy as np

# %%
with open(
    "neuron_simulation/decision_trees_new/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl",
    "rb",
) as f:
    data = pickle.load(f)
layer = 3
neuron_idx = 2018
function_name = list(data[layer].keys())[0]
tree_model = data[layer][function_name]["decision_tree"]["model"].estimators_[
    neuron_idx
]
max_samples = 10000
batch_size = 32

# %%
move = neel_utils.id_to_square([20, 19, 41, 21, 27, 34, 13, 33, 29, 12, 26, 43, 38, 14, 10, 48, 42, 18, 28, 32, 49, 22, 4, 15, 44, 50, 37, 31, 39, 2, 55, 57, 51, 6, 17, 24, 40, 47, 45, 46, 54, 52, 23])
features = games_batch_to_input_tokens_flipped_bs_classifier_input_BLC([move]).numpy()[:, -1]
print(tree_model.decision_path(features))

# %%
def othello_generator(
    dataset_name="adamkarvonen/othello_45MB_games",
    split="train",
    streaming=True,
    max_samples=100,  # New parameter
    token_mapping=None,
    batch_size=32,  # New parameter
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    def gen():
        batch = []
        sample_count = 0
        for x in iter(dataset):
            if sample_count >= max_samples:
                break
            tokens = x["tokens"]
            if token_mapping:
                tokens = [token_mapping(token) for token in tokens]
            batch.append(tokens)
            sample_count += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    return gen()
generator = othello_generator(token_mapping=neel_utils.id_to_square, max_samples=max_samples, batch_size=batch_size)

# %%
node_data = defaultdict(list)
n_batches = math.ceil(max_samples / batch_size)
for batch in tqdm(generator, desc = "dt max activating examples", total = n_batches):
    batch_features = games_batch_to_input_tokens_flipped_bs_classifier_input_BLC(batch)
    # Only use moves 0-58 (exclude last move since model doesn't predict after it)
    batch_input = einops.rearrange(
        batch_features[:, :59].numpy(), "batch seq n_features -> (batch seq) n_features"
    )
    node_ids = tree_model.apply(batch_input) # can retrace branch and add to all parent nodes as well
    for i, node_id in enumerate(node_ids):
        game_idx = i // 59  # Now dividing by 59 instead of 60
        move_idx = i % 59   # move_idx ranges from 0-58
        node_data[node_id].append({
            'game': batch[game_idx][:move_idx+1],  # Include moves 0 through move_idx
            'move_idx': move_idx
        })

# %%
save_data = {
    "node_data": dict(node_data),
    "metadata": {
        "layer": layer,
        "neuron_idx": neuron_idx,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "n_nonempty_leafs": len(node_data),
        "n_total_leafs": (tree_model.tree_.children_left == -1).sum(),
    },
}
with gzip.open(f"leaf_examples_L{layer}_N{neuron_idx}.pkl.gz", "wb") as f:
    pickle.dump(save_data, f)

# %%
with gzip.open(f"leaf_examples_L{layer}_N{neuron_idx}.pkl.gz", "rb") as f:
    loaded_data = pickle.load(f)

# %%
node_data = loaded_data['node_data']

# %%
selected = node_data[326]
for example in selected[:10]:
    game, move_idx = example['game'], example['move_idx']
    focus_states, focus_legal_moves, focus_legal_moves_annotation = (
        get_board_states_and_legal_moves(t.tensor(game))
    )
    # Need to add batch dimension for single board
    neel_utils.plot_board_values(
        focus_states[-2:],  # Use slice to keep dimension
        title="Board states",
        width=800,
        height=500,
        boards_per_row=2,
        board_titles=[
            f"After move {move_idx - 1}, {'white' if (move_idx - 1) % 2 == 0 else 'black'} to play next",
            f"After move {move_idx}, {'white' if move_idx % 2 == 0 else 'black'} to play next"
        ],
        text=focus_legal_moves_annotation[-2:],  # Wrap in list for single board
    )

# %%
def analyze_top_leaf_nodes(tree_model, node_data, top_k=10, min_examples=10):
    """
    Show top k leaf nodes by prediction value (only those with at least min_examples).
    
    Args:
        tree_model: The decision tree model
        node_data: Dictionary mapping leaf node_id to examples
        top_k: Number of top leaf nodes to display
        min_examples: Minimum number of examples required (default 10)
    """
    # Get all leaf nodes
    leaf_nodes = np.where(tree_model.tree_.children_left == -1)[0]
    all_predictions = tree_model.tree_.value.flatten()
    
    # Get predictions for leaf nodes that have at least min_examples
    leaf_predictions = []
    for node_id in leaf_nodes:
        if node_id in node_data and len(node_data[node_id]) >= min_examples:
            n_examples = len(node_data[node_id])
            prediction = all_predictions[node_id]
            leaf_predictions.append((node_id, prediction, n_examples))
    
    # Sort by prediction value (largest first)
    leaf_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"\nTop {top_k} leaf nodes by prediction value (≥{min_examples} examples):")
    print(f"Total leaf nodes: {len(leaf_nodes)}, with ≥{min_examples} examples: {len(leaf_predictions)}")
    print("-" * 60)
    
    k = min(top_k, len(leaf_predictions))
    for i, (node_id, prediction, n_examples) in enumerate(leaf_predictions[:k]):
        print(f"Rank {i+1:2d}: Node {node_id:4d} | Prediction: {prediction:8.4f} | Examples: {n_examples:5d}")

# Example usage:
analyze_top_leaf_nodes(tree_model, node_data, top_k=10)

# %%
