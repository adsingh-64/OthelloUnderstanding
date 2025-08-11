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
neuron_idx = 1199
function_name = list(data[layer].keys())[0]
tree_model = data[layer][function_name]["decision_tree"]["model"].estimators_[
    neuron_idx
]
max_samples = 10000
batch_size = 32

# %%
def othello_generator(
    dataset_name="taufeeque/othellogpt",
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
    node_ids = tree_model.apply(batch_input)
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
selected = node_data[347]

# %%
for example in selected:
    game, move_idx = example['game'], example['move_idx']
    focus_states, focus_legal_moves, focus_legal_moves_annotation = (
        get_board_states_and_legal_moves(t.tensor(game))
    )
    # Need to add batch dimension for single board
    neel_utils.plot_board_values(
        focus_states[-1:],  # Use slice to keep dimension
        title="Board states",
        width=500,
        height = 500,
        boards_per_row=1,
        board_titles=[
            f"After move {move_idx}, {'white' if move_idx % 2 == 0 else 'black'} to play next"
        ],
        text=[focus_legal_moves_annotation[-1]],  # Wrap in list for single board
    )

# %%
