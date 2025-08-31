# %%
import torch as t
import numpy as np
import einops
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy
import transformer_lens
from torch import Tensor
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
import neel_utils as neel_utils
import matplotlib.pyplot as plt
import seaborn as sns
import json
import circuitsvis as cv

# from ablate_probe import directional_ablation_single_square, plot_cosine_sim

device = "cuda" if t.cuda.is_available() else "cpu"

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_size = 50
custom_functions = [
    othello_utils.games_batch_to_flipped_classifier_input_BLC,
]
model = utils.get_model(model_name, device)
train_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size,
    split="test",
    device=device,
)

# %%
input_ids = t.tensor(train_data["encoded_inputs"], dtype=t.long, device=device)
input_ids = input_ids[:, :30]
print(input_ids.shape)

# %%
keys = [transformer_lens.utils.get_act_name("pattern", i) for i in range(model.cfg.n_layers)]
logits, cache = model.run_with_cache(
    input_ids,
    names_filter=lambda name: name in keys
)

# %%
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    mean_attention_pattern = einops.reduce(attention_pattern, "n_games head row col -> head row col", "mean")
    display(
        cv.attention.attention_patterns(tokens=["_"]*30, attention=mean_attention_pattern)
    )

# %%
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import base64
from io import BytesIO
from PIL import Image
import imgkit  # You'll need to install this: pip install imgkit

# Option A: Using matplotlib to recreate as static plot
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    mean_attention_pattern = einops.reduce(
        attention_pattern, "n_games head row col -> head row col", "mean"
    )

    # Convert to numpy for plotting
    pattern_np = to_numpy(mean_attention_pattern)

    # Create figure with subplots for each head
    n_heads = pattern_np.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # Adjust based on n_heads
    axes = axes.flatten()

    for head in range(n_heads):
        ax = axes[head]
        im = ax.imshow(pattern_np[head], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {head}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")

    #plt.colorbar(im, ax=axes)
    plt.suptitle(f"Layer {layer} Attention Patterns")
    plt.tight_layout()

    # Save as PNG
    plt.savefig(f"attention_layer_{layer}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# %%
