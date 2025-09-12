import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight.models import NNsightModel
from nnsight import LanguageModel
from transformer_lens import HookedTransformer

import neel_utils as neel_utils
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset

import json
import sys
from jaxtyping import Bool, Float, Int
from typing import Tuple
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table
from pprint import pprint
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from fine_grained_intervention.utils import (
    load_model,
)

device = t.device('cuda')

model = load_model(device=device)

W_Q = model.W_Q.detach().clone()
W_K = model.W_K.detach().clone()
W_V = model.W_V.detach().clone()
W_O = model.W_O.detach().clone()

QK = einops.einsum(
    W_Q, W_K, "... d_model d_head, ... d_model_2 d_head -> ... d_model d_model_2"
)

OV = einops.einsum(
    W_V, W_O, "... d_model d_head, ... d_head d_model_2 -> ... d_model d_model_2"
)

QK_norms = QK.norm(dim=(-1, -2), p="fro")
OV_norms = OV.norm(dim=(-1, -2), p="fro")

product_of_norms = einops.einsum(
    OV_norms,
    QK_norms,
    "layer_ov head_ov, layer_qk head_qk -> layer_ov head_ov layer_qk head_qk",
)

products = einops.einsum(
    OV,
    QK,
    "layer_ov head_ov d_model d_model_2, layer_qk head_qk d_model_2 d_model_3 -> layer_ov head_ov layer_qk head_qk d_model d_model_3",
)
product_norms = products.norm(dim=(-1, -2), p="fro")

layer_qk = 4
head_qk = 0

composition_scores = t.empty((layer_qk, model.cfg.n_heads), device=device)
for layer in range(layer_qk):
    for head in range(model.cfg.n_heads):
        composition_scores[layer, head] = (
            product_norms[layer, head, layer_qk, head_qk]  # Note the order change
            / product_of_norms[layer, head, layer_qk, head_qk]
        )

scores = composition_scores.detach().cpu().numpy()

# Create hover text with precise values
hover_text = []
for layer in range(layer_qk):
    row_text = []
    for head in range(model.cfg.n_heads):
        row_text.append(f'Layer: {layer}<br>Head: {head}<br>Score: {scores[layer, head]:.6f}')
    hover_text.append(row_text)

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=scores,
    x=list(range(model.cfg.n_heads)),
    y=list(range(layer_qk)),
    colorscale='Blues',
    text=hover_text,
    hovertemplate='%{text}<extra></extra>',
    colorbar=dict(title='Composition Score')
))

# Update layout
fig.update_layout(
    title=f'Composition Scores to Head {layer_qk}.{head_qk}',
    xaxis=dict(title='Head', tickmode='linear', tick0=0, dtick=1),
    yaxis=dict(title='Layer', tickmode='linear', tick0=0, dtick=1, autorange='reversed'),
    width=800,
    height=600,
)

fig.show()