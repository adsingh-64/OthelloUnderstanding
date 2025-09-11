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

from fine_grained_intervention.utils import (
    load_model,
)

device = t.device('cuda')

model = HookedTransformer.from_pretrained("gpt2-small", dtype=t.bfloat16)
model = model.to(device)

W_Q = model.W_Q.detach().clone()
W_K = model.W_K.detach().clone()
W_V = model.W_V.detach().clone()
W_O = model.W_O.detach().clone()

QK = einops.einsum(W_Q, W_K, "... d_model d_head, ... d_model_2 d_head -> ... d_model d_model_2")

OV = einops.einsum(W_V, W_O, "... d_model d_head, ... d_head d_model_2 -> ... d_model d_model_2")

QK_norms = QK.norm(dim = (-1, -2), p = 'fro')
OV_norms = OV.norm(dim = (-1, -2), p = 'fro')

product_of_norms = einops.einsum(
    QK_norms, 
    OV_norms, 
    'layer_qk head_qk, layer_ov head_ov -> layer_qk head_qk layer_ov head_ov'
)

products = einops.einsum(
    QK, 
    OV, 
    'layer_qk head_qk d_model_qk d_model, layer_ov head_ov d_model d_model_ov -> layer_qk head_qk layer_ov head_ov d_model_qk d_model_ov'
)
product_norms = products.norm(dim=(-1, -2), p='fro')

layer_qk = 9
head_qk = 9

composition_scores = t.empty((layer_qk, model.cfg.n_heads), device=device)
for layer in range(layer_qk):
    for head in range(model.cfg.n_heads):
        composition_scores[layer, head] = product_norms[layer_qk, head_qk, layer, head] / product_of_norms[layer_qk, head_qk, layer, head]

plt.figure(figsize=(10, 6))
plt.imshow(composition_scores.cpu().numpy(), aspect='auto', cmap='viridis')
plt.colorbar(label='Composition Score')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.title('Composition Scores by Layer and Head')
plt.yticks(range(layer_qk))
plt.show()














