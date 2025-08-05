#%% 
import torch as t
from torch import Tensor
import einops
from jaxtyping import Float
from typing import Tuple

device = t.device("cuda")

# %%
activations = t.load("linear_probes/dataset_cache/activations_100000_layers_6.pt")['layer_activations'][6].double().to(device)

# %%
probe = t.load("20250804_174921/probe_6.pt").double().to(device) # [d_model, row, col, mode]

# %%
def gram_schmidt(vectors):
    """Gram-Schmidt for a single set of vectors"""
    Q, _ = t.linalg.qr(vectors.T)
    return Q.T.double()

def get_difference_vectors(probe : Float[Tensor, 'd_model row col mode']):
    d_model, row, col, mode = probe.shape
    difference_vectors = t.stack((probe[..., 0] - probe[..., 1], probe[..., 0] - probe[..., 2]))
    return einops.rearrange(difference_vectors, "two d_model row col -> (two row col) d_model", two=2, row=row, col=col)

def get_difference_vectors_single_square(probe : Float[Tensor, 'd_model mode']):
    d_model, mode = probe.shape
    difference_vectors = t.stack((probe[..., 0] - probe[..., 1], probe[..., 0] - probe[..., 2]))
    return difference_vectors

def directional_ablation(activations: Float[Tensor, 'games moves d_model'], probe : Float[Tensor, 'd_model row col mode']) -> Float[Tensor, 'games moves d_model']:
    difference_vectors = get_difference_vectors(probe)
    orthonormal_basis = gram_schmidt(difference_vectors.double())
    for i in range(orthonormal_basis.shape[0]):
        projection = einops.einsum(activations, orthonormal_basis[i], 'games moves d_model, d_model -> games moves')
        activations -= projection.unsqueeze(dim = -1) * orthonormal_basis[i].unsqueeze(dim = 0).unsqueeze(dim = 0)
    return activations

def directional_ablation_single_square(activations: Float[Tensor, 'games moves d_model'], probe : Float[Tensor, 'd_model mode']) -> Float[Tensor, 'games moves d_model']:
    difference_vectors = get_difference_vectors_single_square(probe)
    orthonormal_basis = gram_schmidt(difference_vectors.double())
    for i in range(orthonormal_basis.shape[0]):
        projection = einops.einsum(activations, orthonormal_basis[i], 'games moves d_model, d_model -> games moves')
        activations -= projection.unsqueeze(dim = -1) * orthonormal_basis[i].unsqueeze(dim = 0).unsqueeze(dim = 0)
    return activations

def test_against_original_differences(ablated_activations, probe):
    """Test against the exact difference vectors used for ablation"""
    difference_vectors = get_difference_vectors(probe)  # (2*row*col, d_model)
    # Project ablated activations onto each difference vector
    projections = einops.einsum(ablated_activations, difference_vectors, 
                               'games moves d_model, vectors d_model -> games moves vectors')
    
    assert t.allclose(projections, t.zeros_like(projections), atol=1e-8, rtol=1e-8)

def test_against_original_differences_single_square(ablated_activations, probe):
    """Test against the exact difference vectors used for ablation"""
    difference_vectors = get_difference_vectors_single_square(probe)  # (2*row*col, d_model)
    # Project ablated activations onto each difference vector
    projections = einops.einsum(ablated_activations, difference_vectors, 
                               'games moves d_model, vectors d_model -> games moves vectors')
    assert t.allclose(projections, t.zeros_like(projections), atol=1e-8, rtol=1e-8)

#%%
ablated_activations = directional_ablation_single_square(activations, probe[:, 0, 0, :])
test_against_original_differences_single_square(ablated_activations, probe[:, 0, 0, :])

#%%
t.save({'layer_activations': {6: ablated_activations.float()}}, "linear_probes/dataset_cache/activations_100000_layers_6_A0_ablated.pt")

#%%
ablated_activations = directional_ablation(activations, probe)
test_against_original_differences(ablated_activations, probe)

# %%
t.save({'layer_activations': {6: ablated_activations.float()}}, "linear_probes/dataset_cache/activations_100000_layers_6_once_ablated.pt")
# %%
