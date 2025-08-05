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
probe = t.load("20250805_135718/probe_6.pt").double().to(device) # [d_model, row, col, mode]

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

def plot_cosine_sim(probe):
    difference_vectors = get_difference_vectors(probe)
    difference_vectors = einops.rearrange(difference_vectors, "(two row col) d_model -> row col two d_model", two=2, row=8, col=8)
    
    # Reshape for pairwise computation
    n_pos = 64
    d_model = difference_vectors.shape[-1]
    vecs_flat = einops.rearrange(difference_vectors, "row col two d_model -> (row col) two d_model")
    
    # Compute QR decomposition for orthonormal bases
    Q_all = []
    for i in range(n_pos):
        vecs = vecs_flat[i].T  # (d_model, 2)
        Q, _ = t.qr(vecs)
        Q_all.append(Q[:, :2])  # Keep first 2 columns
    
    Q_all = t.stack(Q_all)  # (n_pos, d_model, 2)
    
    # Compute all pairwise products Q_i^T @ Q_j efficiently
    # Q_all: (n_pos, d_model, 2)
    # We want: (n_pos, n_pos, 2, 2) where [i,j] = Q_i^T @ Q_j
    
    Q_all_T = Q_all.transpose(-2, -1)  # (n_pos, 2, d_model)
    
    # Use einsum for efficient computation
    M_all = t.einsum('i a d, j d b -> i j a b', Q_all_T, Q_all)  # (n_pos, n_pos, 2, 2)
    
    # Compute SVD for each 2x2 matrix
    similarities = t.zeros(n_pos, n_pos)
    
    for i in range(n_pos):
        for j in range(n_pos):
            _, S, _ = t.svd(M_all[i, j])
            angles = t.acos(S.clamp(-1, 1))
            similarities[i, j] = 1 - (angles.mean() / (t.pi / 2))
    
    return einops.rearrange(similarities, "squares (row col) -> squares row col", row=8, col=8)

#%%
### Single square ablation
ablated_activations = directional_ablation_single_square(activations, probe[:, 0, 3, :])
test_against_original_differences_single_square(ablated_activations, probe[:, 0, 3, :])

#%%
t.save({'layer_activations': {6: ablated_activations.float()}}, "linear_probes/dataset_cache/activations_100000_layers_6_A3_ablated.pt")

#%%
### Full ablation
ablated_activations = directional_ablation(activations, probe)
test_against_original_differences(ablated_activations, probe)

# %%
t.save({'layer_activations': {6: ablated_activations.float()}}, "linear_probes/dataset_cache/activations_100000_layers_6_once_ablated.pt")
# %%
