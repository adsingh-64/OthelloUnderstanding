import torch as t
import einops

D_MODEL = 512

matrices_1 = t.randn((50, D_MODEL, D_MODEL))
matrices_2 = t.randn((50, D_MODEL, D_MODEL))

products = einops.einsum(matrices_1, matrices_2, "... d_model_1 d_model, ... d_model d_model_2 -> ... d_model_1 d_model_2")
norm_of_products = products.norm(dim=(-1, -2), p='fro')

matrices_1_norms = matrices_1.norm(dim=(-1, -2), p='fro')
matrices_2_norms = matrices_2.norm(dim=(-1, -2), p='fro')
product_of_norms = matrices_1_norms * matrices_2_norms

composition_scores = norm_of_products / product_of_norms
print(composition_scores.mean())

