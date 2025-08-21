# %%
import json
import matplotlib.pyplot as plt
import numpy as np

# %%
# Load the data
with open('flipped_square_ablation_results/g2_ablation_results_dla.json', 'r') as f:
    dla_data = json.load(f)

with open('flipped_square_ablation_results/g2_ablation_results_dt.json', 'r') as f:
    dt_data = json.load(f)

with open('flipped_square_ablation_results/g2_ablation_results_random.json', 'r') as f:
    random_data = json.load(f)

# %%
# Create the plot
plt.figure(figsize=(10, 6))

# Extract data for DLA (skip 0 for log scale)
dla_data_filtered = [d for d in dla_data if d['n_ablated'] > 0]
dla_n_ablated = [d['n_ablated'] for d in dla_data_filtered]
dla_accuracy = [d['accuracy'] for d in dla_data_filtered]

# Extract data for DT (skip 0 for log scale)
dt_data_filtered = [d for d in dt_data if d['n_ablated'] > 0]
dt_n_ablated = [d['n_ablated'] for d in dt_data_filtered]
dt_accuracy = [d['accuracy'] for d in dt_data_filtered]

# Extract data for Random (only up to 2^9 = 512, skip 0)
random_limited = [d for d in random_data if d["n_ablated"] > 0]
random_n_ablated = [d['n_ablated'] for d in random_limited]
random_accuracy = [d['accuracy'] for d in random_limited]

# Plot the lines
plt.plot(dla_n_ablated, dla_accuracy, 'o-', label='Direct Probe Attribution', linewidth=2, markersize=6)
plt.plot(dt_n_ablated, dt_accuracy, 's-', label='Decision Tree', linewidth=2, markersize=6)
plt.plot(random_n_ablated, random_accuracy, '^-', label='Random', linewidth=2, markersize=6)

# Set x-axis to log scale
plt.xscale('log', base=2)

# Customize the plot
plt.xlabel('Number of Neurons Ablated', fontsize=12)
plt.ylabel('G2 Probe Accuracy', fontsize=12)
plt.title('Neuron Ablation Results Comparison', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Set x-axis ticks to powers of 2
x_ticks = [2**i for i in range(0, 14)]
plt.xticks(x_ticks, [str(x) for x in x_ticks])

plt.tight_layout()
plt.show()

# %%
plt.savefig("g2_ablation_comparison.png", dpi=300, bbox_inches='tight')

# %%
