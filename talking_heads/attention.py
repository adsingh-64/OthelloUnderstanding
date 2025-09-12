import torch as t
import numpy as np
import einops
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fine_grained_intervention.utils import load_model
import math

N_LAYERS = 8

class CompositionAnalyzer:
    def __init__(self, device='cuda'):
        """Initialize with model and precompute matrices."""
        self.device = t.device(device)
        self.model = load_model(device=self.device)
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        
        # Precompute QK and OV matrices once
        self.QK, self.OV = self._compute_matrices()
        
    def _compute_matrices(self):
        """Extract and compute QK and OV matrices from model weights."""
        W_Q = self.model.W_Q.detach()
        W_K = self.model.W_K.detach()
        W_V = self.model.W_V.detach()
        W_O = self.model.W_O.detach()
        
        QK = einops.einsum(
            W_Q, W_K, "... d_model d_head, ... d_model_2 d_head -> ... d_model d_model_2"
        )
        
        OV = einops.einsum(
            W_V, W_O, "... d_model d_head, ... d_head d_model_2 -> ... d_model d_model_2"
        )
        
        return QK, OV
    
    def _compute_svd_scores(self, source_matrices, target_matrix, max_layer):
        """Compute max composition scores using SVD decomposition."""
        target_norm = target_matrix.norm(p="fro")
        max_scores = t.zeros((max_layer, self.n_heads), device=self.device)
        
        for layer in range(max_layer):
            for head in range(self.n_heads):
                source = source_matrices[layer, head]
                U, S, Vt = t.linalg.svd(source, full_matrices=False)
                
                component_scores = []
                for i in range(min(U.shape[1], 64)):
                    component = S[i] * t.outer(U[:, i], Vt[i, :])
                    product = component @ target_matrix
                    numerator = product.norm(p="fro")
                    denominator = component.norm(p="fro") * target_norm
                    
                    if denominator > 0:
                        component_scores.append((numerator / denominator).item())
                
                if component_scores:
                    max_scores[layer, head] = max(component_scores)
        
        return max_scores
    
    def visualize_head(self, layer_target, head_target):
        """Create side-by-side visualization for specified target head."""
        # Compute scores
        ov_qk_scores = self._compute_svd_scores(
            self.OV, self.QK[layer_target, head_target], layer_target
        ).cpu().numpy()
        
        ov_ov_scores = self._compute_svd_scores(
            self.OV, self.OV[layer_target, head_target], layer_target
        ).cpu().numpy()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'OV→QK to L{layer_target}H{head_target}',
                f'OV→OV to L{layer_target}H{head_target}'
            ),
            horizontal_spacing=0.12
        )
        
        # Add heatmaps
        for col, (scores, name) in enumerate([(ov_qk_scores, 'OV→QK'), 
                                                (ov_ov_scores, 'OV→OV')], 1):
            hover_text = [[f'L{l}H{h}: {scores[l, h]:.4f}' 
                          for h in range(self.n_heads)] 
                          for l in range(layer_target)]
            
            heatmap = go.Heatmap(
                z=scores,
                x=list(range(self.n_heads)),
                y=list(range(layer_target)),
                colorscale='Blues',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showscale=(col == 1),
                colorbar=dict(title='Score', len=0.5) if col == 1 else None
            )
            fig.add_trace(heatmap, row=1, col=col)
        
        # Update axes
        for col in [1, 2]:
            fig.update_xaxes(title_text="Head", tickmode='linear', 
                           tick0=0, dtick=1, row=1, col=col)
            fig.update_yaxes(title_text="Layer", tickmode='linear', 
                           tick0=0, dtick=1, autorange='reversed', row=1, col=col)
        
        fig.update_layout(
            title_text=f"Max SVD Composition Scores to L{layer_target}H{head_target}",
            width=1200,
            height=600
        )
        
        fig.show()
    
    def visualize_layer(self, layer_target, composition_type='OV_QK', 
                       cols_per_row=4, separate_plots=False):
        """
        Visualize composition scores to all heads in a target layer.
        
        Args:
            layer_target: Target layer to analyze
            composition_type: Either 'OV_QK', 'OV_OV', or 'both'
            cols_per_row: Number of columns per row (default 4 for better visibility)
            separate_plots: If True, create separate figures for OV_QK and OV_OV when type='both'
        """
        if composition_type == 'both' and separate_plots:
            # Create two separate plots
            self.visualize_layer(layer_target, 'OV_QK', cols_per_row, False)
            self.visualize_layer(layer_target, 'OV_OV', cols_per_row, False)
            return
        
        # Calculate grid dimensions
        if composition_type == 'both':
            total_plots = self.n_heads * 2
            n_cols = min(cols_per_row, self.n_heads)
            n_rows = math.ceil(self.n_heads / n_cols) * 2
        else:
            total_plots = self.n_heads
            n_cols = min(cols_per_row, self.n_heads)
            n_rows = math.ceil(self.n_heads / n_cols)
        
        # Create subplot titles
        titles = []
        if composition_type == 'both':
            # First set of rows for OV_QK
            for h in range(self.n_heads):
                titles.append(f'OV→QK to L{layer_target}H{h}')
            # Second set of rows for OV_OV
            for h in range(self.n_heads):
                titles.append(f'OV→OV to L{layer_target}H{h}')
        else:
            comp_label = 'OV→QK' if composition_type == 'OV_QK' else 'OV→OV'
            titles = [f'{comp_label} to L{layer_target}H{h}' for h in range(self.n_heads)]
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=titles[:total_plots],
            vertical_spacing=0.12 if n_rows > 1 else 0.1,
            horizontal_spacing=0.08
        )
        
        # Helper function to add heatmap
        def add_heatmap(scores, plot_idx, show_colorbar=False):
            row = (plot_idx // n_cols) + 1
            col = (plot_idx % n_cols) + 1
            
            hover_text = [[f'L{l}H{h}: {scores[l, h]:.4f}' 
                          for h in range(self.n_heads)] 
                          for l in range(layer_target)]
            
            heatmap = go.Heatmap(
                z=scores,
                colorscale='Blues',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showscale=show_colorbar,
                colorbar=dict(title='Score', len=0.8) if show_colorbar else None
            )
            fig.add_trace(heatmap, row=row, col=col)
            
            return row, col
        
        # Compute and plot
        plot_idx = 0
        
        if composition_type in ['OV_QK', 'both']:
            for head in range(self.n_heads):
                scores = self._compute_svd_scores(
                    self.OV, self.QK[layer_target, head], layer_target
                ).cpu().numpy()
                
                # Show colorbar only on the last plot of this type
                show_colorbar = (head == self.n_heads - 1) and (composition_type == 'OV_QK')
                row, col = add_heatmap(scores, plot_idx, show_colorbar)
                plot_idx += 1
        
        if composition_type in ['OV_OV', 'both']:
            if composition_type == 'both':
                # Skip to next set of rows if needed
                while plot_idx % n_cols != 0 and plot_idx < n_rows * n_cols // 2:
                    plot_idx += 1
                if plot_idx < n_rows * n_cols // 2:
                    plot_idx = n_rows * n_cols // 2
            
            for head in range(self.n_heads):
                scores = self._compute_svd_scores(
                    self.OV, self.OV[layer_target, head], layer_target
                ).cpu().numpy()
                
                # Show colorbar only on the last plot
                show_colorbar = (head == self.n_heads - 1)
                row, col = add_heatmap(scores, plot_idx, show_colorbar)
                plot_idx += 1
        
        # Update axes labels
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                # Calculate which plot this is
                current_plot = (row - 1) * n_cols + (col - 1)
                if current_plot < total_plots:
                    fig.update_xaxes(
                        title_text="Head" if row == n_rows or 
                                  (composition_type == 'both' and row == n_rows // 2) else "",
                        tickmode='linear', tick0=0, dtick=2,
                        row=row, col=col
                    )
                    fig.update_yaxes(
                        title_text="Layer" if col == 1 else "",
                        tickmode='linear', tick0=0, dtick=1,
                        autorange='reversed',
                        row=row, col=col
                    )
        
        # Calculate appropriate figure size
        subplot_width = 250
        subplot_height = 200
        width = subplot_width * n_cols + 100  # Extra space for colorbar
        height = subplot_height * n_rows + 150  # Extra space for titles
        
        title = f"Layer {layer_target} Composition Analysis"
        if composition_type != 'both':
            title += f" ({composition_type.replace('_', '→')})"
        
        fig.update_layout(
            title_text=title,
            width=width,
            height=height,
            showlegend=False
        )
        
        fig.show()
    
    def visualize_layer_separate(self, layer_target):
        """
        Create separate, well-formatted plots for OV_QK and OV_OV compositions.
        Each plot uses an optimal grid layout.
        """
        for comp_type in ['OV_QK', 'OV_OV']:
            self.visualize_layer(layer_target, comp_type, cols_per_row=4)

# Global analyzer instance to avoid reloading
_analyzer = None

def get_analyzer():
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CompositionAnalyzer()
    return _analyzer

if __name__ == "__main__":
    # Get analyzer (loads model only once)
    analyzer = get_analyzer()
    
    # Example: Visualize layer 5 with better layout
    for LAYER in range(1, N_LAYERS):
    
        # Option 1: Single plot with good layout (4 columns)
        analyzer.visualize_layer(LAYER, composition_type='OV_QK', cols_per_row=4)
        
        # Option 2: Both types in one figure (will be taller)
        # analyzer.visualize_layer(LAYER, composition_type='both', cols_per_row=4)
        
        # Option 3: Separate plots for each type (clearest)
        #analyzer.visualize_layer_separate(LAYER)