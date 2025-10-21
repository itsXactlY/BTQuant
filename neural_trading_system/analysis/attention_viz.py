import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class AttentionAnalyzer:
    """
    Analyze what the neural network is paying attention to.
    Visualize which indicators matter most for decisions.
    """
    
    def __init__(self, model, feature_extractor, device='cpu'):
        self.model = model
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = device
    
    def extract_attention_weights(self, feature_sequence: np.ndarray) -> List[np.ndarray]:
        """
        Extract attention weights from all transformer layers.
        
        Returns:
            List of attention weight matrices [num_layers]
            Each matrix is [num_heads, seq_len, seq_len]
        """
        # Prepare input
        feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(feature_tensor)
        
        # Extract attention weights from all layers
        attention_weights = predictions['attention_weights']
        
        # Convert to numpy
        attention_np = [
            attn.cpu().numpy()[0]  # [num_heads, seq_len, seq_len]
            for attn in attention_weights
        ]
        
        return attention_np, predictions
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        layer_idx: int = -1,
        head_idx: int = 0,
        figsize=(12, 10)
    ):
        """
        Plot attention heatmap for a specific layer and head.
        
        Args:
            attention_weights: [num_heads, seq_len, seq_len]
            layer_idx: Which transformer layer (-1 = last)
            head_idx: Which attention head
        """
        attn = attention_weights[head_idx]  # [seq_len, seq_len]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            attn,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=range(attn.shape[1]),
            yticklabels=range(attn.shape[0])
        )
        plt.title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position (Historical Bars)')
        plt.ylabel('Query Position (Historical Bars)')
        plt.tight_layout()
        plt.show()
    
    def plot_attention_timeline(
        self,
        attention_weights_list: List[np.ndarray],
        figsize=(15, 8)
    ):
        """
        Plot how attention evolves across layers.
        Shows which historical bars are most important.
        """
        num_layers = len(attention_weights_list)
        
        fig, axes = plt.subplots(num_layers, 1, figsize=figsize, sharex=True)
        
        for layer_idx, attn_weights in enumerate(attention_weights_list):
            # Average across heads
            attn_avg = attn_weights.mean(axis=0)  # [seq_len, seq_len]
            
            # Focus on attention to the last position (current decision)
            attention_to_now = attn_avg[-1, :]  # [seq_len]
            
            ax = axes[layer_idx] if num_layers > 1 else axes
            ax.plot(attention_to_now, linewidth=2)
            ax.set_ylabel(f'Layer {layer_idx}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(attention_to_now))
        
        axes[-1].set_xlabel('Historical Bar Index (0 = oldest, -1 = current)')
        fig.suptitle('Attention Evolution Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def compute_feature_importance(
        self,
        test_sequences: np.ndarray,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute feature importance using gradient-based attribution.
        
        Shows which indicator features matter most for predictions.
        """
        self.model.eval()
        importances = []
        
        for i in range(min(num_samples, len(test_sequences))):
            seq = test_sequences[i]
            
            # Prepare input
            feature_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            feature_tensor.requires_grad = True
            
            # Forward pass
            predictions = self.model(feature_tensor)
            
            # Use entry probability as the target
            target = predictions['entry_prob']
            
            # Backward pass
            target.backward()
            
            # Get gradients
            gradients = feature_tensor.grad.cpu().numpy()[0]  # [seq_len, feature_dim]
            
            # Average gradient magnitude across sequence
            importance = np.abs(gradients).mean(axis=0)  # [feature_dim]
            importances.append(importance)
        
        # Average across samples
        avg_importance = np.mean(importances, axis=0)
        
        # Create feature names (you'll need to map these to actual indicator names)
        feature_names = [f'feature_{i}' for i in range(len(avg_importance))]
        
        importance_dict = dict(zip(feature_names, avg_importance))
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 30,
        figsize=(12, 8)
    ):
        """Plot top N most important features."""
        # Get top N
        top_features = list(importance_dict.items())[:top_n]
        names, values = zip(*top_features)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(names)), values)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()
    
    def visualize_regime_space(
        self,
        test_sequences: np.ndarray,
        test_returns: np.ndarray,
        figsize=(10, 8)
    ):
        """
        Visualize the learned market regime space (VAE latent space).
        Shows how the model clusters different market conditions.
        """
        regime_embeddings = []
        returns_list = []
        
        for seq, ret in zip(test_sequences[:500], test_returns[:500]):
            feature_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(feature_tensor)
                regime_z = predictions['regime_z'].cpu().numpy()[0]
                regime_embeddings.append(regime_z)
                returns_list.append(ret)
        
        regime_embeddings = np.array(regime_embeddings)
        returns_list = np.array(returns_list)
        
        # Use PCA for 2D visualization if latent_dim > 2
        from sklearn.decomposition import PCA
        
        if regime_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            regime_2d = pca.fit_transform(regime_embeddings)
        else:
            regime_2d = regime_embeddings
        
        # Color by future returns
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            regime_2d[:, 0],
            regime_2d[:, 1],
            c=returns_list,
            cmap='RdYlGn',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label='Future Return')
        plt.xlabel('Regime Dimension 1')
        plt.ylabel('Regime Dimension 2')
        plt.title('Learned Market Regime Space (colored by future returns)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_decision_boundary(
        self,
        test_sequences: np.ndarray,
        test_returns: np.ndarray
    ):
        """
        Analyze the model's decision boundary.
        Shows entry probability vs actual returns.
        """
        entry_probs = []
        actual_returns = []
        
        for seq, ret in zip(test_sequences[:1000], test_returns[:1000]):
            feature_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(feature_tensor)
                entry_prob = predictions['entry_prob'].cpu().item()
                entry_probs.append(entry_prob)
                actual_returns.append(ret)
        
        entry_probs = np.array(entry_probs)
        actual_returns = np.array(actual_returns)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(entry_probs, actual_returns, alpha=0.3, s=10)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0.5, color='b', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Entry Probability')
        axes[0].set_ylabel('Actual Return')
        axes[0].set_title('Entry Probability vs Actual Return')
        axes[0].grid(True, alpha=0.3)
        
        # Binned analysis
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_returns = []
        
        for i in range(len(bins) - 1):
            mask = (entry_probs >= bins[i]) & (entry_probs < bins[i+1])
            if mask.sum() > 0:
                bin_returns.append(actual_returns[mask].mean())
            else:
                bin_returns.append(0)
        
        axes[1].bar(bin_centers, bin_returns, width=0.08, alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Entry Probability Bin')
        axes[1].set_ylabel('Average Actual Return')
        axes[1].set_title('Average Return by Confidence Level')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print calibration stats
        high_conf = actual_returns[entry_probs > 0.7]
        low_conf = actual_returns[entry_probs < 0.3]
        
        print("\n📊 Decision Boundary Analysis:")
        print(f"High confidence (>0.7) samples: {len(high_conf)}")
        print(f"  - Win rate: {(high_conf > 0).mean():.2%}")
        print(f"  - Avg return: {high_conf.mean():.2%}")
        print(f"\nLow confidence (<0.3) samples: {len(low_conf)}")
        print(f"  - Win rate: {(low_conf > 0).mean():.2%}")
        print(f"  - Avg return: {low_conf.mean():.2%}")