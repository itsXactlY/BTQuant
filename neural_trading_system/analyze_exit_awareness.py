# analyze_exit_awareness.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from rich.console import Console
from rich.table import Table

console = Console()

class ExitAwarenessAnalyzer:
    """
    🔍 Analyze how well the model learned exit management
    """
    
    def __init__(self, model_path: str, feature_extractor_path: str, test_data: dict, device='cuda'):
        self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        from models.architecture import NeuralTradingModel
        
        config = checkpoint.get('config', {})
        self.model = NeuralTradingModel(
            feature_dim=config.get('feature_dim', 500),
            d_model=config.get('d_model', 512),
            num_heads=config.get('num_heads', 16),
            num_layers=config.get('num_layers', 8),
            d_ff=config.get('d_ff', 2048),
            dropout=0.0,
            latent_dim=config.get('latent_dim', 16),
            seq_len=config.get('seq_len', 100)
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load feature extractor
        with open(feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        self.test_features = test_data['features']
        self.test_returns = test_data['returns']
        self.test_prices = test_data['prices']
        
        console.print("[green]✅ Exit Awareness Analyzer loaded[/green]")
    
    def analyze_take_profit_timing(self, num_samples=500):
        """
        Analyze how well model identifies optimal take-profit moments
        """
        console.print("\n[cyan]📊 Analyzing Take-Profit Timing...[/cyan]")
        
        results = {
            'took_profit_at_peak': [],
            'took_profit_too_early': [],
            'missed_peaks': [],
            'take_profit_probs': [],
            'hold_scores': [],
            'actual_future_returns': []
        }
        
        for i in range(min(num_samples, len(self.test_features) - 50)):
            seq = self.test_features[i]
            current_price = self.test_prices[i]
            
            # Simulate being in a profitable position
            entry_price = current_price * 0.98  # Simulated 2% profit
            unrealized_pnl = (current_price - entry_price) / entry_price
            
            if unrealized_pnl < 0.01:  # Skip if not enough profit
                continue
            
            # Get model prediction
            feature_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            position_context = {
                'unrealized_pnl': torch.tensor([[unrealized_pnl]], device=self.device),
                'time_in_position': torch.tensor([[10.0]], device=self.device),
            }
            
            with torch.no_grad():
                pred = self.model(feature_tensor, position_context=position_context)
            
            # Extract signals
            if pred['exit_signals']['profit_taking'] is not None:
                tp_prob = pred['exit_signals']['profit_taking']['take_profit_prob'].cpu().item()
                hold_score = pred['exit_signals']['let_winner_run']['hold_score'].cpu().item()
                
                # Look ahead to see what happened
                future_idx = min(i + 20, len(self.test_prices) - 1)
                future_prices = self.test_prices[i:future_idx]
                future_returns = (future_prices - current_price) / current_price
                
                peak_return = future_returns.max()
                bars_to_peak = future_returns.argmax()
                
                results['take_profit_probs'].append(tp_prob)
                results['hold_scores'].append(hold_score)
                results['actual_future_returns'].append(peak_return)
                
                # Classify decision quality
                if tp_prob > 0.7:  # Model says take profit
                    if bars_to_peak <= 3:  # Peak was near
                        results['took_profit_at_peak'].append(1)
                    else:  # Peak was further
                        results['took_profit_too_early'].append(1)
                else:  # Model says hold
                    if bars_to_peak > 5 and peak_return > unrealized_pnl * 1.5:
                        results['took_profit_at_peak'].append(1)  # Correctly held
                    elif bars_to_peak <= 3:
                        results['missed_peaks'].append(1)  # Should have exited
        
        # Calculate metrics
        total_decisions = len(results['take_profit_probs'])
        good_exits = len(results['took_profit_at_peak'])
        early_exits = len(results['took_profit_too_early'])
        missed_exits = len(results['missed_peaks'])
        
        # Display results
        table = Table(title="Take-Profit Timing Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Percentage", style="yellow")
        
        table.add_row("Total Profit Situations", str(total_decisions), "100%")
        table.add_row("Optimal Exits", str(good_exits), f"{good_exits/total_decisions*100:.1f}%")
        table.add_row("Too Early Exits", str(early_exits), f"{early_exits/total_decisions*100:.1f}%")
        table.add_row("Missed Exits", str(missed_exits), f"{missed_exits/total_decisions*100:.1f}%")
        
        console.print(table)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Take-profit probability distribution
        axes[0, 0].hist(results['take_profit_probs'], bins=30, alpha=0.7, color='green')
        axes[0, 0].set_xlabel('Take-Profit Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Take-Profit Signal Distribution')
        axes[0, 0].axvline(0.7, color='red', linestyle='--', label='Decision Threshold')
        axes[0, 0].legend()
        
        # 2. Hold score distribution
        axes[0, 1].hist(results['hold_scores'], bins=30, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Hold Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Let-Winner-Run Signal Distribution')
        axes[0, 1].axvline(0.7, color='red', linestyle='--', label='Strong Hold Threshold')
        axes[0, 1].legend()
        
        # 3. TP prob vs actual future return
        axes[1, 0].scatter(results['take_profit_probs'], results['actual_future_returns'], 
                          alpha=0.5, s=10)
        axes[1, 0].set_xlabel('Take-Profit Probability')
        axes[1, 0].set_ylabel('Actual Future Return (20 bars)')
        axes[1, 0].set_title('TP Signal vs Future Performance')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.3)
        axes[1, 0].axvline(0.7, color='red', linestyle='--', alpha=0.3)
        
        # 4. Hold score vs actual future return
        axes[1, 1].scatter(results['hold_scores'], results['actual_future_returns'], 
                          alpha=0.5, s=10, color='blue')
        axes[1, 1].set_xlabel('Hold Score')
        axes[1, 1].set_ylabel('Actual Future Return (20 bars)')
        axes[1, 1].set_title('Hold Signal vs Future Performance')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/take_profit_analysis.png', dpi=150, bbox_inches='tight')
        console.print("[green]💾 Saved: analysis/take_profit_analysis.png[/green]")
        plt.show()
        
        return results
    
    def analyze_stop_loss_timing(self, num_samples=500):
        """
        Analyze how well model cuts losses at the right time
        """
        console.print("\n[cyan]📊 Analyzing Stop-Loss Timing...[/cyan]")
        
        results = {
            'cut_before_crash': [],
            'cut_too_early': [],
            'held_through_recovery': [],
            'stop_loss_probs': [],
            'pattern_failure_scores': [],
            'actual_future_returns': []
        }
        
        for i in range(min(num_samples, len(self.test_features) - 50)):
            seq = self.test_features[i]
            current_price = self.test_prices[i]
            
            # Simulate being in a losing position
            entry_price = current_price * 1.02  # Simulated 2% loss
            unrealized_pnl = (current_price - entry_price) / entry_price
            
            if unrealized_pnl > -0.01:  # Skip if not in loss
                continue
            
            # Get model prediction
            feature_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            position_context = {
                'unrealized_pnl': torch.tensor([[unrealized_pnl]], device=self.device),
                'time_in_position': torch.tensor([[10.0]], device=self.device),
            }
            
            with torch.no_grad():
                pred = self.model(feature_tensor, position_context=position_context)
            
            # Extract signals
            if pred['exit_signals']['stop_loss'] is not None:
                sl_prob = pred['exit_signals']['stop_loss']['stop_loss_prob'].cpu().item()
                pattern_fail = pred['exit_signals']['stop_loss']['pattern_failure'].cpu().item()
                
                # Look ahead
                future_idx = min(i + 20, len(self.test_prices) - 1)
                future_prices = self.test_prices[i:future_idx]
                future_returns = (future_prices - current_price) / current_price
                
                min_return = future_returns.min()
                recovery = future_returns[-1]  # Final return after 20 bars
                
                results['stop_loss_probs'].append(sl_prob)
                results['pattern_failure_scores'].append(pattern_fail)
                results['actual_future_returns'].append(recovery)
                
                # Classify decision quality
                if sl_prob > 0.7:  # Model says cut loss
                    if min_return < unrealized_pnl * 1.5:  # Loss accelerated
                        results['cut_before_crash'].append(1)  # Good exit
                    elif recovery > 0:  # Would have recovered
                        results['cut_too_early'].append(1)  # Premature exit
                else:  # Model says hold
                    if recovery > unrealized_pnl * 0.5:  # Recovered somewhat
                        results['held_through_recovery'].append(1)  # Good hold
        
        # Calculate metrics
        total_decisions = len(results['stop_loss_probs'])
        avoided_crashes = len(results['cut_before_crash'])
        premature_cuts = len(results['cut_too_early'])
        successful_holds = len(results['held_through_recovery'])
        
        # Display results
        table = Table(title="Stop-Loss Timing Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Percentage", style="yellow")
        
        table.add_row("Total Loss Situations", str(total_decisions), "100%")
        table.add_row("Avoided Bigger Losses", str(avoided_crashes), f"{avoided_crashes/total_decisions*100:.1f}%")
        table.add_row("Premature Cuts", str(premature_cuts), f"{premature_cuts/total_decisions*100:.1f}%")
        table.add_row("Held Through Recovery", str(successful_holds), f"{successful_holds/total_decisions*100:.1f}%")
        
        console.print(table)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Stop-loss probability distribution
        axes[0, 0].hist(results['stop_loss_probs'], bins=30, alpha=0.7, color='red')
        axes[0, 0].set_xlabel('Stop-Loss Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Stop-Loss Signal Distribution')
        axes[0, 0].axvline(0.7, color='darkred', linestyle='--', label='Decision Threshold')
        axes[0, 0].legend()
        
        # 2. Pattern failure score distribution
        axes[0, 1].hist(results['pattern_failure_scores'], bins=30, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Pattern Failure Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Pattern Failure Detection')
        axes[0, 1].legend()
        
        # 3. SL prob vs actual future return
        axes[1, 0].scatter(results['stop_loss_probs'], results['actual_future_returns'], 
                          alpha=0.5, s=10, color='red')
        axes[1, 0].set_xlabel('Stop-Loss Probability')
        axes[1, 0].set_ylabel('Actual Future Return (20 bars)')
        axes[1, 0].set_title('SL Signal vs Future Performance')
        axes[1, 0].axhline(0, color='green', linestyle='--', alpha=0.3, label='Recovery')
        axes[1, 0].axvline(0.7, color='darkred', linestyle='--', alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Decision quality heatmap
        sl_bins = np.linspace(0, 1, 6)
        ret_bins = np.linspace(results['actual_future_returns'].min() if results['actual_future_returns'] else -0.1, 
                               results['actual_future_returns'].max() if results['actual_future_returns'] else 0.1, 6)
        
        if results['stop_loss_probs'] and results['actual_future_returns']:
            H, xedges, yedges = np.histogram2d(results['stop_loss_probs'], results['actual_future_returns'], 
                                               bins=[sl_bins, ret_bins])
            im = axes[1, 1].imshow(H.T, origin='lower', aspect='auto', cmap='RdYlGn',
                                  extent=[sl_bins[0], sl_bins[-1], ret_bins[0], ret_bins[-1]])
            axes[1, 1].set_xlabel('Stop-Loss Probability')
            axes[1, 1].set_ylabel('Future Return')
            axes[1, 1].set_title('Decision Quality Heatmap')
            plt.colorbar(im, ax=axes[1, 1], label='Frequency')
        
        plt.tight_layout()
        plt.savefig('analysis/stop_loss_analysis.png', dpi=150, bbox_inches='tight')
        console.print("[green]💾 Saved: analysis/stop_loss_analysis.png[/green]")
        plt.show()
        
        return results
    
    def analyze_regime_change_detection(self, num_samples=500):
        """
        Analyze regime change detection accuracy
        """
        console.print("\n[cyan]📊 Analyzing Regime Change Detection...[/cyan]")
        
        results = {
            'regime_scores': [],
            'vol_spikes': [],
            'volume_anomalies': [],
            'actual_vol_changes': [],
            'detected_before_crash': 0,
            'false_alarms': 0
        }
        
        for i in range(min(num_samples, len(self.test_features) - 20)):
            seq = self.test_features[i]
            
            feature_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(feature_tensor, position_context=None)
            
            # Extract regime signals
            regime_score = pred['regime_change']['regime_change_score'].cpu().item()
            vol_spike = pred['regime_change']['vol_change'].cpu().item()
            vol_anomaly = pred['regime_change']['volume_anomaly'].cpu().item()
            
            # Calculate actual volatility change
            future_returns = self.test_returns[i:i+20]
            if len(future_returns) >= 20:
                recent_vol = np.std(future_returns[:10])
                future_vol = np.std(future_returns[10:20])
                vol_change_ratio = future_vol / (recent_vol + 1e-10)
                
                results['regime_scores'].append(regime_score)
                results['vol_spikes'].append(vol_spike)
                results['volume_anomalies'].append(vol_anomaly)
                results['actual_vol_changes'].append(vol_change_ratio)
                
                # Check detection quality
                if regime_score > 0.7:  # High regime change signal
                    if vol_change_ratio > 1.5:  # Actual volatility increased
                        results['detected_before_crash'] += 1
                    else:
                        results['false_alarms'] += 1
        
        # Display results
        total_signals = results['detected_before_crash'] + results['false_alarms']
        precision = results['detected_before_crash'] / total_signals if total_signals > 0 else 0
        
        table = Table(title="Regime Change Detection Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Regime Change Signals", str(total_signals))
        table.add_row("Correctly Detected Changes", str(results['detected_before_crash']))
        table.add_row("False Alarms", str(results['false_alarms']))
        table.add_row("Precision", f"{precision*100:.1f}%")
        
        console.print(table)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Regime score vs actual vol change
        axes[0, 0].scatter(results['regime_scores'], results['actual_vol_changes'], alpha=0.5, s=10)
        axes[0, 0].set_xlabel('Regime Change Score')
        axes[0, 0].set_ylabel('Actual Vol Change Ratio')
        axes[0, 0].set_title('Regime Detection vs Actual Volatility Change')
        axes[0, 0].axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Significant Change')
        axes[0, 0].axvline(0.7, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # 2. Distribution of regime scores
        axes[0, 1].hist(results['regime_scores'], bins=30, alpha=0.7, color='purple')
        axes[0, 1].set_xlabel('Regime Change Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Regime Change Signal Distribution')
        axes[0, 1].axvline(0.7, color='red', linestyle='--', label='Alert Threshold')
        axes[0, 1].legend()
        
        # 3. Vol spike detection
        axes[1, 0].scatter(results['vol_spikes'], results['actual_vol_changes'], alpha=0.5, s=10, color='orange')
        axes[1, 0].set_xlabel('Vol Spike Score')
        axes[1, 0].set_ylabel('Actual Vol Change')
        axes[1, 0].set_title('Volatility Spike Detection')
        axes[1, 0].axhline(1.5, color='red', linestyle='--', alpha=0.5)
        
        # 4. Volume anomaly detection
        axes[1, 1].hist(results['volume_anomalies'], bins=30, alpha=0.7, color='brown')
        axes[1, 1].set_xlabel('Volume Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Volume Anomaly Distribution')
        
        plt.tight_layout()
        plt.savefig('analysis/regime_change_analysis.png', dpi=150, bbox_inches='tight')
        console.print("[green]💾 Saved: analysis/regime_change_analysis.png[/green]")
        plt.show()
        
        return results
    
    def run_full_analysis(self):
        """Run complete exit awareness analysis"""
        console.print("\n[bold cyan]🔬 Running Full Exit Awareness Analysis[/bold cyan]\n")
        
        Path("analysis").mkdir(exist_ok=True)
        
        tp_results = self.analyze_take_profit_timing()
        sl_results = self.analyze_stop_loss_timing()
        regime_results = self.analyze_regime_change_detection()
        
        console.print("\n[bold green]✅ Analysis Complete![/bold green]")
        console.print("📁 Results saved to ./analysis/")
        
        return {
            'take_profit': tp_results,
            'stop_loss': sl_results,
            'regime_change': regime_results
        }


# Usage example
if __name__ == '__main__':
    # Load test data
    import pickle
    with open('neural_data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    analyzer = ExitAwarenessAnalyzer(
        model_path='models/best_exit_aware_model.pt',
        feature_extractor_path='models/best_exit_aware_model_feature_extractor.pkl',
        test_data=test_data
    )
    
    results = analyzer.run_full_analysis()