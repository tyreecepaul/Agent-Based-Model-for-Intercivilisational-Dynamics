"""
Batch simulation environment for Dark Forest simulations.
Runs multiple simulations, collects data, and performs statistical analysis.
"""
import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass, field
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from agent import Civilisation
from env import Galaxy
from config import GALAXY_SIZE, NUM_STEPS

# Output directory for all results
OUTPUT_DIR = "res"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SpendingTracker:
    """Track spending patterns for a civilization during simulation."""
    
    def __init__(self, civ_name: str):
        self.civ_name = civ_name
        self.weapons_spent = []
        self.search_spent = []
        self.camo_spent = []
        self.growth_spent = []
        self.dismantling_events = []
    
    def record_spending(self, allocation: Dict[str, float]):
        """Record spending for this time step."""
        self.weapons_spent.append(allocation.get('weapons', 0.0))
        self.search_spent.append(allocation.get('search', 0.0))
        self.camo_spent.append(allocation.get('camo', 0.0))
        self.growth_spent.append(allocation.get('growth', 0.0))
    
    def record_dismantling(self, step: int, category: str, amount: float, recovered: float):
        """Record a dismantling event."""
        self.dismantling_events.append({
            'step': step,
            'category': category,
            'amount': amount,
            'recovered': recovered
        })
    
    def get_total_spent(self) -> Dict[str, float]:
        """Get total spending by category."""
        return {
            'weapons': sum(self.weapons_spent),
            'search': sum(self.search_spent),
            'camo': sum(self.camo_spent),
            'growth': sum(self.growth_spent)
        }
    
    def get_spending_history(self) -> Dict[str, List[float]]:
        """Get full spending history."""
        return {
            'weapons': self.weapons_spent.copy(),
            'search': self.search_spent.copy(),
            'camo': self.camo_spent.copy(),
            'growth': self.growth_spent.copy()
        }


@dataclass
class SimulationResult:
    """Store results from a single simulation run."""
    simulation_id: int
    winner_name: str = None
    total_steps: int = 0
    final_civs: int = 0
    
    # Winner statistics
    winner_resources: float = 0.0
    winner_tech: float = 0.0
    winner_weapon_investment: float = 0.0
    winner_search_investment: float = 0.0
    winner_camo_investment: float = 0.0
    winner_survival_drive: float = 0.0
    winner_known_civs: int = 0
    
    # Winner spending patterns
    winner_total_weapons_spent: float = 0.0
    winner_total_search_spent: float = 0.0
    winner_total_camo_spent: float = 0.0
    winner_total_growth_spent: float = 0.0
    winner_dismantling_events: int = 0
    winner_resources_recovered: float = 0.0
    
    # Aggregate statistics
    total_interactions: int = 0
    total_attacks: int = 0
    total_cooperations: int = 0
    avg_resource_pressure: float = 0.0
    max_resource_pressure: float = 0.0
    
    # Time series data
    resource_pressure_history: List[float] = field(default_factory=list)
    active_civs_history: List[int] = field(default_factory=list)
    total_resources_history: List[float] = field(default_factory=list)
    
    # Spending pattern time series (per-civilization tracking)
    spending_history: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # Format: {civ_name: {'weapons': [...], 'search': [...], 'camo': [...], 'growth': [...]}}
    
    dismantling_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Format: {civ_name: [{'step': int, 'category': str, 'amount': float, 'recovered': float}, ...]}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return {
            'simulation_id': self.simulation_id,
            'winner_name': self.winner_name,
            'total_steps': self.total_steps,
            'final_civs': self.final_civs,
            'winner_resources': self.winner_resources,
            'winner_tech': self.winner_tech,
            'winner_weapon_investment': self.winner_weapon_investment,
            'winner_search_investment': self.winner_search_investment,
            'winner_camo_investment': self.winner_camo_investment,
            'winner_survival_drive': self.winner_survival_drive,
            'winner_known_civs': self.winner_known_civs,
            'winner_total_weapons_spent': self.winner_total_weapons_spent,
            'winner_total_search_spent': self.winner_total_search_spent,
            'winner_total_camo_spent': self.winner_total_camo_spent,
            'winner_total_growth_spent': self.winner_total_growth_spent,
            'winner_dismantling_events': self.winner_dismantling_events,
            'winner_resources_recovered': self.winner_resources_recovered,
            'total_interactions': self.total_interactions,
            'total_attacks': self.total_attacks,
            'total_cooperations': self.total_cooperations,
            'avg_resource_pressure': self.avg_resource_pressure,
            'max_resource_pressure': self.max_resource_pressure,
        }


class BatchSimulator:
    """Run multiple simulations and collect aggregate statistics."""
    
    def __init__(self, n_simulations: int = 10, num_civs: int = 4, verbose: bool = False):
        """
        Initialize batch simulator.
        
        Args:
            n_simulations: Number of simulations to run
            num_civs: Number of civilizations per simulation
            verbose: Print detailed output during simulations
        """
        self.n_simulations = n_simulations
        self.num_civs = num_civs
        self.verbose = verbose
        self.results: List[SimulationResult] = []
        
        self.civ_names = [
            "Sol", "Alpha Centauri", "Proxima B", "Xylos", 
            "Kepler-442", "TRAPPIST-1", "Gliese 667", "Tau Ceti",
            "Epsilon Eridani", "Barnard's Star"
        ]
        
        self.civ_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
            "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2",
            "#F8B739", "#52B788"
        ]
    
    def run_single_simulation(self, sim_id: int) -> SimulationResult:
        """
        Run a single simulation and collect data.
        
        Args:
            sim_id: Unique identifier for this simulation
            
        Returns:
            SimulationResult with collected data
        """
        result = SimulationResult(simulation_id=sim_id)
        
        # Create galaxy with suppressed output if not verbose
        if not self.verbose:
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        try:
            galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE)
            
            # Add civilizations at random positions
            selected_indices = random.sample(range(len(self.civ_names)), self.num_civs)
            for idx in selected_indices:
                name = self.civ_names[idx]
                color = self.civ_colors[idx]
                x = random.uniform(0, GALAXY_SIZE)
                y = random.uniform(0, GALAXY_SIZE)
                galaxy.import_civilisation_default(name, x, y, color)
            
            # Create spending trackers for each civilization
            spending_trackers = {civ.name: SpendingTracker(civ.name) for civ in galaxy.civilisations}
            
            # Track initial investments for each civilization
            initial_investments = {
                civ.name: {
                    'weapons': civ.total_invested_in_weapons,
                    'search': civ.total_invested_in_search,
                    'camo': civ.total_invested_in_camo
                }
                for civ in galaxy.civilisations
            }
            
            # Run simulation
            step = 0
            for step in range(NUM_STEPS):
                # Collect time series data
                result.resource_pressure_history.append(galaxy.get_resource_pressure())
                result.active_civs_history.append(len([c for c in galaxy.civilisations if c.check_is_active()]))
                result.total_resources_history.append(galaxy.get_total_active_resources())
                
                # Track spending for each civilization before allocation
                pre_step_investments = {
                    civ.name: {
                        'weapons': civ.total_invested_in_weapons,
                        'search': civ.total_invested_in_search,
                        'camo': civ.total_invested_in_camo,
                        'resources': civ.resources
                    }
                    for civ in galaxy.civilisations
                }
                
                # Run one step
                if not galaxy.run_simulation():
                    break
                
                # Track spending changes after step
                for civ in galaxy.civilisations:
                    if civ.name in spending_trackers:
                        tracker = spending_trackers[civ.name]
                        pre = pre_step_investments.get(civ.name, {})
                        
                        # Calculate spending this step (increase in total_invested)
                        weapons_spent = civ.total_invested_in_weapons - pre.get('weapons', 0)
                        search_spent = civ.total_invested_in_search - pre.get('search', 0)
                        camo_spent = civ.total_invested_in_camo - pre.get('camo', 0)
                        
                        # Estimate growth spending (resource increase from growth)
                        resource_change = civ.resources - pre.get('resources', 0)
                        # Growth would increase resources, investments would decrease them
                        # This is an approximation
                        growth_spent = max(0, -resource_change - weapons_spent - search_spent - camo_spent)
                        
                        # Detect dismantling (decrease in total_invested)
                        if weapons_spent < -0.01:  # Significant decrease
                            recovered = -weapons_spent * 0.8  # Approximate recovery
                            tracker.record_dismantling(step, 'weapons', -weapons_spent, recovered)
                            weapons_spent = 0  # Don't count as spending
                        
                        if search_spent < -0.01:
                            recovered = -search_spent * 0.8
                            tracker.record_dismantling(step, 'search', -search_spent, recovered)
                            search_spent = 0
                        
                        if camo_spent < -0.01:
                            recovered = -camo_spent * 0.8
                            tracker.record_dismantling(step, 'camo', -camo_spent, recovered)
                            camo_spent = 0
                        
                        # Record positive spending only
                        tracker.record_spending({
                            'weapons': max(0, weapons_spent),
                            'search': max(0, search_spent),
                            'camo': max(0, camo_spent),
                            'growth': max(0, growth_spent)
                        })
            
            result.total_steps = step + 1
            result.final_civs = len(galaxy.civilisations)
            
            # Calculate aggregate statistics
            if result.resource_pressure_history:
                result.avg_resource_pressure = np.mean(result.resource_pressure_history)
                result.max_resource_pressure = np.max(result.resource_pressure_history)
            
            # Store spending histories for all civilizations
            for civ_name, tracker in spending_trackers.items():
                result.spending_history[civ_name] = tracker.get_spending_history()
                result.dismantling_history[civ_name] = tracker.dismantling_events.copy()
            
            # Collect winner data
            if galaxy.civilisations:
                winner = galaxy.civilisations[0]
                result.winner_name = winner.name
                result.winner_resources = winner.resources
                result.winner_tech = winner.tech
                result.winner_weapon_investment = winner.weapon_investment
                result.winner_search_investment = winner.search_investment
                result.winner_camo_investment = winner.camo_investment
                result.winner_survival_drive = winner.survival_drive
                result.winner_known_civs = len(winner.known_civs)
                
                # Add spending pattern data for winner
                if winner.name in spending_trackers:
                    tracker = spending_trackers[winner.name]
                    totals = tracker.get_total_spent()
                    result.winner_total_weapons_spent = totals['weapons']
                    result.winner_total_search_spent = totals['search']
                    result.winner_total_camo_spent = totals['camo']
                    result.winner_total_growth_spent = totals['growth']
                    result.winner_dismantling_events = len(tracker.dismantling_events)
                    result.winner_resources_recovered = sum(
                        event['recovered'] for event in tracker.dismantling_events
                    )
        
        finally:
            if not self.verbose:
                sys.stdout = old_stdout
        
        return result
    
    def run_batch(self) -> pd.DataFrame:
        """
        Run all simulations and return results as DataFrame.
        
        Returns:
            DataFrame with all simulation results
        """
        print(f"\n{'='*70}")
        print(f"BATCH SIMULATION: Running {self.n_simulations} simulations")
        print(f"{'='*70}\n")
        
        self.results = []
        
        for i in range(self.n_simulations):
            print(f"Running simulation {i+1}/{self.n_simulations}...", end=' ')
            result = self.run_single_simulation(i)
            self.results.append(result)
            print(f"✓ Completed in {result.total_steps} steps (Winner: {result.winner_name or 'None'})")
        
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE: {len(self.results)} simulations finished")
        print(f"{'='*70}\n")
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])
        return df
    
    def save_results(self, filename: str = None):
        """Save results to JSON file in res directory."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
        
        # Ensure filename is in res directory
        if not filename.startswith(OUTPUT_DIR):
            filename = os.path.join(OUTPUT_DIR, filename)
        
        data = {
            'metadata': {
                'n_simulations': self.n_simulations,
                'num_civs': self.num_civs,
                'timestamp': datetime.now().isoformat()
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")


class SimulationAnalyzer:
    """Analyze and visualize batch simulation results."""
    
    def __init__(self, results: List[SimulationResult], df: pd.DataFrame):
        """
        Initialize analyzer with simulation results.
        
        Args:
            results: List of SimulationResult objects
            df: DataFrame with simulation data
        """
        self.results = results
        self.df = df
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
    
    def print_summary_statistics(self):
        """Print summary statistics of all simulations."""
        print(f"\n{'='*70}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*70}\n")
        
        print(f"Total Simulations: {len(self.results)}")
        print(f"\nWinner Distribution:")
        winner_counts = self.df['winner_name'].value_counts()
        for name, count in winner_counts.items():
            percentage = (count / len(self.results)) * 100
            print(f"  {name}: {count} ({percentage:.1f}%)")
        
        print(f"\nSimulation Duration:")
        print(f"  Mean steps: {self.df['total_steps'].mean():.1f}")
        print(f"  Median steps: {self.df['total_steps'].median():.1f}")
        print(f"  Std dev: {self.df['total_steps'].std():.1f}")
        
        print(f"\nWinner Characteristics:")
        numeric_cols = ['winner_resources', 'winner_tech', 'winner_weapon_investment', 
                       'winner_search_investment', 'winner_camo_investment']
        
        for col in numeric_cols:
            print(f"  {col.replace('winner_', '').replace('_', ' ').title()}:")
            print(f"    Mean: {self.df[col].mean():.2f}")
            print(f"    Std: {self.df[col].std():.2f}")
        
        print(f"\nWinner Spending Patterns:")
        spending_cols = ['winner_total_weapons_spent', 'winner_total_search_spent',
                        'winner_total_camo_spent', 'winner_total_growth_spent']
        
        for col in spending_cols:
            if col in self.df.columns:
                print(f"  {col.replace('winner_total_', '').replace('_spent', '').title()}:")
                print(f"    Mean: {self.df[col].mean():.2f}")
                print(f"    Std: {self.df[col].std():.2f}")
        
        if 'winner_dismantling_events' in self.df.columns:
            print(f"\nDismantling Activity:")
            print(f"  Winners with dismantling: {(self.df['winner_dismantling_events'] > 0).sum()}")
            print(f"  Avg dismantling events: {self.df['winner_dismantling_events'].mean():.2f}")
            print(f"  Avg resources recovered: {self.df['winner_resources_recovered'].mean():.2f}")
        
        print(f"\nResource Pressure:")
        print(f"  Avg pressure (mean): {self.df['avg_resource_pressure'].mean():.2%}")
        print(f"  Max pressure (mean): {self.df['max_resource_pressure'].mean():.2%}")
        
        print(f"\n{'='*70}\n")
    
    def plot_winner_characteristics(self):
        """Plot distribution of winner characteristics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('🏆 Winner Characteristics Distribution', fontsize=18, fontweight='bold', color='#ffffff')
        
        characteristics = [
            ('winner_resources', '💰 Resources'),
            ('winner_tech', '⚗️ Technology'),
            ('winner_weapon_investment', '⚔️ Weapon Investment'),
            ('winner_search_investment', '🔍 Search Investment'),
            ('winner_camo_investment', '🎭 Camouflage Investment'),
            ('winner_known_civs', '👁️ Known Civilizations')
        ]
        
        colors = ['#00ff88', '#00ddff', '#ff3366', '#ffcc00', '#aa44ff', '#ff8800']
        
        for idx, (col, title) in enumerate(characteristics):
            ax = axes[idx // 3, idx % 3]
            ax.set_facecolor('#0f0f0f')
            
            # Histogram with enhanced styling
            n, bins, patches = ax.hist(self.df[col], bins=15, alpha=0.8, 
                                      edgecolor='#ffffff', linewidth=1.5, 
                                      color=colors[idx])
            
            # Color gradient for bars
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[idx])
                patch.set_alpha(0.7 + 0.3 * (i / len(patches)))
            
            ax.set_title(title, fontweight='bold', fontsize=12, color='#ffffff', pad=10)
            ax.set_xlabel(title.split()[-1], fontsize=10, color='#cccccc')
            ax.set_ylabel('Frequency', fontsize=10, color='#cccccc')
            ax.tick_params(colors='#999999', labelsize=9)
            ax.grid(True, alpha=0.1, color='#666666')
            
            # Add mean line with glow effect
            mean_val = self.df[col].mean()
            ax.axvline(mean_val, color='#ffffff', linestyle='--', linewidth=2.5, 
                      label=f'Mean: {mean_val:.2f}', alpha=0.9, zorder=10)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')
                spine.set_linewidth(1.5)
            
            legend = ax.legend(framealpha=0.9, facecolor='#1a1a1a', 
                             edgecolor='#666666', labelcolor='#ffffff')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'winner_characteristics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of winner characteristics."""
        numeric_cols = [
            'total_steps', 'winner_resources', 'winner_tech',
            'winner_weapon_investment', 'winner_search_investment', 
            'winner_camo_investment', 'avg_resource_pressure'
        ]
        
        corr_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0f0f0f')
        
        # Enhanced heatmap with custom colormap
        im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=-1, vmax=1, interpolation='nearest')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_facecolor('#1a1a1a')
        cbar.ax.tick_params(colors='#ffffff', labelsize=10)
        cbar.outline.set_edgecolor('#444444')
        cbar.outline.set_linewidth(1.5)
        
        # Tick labels
        labels = ['Steps', 'Resources', 'Tech', 'Weapon', 'Search', 'Camo', 'Pressure']
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=11, color='#ffffff', rotation=45, ha='right')
        ax.set_yticklabels(labels, fontsize=11, color='#ffffff')
        
        # Annotate cells
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                value = corr_matrix.iloc[i, j]
                color = '#000000' if abs(value) > 0.5 else '#ffffff'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=color, fontsize=10, fontweight='bold')
        
        # Grid
        ax.set_xticks(np.arange(len(labels))-.5, minor=True)
        ax.set_yticks(np.arange(len(labels))-.5, minor=True)
        ax.grid(which='minor', color='#444444', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)
        
        # Title
        ax.set_title('📊 Correlation Matrix: Simulation Variables', 
                    fontsize=16, fontweight='bold', pad=20, color='#ffffff')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def plot_time_series_comparison(self):
        """Plot time series data from multiple simulations."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('📈 Time Series Comparison Across Simulations', 
                    fontsize=18, fontweight='bold', color='#ffffff')
        
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
        # Resource Pressure
        ax1 = axes[0]
        ax1.set_facecolor('#0f0f0f')
        for i, result in enumerate(self.results[:10]):  # Plot first 10 for clarity
            ax1.plot(result.resource_pressure_history, alpha=0.7, linewidth=1.5, 
                    color=colors[i], label=f'Sim {i+1}')
        ax1.set_title('📊 Resource Pressure Over Time', fontweight='bold', 
                     fontsize=13, color='#ffffff', pad=10)
        ax1.set_xlabel('Time Step', fontsize=11, color='#cccccc')
        ax1.set_ylabel('Resource Pressure', fontsize=11, color='#cccccc')
        ax1.set_ylim(0, 1)
        ax1.tick_params(colors='#999999', labelsize=9)
        ax1.grid(True, alpha=0.15, color='#666666', linestyle='--')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.5)
        
        # Active Civilizations
        ax2 = axes[1]
        ax2.set_facecolor('#0f0f0f')
        for i, result in enumerate(self.results[:10]):
            ax2.plot(result.active_civs_history, alpha=0.7, linewidth=1.5, 
                    color=colors[i])
        ax2.set_title('🌌 Active Civilizations Over Time', fontweight='bold', 
                     fontsize=13, color='#ffffff', pad=10)
        ax2.set_xlabel('Time Step', fontsize=11, color='#cccccc')
        ax2.set_ylabel('Active Civilizations', fontsize=11, color='#cccccc')
        ax2.tick_params(colors='#999999', labelsize=9)
        ax2.grid(True, alpha=0.15, color='#666666', linestyle='--')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.5)
        
        # Total Resources
        ax3 = axes[2]
        ax3.set_facecolor('#0f0f0f')
        for i, result in enumerate(self.results[:10]):
            ax3.plot(result.total_resources_history, alpha=0.7, linewidth=1.5, 
                    color=colors[i])
        ax3.set_title('💰 Total Resources in Use Over Time', fontweight='bold', 
                     fontsize=13, color='#ffffff', pad=10)
        ax3.set_xlabel('Time Step', fontsize=11, color='#cccccc')
        ax3.set_ylabel('Total Resources', fontsize=11, color='#cccccc')
        ax3.tick_params(colors='#999999', labelsize=9)
        ax3.grid(True, alpha=0.15, color='#666666', linestyle='--')
        for spine in ax3.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'time_series_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def plot_spending_patterns(self):
        """Plot winner spending patterns across simulations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('💸 Winner Spending Patterns Analysis', fontsize=18, 
                    fontweight='bold', color='#ffffff')
        
        spending_categories = [
            ('winner_total_weapons_spent', '⚔️ Weapons Spending', axes[0, 0], '#ff3366'),
            ('winner_total_search_spent', '🔍 Search Spending', axes[0, 1], '#ffcc00'),
            ('winner_total_camo_spent', '🎭 Camouflage Spending', axes[1, 0], '#aa44ff'),
            ('winner_total_growth_spent', '📈 Growth Spending', axes[1, 1], '#00ff88')
        ]
        
        for col, title, ax, color in spending_categories:
            ax.set_facecolor('#0f0f0f')
            
            if col in self.df.columns:
                n, bins, patches = ax.hist(self.df[col], bins=20, alpha=0.8, 
                                          edgecolor='#ffffff', linewidth=1.5, color=color)
                
                # Gradient effect
                for i, patch in enumerate(patches):
                    patch.set_alpha(0.6 + 0.4 * (i / len(patches)))
                
                ax.set_title(title, fontweight='bold', fontsize=12, color='#ffffff', pad=10)
                ax.set_xlabel('Total Resources Spent', fontsize=10, color='#cccccc')
                ax.set_ylabel('Frequency', fontsize=10, color='#cccccc')
                ax.tick_params(colors='#999999', labelsize=9)
                ax.grid(True, alpha=0.1, color='#666666')
                
                mean_val = self.df[col].mean()
                ax.axvline(mean_val, color='#ffffff', linestyle='--', linewidth=2.5, 
                          label=f'Mean: {mean_val:.1f}', alpha=0.9, zorder=10)
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444444')
                    spine.set_linewidth(1.5)
                
                legend = ax.legend(framealpha=0.9, facecolor='#1a1a1a', 
                                 edgecolor='#666666', labelcolor='#ffffff')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'spending_patterns.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def plot_spending_vs_outcome(self):
        """Plot relationship between spending and winning."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('📊 Spending vs Final Investment Levels', fontsize=18, 
                    fontweight='bold', color='#ffffff')
        
        comparisons = [
            ('winner_total_weapons_spent', 'winner_weapon_investment', 'Weapons', '#ff3366'),
            ('winner_total_search_spent', 'winner_search_investment', 'Search', '#ffcc00'),
            ('winner_total_camo_spent', 'winner_camo_investment', 'Camouflage', '#aa44ff'),
            ('winner_total_growth_spent', 'winner_resources', 'Growth → Resources', '#00ff88')
        ]
        
        for idx, (spent_col, final_col, category, color) in enumerate(comparisons):
            ax = axes[idx // 2, idx % 2]
            ax.set_facecolor('#0f0f0f')
            
            if spent_col in self.df.columns and final_col in self.df.columns:
                # Scatter plot with glow
                ax.scatter(self.df[spent_col], self.df[final_col], 
                          alpha=0.3, s=100, color=color, edgecolors='none')
                ax.scatter(self.df[spent_col], self.df[final_col], 
                          alpha=0.8, s=50, color=color, edgecolors='#ffffff', linewidth=1)
                
                # Trend line
                if len(self.df) > 1:
                    z = np.polyfit(self.df[spent_col], self.df[final_col], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(self.df[spent_col].min(), self.df[spent_col].max(), 100)
                    ax.plot(x_line, p(x_line), color='#ffffff', linestyle='--', 
                           alpha=0.9, linewidth=2.5, label='Trend', zorder=10)
                
                ax.set_xlabel(f'Total {category} Spent', fontsize=11, color='#cccccc')
                ax.set_ylabel(f'Final {category} Level', fontsize=11, color='#cccccc')
                ax.set_title(f'{category}: Spending vs Final Level', 
                           fontweight='bold', fontsize=12, color='#ffffff', pad=10)
                ax.tick_params(colors='#999999', labelsize=9)
                ax.grid(True, alpha=0.1, color='#666666')
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444444')
                    spine.set_linewidth(1.5)
                
                legend = ax.legend(framealpha=0.9, facecolor='#1a1a1a', 
                                 edgecolor='#666666', labelcolor='#ffffff')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'spending_vs_outcome.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def plot_spending_time_series(self, max_simulations: int = 5):
        """Plot spending over time for winners from selected simulations."""
        # Select simulations with winners
        winner_results = [r for r in self.results if r.winner_name and r.winner_name in r.spending_history]
        
        if not winner_results:
            print("⚠️  No winner spending data available")
            return
        
        # Take up to max_simulations
        selected_results = winner_results[:min(max_simulations, len(winner_results))]
        
        fig, axes = plt.subplots(len(selected_results), 1, figsize=(16, 4 * len(selected_results)))
        fig.patch.set_facecolor('#0a0a0a')
        if len(selected_results) == 1:
            axes = [axes]
        
        fig.suptitle('💸 Winner Spending Patterns Over Time', fontsize=18, 
                    fontweight='bold', color='#ffffff')
        
        colors = {'weapons': '#ff3366', 'search': '#ffcc00', 'camo': '#aa44ff', 'growth': '#00ff88'}
        
        for idx, result in enumerate(selected_results):
            ax = axes[idx]
            ax.set_facecolor('#0f0f0f')
            winner_spending = result.spending_history.get(result.winner_name, {})
            
            if not winner_spending:
                continue
            
            # Plot cumulative spending
            steps = range(len(winner_spending.get('weapons', [])))
            
            for category in ['weapons', 'search', 'camo', 'growth']:
                if category in winner_spending:
                    spending = winner_spending[category]
                    cumulative = np.cumsum(spending)
                    ax.plot(steps, cumulative, label=category.title(), 
                           color=colors[category], linewidth=2.5, alpha=0.9)
            
            ax.set_title(f'🏆 Simulation {result.simulation_id}: {result.winner_name}', 
                        fontweight='bold', fontsize=12, color='#ffffff', pad=10)
            ax.set_xlabel('Time Step', fontsize=11, color='#cccccc')
            ax.set_ylabel('Cumulative Resources Spent', fontsize=11, color='#cccccc')
            ax.tick_params(colors='#999999', labelsize=9)
            ax.grid(True, alpha=0.1, color='#666666')
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')
                spine.set_linewidth(1.5)
            
            # Add dismantling markers
            if result.winner_name in result.dismantling_history:
                dismantling_events = result.dismantling_history[result.winner_name]
                for event in dismantling_events:
                    ax.axvline(event['step'], color='#ff00ff', linestyle=':', 
                             alpha=0.7, linewidth=2)
                
                if dismantling_events:
                    ax.plot([], [], color='#ff00ff', linestyle=':', linewidth=2, 
                           label=f'💥 Dismantling ({len(dismantling_events)} events)')
            
            legend = ax.legend(loc='upper left', framealpha=0.9, facecolor='#1a1a1a', 
                             edgecolor='#666666', labelcolor='#ffffff')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'spending_time_series.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def plot_dismantling_analysis(self):
        """Analyze dismantling behavior across simulations."""
        if 'winner_dismantling_events' not in self.df.columns:
            print("⚠️  No dismantling data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('💥 Dismantling Behavior Analysis', fontsize=18, 
                    fontweight='bold', color='#ffffff')
        
        # Dismantling frequency
        ax1 = axes[0]
        ax1.set_facecolor('#0f0f0f')
        dismantling_counts = self.df['winner_dismantling_events'].value_counts().sort_index()
        
        bars = ax1.bar(dismantling_counts.index, dismantling_counts.values, 
                      alpha=0.85, edgecolor='#ffffff', linewidth=1.5, color='#ff6b35')
        
        # Gradient effect on bars
        for i, bar in enumerate(bars):
            bar.set_alpha(0.7 + 0.3 * (i / len(bars)))
        
        ax1.set_xlabel('Number of Dismantling Events', fontsize=11, color='#cccccc')
        ax1.set_ylabel('Number of Winners', fontsize=11, color='#cccccc')
        ax1.set_title('Dismantling Event Distribution', fontweight='bold', 
                     fontsize=12, color='#ffffff', pad=10)
        ax1.tick_params(colors='#999999', labelsize=9)
        ax1.grid(axis='y', alpha=0.1, color='#666666')
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.5)
        
        # Resources recovered vs final resources
        ax2 = axes[1]
        ax2.set_facecolor('#0f0f0f')
        
        # Scatter with glow effect
        ax2.scatter(self.df['winner_resources_recovered'], self.df['winner_resources'],
                   alpha=0.3, s=120, color='#00ff88', edgecolors='none')
        ax2.scatter(self.df['winner_resources_recovered'], self.df['winner_resources'],
                   alpha=0.8, s=60, edgecolors='#ffffff', linewidth=1, color='#00ff88')
        
        ax2.set_xlabel('Resources Recovered from Dismantling', fontsize=11, color='#cccccc')
        ax2.set_ylabel('Final Resources', fontsize=11, color='#cccccc')
        ax2.set_title('Dismantling Recovery vs Final Resources', fontweight='bold', 
                     fontsize=12, color='#ffffff', pad=10)
        ax2.tick_params(colors='#999999', labelsize=9)
        ax2.grid(True, alpha=0.1, color='#666666')
        
        # Add trend line if enough data
        if len(self.df) > 1 and self.df['winner_resources_recovered'].sum() > 0:
            mask = self.df['winner_resources_recovered'] > 0
            if mask.sum() > 1:
                z = np.polyfit(self.df.loc[mask, 'winner_resources_recovered'], 
                             self.df.loc[mask, 'winner_resources'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(self.df['winner_resources_recovered'].min(), 
                                    self.df['winner_resources_recovered'].max(), 100)
                ax2.plot(x_line, p(x_line), color='#ffffff', linestyle='--', 
                        alpha=0.9, linewidth=2.5, label='Trend', zorder=10)
                legend = ax2.legend(framealpha=0.9, facecolor='#1a1a1a', 
                                  edgecolor='#666666', labelcolor='#ffffff')
        
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'dismantling_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"✅ Saved: {output_path}")
        plt.show()
    
    def perform_pca_analysis(self):
        """Perform PCA on winner characteristics."""
        # Filter out simulations with no winner
        df_winners = self.df[self.df['winner_name'].notna()].copy()
        
        if len(df_winners) < 2:
            print("⚠️  Not enough winners for PCA analysis (need at least 2)")
            return
        
        features = [
            'winner_resources', 'winner_tech', 'winner_weapon_investment',
            'winner_search_investment', 'winner_camo_investment', 
            'winner_survival_drive', 'avg_resource_pressure'
        ]
        
        X = df_winners[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=df_winners['total_steps'], cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, label='Total Steps')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title('PCA: Winner Characteristics', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # Add winner names as annotations
        for i, name in enumerate(df_winners['winner_name']):
            plt.annotate(name[:3], (X_pca[i, 0], X_pca[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'pca_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.show()
        
        # Print PCA components
        print("\nPCA Component Analysis:")
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=features
        )
        print(components_df.round(3))
    
    def perform_clustering(self, n_clusters: int = 3):
        """Perform K-means clustering on winner characteristics."""
        # Filter out simulations with no winner
        df_winners = self.df[self.df['winner_name'].notna()].copy()
        
        if len(df_winners) < n_clusters:
            print(f"⚠️  Not enough winners for clustering (need at least {n_clusters})")
            return
        
        features = [
            'winner_weapon_investment', 'winner_search_investment', 
            'winner_camo_investment'
        ]
        
        X = df_winners[features].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        df_winners['cluster'] = clusters
        
        # 3D scatter plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df_winners['winner_weapon_investment'],
                           df_winners['winner_search_investment'],
                           df_winners['winner_camo_investment'],
                           c=clusters, cmap='viridis', s=100, 
                           alpha=0.6, edgecolors='black')
        
        ax.set_xlabel('Weapon Investment', fontsize=12)
        ax.set_ylabel('Search Investment', fontsize=12)
        ax.set_zlabel('Camo Investment', fontsize=12)
        ax.set_title('K-Means Clustering: Winner Strategies', fontsize=16, fontweight='bold')
        
        plt.colorbar(scatter, label='Cluster', ax=ax, shrink=0.5)
        output_path = os.path.join(OUTPUT_DIR, 'clustering_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.show()
        
        # Print cluster characteristics
        print("\nCluster Characteristics:")
        cluster_summary = df_winners.groupby('cluster')[features].mean()
        print(cluster_summary.round(2))
    
    def statistical_tests(self):
        """Perform statistical tests on the data."""
        print(f"\n{'='*70}")
        print(f"STATISTICAL TESTS")
        print(f"{'='*70}\n")
        
        # Filter out simulations with no winner
        df_winners = self.df[self.df['winner_name'].notna()].copy()
        
        if len(df_winners) < 2:
            print("⚠️  Not enough winners for statistical tests (need at least 2)")
            print(f"\n{'='*70}\n")
            return
        
        # Test if weapon investment correlates with winning
        print("Correlation Tests:")
        investments = ['winner_weapon_investment', 'winner_search_investment', 'winner_camo_investment']
        
        for inv in investments:
            corr, p_value = stats.pearsonr(df_winners[inv], df_winners['total_steps'])
            print(f"  {inv} vs total_steps:")
            print(f"    Correlation: {corr:.3f}, p-value: {p_value:.4f}")
        
        # Test normality of winner resources
        print(f"\nNormality Test (Shapiro-Wilk) for Winner Resources:")
        stat, p_value = stats.shapiro(df_winners['winner_resources'])
        print(f"  Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        print(f"  Result: {'Normal distribution' if p_value > 0.05 else 'Not normal distribution'}")
        
        print(f"\n{'='*70}\n")
    
    def generate_full_report(self):
        """Generate complete analysis report with all visualizations."""
        print(f"\n{'='*70}")
        print(f"GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print(f"{'='*70}\n")
        
        self.print_summary_statistics()
        self.statistical_tests()
        self.plot_winner_characteristics()
        self.plot_spending_patterns()
        self.plot_spending_vs_outcome()
        self.plot_spending_time_series()
        self.plot_dismantling_analysis()
        self.plot_correlation_matrix()
        self.plot_time_series_comparison()
        self.perform_pca_analysis()
        self.perform_clustering()
        
        print(f"\n{'='*70}")
        print(f"REPORT GENERATION COMPLETE")
        print(f"{'='*70}\n")


def main():
    """Main entry point for batch simulation."""
    # Configuration
    N_SIMULATIONS = 100
    NUM_CIVS = 4
    VERBOSE = False
    
    # Run batch simulations
    simulator = BatchSimulator(
        n_simulations=N_SIMULATIONS,
        num_civs=NUM_CIVS,
        verbose=VERBOSE
    )
    
    df = simulator.run_batch()
    
    # Save results
    simulator.save_results()
    
    # Analyze results
    analyzer = SimulationAnalyzer(simulator.results, df)
    analyzer.generate_full_report()
    
    # Save DataFrame
    csv_path = os.path.join(OUTPUT_DIR, 'simulation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")


if __name__ == "__main__":
    main()