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
            
            # Run simulation
            step = 0
            for step in range(NUM_STEPS):
                # Collect time series data
                result.resource_pressure_history.append(galaxy.get_resource_pressure())
                result.active_civs_history.append(len([c for c in galaxy.civilisations if c.check_is_active()]))
                result.total_resources_history.append(galaxy.get_total_active_resources())
                
                # Run one step
                if not galaxy.run_simulation():
                    break
            
            result.total_steps = step + 1
            result.final_civs = len(galaxy.civilisations)
            
            # Calculate aggregate statistics
            if result.resource_pressure_history:
                result.avg_resource_pressure = np.mean(result.resource_pressure_history)
                result.max_resource_pressure = np.max(result.resource_pressure_history)
            
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
        
        print(f"\nResource Pressure:")
        print(f"  Avg pressure (mean): {self.df['avg_resource_pressure'].mean():.2%}")
        print(f"  Max pressure (mean): {self.df['max_resource_pressure'].mean():.2%}")
        
        print(f"\n{'='*70}\n")
    
    def plot_winner_characteristics(self):
        """Plot distribution of winner characteristics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Winner Characteristics Distribution', fontsize=16, fontweight='bold')
        
        characteristics = [
            ('winner_resources', 'Resources'),
            ('winner_tech', 'Technology'),
            ('winner_weapon_investment', 'Weapon Investment'),
            ('winner_search_investment', 'Search Investment'),
            ('winner_camo_investment', 'Camouflage Investment'),
            ('winner_known_civs', 'Known Civilizations')
        ]
        
        for idx, (col, title) in enumerate(characteristics):
            ax = axes[idx // 3, idx % 3]
            
            # Histogram with KDE
            self.df[col].hist(bins=15, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            
            # Add mean line
            mean_val = self.df[col].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'winner_characteristics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.show()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of winner characteristics."""
        numeric_cols = [
            'total_steps', 'winner_resources', 'winner_tech',
            'winner_weapon_investment', 'winner_search_investment', 
            'winner_camo_investment', 'avg_resource_pressure'
        ]
        
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: Simulation Variables', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.show()
    
    def plot_time_series_comparison(self):
        """Plot time series data from multiple simulations."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Time Series Comparison Across Simulations', fontsize=16, fontweight='bold')
        
        # Resource Pressure
        ax1 = axes[0]
        for result in self.results[:10]:  # Plot first 10 for clarity
            ax1.plot(result.resource_pressure_history, alpha=0.5, linewidth=1)
        ax1.set_title('Resource Pressure Over Time', fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Resource Pressure')
        ax1.set_ylim(0, 1)
        
        # Active Civilizations
        ax2 = axes[1]
        for result in self.results[:10]:
            ax2.plot(result.active_civs_history, alpha=0.5, linewidth=1)
        ax2.set_title('Active Civilizations Over Time', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Active Civilizations')
        
        # Total Resources
        ax3 = axes[2]
        for result in self.results[:10]:
            ax3.plot(result.total_resources_history, alpha=0.5, linewidth=1)
        ax3.set_title('Total Resources in Use Over Time', fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Total Resources')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'time_series_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
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