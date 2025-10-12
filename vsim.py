"""
Enhanced vsim.py - Visualization with Dark Forest Axioms Display
Shows resource pressure, suspicion levels, and axiom dynamics in real-time
"""
from env import Galaxy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random

from config import (
    GALAXY_SIZE, NUM_STEPS, FIGURE_SIZE, PLOT_INTERVAL_MS,
    FIXED_PLANET_SIZE, SHOW_INTERACTION_RADII, TOTAL_UNIVERSE_RESOURCES
)


def run_animation():
    """Sets up and runs the enhanced visualization animation with axiom displays."""
    
    # Create figure with subplots for additional information
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    # Main galaxy view (larger)
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    
    # Side panel for statistics
    ax_stats = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax_stats.axis('off')
    
    # Bottom panel for resource pressure gauge
    ax_pressure = plt.subplot2grid((3, 3), (2, 2))
    
    fig.canvas.manager.set_window_title("Dark Forest Simulation - Three Axioms")
    
    # Configure main plot
    ax_main.set_xlim(0, GALAXY_SIZE)
    ax_main.set_ylim(0, GALAXY_SIZE)
    ax_main.set_title("Galactic Civilization Dynamics", color='white', fontsize=14, fontweight='bold')
    ax_main.set_aspect('equal', adjustable='box')
    
    # Set dark theme
    plt.style.use('dark_background')
    fig.patch.set_facecolor('black')
    ax_main.set_facecolor('black')

    def generate_random_hex_color():
        """Generates a random hexadecimal color code."""
        r = random.randint(50, 205)
        g = random.randint(50, 205)
        b = random.randint(50, 205)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        return hex_color

    # Initialize galaxy
    galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE, total_resources=TOTAL_UNIVERSE_RESOURCES)
    
    print("\n" + "="*70)
    print("INITIALIZING DARK FOREST SIMULATION")
    print("="*70)
    
    # Create civilizations
    civs_to_create = [
        "Sol",
        "Alpha Centauri",
        "Proxima B",
        "Xylos",
        "Kepler-442"
    ]
    
    for name in civs_to_create:
        x = random.uniform(50, GALAXY_SIZE - 50)
        y = random.uniform(50, GALAXY_SIZE - 50)
        col = generate_random_hex_color()
        
        galaxy.import_civilisation_default(name, x, y, col)
    
    print("="*70)
    print(f"\n🎮 Starting simulation with {len(galaxy.civilisations)} civilizations")
    print("Watch as the three axioms drive intercivilizational dynamics!\n")

    def update(frame):
        """Animation update function - called each frame."""
        nonlocal galaxy
        
        # Check if simulation is still running
        if not galaxy.run_simulation():
            ani.event_source.stop()
            return
        
        ax_main.clear()
        ax_main.set_xlim(0, GALAXY_SIZE)
        ax_main.set_ylim(0, GALAXY_SIZE)
        ax_main.set_facecolor('black')
        
        # Title with current timestep and resource pressure
        pressure = galaxy.get_resource_pressure()
        pressure_color = '#00ff00' if pressure < 0.4 else '#ffff00' if pressure < 0.7 else '#ff0000'
        
        ax_main.set_title(
            f"Dark Forest Simulation - Time Step: {galaxy.time_step}\n"
            f"Resource Pressure: {pressure:.1%}",
            color='white', fontsize=12, fontweight='bold'
        )
        
        if not galaxy.civilisations:
            return
        
        # Extract data for plotting
        x_coords = [c.x for c in galaxy.civilisations]
        y_coords = [c.y for c in galaxy.civilisations]
        colours = [c.colour for c in galaxy.civilisations]
        
        # Plot civilizations (fixed size planets)
        ax_main.scatter(x_coords, y_coords, s=FIXED_PLANET_SIZE, 
                       c=colours, alpha=0.9, edgecolors='white', linewidth=2)
        
        # Draw interaction radii and suspicion indicators
        if SHOW_INTERACTION_RADII:
            for civ in galaxy.civilisations:
                radius = civ.get_interaction_radius()
                
                # Color radius by threat level
                threat = civ.survival_threat_level(galaxy)
                if threat > 0.7:
                    radius_color = '#ff4444'
                    alpha = 0.3
                elif threat > 0.4:
                    radius_color = '#ffaa44'
                    alpha = 0.2
                else:
                    radius_color = '#44ff44'
                    alpha = 0.15
                
                circle = plt.Circle((civ.x, civ.y), radius, 
                                  color=radius_color, fill=False, 
                                  linestyle='--', alpha=alpha, linewidth=1.5)
                ax_main.add_artist(circle)
        
        # Draw suspicion lines between known civilizations
        for civ in galaxy.civilisations:
            for known_name, suspicion in civ.known_civs.items():
                # Find the other civilization
                other = next((c for c in galaxy.civilisations if c.name == known_name), None)
                if other and suspicion > 0.5:  # Only show high suspicion
                    # Draw line with thickness/color based on suspicion
                    line_color = '#ff0000' if suspicion > 0.8 else '#ff8800'
                    line_alpha = min(0.5, suspicion)
                    line_width = 1 + suspicion * 2
                    
                    ax_main.plot([civ.x, other.x], [civ.y, other.y],
                               color=line_color, alpha=line_alpha, 
                               linewidth=line_width, linestyle=':')
        
        # Add labels with detailed info
        for civ in galaxy.civilisations:
            # Calculate display metrics
            threat = civ.survival_threat_level(galaxy)
            power = civ.resources * civ.tech * civ.weapon_investment
            
            # Choose label color based on threat
            if threat > 0.7:
                label_color = '#ff4444'
            elif threat > 0.4:
                label_color = '#ffaa44'
            else:
                label_color = '#44ff44'
            
            # Create label with key info including economic investments
            label = (f"{civ.name}\n"
                    f"R: {civ.resources:.0f} | T: {civ.tech:.1f}\n"
                    f"W: {civ.weapon_investment:.1f} | S: {civ.search_investment:.1f}\n"
                    f"C: {civ.camo_investment:.1f}")
            
            if civ.known_civs:
                label += f" | Known: {len(civ.known_civs)}"
            
            ax_main.text(civ.x + 15, civ.y + 15, label,
                        color=label_color, fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7, edgecolor=label_color))

        ax_stats.clear()
        ax_stats.axis('off')
        
        stats_text = "🌌 DARK FOREST STATUS\n"
        stats_text += "="*25 + "\n\n"
        
        # Axiom 2: Resource Statistics
        stats_text += "📊 AXIOM 2: RESOURCES\n"
        stats_text += f"Total: {galaxy.get_total_active_resources():.0f}/{galaxy.total_resources:.0f}\n"
        stats_text += f"Pressure: {pressure:.1%}\n"
        
        pressure_status = "🟢 Abundant" if pressure < 0.4 else "🟡 Scarce" if pressure < 0.7 else "🔴 Critical"
        stats_text += f"Status: {pressure_status}\n\n"
        
        # Axiom 1 & 3: Civilization Rankings
        stats_text += "👑 POWER RANKINGS\n"
        ranked = sorted(galaxy.civilisations, 
                       key=lambda c: c.resources * c.tech * c.weapon_investment, 
                       reverse=True)
        
        for i, civ in enumerate(ranked[:5], 1):  # Top 5
            threat = civ.survival_threat_level(galaxy)
            power = civ.resources * civ.tech * civ.weapon_investment
            
            threat_icon = '🔴' if threat > 0.7 else '🟡' if threat > 0.4 else '🟢'
            
            stats_text += f"{i}. {civ.name[:12]}\n"
            stats_text += f"   Pwr: {power:.0f} {threat_icon}\n"
            stats_text += f"   W:{civ.weapon_investment:.1f} S:{civ.search_investment:.1f} C:{civ.camo_investment:.1f}\n"
        
        stats_text += "\n" + "="*25 + "\n\n"
        
        # Axiom 3: Suspicion Levels
        stats_text += "🔍 AXIOM 3: SUSPICION\n"
        
        high_suspicion_pairs = []
        for civ in galaxy.civilisations:
            for name, susp in civ.known_civs.items():
                if susp > 0.7:
                    high_suspicion_pairs.append((civ.name[:8], name[:8], susp))
        
        if high_suspicion_pairs:
            stats_text += "High Tension:\n"
            for civ1, civ2, susp in high_suspicion_pairs[:3]:  # Top 3
                stats_text += f"  {civ1}↔{civ2}: {susp:.0%}\n"
        else:
            stats_text += "No high tensions\n"
        
        ax_stats.text(0.05, 0.95, stats_text, 
                     transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top',
                     color='white', family='monospace')
 
        ax_pressure.clear()
        ax_pressure.set_xlim(0, 1)
        ax_pressure.set_ylim(0, 1)
        ax_pressure.axis('off')
        ax_pressure.set_facecolor('black')
        
        # Draw gauge background
        gauge_rect = mpatches.Rectangle((0.1, 0.3), 0.8, 0.2, 
                                       fill=True, facecolor='#222222', 
                                       edgecolor='white', linewidth=2)
        ax_pressure.add_patch(gauge_rect)
        
        # Draw pressure fill
        pressure_width = 0.8 * pressure
        pressure_color = '#00ff00' if pressure < 0.4 else '#ffff00' if pressure < 0.7 else '#ff0000'
        
        pressure_rect = mpatches.Rectangle((0.1, 0.3), pressure_width, 0.2,
                                          fill=True, facecolor=pressure_color,
                                          alpha=0.8)
        ax_pressure.add_patch(pressure_rect)
        
        # Add label
        ax_pressure.text(0.5, 0.7, "RESOURCE PRESSURE (AXIOM 2)",
                        ha='center', va='center', color='white',
                        fontsize=9, fontweight='bold')
        ax_pressure.text(0.5, 0.15, f"{pressure:.1%}",
                        ha='center', va='center', color=pressure_color,
                        fontsize=12, fontweight='bold')

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS, 
                                 repeat=False, interval=PLOT_INTERVAL_MS)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#44ff44', label='Low Threat'),
        mpatches.Patch(color='#ffaa44', label='Medium Threat'),
        mpatches.Patch(color='#ff4444', label='High Threat'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "DARK FOREST HYPOTHESIS SIMULATION")
    print(" "*20 + "Three Axioms in Action")
    print("="*70)
    print("\nAXIOM 1: Survival is the primary need of civilization")
    print("AXIOM 2: Resources are finite (total matter is constant)")
    print("AXIOM 3: Chain of suspicion & technological explosion\n")
    print("="*70 + "\n")
    
    run_animation()