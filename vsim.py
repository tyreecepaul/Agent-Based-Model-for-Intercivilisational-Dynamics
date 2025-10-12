"""
Visualization with Dark Forest Axioms Display
Shows resource pressure, suspicion levels, and axiom dynamics in real-time
"""
from env import Galaxy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import numpy as np
import random

from config import (
    GALAXY_SIZE, NUM_STEPS, FIGURE_SIZE, PLOT_INTERVAL_MS,
    FIXED_PLANET_SIZE, SHOW_INTERACTION_RADII, TOTAL_UNIVERSE_RESOURCES
)


def run_animation():
    """Sets up and runs the enhanced visualization animation with axiom displays."""
    
    # Create figure with better layout
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0a')
    
    # Create GridSpec for better control
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, 
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Main galaxy view (larger, cleaner)
    ax_main = fig.add_subplot(gs[:, :2])
    
    # Side panel for statistics
    ax_stats = fig.add_subplot(gs[:2, 2])
    ax_stats.axis('off')
    
    # Bottom panel for resource pressure gauge
    ax_pressure = fig.add_subplot(gs[2, 2])
    ax_pressure.axis('off')
    
    fig.canvas.manager.set_window_title("Dark Forest Simulations")
    
    # Configure main plot with better styling
    ax_main.set_xlim(0, GALAXY_SIZE)
    ax_main.set_ylim(0, GALAXY_SIZE)
    ax_main.set_title("Galactic Civilization Dynamics", 
                     color='#ffffff', fontsize=16, fontweight='bold', pad=20)
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.grid(True, alpha=0.1, color='#333333', linestyle=':', linewidth=0.5)
    ax_main.set_xlabel('Galactic X-Coordinate', color='#888888', fontsize=10)
    ax_main.set_ylabel('Galactic Y-Coordinate', color='#888888', fontsize=10)
    ax_main.tick_params(colors='#666666', labelsize=8)
    
    # Set dark theme
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0a0a0a')
    ax_main.set_facecolor('#0f0f0f')

    def generate_random_hex_color():
        """Generates a random hexadecimal color code."""
        r = random.randint(50, 205)
        g = random.randint(50, 205)
        b = random.randint(50, 205)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        return hex_color

    # Initialize galaxy
    galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE, total_resources=TOTAL_UNIVERSE_RESOURCES)
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "DARK FOREST SIMULATION" + " "*31 + "║")
    print("╠" + "="*68 + "╣")
    print("║  🌌 Three Axioms in Action" + " "*40 + "║")
    print("║  1️⃣  Survival is the primary need" + " "*36 + "║")
    print("║  2️⃣  Resources are finite" + " "*42 + "║")
    print("║  3️⃣  Chain of suspicion & tech explosion" + " "*26 + "║")
    print("╚" + "="*68 + "╝\n")
    
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
    
    print(f"✨ Initialized {len(galaxy.civilisations)} civilizations")
    for i, civ in enumerate(galaxy.civilisations, 1):
        print(f"   {i}. {civ.name:20} @ ({civ.x:>6.0f}, {civ.y:>6.0f})")
    
    print(f"\n🎮 Starting simulation...")
    print(f"   Universe resources: {TOTAL_UNIVERSE_RESOURCES:,.0f}")
    print(f"   Maximum steps: {NUM_STEPS:,}")
    print(f"   Watch as axioms drive intercivilizational dynamics!\n")

    # Track extra iterations after simulation completion
    extra_iterations_remaining = [5]  # Using list to allow modification in nested function
    simulation_ended = [False]

    def update(frame):
        """Animation update function - called each frame."""
        nonlocal galaxy
        
        # Check if simulation is still running
        if not galaxy.run_simulation():
            if not simulation_ended[0]:
                simulation_ended[0] = True
                print(f"\n🏁 Simulation completed at step {galaxy.time_step}. Running 5 more iterations...")
            
            if extra_iterations_remaining[0] > 0:
                extra_iterations_remaining[0] -= 1
                print(f"   Extra iteration {5 - extra_iterations_remaining[0]}/5")
                # Continue to render the final state
            else:
                print("✅ All extra iterations completed. Stopping animation.")
                ani.event_source.stop()
                return
        
        ax_main.clear()
        ax_main.set_xlim(0, GALAXY_SIZE)
        ax_main.set_ylim(0, GALAXY_SIZE)
        ax_main.set_facecolor('#0f0f0f')
        ax_main.grid(True, alpha=0.1, color='#333333', linestyle=':', linewidth=0.5)
        ax_main.tick_params(colors='#666666', labelsize=8)
        
        # Title with current timestep and resource pressure
        pressure = galaxy.get_resource_pressure()
        pressure_color = '#00ff88' if pressure < 0.4 else '#ffcc00' if pressure < 0.7 else '#ff3366'
        pressure_icon = '🟢' if pressure < 0.4 else '🟡' if pressure < 0.7 else '🔴'
        
        title_text = f"Step {galaxy.time_step:>4}/{NUM_STEPS}  •  Resource Pressure: {pressure:.1%} {pressure_icon}"
        if simulation_ended[0]:
            title_text = f"⚠️  SIMULATION ENDED  •  Final State  •  Step {galaxy.time_step}"
        
        ax_main.set_title(title_text, 
                         color='white', fontsize=14, fontweight='bold', pad=15)
        
        if not galaxy.civilisations:
            return
        
        # Extract data for plotting
        x_coords = [c.x for c in galaxy.civilisations]
        y_coords = [c.y for c in galaxy.civilisations]
        colours = [c.colour for c in galaxy.civilisations]
        
        # Plot civilizations with glowing effect
        for civ in galaxy.civilisations:
            # Glow effect
            ax_main.scatter(civ.x, civ.y, s=FIXED_PLANET_SIZE * 2, 
                          c=civ.colour, alpha=0.15, edgecolors='none')
            # Main planet
            ax_main.scatter(civ.x, civ.y, s=FIXED_PLANET_SIZE, 
                          c=civ.colour, alpha=0.95, edgecolors='white', linewidth=2.5, zorder=10)
        
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
            
            # Create label with key info - more compact and readable
            label = (f"{civ.name}\n"
                    f"💰{civ.resources:.0f} ⚗️{civ.tech:.1f}\n"
                    f"⚔️{civ.weapon_investment:.1f} 🔍{civ.search_investment:.1f} 🎭{civ.camo_investment:.1f}")
            
            if civ.known_civs:
                label += f"\n👁️ {len(civ.known_civs)}"
            
            # Better text styling
            ax_main.text(civ.x + 25, civ.y + 25, label,
                        color=label_color, fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='#000000', alpha=0.85, 
                                edgecolor=label_color, linewidth=2),
                        zorder=15)

        ax_stats.clear()
        ax_stats.axis('off')
        
        stats_text = "╔═══ 🌌 GALAXY STATUS ═══╗\n\n"
        
        # Axiom 2: Resource Statistics
        stats_text += "📊 AXIOM 2: FINITE RESOURCES\n"
        stats_text += f"Total: {galaxy.get_total_active_resources():.0f}/{galaxy.total_resources:.0f}\n"
        stats_text += f"Pressure: {pressure:.1%} {pressure_icon}\n"
        
        pressure_status = "🟢 Abundant" if pressure < 0.4 else "🟡 Scarce" if pressure < 0.7 else "🔴 Critical"
        stats_text += f"Status: {pressure_status}\n\n"
        
        # Axiom 1 & 3: Civilization Rankings
        stats_text += "👑 POWER RANKINGS\n"
        stats_text += "─" * 23 + "\n"
        ranked = sorted(galaxy.civilisations, 
                       key=lambda c: c.resources * c.tech * c.weapon_investment, 
                       reverse=True)
        
        for i, civ in enumerate(ranked[:5], 1):  # Top 5
            threat = civ.survival_threat_level(galaxy)
            power = civ.resources * civ.tech * civ.weapon_investment
            
            threat_icon = '🔴' if threat > 0.7 else '🟡' if threat > 0.4 else '🟢'
            
            # Truncate name if too long
            name_display = civ.name[:10] + '...' if len(civ.name) > 10 else civ.name
            stats_text += f"{i}. {name_display:12} {threat_icon}\n"
            stats_text += f"   Pwr: {power:>6.0f}\n"
            stats_text += f"   W:{civ.weapon_investment:>4.1f} S:{civ.search_investment:>4.1f} C:{civ.camo_investment:>4.1f}\n"
        
        stats_text += "\n" + "─" * 23 + "\n\n"
        
        # Axiom 3: Suspicion Levels
        stats_text += "🔍 AXIOM 3: SUSPICION\n"
        stats_text += "─" * 23 + "\n"
        
        high_suspicion_pairs = []
        for civ in galaxy.civilisations:
            for name, susp in civ.known_civs.items():
                if susp > 0.6:
                    high_suspicion_pairs.append((civ.name[:6], name[:6], susp))
        
        if high_suspicion_pairs:
            for civ1, civ2, susp in sorted(high_suspicion_pairs, key=lambda x: x[2], reverse=True)[:4]:
                bar_length = int(susp * 10)
                bar = '█' * bar_length + '░' * (10 - bar_length)
                stats_text += f"{civ1}↔{civ2}\n{bar} {susp:.0%}\n"
        else:
            stats_text += "✨ No high tensions\n"
        
        stats_text += "\n╚" + "═" * 21 + "╝"
        
        ax_stats.text(0.05, 0.98, stats_text, 
                     transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top',
                     color='#eeeeee', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.8', 
                              facecolor='#1a1a1a', alpha=0.95, 
                              edgecolor='#444444', linewidth=1.5))
 
        # Resource pressure gauge with modern circular design
        ax_pressure.clear()
        ax_pressure.set_xlim(-1.2, 1.2)
        ax_pressure.set_ylim(-1.2, 1.2)
        ax_pressure.set_aspect('equal')
        ax_pressure.axis('off')
        
        # Background circle
        circle_bg = Circle((0, 0), 1, color='#1a1a1a', alpha=0.8, zorder=1)
        ax_pressure.add_patch(circle_bg)
        
        # Outer ring
        circle_outer = Circle((0, 0), 1, fill=False, edgecolor='#444444', linewidth=3, zorder=2)
        ax_pressure.add_patch(circle_outer)
        
        # Pressure arc (colored based on level)
        theta = np.linspace(0, 2 * np.pi * pressure, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Color gradient based on pressure
        if pressure < 0.4:
            color = '#00ff88'
            glow_color = '#00ff8844'
        elif pressure < 0.7:
            color = '#ffcc00'
            glow_color = '#ffcc0044'
        else:
            color = '#ff3366'
            glow_color = '#ff336644'
        
        # Draw glow effect
        for i in range(3, 0, -1):
            circle_glow = Circle((0, 0), 0.92 + i*0.03, fill=False, 
                               edgecolor=glow_color, linewidth=8-i*2, zorder=3)
            ax_pressure.add_patch(circle_glow)
        
        # Draw pressure arc
        if pressure > 0:
            ax_pressure.plot(x, y, color=color, linewidth=8, zorder=5, 
                           solid_capstyle='round')
        
        # Center text
        ax_pressure.text(0, 0.15, f"{pressure:.0%}", 
                        ha='center', va='center', 
                        fontsize=24, fontweight='bold', 
                        color='#ffffff', zorder=10)
        
        ax_pressure.text(0, -0.25, "RESOURCE\nPRESSURE", 
                        ha='center', va='center', 
                        fontsize=9, color='#999999', zorder=10)
        
        # Tick marks
        for i in range(0, 101, 25):
            angle = (i / 100) * 2 * np.pi - np.pi/2
            x_inner = 0.85 * np.cos(angle)
            y_inner = 0.85 * np.sin(angle)
            x_outer = 1.0 * np.cos(angle)
            y_outer = 1.0 * np.sin(angle)
            ax_pressure.plot([x_inner, x_outer], [y_inner, y_outer], 
                           color='#666666', linewidth=2, zorder=4)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS, 
                                 repeat=False, interval=PLOT_INTERVAL_MS)
    
    # Add enhanced legend
    legend_elements = [
        mpatches.Patch(facecolor='#44ff44', edgecolor='#ffffff', linewidth=1.5, label='Low Threat'),
        mpatches.Patch(facecolor='#ffaa44', edgecolor='#ffffff', linewidth=1.5, label='Medium Threat'),
        mpatches.Patch(facecolor='#ff4444', edgecolor='#ffffff', linewidth=1.5, label='High Threat'),
        mpatches.Patch(facecolor='none', edgecolor='#00ddff', linewidth=2, label='Detection Range'),
        mpatches.Patch(facecolor='none', edgecolor='#ff8800', linewidth=2, label='Interaction Range'),
    ]
    legend = ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8,
                           framealpha=0.9, facecolor='#1a1a1a', edgecolor='#444444',
                           labelcolor='#eeeeee', frameon=True)
    
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