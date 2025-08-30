from civilisation import Civilisation
from galaxy import Galaxy

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import GALAXY_SIZE, NUM_STEPS, FIGURE_SIZE, PLOT_INTERVAL_MS

def run_animation():
    """Sets up and runs the visualization animation."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.canvas.manager.set_window_title("Galactic Simulation")
    ax.set_xlim(0, GALAXY_SIZE)
    ax.set_ylim(0, GALAXY_SIZE)
    ax.set_title("Galactic Civilization Dynamics")
    ax.set_aspect('equal', adjustable='box')
    plt.style.use('dark_background')

    # Initial plot data
    galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE)
    galaxy.import_civilisation_default("Sol", 100, 100)
    galaxy.import_civilisation_default("Alpha Centauri", 120, 120)
    galaxy.import_civilisation_default("Proxima B", 150, 200)
    galaxy.import_civilisation_default("Xylos", 800, 900)
    
    # Store plot elements for animation
    civilization_points = ax.scatter([], [])
    civilization_radii = []
    texts = []

    def update(frame):
        nonlocal galaxy
        # Check if simulation is still running
        if galaxy.run_simulation():
            # Extract data for plotting
            x_coords = [c.x for c in galaxy.civilisations]
            y_coords = [c.y for c in galaxy.civilisations]
            tech_levels = [c.tech for c in galaxy.civilisations]

            # Explicitly set the background to black for the figure and axes
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Clear previous plots
            ax.clear()
            ax.set_xlim(0, GALAXY_SIZE)
            ax.set_ylim(0, GALAXY_SIZE)
            # Set title with a white color for readability on dark background
            ax.set_title(f"Galactic Civilization Dynamics\nTime Step: {galaxy.time_step}", color='white')
            ax.set_aspect('equal', adjustable='box')
            
            # Draw civilizations
            ax.scatter(x_coords, y_coords, s=[tech * 2 for tech in tech_levels], c='cyan', alpha=0.8, edgecolors='white', linewidth=1)
            
            
            # Draw interaction radii
            for civ in galaxy.civilisations:
                radius = civ.get_interaction_radius()
                circle = plt.Circle((civ.x, civ.y), radius, color='red', fill=False, linestyle='--', alpha=0.2)
                ax.add_artist(circle)
            
            # Add labels
            for i, civ in enumerate(galaxy.civilisations):
                ax.text(civ.x + 10, civ.y + 10, f"{civ.name}\nRes: {civ.resources:.0f}\nTech: {civ.tech:.1f}", color='white', fontsize=8)

        else:
            # If simulation is finished, stop animation
            ani.event_source.stop()

    ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS, repeat=False, interval=PLOT_INTERVAL_MS)
    plt.show()

if __name__ == "__main__":
    run_animation()