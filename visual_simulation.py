from civilisation import Civilisation
from galaxy import Galaxy

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random

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

    
    def generate_random_hex_color():
        """Generates a random hexadecimal color code."""
        r = random.randint(50, 205)
        g = random.randint(50, 205)
        b = random.randint(50, 205)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        return hex_color

    # Initial plot data
    galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE)
    for n in range(20):
        galaxy.import_civilisation_default(f"{n}", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE), col=generate_random_hex_color())

    ''' 
    galaxy.import_civilisation_default("Sol", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE), col=generate_random_hex_color())
    galaxy.import_civilisation_default("Alpha Centauri", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE), col=generate_random_hex_color())
    galaxy.import_civilisation_default("Proxima B", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE), col=generate_random_hex_color())
    galaxy.import_civilisation_default("Xylos", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE), col=generate_random_hex_color())
    '''

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
            colours = [c.colour for c in galaxy.civilisations]

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

            ax.scatter(x_coords, y_coords, s=[tech * 2 for tech in tech_levels], c=colours, alpha=0.8, edgecolors='white', linewidth=1)
            
            # Draw interaction radii
            for civ in galaxy.civilisations:
                radius = civ.get_interaction_radius()
                circle = plt.Circle((civ.x, civ.y), radius, color='white', fill=False, linestyle='--', alpha=0.2)
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

# for each timestep
    # for each civilistion
        # action = civ.get_action(state)
        # civ.update_env(action)

# class Civ
#   init takes in:
    # - attaching func
    # - defensive func
    # - passive func

# def make_action(state):
    # do some stuff....
    # choose one of attacking func, defensive func, passive fuunc