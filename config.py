"""
Central configuration file for the Galactic Simulation project.
Stores all tunable parameters in one place for easy management.
"""

# Simulation controls
NUM_STEPS = 1000
GALAXY_SIZE = 1000

# Civilisation constants
DEFAULT_RESOURCES = 100.0
DEFAULT_TECH = 1.0
RESOURCE_GROWTH_RATE = 5.0
TECH_GROWTH_RATE = 0.01
INTERACTION_RADIUS_MULTIPLIER = 2.0

# Visualisation
PLOT_INTERVAL_MS =  100   # update interval in milliseconds
FIGURE_SIZE = (8, 8)