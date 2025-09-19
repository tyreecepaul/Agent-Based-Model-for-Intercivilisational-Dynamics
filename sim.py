from agent import Civilisation
from env import Galaxy

import random

GALAXY_SIZE = 1000
NUM_STEPS = 1000

if __name__ == "__main__":
    # Example usage to test the full simulation
    galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE)
    galaxy.import_civilisation_default("Sol", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE))
    galaxy.import_civilisation_default("Alpha Centauri", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE))
    galaxy.import_civilisation_default("Proxima B", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE))
    galaxy.import_civilisation_default("Xylos", random.uniform(0, GALAXY_SIZE), random.uniform(0, GALAXY_SIZE))

    for _ in range(NUM_STEPS):
        if not galaxy.run_simulation():
            break