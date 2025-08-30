from civilisation import Civilisation
from galaxy import Galaxy

GALAXY_SIZE = 1000
NUM_STEPS = 1000

if __name__ == "__main__":
    # Example usage to test the full simulation
    galaxy = Galaxy(GALAXY_SIZE, GALAXY_SIZE)
    galaxy.import_civilisation_default("Sol", 100, 100)
    galaxy.import_civilisation_default("Alpha Centauri", 120, 120)
    galaxy.import_civilisation_default("Proxima B", 150, 200)
    galaxy.import_civilisation_default("Xylos", 800, 900)

    for _ in range(NUM_STEPS):
        if not galaxy.run_simulation():
            break