"""
This file defines the Galaxy class, which serves as the simulation engine.
It manages all civilization agents and orchestrates the main simulation loop.
"""
from agent import Civilisation
from interaction import InteractionManager
class Galaxy:
    """
    Manages simulation of civs within a defined space.

    Attributes:
        width (float): width of plane
        height (float): height of plane
        civilisations (List[Civilisation]): list of all C agents
        time_step (int): current time step of the simulation
    """
    def __init__(self, width: int, height: int):
        """Initialise galaxy with specified size"""
        self.width = width
        self.height = height
        self.time_step = 0
        self.civilisations = []
        print(f"Galaxy created ({self.width, self.height})")

    def import_civilisation_default(self, name: str, x: float, y: float, col: str):
        new_civ = Civilisation(name, x, y, col)
        self.civilisations.append(new_civ)

    def import_civilisation(self, name: str, x: float, y: float, r: float, t: float, col: str):
        new_civ = Civilisation(name, x, y, r, t, col)
        self.civilisations.append(new_civ)

    def get_size_coordinates(self): return (self.width, self.height)

    def get_size(self): return self.width * self.height
    
    def get_civilisations(self): print(*self.civilisations)

    # Interacting between civilisations
    def get_neighbours(self, civ: Civilisation):
        """Finds all active civilisations within range of specified civ"""
        neighbours = []
        radius = civ.get_interaction_radius()
        for other_civ in self.civilisations:
            if civ != other_civ and other_civ.check_is_active():
                if civ.distance_to(other_civ) <= radius:
                    neighbours.append(other_civ)
        return neighbours
    
    def run_simulation(self):
        """
        The main simulation loop. This is the core engine.
        It runs one full time step.
        """
        self.time_step += 1
        print(f"\n--- Time Step {self.time_step} ---")

        # Phase 1: Internal State Updates
        for civ in self.civilisations:
            if civ.check_is_active():
                civ.set_state()
                civ.check_extinction()

        # Phase 2: Inter-civilization Interactions
        active_civs = [c for c in self.civilisations if c.check_is_active()]
        for civ_a in active_civs:
            if not civ_a.check_is_active(): continue
            
            neighbors = self.get_neighbours(civ_a)
            
            for civ_b in neighbors:
                if civ_b.check_is_active():
                    InteractionManager.resolve_interaction(civ_a, civ_b, self)

        # Phase 3: Post-step Cleanup
        self.civilisations = [c for c in self.civilisations if c.check_is_active()]
        print(f"Total active civilizations: {len(self.civilisations)}")
        if len(self.civilisations) <= 1:
            print("Simulation ended. Only one civilization or none left.")
            if len(self.civilisations) == 1:
                print(f"Winner: {self.civilisations[0]}")
            return False
        
        return True
