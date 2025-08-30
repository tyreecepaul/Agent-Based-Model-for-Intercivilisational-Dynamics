"""
This file defines the Galaxy class, which serves as the simulation engine.
It manages all civilization agents and orchestrates the main simulation loop.
"""
from civilisation import Civilisation

class Galaxy:
    """
    Manages simulation of C within a defined space.

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

    def import_civilisation_default(self, name: str, x: float, y: float):
        new_civ = Civilisation(name, x, y, )
        self.civilisations.append(new_civ)

    def import_civilisation(self, name: str, x: float, y: float, r=float, t=float):
        new_civ = Civilisation(name, x, y, r, t)
        self.civilisations.append(new_civ)

    def get_size(self): return (self.width, self.height)
    
    def get_civilisations(self): print(*self.civilisations)

    # Interacting between civilisations
    def get_neighbours(self, civ: Civilisation, radius: float):
        """Finds all active civilisations wihtin given radius of specified civ"""
        neighbours = []
        for other_civ in self.civilisations:
            if civ != other_civ and other_civ.check_is_active():
                if civ.distance_to(other_civ) <= radius:
                    neighbours.append(other_civ)
        return neighbours
    
    
