"""
Defines a civilisation, fundamental agent in the intercivilisational dynamics simulation.
Each instance of this class represents a single civilisation with its own state and properties.
"""
import math

class Civilisation:
    """
    Represents a single civilisation (C) within the simulation.

    Attributes:
        name (str): unique name of C
        x (float): x-coordinate of C in G
        y (float): y-coordinate of C in G
        resources (float): current amount of accumulated resources
        tech_level (float): value representing C tech advancement
        is_active (bool): flag to check if C is still active in simulation
     """
    
    def __init__(self, name: str, x: float, y: float, r=100.0, t=1.0):
        """Initialises new C with starting properties"""
        self.name = name
        self.x = x
        self.y = y

        self.resources = r
        self.tech = t
        self.is_active = True

        self.resource_modifier = 5.0
        self.tech_modifier = 0.01

        print(f"Civilisation Created: {self.name}")

    def __repr__(self) -> str:
        """String representation for easy debugging"""
        return (f"Civilization('{self.name}', "
                f"Pos: ({self.x:.2f}, {self.y:.2f}), "
                f"Res: {self.resources:.2f}, "
                f"Tech: {self.tech:.2f})")
    
    def get_coordinates(self): return (self.x, self.y)
    
    # Setters
    def set_coordinates(self, x: float, y: float):
        self.x, self.y = x, y
        return f"New Coordinates: ({self.x, self.y})"
    
    def set_resources(self, resource: float):
        self.resources = resource
        return f"Resource Level: {self.resources}"
    
    def set_tech(self, new_tech):
        self.tech = new_tech
        return f"Tech Level: {self.tech}"
    
    def set_active(self, bool):
        self.is_active = bool
        return f"Is Active: {self.is_active}"
    
    def check_extinction(self):
        if self.resources <= 0:
            print(f"[{self.name}] has collapsed and is now extinct")
            self.is_active = False

    # Internal Civilisation Factors
    def set_state(self):
        """
        Function to update the civilisation's internal state.
        For now, resources and tech grow with a simple function.
        """
        self.resources += self.resource_modifier
        self.tech += self.resources * self.tech_modifier
        pass
    
    def set_resource_modifier(self, new_value):
        self.resource_modifier = new_value
        return f"New Resource Modifier: {self.resource_modifier}"
    
    def set_tech_modifier(self, new_value):
        self.tech_modifier = new_value
        return f"New Tech Modifier: {self.tech_modifier}"
    
    # External Civilisation Factors
    def distance_to(self, other_civilisation) -> float:
        """Calculates Euclidian distance to another civilisation"""
        other_x, other_y = other_civilisation.get_coordinates()
        dx = self.x - other_x
        dy = self.y - other_y
        return math.sqrt(dx**2 + dy**2)
    
    def interact(self, other_civ):
        """
        Defines how this civ interacts with another civ
        """
        pass
    