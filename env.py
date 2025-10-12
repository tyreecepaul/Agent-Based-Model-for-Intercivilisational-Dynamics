"""
Galaxy environment implementing Dark Forest axioms.
Manages civilization agents and orchestrates the simulation loop.
"""
import math
from agent import Civilisation
from interaction import InteractionManager
from config import (
    RESOURCE_GROWTH_RATE, 
    TECH_GROWTH_RATE,
    RESOURCE_ALLOCATION_PERCENTAGE
)


class Galaxy:
    """
    Galaxy class incorporating Dark Forest axioms and finite resources.
    
    Attributes:
        width (float): Width of simulation space
        height (float): Height of simulation space
        civilisations (List[Civilisation]): All civilization agents
        time_step (int): Current simulation time step
        total_resources (float): Finite total resources in universe
        initial_resources (float): Starting total resources
    """
    
    def __init__(self, width: int, height: int, total_resources: float = 50000.0):
        """
        Initialize galaxy with specified size and finite resources.
        
        Args:
            width: Width of simulation space
            height: Height of simulation space
            total_resources: Total finite resources available
        """
        self.width = width
        self.height = height
        self.time_step = 0
        self.civilisations = []
        
        # Finite total resources in universe
        self.total_resources = total_resources
        self.initial_resources = total_resources
        
        print(f"Galaxy created ({self.width} × {self.height})")
        print(f"Total universe resources: {self.total_resources:.2f}")
    
    def get_total_active_resources(self) -> float:
        """
        Calculate sum of all resources held by active civilizations.
        
        Returns:
            float: Total resources held by active civilizations
        """
        return sum(c.resources for c in self.civilisations if c.check_is_active())
    
    def get_resource_pressure(self) -> float:
        """
        Calculate resource pressure in the universe (0-1).
        
        Formula: P = Σ(resources_i) / total_resources
        
        Returns:
            float: Resource pressure (0 = abundant, 1 = maximum scarcity)
        """
        used = self.get_total_active_resources()
        pressure = min(1.0, used / self.total_resources)
        return pressure
    
    def enforce_resource_conservation(self):
        """
        Enforce universe resource limit (AXIOM 2: Resource chains are finite).
        
        If total resources exceed limit, proportionally reduce all 
        civilizations' resources to maintain conservation law.
        
        This simulates the hard cap on galactic resources - civilizations
        must compete for a fixed resource pool.
        """
        total_used = self.get_total_active_resources()
        
        if total_used > self.total_resources:
            # Calculate how much over the limit we are
            excess = total_used - self.total_resources
            reduction_factor = self.total_resources / total_used
            
            print(f"\n⚠️  AXIOM 2 ENFORCEMENT: Resource Cap Exceeded!")
            print(f"   Total Resources Used: {total_used:.0f} / {self.total_resources:.0f}")
            print(f"   Excess: {excess:.0f} (over limit by {(excess/self.total_resources)*100:.1f}%)")
            print(f"   Applying {(1-reduction_factor)*100:.1f}% reduction to all civilizations")
            
            # Proportionally reduce all civilizations' resources
            for civ in self.civilisations:
                if civ.check_is_active():
                    old_resources = civ.resources
                    civ.resources *= reduction_factor
                    lost = old_resources - civ.resources
                    if lost > 1.0:  # Only print significant losses
                        print(f"      • {civ.name}: {old_resources:.0f} → {civ.resources:.0f} (-{lost:.0f})")
            
            print(f"   New Total: {self.get_total_active_resources():.0f} / {self.total_resources:.0f} ✓")
    
    def import_civilisation_default(self, name: str, x: float, y: float, col: str):
        """
        Add a new civilization with default starting parameters.
        
        Args:
            name: Unique name for the civilization
            x: X-coordinate position
            y: Y-coordinate position
            col: Color for visualization (hex string)
        """
        new_civ = Civilisation(name, x, y, col)
        self.civilisations.append(new_civ)
        print(f"  Added {name} at ({x:.1f}, {y:.1f})")

    def import_civilisation(self, name: str, x: float, y: float, col: str, r: float, t: float):
        """
        Add a new civilization with custom starting parameters.
        
        Args:
            name: Unique name for the civilization
            x: X-coordinate position
            y: Y-coordinate position
            col: Color for visualization (hex string)
            r: Starting resources
            t: Starting tech level
        """
        new_civ = Civilisation(name, x, y, col, r, t)
        self.civilisations.append(new_civ)
        print(f"  Added {name} at ({x:.1f}, {y:.1f}) - Resources: {r:.0f}, Tech: {t:.1f}")

    def get_size_coordinates(self):
        """Get galaxy dimensions as tuple."""
        return (self.width, self.height)

    def get_size(self):
        """Get total area of galaxy."""
        return self.width * self.height
    
    def get_civilisations(self):
        """Print all civilizations (for debugging)."""
        print(*self.civilisations)

    def get_neighbours(self, civ: Civilisation):
        """
        Find all active civilizations within detection range of specified civilization.
        
        Implements detection and suspicion mechanics with camouflage modifiers.
        Detection is modified by target's camo investment.
        
        Args:
            civ: Detecting civilization
            
        Returns:
            List[Civilisation]: Detected neighboring civilizations
        """
        neighbours = []
        base_radius = civ.get_interaction_radius()
        galaxy_size = math.sqrt(self.width**2 + self.height**2)
        
        for other_civ in self.civilisations:
            if civ != other_civ and other_civ.check_is_active():
                distance = civ.distance_to(other_civ)
                
                # Apply camo modifier: target's camo reduces detection range
                camo_modifier = other_civ.get_camo_detection_modifier(civ)
                effective_radius = base_radius * camo_modifier
                
                if distance <= effective_radius:
                    # Update knowledge and calculate suspicion
                    civ.update_known_civilization(other_civ, distance, galaxy_size)
                    neighbours.append(other_civ)
        
        return neighbours
    
    def run_simulation(self):
        """
        Main simulation loop incorporating all Dark Forest axioms.
        Runs one full time step.
        
        Phases:
        0. Economic Decisions - Resource allocation
        1. Internal Updates - Civilization growth
        2. Resource Conservation - Enforce finite resources
        3. Interactions - Detection and strategic decisions
        4. Cleanup - Remove extinct civilizations
        
        Returns:
            bool: True if simulation should continue
        """
        self.time_step += 1
        
        # Header
        print(f"\n{'='*70}")
        print(f"⏱️  TIME STEP {self.time_step}")
        print(f"🌍 Resource Pressure: {self.get_resource_pressure():.1%} "
              f"({self.get_total_active_resources():.0f}/{self.total_resources:.0f})")
        print(f"{'='*70}")

        print(f"\n💰 Phase 0: Economic Resource Allocation")
        for civ in self.civilisations:
            if civ.check_is_active() and civ.resources > 0:
                # Check if civilization should dismantle due to resource pressure
                should_dismantle, category = civ.should_dismantle(self)
                if should_dismantle:
                    print(f"  ⚠️ Resource Pressure {self.get_resource_pressure():.1%}: {civ.name} considering dismantling...")
                    recovered = civ.dismantle_investment(category)
                    if recovered > 0:
                        print(f"     💰 Recovered {recovered:.1f} resources (now have {civ.resources:.1f})")
                
                # Determine how much to allocate (% of current resources)
                available_for_allocation = civ.resources * RESOURCE_ALLOCATION_PERCENTAGE
                resources_before = civ.resources
                
                # Decide allocation strategy based on situation
                allocation = civ.decide_resource_allocation(self, available_for_allocation)
                
                # Invest the allocated resources (now with real costs and diminishing returns)
                investment_summary, actual_costs = civ.invest_resources(allocation)
                
                # Calculate total spent
                total_cost = sum(actual_costs.values())
                resources_after = civ.resources
                
                print(f"  • {civ.name}: {investment_summary}")
                print(f"    Total spent: ${total_cost:.1f} (had {resources_before:.0f} → now {resources_after:.0f})")
                print(f"    Levels: Wpn={civ.weapon_investment:.1f}, Sch={civ.search_investment:.1f}, "
                      f"Cam={civ.camo_investment:.1f}")

        print(f"\n📈 Phase 1: Internal Growth")
        for civ in self.civilisations:
            if civ.check_is_active():
                old_resources = civ.resources
                old_tech = civ.tech
                
                # Update state with axiom-based growth
                civ.set_state(self, RESOURCE_GROWTH_RATE, TECH_GROWTH_RATE)
                
                # Check for extinction
                civ.check_extinction()
                
                if civ.check_is_active():
                    resource_change = civ.resources - old_resources
                    tech_change = civ.tech - old_tech
                    print(f"  • {civ.name}: R +{resource_change:.1f} → {civ.resources:.0f}, "
                          f"T +{tech_change:.2f} → {civ.tech:.1f}")
        
        self.enforce_resource_conservation()

        print(f"\n🌐 Phase 2: Inter-Civilization Interactions")
        
        active_civs = [c for c in self.civilisations if c.check_is_active()]
        interaction_count = 0
        
        for civ_a in active_civs:
            if not civ_a.check_is_active(): 
                continue
            
            neighbors = self.get_neighbours(civ_a)
            
            if neighbors:
                print(f"\n  [{civ_a.name}] detected {len(neighbors)} civilization(s):")
                
            for civ_b in neighbors:
                if civ_b.check_is_active():
                    interaction_count += 1
                    # Enhanced interaction with Dark Forest logic
                    InteractionManager.resolve_interaction(civ_a, civ_b, self)
        
        if interaction_count == 0:
            print("  No interactions this turn (civilizations isolated)")

        # Remove extinct civilizations
        extinct_civs = [c for c in self.civilisations if not c.check_is_active()]
        self.civilisations = [c for c in self.civilisations if c.check_is_active()]
        
        # Final status report
        print(f"\n{'─'*70}")
        print(f"📊 END OF TURN STATUS:")
        print(f"{'─'*70}")
        print(f"   Active Civilizations: {len(self.civilisations)}")
        print(f"   Resources in Use: {self.get_total_active_resources():.0f}/{self.total_resources:.0f} "
              f"({self.get_resource_pressure():.1%})")
        
        if self.civilisations:
            print(f"\n   Civilization Rankings:")
            # Sort by power (resources * tech * weapons)
            ranked = sorted(self.civilisations, 
                          key=lambda c: c.resources * c.tech * c.weapon_investment, 
                          reverse=True)
            
            for i, civ in enumerate(ranked, 1):
                threat = civ.survival_threat_level(self)
                power = civ.resources * civ.tech * civ.weapon_investment
                
                threat_icon = '🔴' if threat > 0.7 else '🟡' if threat > 0.4 else '🟢'
                
                print(f"   {i}. {civ.name}: Power={power:.0f} "
                      f"(R={civ.resources:.0f} × T={civ.tech:.1f} × W={civ.weapon_investment:.1f}) "
                      f"| Threat={threat_icon}{threat:.0%} "
                      f"| Known={len(civ.known_civs)}")
        
        # Check for simulation end conditions
        if len(self.civilisations) <= 1:
            print("\n" + "="*70)
            print("🏁 SIMULATION ENDED")
            print("="*70)
            
            if len(self.civilisations) == 1:
                winner = self.civilisations[0]
                print(f"👑 WINNER: {winner.name}")
                print(f"   Final Resources: {winner.resources:.2f}")
                print(f"   Final Tech Level: {winner.tech:.2f}")
                print(f"   Weapon Investment: {winner.weapon_investment:.2f}")
                print(f"   Search Investment: {winner.search_investment:.2f}")
                print(f"   Camo Investment: {winner.camo_investment:.2f}")
                print(f"   Survival Drive: {winner.survival_drive:.2f}")
                print(f"   Known Civilizations: {len(winner.known_civs)}")
            else:
                print("💀 All civilizations have gone extinct")
                print("   The Dark Forest is silent once more...")
            
            print("="*70)
            return False
        
        return True