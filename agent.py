"""
Civilization agent implementing Dark Forest Three Axioms.

Implements economic system, resource allocation, and strategic decision-making
based on survival drive, resource scarcity, and technological uncertainty.
"""
from __future__ import annotations
import math
import random

# Import config values if available, otherwise use defaults
try:
    from config import (
        TECH_EXPLOSION_BASE_PROBABILITY,
        TECH_EXPLOSION_MIN_GAIN,
        TECH_EXPLOSION_MAX_GAIN,
        TECH_EXPLOSION_MOMENTUM_CAP,
        WEAPON_INVESTMENT_EFFICIENCY,
        SEARCH_INVESTMENT_EFFICIENCY,
        CAMO_INVESTMENT_EFFICIENCY,
        WEAPON_COST_SCALING,
        SEARCH_COST_SCALING,
        CAMO_COST_SCALING,
        WEAPON_BASE_COST,
        SEARCH_BASE_COST,
        CAMO_BASE_COST,
        BASE_DETECTION_RADIUS,
        SEARCH_RANGE_PER_POINT,
        CAMO_MAX_REDUCTION,
        CAMO_EFFECTIVENESS_FACTOR,
        RESOURCE_ALLOCATION_PERCENTAGE,
        WEIGHT_SUSPICION,
        WEIGHT_RESOURCE_NEED,
        WEIGHT_TECH_ADVANTAGE,
        WEIGHT_SURVIVAL_THREAT
    )
except ImportError:
    TECH_EXPLOSION_BASE_PROBABILITY = 0.05
    TECH_EXPLOSION_MIN_GAIN = 0.10
    TECH_EXPLOSION_MAX_GAIN = 0.50
    TECH_EXPLOSION_MOMENTUM_CAP = 2.0
    WEAPON_INVESTMENT_EFFICIENCY = 0.5
    SEARCH_INVESTMENT_EFFICIENCY = 2.0
    CAMO_INVESTMENT_EFFICIENCY = 0.3
    WEAPON_COST_SCALING = 1.15
    SEARCH_COST_SCALING = 1.20
    CAMO_COST_SCALING = 1.18
    WEAPON_BASE_COST = 1.0
    SEARCH_BASE_COST = 1.0
    CAMO_BASE_COST = 1.0
    BASE_DETECTION_RADIUS = 20.0
    SEARCH_RANGE_PER_POINT = 5.0
    CAMO_MAX_REDUCTION = 0.5
    CAMO_EFFECTIVENESS_FACTOR = 0.1
    RESOURCE_ALLOCATION_PERCENTAGE = 0.3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env import Galaxy


class Civilisation:
    """
    Civilization agent with Dark Forest axioms.
    
    Axioms:
    1. Survival is primary - drives resource acquisition and defense
    2. Resources are finite - creates competition and scarcity
    3. Suspicion and tech explosions - makes trust impossible
    """
    
    def __init__(self, name: str, x: float, y: float, col: str, r: float=100.0, t: float=1.0):
        """Initialize civilization with starting properties."""
        self.name = name
        self.x = x
        self.y = y
        self.resources = r
        self.tech = t
        self.is_active = True
        self.colour = col
        
        # Survival drive varies by civilization
        self.survival_drive = 0.7
        
        # Known civilizations and suspicion levels
        self.known_civs = {}
        self.tech_explosion_probability = TECH_EXPLOSION_BASE_PROBABILITY
        
        # Combat history
        self.has_been_attacked = False
        self.attack_count = 0
        
        # Economic investments
        self.weapon_investment = 1.0
        self.search_investment = 1.0
        self.camo_investment = 0.0
        
        # Economic strategy preferences
        self.strategy_weights = {
            'weapons': random.uniform(0.2, 0.4),
            'search': random.uniform(0.2, 0.4),
            'camo': random.uniform(0.1, 0.3),
            'growth': random.uniform(0.2, 0.4)
        }
        # Normalize to sum to 1.0
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v/total for k, v in self.strategy_weights.items()}

    def __repr__(self) -> str:
        """String representation for easy debugging"""
        return (f"Civilization('{self.name}', "
                f"Pos: ({self.x:.2f}, {self.y:.2f}), "
                f"Res: {self.resources:.2f}, "
                f"Tech: {self.tech:.2f}, "
                f"Survival: {self.survival_drive:.2f})")
    
    def survival_utility(self, galaxy: Galaxy) -> float:
        """
        Calculate survival utility for this civilization.
        
        Mathematical formulation:
        U_survival = resources / (1 + known_threats)
        
        Higher utility = better survival position
        Lower utility = more threatened
        
        Args:
            galaxy: The Galaxy instance (string type hint to avoid circular import)
        """
        known_threats = sum(1 for civ_name, suspicion in self.known_civs.items() 
                           if suspicion > 0.5)
        
        if known_threats == 0:
            return self.resources
        
        return self.resources / (1 + known_threats)
    
    def survival_threat_level(self, galaxy: Galaxy) -> float:
        """
        Calculate overall threat level to survival (0-1).
        
        Based on:
        - Number of known civilizations
        - Resource scarcity
        - Tech disadvantages
        
        Args:
            galaxy: The Galaxy instance (string type hint to avoid circular import)
        """
        if not self.known_civs:
            return 0.1  # Low baseline threat when isolated
        
        # Threat from known civilizations
        civ_threat = len(self.known_civs) * 0.2
        
        # Resource scarcity threat (Axiom 2)
        resource_pressure = galaxy.get_resource_pressure()
        
        # Tech disadvantage threat
        active_civs = [c for c in galaxy.civilisations if c.check_is_active() and c != self]
        if active_civs:
            avg_tech = sum(c.tech for c in active_civs) / len(active_civs)
            tech_threat = max(0, (avg_tech - self.tech) / avg_tech) if avg_tech > 0 else 0
        else:
            tech_threat = 0
        
        total_threat = min(1.0, 0.3 * civ_threat + 0.4 * resource_pressure + 0.3 * tech_threat)
        return total_threat
    
    def calculate_growth_rate(self, galaxy: Galaxy, base_growth: float) -> float:
        """
        Calculate resource growth rate considering finite universe resources.
        
        Mathematical formulation:
        dr/dt = r * (1 - P) * growth_factor
        
        Where P = resource_pressure = Σ(resources) / total_resources
        
        Args:
            galaxy: The Galaxy instance (string type hint to avoid circular import)
            base_growth: Base growth rate before pressure adjustment
        """
        resource_pressure = galaxy.get_resource_pressure()
        
        # Growth slows as resources become scarce
        effective_growth = base_growth * (1 - resource_pressure)
        
        # Bonus for higher tech (can extract more efficiently)
        tech_bonus = 1 + (self.tech / 100)
        
        return effective_growth * tech_bonus
    
    def resource_competition_modifier(self, galaxy: Galaxy) -> float:
        """
        Calculate how resource competition affects this civilization.
        Returns a value 0-1 where 1 means maximum competition.
        
        Args:
            galaxy: The Galaxy instance (string type hint to avoid circular import)
        """
        pressure = galaxy.get_resource_pressure()
        
        # Civilization's share of total resources
        total_active_resources = sum(c.resources for c in galaxy.civilisations 
                                    if c.check_is_active())
        
        if total_active_resources == 0:
            return 0.5
        
        share = self.resources / total_active_resources
        
        # High pressure + low share = high competition
        competition = pressure * (1 - share)
        
        return min(1.0, competition)
    
    def calculate_suspicion(self, other_civ: 'Civilisation', distance: float, galaxy_size: float) -> float:
        """
        Calculate suspicion level toward another civilization.
        
        Mathematical formulation:
        S(A,B) = 1 - (1 / (1 + |tech_A - tech_B| + normalized_distance))
        
        Suspicion increases with:
        - Technological difference (unpredictability)
        - Proximity (immediate threat)
        - History of aggression
        
        Args:
            other_civ: The other Civilisation
            distance: Distance between civilizations
            galaxy_size: Size of the galaxy for normalization
        """
        tech_diff = abs(self.tech - other_civ.tech)
        normalized_distance = distance / galaxy_size
        
        # Base suspicion formula
        base_suspicion = 1 - (1 / (1 + tech_diff + normalized_distance))
        
        # Modify based on history
        if other_civ.attack_count > 0:
            aggression_modifier = 1 + (0.2 * other_civ.attack_count)
            base_suspicion *= aggression_modifier
        
        if self.has_been_attacked:
            base_suspicion *= 1.3  # More suspicious after being attacked
        
        return min(1.0, base_suspicion)
    
    def update_known_civilization(self, other_civ: 'Civilisation', distance: float, galaxy_size: float):
        """
        Add or update knowledge of another civilization with suspicion level.
        This represents the "detection" phase of the Dark Forest.
        
        Args:
            other_civ: The detected civilization
            distance: Distance to the detected civilization
            galaxy_size: Size of the galaxy for normalization
        """
        suspicion = self.calculate_suspicion(other_civ, distance, galaxy_size)
        self.known_civs[other_civ.name] = suspicion
    
    def attempt_tech_explosion(self) -> bool:
        """
        Attempt a technological explosion event.
        
        Tech explosions are unpredictable and can dramatically change power balance.
        This is a key part of Axiom 3 - the unpredictability that makes trust impossible.
        
        The explosion magnitude is now more balanced:
        - Probability increases with tech level but is capped
        - Gain is uniform random between 10-50% (configurable)
        - Prevents runaway exponential growth
        
        Returns:
            bool: True if explosion occurred, False otherwise
        """
        # Probability increases with current tech level (momentum)
        # But capped to prevent runaway probability
        tech_momentum = min(self.tech / 100, TECH_EXPLOSION_MOMENTUM_CAP)
        adjusted_probability = self.tech_explosion_probability * (1 + tech_momentum)
        
        if random.random() < adjusted_probability:
            # More modest tech jumps: configurable range (default 10-50% increase)
            explosion_magnitude = random.uniform(TECH_EXPLOSION_MIN_GAIN, TECH_EXPLOSION_MAX_GAIN)
            tech_gain = self.tech * explosion_magnitude
            self.tech += tech_gain
            return True
        return False
    
    def decide_resource_allocation(self, galaxy: Galaxy, available_resources: float) -> dict:
        """
        Decide how to allocate available resources among competing priorities.
        
        Strategy is influenced by:
        - Survival threat level (high threat → more weapons/camo)
        - Resource pressure (scarcity → more aggressive expansion)
        - Known neighbors (more neighbors → more search/weapons)
        - Civilization's inherent strategy preferences
        
        Args:
            galaxy: The Galaxy instance
            available_resources: Amount of resources available to allocate
            
        Returns:
            dict: Allocation amounts for each category
        """
        if available_resources <= 0:
            return {'weapons': 0, 'search': 0, 'camo': 0, 'growth': 0}
        
        # Evaluate current situation
        threat = self.survival_threat_level(galaxy)
        pressure = galaxy.get_resource_pressure()
        known_civs_count = len(self.known_civs)
        
        # Adjust weights based on situation
        adjusted_weights = self.strategy_weights.copy()
        
        # High threat → increase weapons and camo
        if threat > 0.7:
            adjusted_weights['weapons'] *= 1.5
            adjusted_weights['camo'] *= 1.3
        elif threat > 0.4:
            adjusted_weights['weapons'] *= 1.2
        
        # High resource pressure → increase search (find more resources) and weapons (take them)
        if pressure > 0.7:
            adjusted_weights['search'] *= 1.4
            adjusted_weights['weapons'] *= 1.3
            adjusted_weights['growth'] *= 0.7  # Less benefit to growing when resources scarce
        
        # Many known neighbors → balance search with camo
        if known_civs_count > 3:
            adjusted_weights['search'] *= 0.8  # Already found many
            adjusted_weights['camo'] *= 1.3    # Need to hide better
        elif known_civs_count == 0:
            adjusted_weights['search'] *= 1.5  # Need to find others
            adjusted_weights['weapons'] *= 0.7 # Less immediate need
        
        # If we've been attacked, prioritize defense and weapons
        if self.has_been_attacked:
            adjusted_weights['weapons'] *= 1.6
            adjusted_weights['camo'] *= 1.4
            adjusted_weights['growth'] *= 0.6
        
        # Normalize adjusted weights
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        # Allocate resources proportionally
        allocation = {
            'weapons': available_resources * adjusted_weights['weapons'],
            'search': available_resources * adjusted_weights['search'],
            'camo': available_resources * adjusted_weights['camo'],
            'growth': available_resources * adjusted_weights['growth']
        }
        
        return allocation
    
    def calculate_investment_cost(self, category: str, desired_gain: float) -> float:
        """
        Calculate the actual resource cost for a desired investment gain.
        Implements diminishing returns - costs increase exponentially with current level.
        
        Formula: 
        cost = (desired_gain / efficiency) * (1 + current_level) ^ scaling_exponent
        
        This ensures:
        - First investments are cheap
        - Each successive investment costs more
        - High-level investments become very expensive
        
        Args:
            category: 'weapons', 'search', or 'camo'
            desired_gain: Amount of capability desired to gain
            
        Returns:
            float: Actual resource cost (may be higher than simple gain/efficiency)
        """
        if category == 'weapons':
            efficiency = WEAPON_INVESTMENT_EFFICIENCY
            scaling = WEAPON_COST_SCALING
            current_level = self.weapon_investment
            base_cost = WEAPON_BASE_COST
        elif category == 'search':
            efficiency = SEARCH_INVESTMENT_EFFICIENCY
            scaling = SEARCH_COST_SCALING
            current_level = self.search_investment
            base_cost = SEARCH_BASE_COST
        elif category == 'camo':
            efficiency = CAMO_INVESTMENT_EFFICIENCY
            scaling = CAMO_COST_SCALING
            current_level = self.camo_investment
            base_cost = CAMO_BASE_COST
        else:
            return desired_gain  # Unknown category, just return the amount
        
        # Base cost calculation (without diminishing returns)
        base_resource_cost = (desired_gain / efficiency) * base_cost
        
        # Apply diminishing returns multiplier
        # Higher current level = higher multiplier = more expensive
        diminishing_returns_multiplier = pow(1 + current_level, scaling - 1.0)
        
        actual_cost = base_resource_cost * diminishing_returns_multiplier
        
        return actual_cost
    
    def invest_resources(self, allocation: dict) -> tuple[str, dict]:
        """
        Apply resource allocation to actual investments with PROPER resource deduction.
        Now implements diminishing returns - successive investments cost more.
        
        AXIOM 2 ENFORCEMENT: All investments (except growth) are deducted from resources.
        Growth allocation stays as reserves for natural expansion.
        
        Args:
            allocation: Dictionary of resource amounts ALLOCATED (not necessarily spent)
            
        Returns:
            tuple: (summary_string, actual_costs_dict)
        """
        investments_made = []
        actual_costs = {'weapons': 0, 'search': 0, 'camo': 0, 'growth': 0}
        total_spent = 0
        
        if allocation['weapons'] > 0:
            # Calculate desired gain based on allocation
            desired_gain = allocation['weapons'] * WEAPON_INVESTMENT_EFFICIENCY
            
            # Calculate actual cost with diminishing returns
            actual_cost = self.calculate_investment_cost('weapons', desired_gain)
            
            # Invest if resources are sufficient
            if actual_cost <= self.resources:
                self.weapon_investment += desired_gain
                self.resources -= actual_cost
                total_spent += actual_cost
                actual_costs['weapons'] = actual_cost
                investments_made.append(f"Wpn+{desired_gain:.1f}(${actual_cost:.1f})")
            else:
                # Partial investment with available resources
                affordable_gain = desired_gain * (self.resources / actual_cost)
                affordable_cost = self.resources
                self.weapon_investment += affordable_gain
                self.resources -= affordable_cost
                total_spent += affordable_cost
                actual_costs['weapons'] = affordable_cost
                investments_made.append(f"Wpn+{affordable_gain:.1f}(${affordable_cost:.1f},LIMITED)")
        
        if allocation['search'] > 0 and self.resources > 0:
            desired_gain = allocation['search'] * SEARCH_INVESTMENT_EFFICIENCY
            actual_cost = self.calculate_investment_cost('search', desired_gain)
            
            if actual_cost <= self.resources:
                self.search_investment += desired_gain
                self.resources -= actual_cost
                total_spent += actual_cost
                actual_costs['search'] = actual_cost
                investments_made.append(f"Sch+{desired_gain:.1f}(${actual_cost:.1f})")
            else:
                affordable_gain = desired_gain * (self.resources / actual_cost)
                affordable_cost = self.resources
                self.search_investment += affordable_gain
                self.resources -= affordable_cost
                total_spent += affordable_cost
                actual_costs['search'] = affordable_cost
                investments_made.append(f"Sch+{affordable_gain:.1f}(${affordable_cost:.1f},LIMITED)")
        
        if allocation['camo'] > 0 and self.resources > 0:
            desired_gain = allocation['camo'] * CAMO_INVESTMENT_EFFICIENCY
            actual_cost = self.calculate_investment_cost('camo', desired_gain)
            
            if actual_cost <= self.resources:
                self.camo_investment += desired_gain
                self.resources -= actual_cost
                total_spent += actual_cost
                actual_costs['camo'] = actual_cost
                investments_made.append(f"Cam+{desired_gain:.1f}(${actual_cost:.1f})")
            else:
                affordable_gain = desired_gain * (self.resources / actual_cost)
                affordable_cost = self.resources
                self.camo_investment += affordable_gain
                self.resources -= affordable_cost
                total_spent += affordable_cost
                actual_costs['camo'] = affordable_cost
                investments_made.append(f"Cam+{affordable_gain:.1f}(${affordable_cost:.1f},LIMITED)")
        
        if allocation['growth'] > 0:
            # Growth allocation is kept as resources (economic infrastructure)
            actual_costs['growth'] = 0
            investments_made.append(f"Grw={allocation['growth']:.1f}(saved)")
        
        summary = " | ".join(investments_made) if investments_made else "No investments"
        
        return summary, actual_costs
    
    def get_effective_detection_range(self) -> float:
        """
        Calculate the effective detection range considering search investment.
        Replaces the old tech-only based system.
        
        Returns:
            float: Detection radius
        """
        search_bonus = self.search_investment * SEARCH_RANGE_PER_POINT
        tech_bonus = self.tech * 0.05  # Small tech influence remains
        
        return BASE_DETECTION_RADIUS + search_bonus + tech_bonus
    
    def get_camo_detection_modifier(self, other_civ: Civilisation) -> float:
        """
        Calculate how much this civ's camo reduces detection by another civ.
        
        Args:
            other_civ: The civilization trying to detect this one
            
        Returns:
            float: Multiplier on detection range (0.5 = detected at half range)
        """
        # Higher camo reduces detection
        # Other civ's search investment counteracts camo
        camo_effectiveness = self.camo_investment / (1 + other_civ.search_investment * 0.1)
        
        # Map camo to detection multiplier using config max reduction
        # 0 camo = 1.0 (normal detection)
        # High camo = CAMO_MAX_REDUCTION (reduced detection range)
        detection_multiplier = max(CAMO_MAX_REDUCTION, 1.0 - (camo_effectiveness * CAMO_EFFECTIVENESS_FACTOR))
        
        return detection_multiplier
    
    def calculate_first_strike_probability(self, target: Civilisation, galaxy: Galaxy) -> float:
        """
        Calculate probability of launching first strike on target.
        Integrates all three axioms with economic considerations.
        
        P(strike) = survival_drive * Σ(weighted_factors)
        
        Factors:
        - Suspicion (Axiom 3)
        - Resource need (Axiom 2)
        - Military advantage (weapon investment)
        - Survival threat (Axiom 1)
        
        Args:
            target: Target civilization
            galaxy: Galaxy instance
        """
        if target.name not in self.known_civs:
            return 0.0
        
        # Factor 1: Suspicion level (Axiom 3)
        suspicion = self.known_civs[target.name]
        
        # Factor 2: Resource need (Axiom 2)
        competition = self.resource_competition_modifier(galaxy)
        resource_need = competition * (target.resources / (self.resources + 1))
        resource_need = min(1.0, resource_need)
        
        # Factor 3: Military advantage (weapon investment + tech)
        power = self.weapon_investment * self.tech
        target_power = target.weapon_investment * target.tech
        if target_power > 0:
            military_advantage = max(0, (power - target_power) / target_power)
            military_advantage = min(1.0, military_advantage * 0.5)
        else:
            military_advantage = 1.0
        
        # Factor 4: Survival threat (Axiom 1)
        threat = self.survival_threat_level(galaxy)
        
        # Weapon confidence: low weapons = reluctance to attack
        weapon_confidence = min(1.0, self.weapon_investment / 5.0)
        
        # Combine factors with configurable weights
        base_strike_probability = (
            WEIGHT_SUSPICION * suspicion +
            WEIGHT_RESOURCE_NEED * resource_need +
            WEIGHT_TECH_ADVANTAGE * military_advantage +
            WEIGHT_SURVIVAL_THREAT * threat
        )
        
        # Apply survival drive and weapon confidence
        strike_probability = self.survival_drive * base_strike_probability * weapon_confidence
        
        return min(1.0, max(0.0, strike_probability))
    
    # ============================================================================
    # EXISTING METHODS (Enhanced)
    # ============================================================================
    
        
        return min(1.0, max(0.0, strike_probability))
    
    def get_coordinates(self): 
        return (self.x, self.y)
    
    def set_coordinates(self, x: float, y: float):
        self.x, self.y = x, y
        return f"New Coordinates: ({self.x}, {self.y})"
    
    def set_resources(self, resource: float):
        self.resources = resource
        return f"Resource Level: {self.resources}"
    
    def set_tech(self, new_tech):
        self.tech = new_tech
        return f"Tech Level: {self.tech}"
    
    def set_active(self, bool):
        self.is_active = bool
        return f"Is Active: {self.is_active}"
    
    def check_is_active(self): 
        return self.is_active

    def check_extinction(self):
        if self.resources <= 0:
            print(f"[{self.name}] has collapsed and is now extinct")
            self.is_active = False

    def set_state(self, galaxy: Galaxy, resource_growth_rate: float, tech_growth_rate: float):
        """
        Enhanced state update incorporating Axioms 1 & 2.
        
        Args:
            galaxy: The Galaxy instance (string type hint to avoid circular import)
            resource_growth_rate: Base resource growth rate
            tech_growth_rate: Base tech growth rate
        """
        # AXIOM 2: Resource growth with finite resources
        effective_growth = self.calculate_growth_rate(galaxy, resource_growth_rate)
        self.resources += effective_growth
        
        # Tech growth (base)
        self.tech += self.resources * tech_growth_rate
        
        # AXIOM 3: Chance of tech explosion
        if self.attempt_tech_explosion():
            print(f"[{self.name}] experienced a TECHNOLOGICAL EXPLOSION! New tech: {self.tech:.2f}")
    
    def distance_to(self, other_civilisation: 'Civilisation') -> float:
        """Calculates Euclidian distance to another civilisation"""
        other_x, other_y = other_civilisation.get_coordinates()
        dx = self.x - other_x
        dy = self.y - other_y
        return math.sqrt(dx**2 + dy**2)
    
    def get_interaction_radius(self) -> float:
        """
        Calculates the civilization's dynamic interaction radius.
        Now based on search investment rather than just tech.
        """
        return self.get_effective_detection_range()