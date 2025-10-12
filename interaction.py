"""
Interaction module implementing Dark Forest strategic decisions.
Civilizations make decisions based on survival, resource scarcity, and suspicion.
"""
from __future__ import annotations
import random

try:
    from config import (
        COOPERATION_BENEFIT_BASE,
        ARMS_RACE_COST,
        ARMS_RACE_TECH_GAIN,
        DEFENSE_COST
    )
except ImportError:
    COOPERATION_BENEFIT_BASE = 15.0
    ARMS_RACE_COST = 10
    ARMS_RACE_TECH_GAIN = 2
    DEFENSE_COST = 8

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Civilisation
    from env import Galaxy


class InteractionManager:
    """
    Manages interactions between civilizations using Dark Forest principles.
    
    When civilizations meet, they must decide:
    - Trust and cooperate (risky due to suspicion)
    - Strike first (driven by survival and resource scarcity)
    - Remain hidden/passive (safest but limits growth)
    """
    
    @staticmethod
    def resolve_interaction(civ_a: Civilisation, civ_b: Civilisation, galaxy: Galaxy):
        """
        Resolve interaction between two civilizations using Dark Forest axioms.
        Each civilization independently evaluates whether to strike first.
        
        Args:
            civ_a: First civilization
            civ_b: Second civilization  
            galaxy: Galaxy instance
        """
        
        # Calculate first strike probabilities (based on all three axioms)
        strike_prob_a = civ_a.calculate_first_strike_probability(civ_b, galaxy)
        strike_prob_b = civ_b.calculate_first_strike_probability(civ_a, galaxy)
        
        # Both civilizations independently decide
        a_strikes = random.random() < strike_prob_a
        b_strikes = random.random() < strike_prob_b
        
        # Scenario 1: Both strike simultaneously (mutual destruction likely)
        if a_strikes and b_strikes:
            print(f"⚔️  MUTUAL FIRST STRIKE: [{civ_a.name}] ⚔️  [{civ_b.name}]")
            InteractionManager._mutual_combat(civ_a, civ_b)
        
        # Scenario 2: Only A strikes (A has advantage)
        elif a_strikes and not b_strikes:
            print(f"🎯 FIRST STRIKE: [{civ_a.name}] → [{civ_b.name}] (prob: {strike_prob_a:.2%})")
            InteractionManager._first_strike(civ_a, civ_b, galaxy)
        
        # Scenario 3: Only B strikes (B has advantage)
        elif b_strikes and not a_strikes:
            print(f"🎯 FIRST STRIKE: [{civ_b.name}] → [{civ_a.name}] (prob: {strike_prob_b:.2%})")
            InteractionManager._first_strike(civ_b, civ_a, galaxy)
        
        # Scenario 4: Neither strikes - evaluate cooperation vs suspicion
        else:
            InteractionManager._cautious_interaction(civ_a, civ_b, galaxy, 
                                                     strike_prob_a, strike_prob_b)
    
    @staticmethod
    def _first_strike(attacker: Civilisation, target: Civilisation, galaxy: Galaxy):
        """
        Execute a first strike attack with element of surprise.
        Attacker has significant advantage due to surprise.
        
        Args:
            attacker: The attacking civilization
            target: The target civilization
            galaxy: The galaxy instance (string type hint)
        """
        # Attacker advantage: 1.5x damage multiplier for surprise
        base_damage = InteractionManager.get_attack_power(attacker)
        surprise_multiplier = 1.5
        total_damage = base_damage * surprise_multiplier
        
        # Apply damage
        target.resources -= total_damage
        target.has_been_attacked = True
        attacker.attack_count += 1
        
        print(f"   💥 [{attacker.name}] dealt {total_damage:.2f} damage (surprise attack)")
        print(f"   💔 [{target.name}] resources: {max(0, target.resources):.2f}")
        
        # Check if target survives
        if target.resources <= 0:
            print(f"   ☠️  [{target.name}] has been ELIMINATED")
            target.check_extinction()
            
            # Winner claims half of victim's initial resources (not more than they had)
            resource_gain = min(100, abs(target.resources) * 0.5)
            attacker.resources += resource_gain
            print(f"   💰 [{attacker.name}] gained {resource_gain:.2f} resources")
        else:
            # Target survived and may retaliate
            print(f"   🛡️  [{target.name}] survived and retaliates!")
            counter_damage = InteractionManager.get_attack_power(target) * 0.7  # Weakened
            attacker.resources -= counter_damage
            print(f"   ⚡ [{target.name}] counter-attacked for {counter_damage:.2f} damage")
            attacker.check_extinction()
    
    @staticmethod
    def _mutual_combat(civ_a: Civilisation, civ_b: Civilisation):
        """
        Both civilizations strike simultaneously - brutal mutual combat.
        No surprise advantage, but both take full damage.
        
        Args:
            civ_a: First civilization
            civ_b: Second civilization
        """
        damage_a = InteractionManager.get_attack_power(civ_a)
        damage_b = InteractionManager.get_attack_power(civ_b)
        
        civ_a.resources -= damage_b
        civ_b.resources -= damage_a
        
        civ_a.has_been_attacked = True
        civ_b.has_been_attacked = True
        civ_a.attack_count += 1
        civ_b.attack_count += 1
        
        print(f"   💥 [{civ_a.name}] took {damage_b:.2f} damage → {max(0, civ_a.resources):.2f} resources")
        print(f"   💥 [{civ_b.name}] took {damage_a:.2f} damage → {max(0, civ_b.resources):.2f} resources")
        
        civ_a.check_extinction()
        civ_b.check_extinction()
        
        if not civ_a.is_active and not civ_b.is_active:
            print(f"   ☠️☠️  MUTUAL DESTRUCTION: Both civilizations eliminated")
        elif not civ_a.is_active:
            print(f"   ☠️  [{civ_a.name}] eliminated, [{civ_b.name}] survives")
        elif not civ_b.is_active:
            print(f"   ☠️  [{civ_b.name}] eliminated, [{civ_a.name}] survives")
    
    @staticmethod
    def _cautious_interaction(civ_a: Civilisation, civ_b: Civilisation,
                             galaxy: Galaxy, strike_prob_a: float, strike_prob_b: float):
        """
        Neither civilization strikes - but trust is impossible (Axiom 3).
        Outcome depends on suspicion levels and relative power.
        
        Args:
            civ_a: First civilization
            civ_b: Second civilization
            galaxy: The galaxy instance (string type hint)
            strike_prob_a: A's strike probability (for display)
            strike_prob_b: B's strike probability (for display)
        """
        suspicion_a = civ_a.known_civs.get(civ_b.name, 0.5)
        suspicion_b = civ_b.known_civs.get(civ_a.name, 0.5)
        avg_suspicion = (suspicion_a + suspicion_b) / 2
        
        # Tech similarity enables limited cooperation
        tech_ratio = min(civ_a.tech, civ_b.tech) / max(civ_a.tech, civ_b.tech)
        
        # High tech similarity + low suspicion = possible cooperation
        if tech_ratio > 0.8 and avg_suspicion < 0.4:
            print(f"🤝 CAUTIOUS COOPERATION: [{civ_a.name}] ↔ [{civ_b.name}]")
            print(f"   Suspicion: {avg_suspicion:.2%}, Tech ratio: {tech_ratio:.2%}")
            
            # Limited resource sharing (mutual benefit, but guarded)
            cooperation_benefit = 15 * (1 - avg_suspicion)
            civ_a.resources += cooperation_benefit
            civ_b.resources += cooperation_benefit
            
            # Small tech exchange
            tech_exchange = 0.5 * (1 - avg_suspicion)
            civ_a.tech += tech_exchange
            civ_b.tech += tech_exchange
            
            print(f"   📈 Both gained {cooperation_benefit:.2f} resources and {tech_exchange:.2f} tech")
        
        # Moderate suspicion = cold war / standoff
        elif avg_suspicion < 0.7:
            print(f"❄️  COLD WAR: [{civ_a.name}] ⚡ [{civ_b.name}]")
            print(f"   Suspicion: {avg_suspicion:.2%}, Strike probs: A={strike_prob_a:.2%}, B={strike_prob_b:.2%}")
            
            # Arms race: both invest in tech at cost of resources
            civ_a.resources -= ARMS_RACE_COST
            civ_b.resources -= ARMS_RACE_COST
            civ_a.tech += ARMS_RACE_TECH_GAIN
            civ_b.tech += ARMS_RACE_TECH_GAIN
            
            print(f"   🚀 Arms race: Both spent {ARMS_RACE_COST:.2f} resources for {ARMS_RACE_TECH_GAIN:.2f} tech")
        
        # High suspicion = tense standoff, preparing for war
        else:
            print(f"😰 TENSE STANDOFF: [{civ_a.name}] 👁️  [{civ_b.name}]")
            print(f"   High suspicion: {avg_suspicion:.2%}")
            print(f"   Strike probabilities: A={strike_prob_a:.2%}, B={strike_prob_b:.2%}")
            
            # Both civilizations prepare defenses
            civ_a.resources -= DEFENSE_COST
            civ_b.resources -= DEFENSE_COST
            
            print(f"   🛡️  Both spent {DEFENSE_COST:.2f} resources on defense")
            
            # Increase mutual suspicion for next encounter
            civ_a.known_civs[civ_b.name] = min(1.0, suspicion_a + 0.1)
            civ_b.known_civs[civ_a.name] = min(1.0, suspicion_b + 0.1)
    
    @staticmethod
    def get_attack_power(civ: Civilisation) -> float:
        """
        Calculate attack power based on weapon investment, tech, and resources.
        
        Based primarily on weapon_investment to prevent tech dominance.
        
        Args:
            civ: Attacking civilization
            
        Returns:
            float: Total attack power
        """
        # Primary factor: weapon investment
        weapon_power = civ.weapon_investment * 10
        
        # Tech multiplier (modest)
        tech_multiplier = 1 + (civ.tech * 0.05)
        
        # Resource factor
        resource_factor = min(1.0, civ.resources / 50)
        
        # Randomness for uncertainty
        randomness = random.uniform(0, 5)
        
        total_power = (weapon_power * tech_multiplier * resource_factor) + randomness
        return total_power