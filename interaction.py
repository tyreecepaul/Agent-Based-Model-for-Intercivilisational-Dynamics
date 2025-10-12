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
        DEFENSE_COST,
        FIRST_STRIKE_ADVANTAGE,
        RETALIATION_STRENGTH,
        RESOURCE_CAPTURE_RATE,
        SILENT_INVASION_COST_MULTIPLIER,
        LOUD_INVASION_EXPOSURE_SUSPICION,
        VERBOSE_COMBAT,
        CAMO_SILENT_INVASION_DISCOUNT,
        CAMO_ACCURACY_REDUCTION,
        CAMO_MAX_ACCURACY_REDUCTION,
    )
except ImportError:
    COOPERATION_BENEFIT_BASE = 15.0
    ARMS_RACE_COST = 10
    ARMS_RACE_TECH_GAIN = 2
    DEFENSE_COST = 8
    SILENT_INVASION_COST_MULTIPLIER = 0.40
    LOUD_INVASION_EXPOSURE_SUSPICION = 0.6

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
    - Shadow reconnaissance (gather intel with stealth)
    """
    
    @staticmethod
    def _attempt_shadow_recon(observer: Civilisation, target: Civilisation):
        """
        Civilization may attempt shadow reconnaissance before deciding on action.
        Only attempted if:
        - Observer has high camo (>= 3)
        - Not on cooldown
        - Has sufficient resources
        - Some suspicion exists (motivation to gather intel)
        
        Args:
            observer: Civilization attempting recon
            target: Target of reconnaissance
        """
        # Check if conditions are met for recon attempt
        has_high_camo = observer.camo_investment >= 3.0
        has_suspicion = target.name in observer.known_civs and observer.known_civs[target.name] > 0.3
        not_on_cooldown = target.name not in observer.recon_cooldowns or observer.recon_cooldowns[target.name] <= 0
        has_resources = observer.resources >= 10.0
        
        # Probabilistic decision: 30% chance if all conditions met
        should_attempt = has_high_camo and has_suspicion and not_on_cooldown and has_resources
        will_attempt = should_attempt and random.random() < 0.3
        
        if will_attempt:
            print(f"   🕵️  [{observer.name}] attempts shadow reconnaissance on [{target.name}]")
            success, intel = observer.shadow_reconnaissance(target)
            
            if success:
                print(f"   ✅ SUCCESS! Gathered intel: R={intel['resources']:.1f}, "
                      f"T={intel['tech']:.2f}, W={intel['weapon_investment']:.1f}")
            else:
                if intel.get('error') == 'detected':
                    print(f"   ❌ DETECTED! Suspicion increased by {intel['suspicion_gain']:.1%}")
                elif intel.get('error') == 'cooldown':
                    pass  # Silent - cooldown active
                elif intel.get('error') == 'insufficient_resources':
                    pass  # Silent - not enough resources
    
    @staticmethod
    def resolve_interaction(civ_a: Civilisation, civ_b: Civilisation, galaxy: Galaxy):
        """
        Resolve interaction between two civilizations using Dark Forest axioms.
        Each civilization independently evaluates whether to strike first.
        May also attempt shadow reconnaissance before attacking.
        
        Args:
            civ_a: First civilization
            civ_b: Second civilization  
            galaxy: Galaxy instance
        """
        
        # Phase 0: Shadow Reconnaissance (if high camo and opportunity)
        # Civilizations with high camo may attempt stealth intel gathering
        InteractionManager._attempt_shadow_recon(civ_a, civ_b)
        InteractionManager._attempt_shadow_recon(civ_b, civ_a)
        
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
            InteractionManager._execute_strike(civ_a, civ_b, galaxy)
        
        # Scenario 3: Only B strikes (B has advantage)
        elif b_strikes and not a_strikes:
            print(f"🎯 FIRST STRIKE: [{civ_b.name}] → [{civ_a.name}] (prob: {strike_prob_b:.2%})")
            InteractionManager._execute_strike(civ_b, civ_a, galaxy)
        
        # Scenario 4: Neither strikes - evaluate cooperation vs suspicion
        else:
            InteractionManager._cautious_interaction(civ_a, civ_b, galaxy, 
                                                     strike_prob_a, strike_prob_b)
    
    @staticmethod
    def _execute_strike(attacker: 'Civilisation', target: 'Civilisation', galaxy: Galaxy):
        """
        Execute a first strike with invasion strategy choice and accuracy system.
        Attacker chooses between silent (costly but hidden) or loud (cheaper but exposes both).
        
        Args:
            attacker: The attacking civilization
            target: The target civilization
            galaxy: The galaxy instance
        """
        # Get state before action
        initial_state = attacker.get_invasion_state(target, galaxy)
        
        # Attacker chooses invasion strategy using Q-learning
        invasion_strategy = attacker.choose_invasion_strategy(target, galaxy)
        
        if invasion_strategy == 'silent':
            reward = InteractionManager._silent_invasion(attacker, target, galaxy)
        else:
            reward = InteractionManager._loud_invasion(attacker, target, galaxy)
        
        # Update Q-value based on outcome
        final_state = attacker.get_invasion_state(target, galaxy)
        attacker.update_q_value(initial_state, invasion_strategy, reward, final_state)
    
    @staticmethod
    def _silent_invasion(attacker: 'Civilisation', target: 'Civilisation', galaxy: Galaxy) -> float:
        """
        Execute a silent invasion - no location exposure but high resource cost.
        Camo investment reduces the cost of silent invasions.
        
        Args:
            attacker: The attacking civilization
            target: The target civilization
            galaxy: The galaxy instance
            
        Returns:
            float: Reward value for Q-learning
        """
        print(f"   🤫 SILENT INVASION: [{attacker.name}] invades covertly")
        
        # Silent invasion base cost
        base_cost_multiplier = SILENT_INVASION_COST_MULTIPLIER
        
        # Camo benefit: reduce silent invasion cost
        camo_discount = min(0.5, attacker.camo_investment * CAMO_SILENT_INVASION_DISCOUNT)
        effective_cost_multiplier = base_cost_multiplier * (1 - camo_discount)
        
        stealth_cost = attacker.resources * effective_cost_multiplier
        attacker.resources -= stealth_cost
        
        if attacker.camo_investment > 0:
            print(f"   🎭 Camo benefit: Cost reduced by {camo_discount:.1%} (camo: {attacker.camo_investment:.1f})")
        print(f"   💸 Stealth operations cost: {stealth_cost:.2f} resources (effective rate: {effective_cost_multiplier:.1%})")
        
        # Calculate hit probability (target's camo reduces accuracy)
        hit_probability = attacker.calculate_hit_probability(target)
        attack_hits = random.random() < hit_probability
        
        if target.camo_investment > 0:
            camo_penalty = min(CAMO_MAX_ACCURACY_REDUCTION, 
                             target.camo_investment * CAMO_ACCURACY_REDUCTION)
            print(f"   🎭 Target camo defense: Accuracy reduced by {camo_penalty:.1%} (camo: {target.camo_investment:.1f})")
        
        # Record the result for learning

        
        # Calculate hit probability
        hit_probability = attacker.calculate_hit_probability()
        attack_hits = random.random() < hit_probability
        
        # Record the result for learning
        attacker.record_attack_result(attack_hits)
        
        if attack_hits:
            # Successful hit - deal damage with surprise bonus
            base_damage = InteractionManager.get_attack_power(attacker)
            surprise_multiplier = 1.5
            total_damage = base_damage * surprise_multiplier
            
            target.resources -= total_damage
            target.has_been_attacked = True
            attacker.attack_count += 1
            
            print(f"   ✅ HIT! Accuracy: {hit_probability:.1%} | Damage: {total_damage:.2f}")
            print(f"   💔 [{target.name}] resources: {max(0, target.resources):.2f}")
            
            # Check if target survives
            if target.resources <= 0:
                print(f"   ☠️  [{target.name}] has been ELIMINATED")
                target.check_extinction()
                
                # Winner claims resources
                resource_gain = min(100, abs(target.resources) * 0.5)
                attacker.resources += resource_gain
                print(f"   💰 [{attacker.name}] gained {resource_gain:.2f} resources")
                
                # High reward for successful elimination
                reward = 100.0 + resource_gain - stealth_cost
            else:
                # Target survived - moderate reward minus stealth cost
                print(f"   🛡️  [{target.name}] survived but is weakened")
                counter_damage = InteractionManager.get_attack_power(target) * 0.7
                attacker.resources -= counter_damage
                print(f"   ⚡ [{target.name}] counter-attacked for {counter_damage:.2f} damage")
                attacker.check_extinction()
                
                reward = total_damage - stealth_cost - counter_damage
        else:
            # Attack missed!
            print(f"   ❌ MISS! Accuracy: {hit_probability:.1%}")
            print(f"   🎯 [{attacker.name}] learns from failure (accuracy: {attacker.successful_hits}/{attacker.total_attacks})")
            
            # Target detects the failed attack and retaliates
            print(f"   ⚠️  [{target.name}] detected the attack and retaliates!")
            counter_damage = InteractionManager.get_attack_power(target) * 0.9  # Strong retaliation
            attacker.resources -= counter_damage
            print(f"   ⚡ [{target.name}] counter-attacked for {counter_damage:.2f} damage")
            attacker.check_extinction()
            
            # Negative reward: stealth cost paid + damage taken, no benefit
            reward = -stealth_cost - counter_damage
        
        # No location exposure in silent invasions
        print(f"   🔒 Both civilizations remain hidden from others")
        
        return reward
    
    @staticmethod
    def _loud_invasion(attacker: 'Civilisation', target: 'Civilisation', galaxy: Galaxy) -> float:
        """
        Execute a loud invasion - cheaper but exposes both civilizations to all others.
        
        Args:
            attacker: The attacking civilization
            target: The target civilization
            galaxy: The galaxy instance
            
        Returns:
            float: Reward value for Q-learning
        """
        print(f"   📢 LOUD INVASION: [{attacker.name}] attacks openly!")
        
        # Calculate hit probability (target's camo reduces accuracy)
        hit_probability = attacker.calculate_hit_probability(target)
        attack_hits = random.random() < hit_probability
        
        if target.camo_investment > 0:
            camo_penalty = min(CAMO_MAX_ACCURACY_REDUCTION, 
                             target.camo_investment * CAMO_ACCURACY_REDUCTION)
            print(f"   🎭 Target camo defense: Accuracy reduced by {camo_penalty:.1%} (camo: {target.camo_investment:.1f})")
        
        # Record the result for learning
        attacker.record_attack_result(attack_hits)
        
        if attack_hits:
            # Successful hit
            base_damage = InteractionManager.get_attack_power(attacker)
            surprise_multiplier = 1.5
            total_damage = base_damage * surprise_multiplier
            
            target.resources -= total_damage
            target.has_been_attacked = True
            attacker.attack_count += 1
            
            print(f"   ✅ HIT! Accuracy: {hit_probability:.1%} | Damage: {total_damage:.2f}")
            print(f"   💔 [{target.name}] resources: {max(0, target.resources):.2f}")
            
            # Check if target survives
            if target.resources <= 0:
                print(f"   ☠️  [{target.name}] has been ELIMINATED")
                target.check_extinction()
                
                resource_gain = min(100, abs(target.resources) * 0.5)
                attacker.resources += resource_gain
                print(f"   💰 [{attacker.name}] gained {resource_gain:.2f} resources")
                
                reward = 100.0 + resource_gain
            else:
                print(f"   🛡️  [{target.name}] survived and retaliates!")
                counter_damage = InteractionManager.get_attack_power(target) * 0.7
                attacker.resources -= counter_damage
                print(f"   ⚡ [{target.name}] counter-attacked for {counter_damage:.2f} damage")
                attacker.check_extinction()
                
                reward = total_damage - counter_damage
        else:
            # Attack missed!
            print(f"   ❌ MISS! Accuracy: {hit_probability:.1%}")
            print(f"   🎯 [{attacker.name}] learns from failure (accuracy: {attacker.successful_hits}/{attacker.total_attacks})")
            
            print(f"   ⚠️  [{target.name}] detected the attack and retaliates!")
            counter_damage = InteractionManager.get_attack_power(target) * 0.9
            attacker.resources -= counter_damage
            print(f"   ⚡ [{target.name}] counter-attacked for {counter_damage:.2f} damage")
            attacker.check_extinction()
            
            reward = -counter_damage
        
        # LOUD INVASION: Potential exposure based on camo investment
        # Calculate detection probabilities for both attacker and target
        attacker_detection_prob = attacker.calculate_detection_probability()
        target_detection_prob = target.calculate_detection_probability()
        
        print(f"   🌍 EXPOSURE CHECK:")
        print(f"      [{attacker.name}] detection probability: {attacker_detection_prob:.1%} (camo: {attacker.camo_investment:.1f})")
        print(f"      [{target.name}] detection probability: {target_detection_prob:.1%} (camo: {target.camo_investment:.1f})")
        
        detected_attacker_count = 0
        detected_target_count = 0
        
        for civ in galaxy.civilisations:
            if civ.is_active and civ.name != attacker.name and civ.name != target.name:
                # Check if this civilization detects the attacker
                attacker_detected = random.random() < attacker_detection_prob
                target_detected = random.random() < target_detection_prob
                
                if attacker_detected:
                    detected_attacker_count += 1
                    if attacker.name not in civ.known_civs:
                        civ.known_civs[attacker.name] = LOUD_INVASION_EXPOSURE_SUSPICION
                        print(f"      👁️  [{civ.name}] DETECTED [{attacker.name}] (suspicion: {LOUD_INVASION_EXPOSURE_SUSPICION:.2f})")
                    else:
                        civ.known_civs[attacker.name] = min(1.0, civ.known_civs[attacker.name] + 0.2)
                        print(f"      👁️  [{civ.name}] DETECTED [{attacker.name}] (suspicion increased)")
                else:
                    print(f"      🔒 [{civ.name}] FAILED to detect [{attacker.name}] (camo effective!)")
                
                if target_detected:
                    detected_target_count += 1
                    if target.name not in civ.known_civs:
                        civ.known_civs[target.name] = LOUD_INVASION_EXPOSURE_SUSPICION
                        print(f"      👁️  [{civ.name}] DETECTED [{target.name}] (suspicion: {LOUD_INVASION_EXPOSURE_SUSPICION:.2f})")
                    else:
                        civ.known_civs[target.name] = min(1.0, civ.known_civs[target.name] + 0.2)
                        print(f"      👁️  [{civ.name}] DETECTED [{target.name}] (suspicion increased)")
                else:
                    print(f"      🔒 [{civ.name}] FAILED to detect [{target.name}] (camo effective!)")
        
        # Penalty for exposure in reward calculation (only for detected civilizations)
        exposure_penalty = 20.0 * detected_attacker_count
        reward -= exposure_penalty
        
        if detected_attacker_count == 0 and detected_target_count == 0:
            print(f"   🎭 STEALTH SUCCESS: No civilizations detected the invasion!")
        elif detected_attacker_count > 0 or detected_target_count > 0:
            print(f"   📡 Detected by {detected_attacker_count + detected_target_count} civilization(s)")
        
        return reward
    
    @staticmethod
    def _first_strike(attacker: Civilisation, target: Civilisation, galaxy: Galaxy):
        """
        DEPRECATED: Legacy method kept for compatibility.
        Use _execute_strike instead which includes invasion strategy and accuracy.
        """
        InteractionManager._execute_strike(attacker, target, galaxy)
    
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