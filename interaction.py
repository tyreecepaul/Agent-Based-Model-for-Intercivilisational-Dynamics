import random

class InteractionManager:
    
    @staticmethod
    def resolve_interaction(civ_a, civ_b, galaxy):
        # Determine initial action from civ_a's perspective
        if civ_a.tech > civ_b.tech * 1.5:
            print(f"[{civ_a.name}] initiates a dominant interaction with [{civ_b.name}].")
            damage_dealt_a = InteractionManager.get_attack_power(civ_a)
            civ_b.resources -= damage_dealt_a
            print(f"[{civ_a.name}] dealt {damage_dealt_a:.2f} damage to [{civ_b.name}].")

            # **New:** civ_b reacts based on its state
            if civ_b.resources <= 0:
                print(f"[{civ_b.name}] is too weak to react and collapses.")
                civ_b.check_extinction()
            else:
                print(f"[{civ_b.name}] retaliates!")
                damage_dealt_b = InteractionManager.get_attack_power(civ_b)
                civ_a.resources -= damage_dealt_b
                print(f"[{civ_b.name}] retaliated with {damage_dealt_b:.2f} damage to [{civ_a.name}].")
                civ_a.check_extinction()

        # Add logic for other types of interactions (trade, etc.)
        elif abs(civ_a.tech - civ_b.tech) < 5:
            print(f"[{civ_a.name}] and [{civ_b.name}] are in a neutral or cooperative interaction.")
            # Example of peaceful outcome
            civ_a.resources += 10
            civ_b.resources += 10
        
        else:
            print(f"[{civ_a.name}] is observing [{civ_b.name}] passively.")

    @staticmethod
    def get_attack_power(civ):
        return civ.tech * 0.5 + random.uniform(0, 5)