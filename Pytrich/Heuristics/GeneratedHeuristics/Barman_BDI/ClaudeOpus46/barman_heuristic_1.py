"""
Barman BDI Domain-Dependent Heuristic

Bottleneck Analysis:
  The single shaker is the critical serializing resource. Every cocktail must be
  mixed in the shaker (add 2 ingredients, shake, pour to shot), then the shaker
  must be cleaned before the next cocktail. This means cocktails are essentially
  serialized through the shaker.

Admissible Lower Bound Terms:

  Term 1 — Per-cocktail production cost:
    Each cocktail that is not yet in its target shot requires at minimum:
    - If the cocktail is not yet in the shaker: clean shaker (up to 3 actions),
      fill 2 shots with ingredients (each ~4 actions: clean shot, grasp, fill, drop),
      pour each to shaker (~3 actions each), shake (1 action) = ~14 actions min.
    - Then pour shaker to shot (~1 action), plus shot prep (~3 actions).
    We estimate a realistic per-cocktail lower bound based on current state.
    This is admissible because these actions are truly required for each cocktail.

  Term 2 — Shaker serialization bound:
    The shaker can only hold one cocktail at a time. For N remaining cocktails,
    we need at least N full shaker cycles. Each cycle requires at minimum:
    clean-shaker + 2*(fill-shot + pour-to-shaker) + shake + pour-to-shot = ~10 actions.
    For the first cocktail we may save the clean if shaker is already clean.
    This is admissible: each cocktail truly needs its own shaker cycle.

  Term 3 — Remaining task network depth:
    Count of primitive + abstract tasks remaining, weighted by minimum
    decomposition depth. This provides a structural lower bound.

Tie-breaking:
  Shots are interchangeable for intermediate use. We add a small penalty
  for states where both hands are occupied (less flexibility).
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic1(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_1"):
        super().__init__(name)
        # Fact ID maps
        self.contains_ids = {}   # (container, beverage) -> global_id
        self.clean_ids = {}      # container -> global_id
        self.empty_ids = {}      # container -> global_id
        self.holding_ids = {}    # (hand, container) -> global_id
        self.hand_empty_ids = {} # hand -> global_id
        self.ontable_ids = {}    # container -> global_id
        self.shaked_ids = {}     # shaker -> global_id
        self.unshaked_ids = {}   # shaker -> global_id
        self.used_ids = {}       # (container, beverage) -> global_id
        
        # Domain structure
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_cocktails = []  # list of (shot, cocktail) pairs from goals
        self.shakers = []         # list of shaker names
        self.shots = []           # list of shot names
        self.hands = []           # list of hand names
        
        # Task name -> abstract task mapping
        self.task_name_map = {}
        
        # Operator costs by type
        self.op_type_map = {}

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to get the base name and arguments."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            base = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return base, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "contains" and len(args) == 2:
                self.contains_ids[(args[0], args[1])] = f.global_id
            elif pred == "clean" and len(args) == 1:
                self.clean_ids[args[0]] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 2:
                self.holding_ids[(args[0], args[1])] = f.global_id
            elif pred == "handEmpty" and len(args) == 1:
                self.hand_empty_ids[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "shaked" and len(args) == 1:
                self.shaked_ids[args[0]] = f.global_id
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_ids[args[0]] = f.global_id
            elif pred == "used" and len(args) == 2:
                self.used_ids[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                cocktail = args[0]
                ingredient = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ingredient
            elif pred == "cocktailPart2" and len(args) == 2:
                cocktail = args[0]
                ingredient = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ingredient
        
        # Identify shakers, shots, hands from fact maps
        seen_shakers = set()
        seen_shots = set()
        seen_hands = set()
        for key in self.shaked_ids:
            seen_shakers.add(key)
        for key in self.hand_empty_ids:
            seen_hands.add(key)
        # Shots: containers that appear in contains but are not shakers
        for (container, _bev) in self.contains_ids:
            if container not in seen_shakers:
                seen_shots.add(container)
        # Also from clean_ids
        for container in self.clean_ids:
            if container not in seen_shakers:
                seen_shots.add(container)
        
        self.shakers = list(seen_shakers)
        self.shots = list(seen_shots)
        self.hands = list(seen_hands)
        
        # Parse goal task network from the initial task network to find target cocktails
        # We look for AchieveContainsShotCocktail tasks
        self._extract_goals_from_task_network(initial_node.task_network)
        
        # Build abstract task name map for quick lookups
        for at in model.abstract_tasks:
            base, args = self._parse_task_name(at.name)
            self.task_name_map[at.global_id] = (base, args)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goals_from_task_network(self, task_network):
        """Extract goal (shot, cocktail) pairs from the initial task network."""
        self.goal_cocktails = []
        for task in task_network:
            base, args = self._parse_task_name(task.name)
            if base == "AchieveContainsShotCocktail" and len(args) == 2:
                self.goal_cocktails.append((args[0], args[1]))

    def _check_fact(self, state, fact_id):
        """Check if a fact is true in the given state using bitwise ops."""
        if fact_id is None:
            return False
        return (state >> fact_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """
        Estimate the minimum number of actions needed to get cocktail into shot.
        This is a state-aware lower bound.
        """
        # Check if already done
        contains_id = self.contains_ids.get((shot, cocktail))
        if contains_id is not None and self._check_fact(state, contains_id):
            return 0
        
        cost = 0
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 1  # fallback
        
        ing1, ing2 = parts  # cocktailPart1, cocktailPart2
        
        # Check if the shaker already contains this cocktail (shaked)
        shaker = self.shakers[0] if self.shakers else None
        if shaker is None:
            return 1
        
        shaker_has_cocktail = False
        shaker_contains_id = self.contains_ids.get((shaker, cocktail))
        shaker_shaked_id = self.shaked_ids.get(shaker)
        
        if shaker_contains_id is not None and self._check_fact(state, shaker_contains_id):
            if shaker_shaked_id is not None and self._check_fact(state, shaker_shaked_id):
                shaker_has_cocktail = True
        
        if shaker_has_cocktail:
            # Just need to pour to shot
            # Need: shot clean & empty, hold shaker
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            
            cost = 0
            if not shot_empty:
                cost += 2  # grasp shot + empty-shot
            if not shot_clean:
                cost += 2  # clean-shot needs: hold shot + handEmpty other + clean action
                # Actually: AchieveCleanShot = hold + empty + handEmpty + clean = up to 4
                # But if already empty, just hold + handEmpty + clean = 3
                cost += 1
            
            # Need to hold shaker: grasp (1) + maybe drop current (1)
            shaker_held = False
            for h in self.hands:
                hid = self.holding_ids.get((h, shaker))
                if hid is not None and self._check_fact(state, hid):
                    shaker_held = True
                    break
            if not shaker_held:
                cost += 1  # grasp shaker
                # May need to drop something first
                any_hand_empty = False
                for h in self.hands:
                    hid = self.hand_empty_ids.get(h)
                    if hid is not None and self._check_fact(state, hid):
                        any_hand_empty = True
                        break
                if not any_hand_empty:
                    cost += 1  # drop something
            
            cost += 1  # pour-shaker-to-shot
            return cost
        
        # Need full cocktail production cycle
        # Minimum actions for a full cycle:
        # 1. Clean shaker if needed (up to 3: hold + empty + handEmpty + clean = 4, or just clean = 1 if clean)
        # 2. For each ingredient: fill shot (clean shot + grasp shot + fill = ~4), pour to shaker (~2)
        # 3. Shake (1)
        # 4. Clean target shot if needed, pour to shot (1)
        
        shaker_clean = self._check_fact(state, self.clean_ids.get(shaker))
        shaker_empty = self._check_fact(state, self.empty_ids.get(shaker))
        
        # Shaker preparation
        if not shaker_clean or not shaker_empty:
            if not shaker_empty:
                # Need to empty shaker first: grasp + empty-shaker + clean-shaker = 3
                cost += 3
            else:
                # Empty but not clean: grasp + clean = 2
                cost += 2
        
        # For each ingredient (2 ingredients):
        # Need a shot with ingredient, then pour to shaker
        # Per ingredient: clean-shot(if needed) + grasp-shot + fill-shot + pour-to-shaker = min 3 actions
        # Plus drops and grasps for hand management
        for ing in [ing1, ing2]:
            if ing is None:
                continue
            # Check if any shot already contains this ingredient
            found_in_shot = False
            for s in self.shots:
                cid = self.contains_ids.get((s, ing))
                if cid is not None and self._check_fact(state, cid):
                    found_in_shot = True
                    break
            
            # Check if shaker already has this ingredient
            shaker_has_ing = False
            cid_shaker = self.contains_ids.get((shaker, ing))
            if cid_shaker is not None and self._check_fact(state, cid_shaker):
                shaker_has_ing = True
            
            if shaker_has_ing:
                continue  # No additional cost for this ingredient
            elif found_in_shot:
                # Just pour from shot to shaker: grasp shot + pour = 2
                cost += 2
            else:
                # Fill a shot then pour: grasp shot + fill + pour = 3
                # But may need to clean shot first: +2 (empty + clean)
                cost += 4  # conservative: grasp + fill + drop_other_if_needed + pour
        
        # Shake: 1 action (need to hold shaker + other hand empty)
        cost += 1
        
        # Pour to target shot
        shot_clean = self._check_fact(state, self.clean_ids.get(shot))
        shot_empty = self._check_fact(state, self.empty_ids.get(shot))
        
        if not shot_empty:
            cost += 2  # empty the shot
        if not shot_clean:
            cost += 2  # clean the shot
        
        cost += 1  # pour-shaker-to-shot
        
        return cost

    def _remaining_cocktails_count(self, state, task_network):
        """
        Count how many distinct cocktails still need to be produced
        (not yet in their target shots).
        """
        count = 0
        seen_cocktails = set()
        
        for task in task_network:
            base, args = self._parse_task_name(task.name)
            if base == "AchieveContainsShotCocktail" and len(args) == 2:
                shot, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is None or not self._check_fact(state, cid):
                    if cocktail not in seen_cocktails:
                        seen_cocktails.add(cocktail)
                        count += 1
        
        return count

    def _shaker_serialization_bound(self, state, task_network):
        """
        Each remaining cocktail needs a full shaker cycle.
        Minimum cycle: clean(1-3) + 2*pour-ingredient(2 each) + shake(1) + pour-out(1) = ~8 min
        But we use a conservative estimate per cocktail.
        """
        remaining = self._remaining_cocktails_count(state, task_network)
        
        if remaining == 0:
            return 0
        
        shaker = self.shakers[0] if self.shakers else None
        if shaker is None:
            return remaining
        
        # Each cocktail after the first needs: empty-shaker + clean-shaker + 
        # 2*(fill-shot + pour-to-shaker) + shake + pour-to-shot
        # = 2 + 2*3 + 1 + 1 = 10 minimum actions per cocktail cycle
        # First cocktail might save on cleaning if shaker is already clean
        
        shaker_clean = self._check_fact(state, self.clean_ids.get(shaker))
        shaker_empty = self._check_fact(state, self.empty_ids.get(shaker))
        
        # Check if shaker currently has a cocktail that's useful
        shaker_has_useful = False
        for task in task_network:
            base, args = self._parse_task_name(task.name)
            if base == "AchieveContainsShotCocktail" and len(args) == 2:
                cocktail = args[1]
                cid = self.contains_ids.get((shaker, cocktail))
                shaked_id = self.shaked_ids.get(shaker)
                if (cid is not None and self._check_fact(state, cid) and
                    shaked_id is not None and self._check_fact(state, shaked_id)):
                    shaker_has_useful = True
                    break
        
        if shaker_has_useful:
            # One cocktail is ready, just needs pouring
            # Remaining (remaining-1) need full cycles
            per_cycle = 10
            cost = 2 + (remaining - 1) * per_cycle  # 2 for the pour of current
        else:
            first_cycle_cost = 8  # slightly less if shaker is clean
            if shaker_clean and shaker_empty:
                first_cycle_cost = 7
            subsequent_cost = 10
            cost = first_cycle_cost + max(0, remaining - 1) * subsequent_cost
        
        return cost

    def _task_network_bound(self, state, task_network):
        """
        Estimate minimum actions from the task network structure.
        For each task, estimate minimum decomposition depth.
        """
        cost = 0
        for task in task_network:
            # Check if it's an operator (primitive task)
            if hasattr(task, 'add_effects'):
                # It's an operator - costs exactly 1
                cost += 1
            elif hasattr(task, 'decompositions'):
                # It's an abstract task
                base, args = self._parse_task_name(task.name)
                task_cost = self._estimate_abstract_task_cost(state, base, args)
                cost += task_cost
        
        return cost

    def _estimate_abstract_task_cost(self, state, base, args):
        """
        State-aware minimum cost estimate for an abstract task.
        """
        if base == "AchieveContainsShotCocktail" and len(args) == 2:
            shot, cocktail = args[0], args[1]
            return self._cocktail_cost(state, shot, cocktail)
        
        elif base == "AchieveContainsShakerCocktail" and len(args) == 2:
            shaker, cocktail = args[0], args[1]
            cid = self.contains_ids.get((shaker, cocktail))
            if cid is not None and self._check_fact(state, cid):
                return 0
            # Need: clean shaker + 2 ingredients + shake = ~8
            return 7
        
        elif base == "AchieveContainsShakerIngredient" and len(args) == 2:
            shaker, ingredient = args[0], args[1]
            cid = self.contains_ids.get((shaker, ingredient))
            if cid is not None and self._check_fact(state, cid):
                return 0
            # Need: fill shot + pour to shaker = ~4
            return 3
        
        elif base == "AchieveContainsShotIngredient" and len(args) == 2:
            shot, ingredient = args[0], args[1]
            cid = self.contains_ids.get((shot, ingredient))
            if cid is not None and self._check_fact(state, cid):
                return 0
            # Need: clean shot + grasp + fill = ~3
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            cost = 1  # fill-shot
            if not shot_clean:
                cost += 2
            if not shot_empty:
                cost += 1
            cost += 1  # grasp
            return cost
        
        elif base == "AchieveCleanShaker" and len(args) == 1:
            shaker = args[0]
            if self._check_fact(state, self.clean_ids.get(shaker)):
                return 0
            if self._check_fact(state, self.empty_ids.get(shaker)):
                return 2  # grasp + clean
            return 4  # grasp + empty + clean + hand management
        
        elif base == "AchieveCleanShot" and len(args) == 1:
            shot = args[0]
            if self._check_fact(state, self.clean_ids.get(shot)):
                return 0
            if self._check_fact(state, self.empty_ids.get(shot)):
                return 2  # grasp + clean
            return 4  # grasp + empty + clean + hand mgmt
        
        elif base == "AchieveHolding" and len(args) == 2:
            hand, container = args[0], args[1]
            hid = self.holding_ids.get((hand, container))
            if hid is not None and self._check_fact(state, hid):
                return 0
            # May need: drop current + grasp = 1-2
            he_id = self.hand_empty_ids.get(hand)
            if he_id is not None and self._check_fact(state, he_id):
                # Check if container is on table
                ot_id = self.ontable_ids.get(container)
                if ot_id is not None and self._check_fact(state, ot_id):
                    return 1  # just grasp
                return 2  # someone else holds it, need drop + grasp
            return 2  # drop + grasp
        
        elif base == "AchieveHandEmpty" and len(args) == 1:
            hand = args[0]
            he_id = self.hand_empty_ids.get(hand)
            if he_id is not None and self._check_fact(state, he_id):
                return 0
            return 1  # drop
        
        elif base == "AchieveOnTable" and len(args) == 1:
            container = args[0]
            ot_id = self.ontable_ids.get(container)
            if ot_id is not None and self._check_fact(state, ot_id):
                return 0
            return 1  # drop
        
        elif base == "DoPourShakerToShot" and len(args) == 3:
            return 1  # pour-shaker-to-shot action
        
        # Default: at least 1 action
        return 1

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0
        
        # Term 1: Sum of per-task costs (additive, state-aware)
        term1 = self._task_network_bound(state, task_network)
        
        # Term 2: Shaker serialization bound
        term2 = self._shaker_serialization_bound(state, task_network)
        
        # Take the max of the two admissible bounds
        h = max(term1, term2)
        
        # Tie-breaking: penalize states where both hands are full (less flexibility)
        both_hands_full = True
        for hand in self.hands:
            he_id = self.hand_empty_ids.get(hand)
            if he_id is not None and self._check_fact(state, he_id):
                both_hands_full = False
                break
        
        if both_hands_full and len(self.hands) > 0:
            h += 0.5  # Small penalty for reduced flexibility
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
