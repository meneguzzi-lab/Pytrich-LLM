"""
Barman BDI Domain-Dependent Heuristic

Bottleneck: Single shaker serializes all cocktail production. Each cocktail
requires exclusive shaker access: clean shaker, add 2 ingredients via shots,
shake, pour to target shot. This creates a sequential bottleneck.

Admissible lower bounds:
1. Unserved cocktails: Each cocktail not in target shot needs ~13-15 actions minimum.
2. Task network cost: State-aware per-task cost estimation for remaining tasks.
3. Shaker state penalty: Current shaker state may require cleanup before next cocktail.

All terms are admissible lower bounds; max preserves admissibility.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator, AbstractTask


class BarmanBDIAnthropicOpus46Heuristic6(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_6"):
        super().__init__(name)
        # Fact ID maps
        self.contains_ids = {}    # (container, beverage) -> global_id
        self.clean_ids = {}       # container -> global_id
        self.empty_ids = {}       # container -> global_id
        self.holding_ids = {}     # (hand, container) -> global_id
        self.hand_empty_ids = {}  # hand -> global_id
        self.ontable_ids = {}     # container -> global_id
        self.shaked_ids = {}      # shaker -> global_id
        self.unshaked_ids = {}    # shaker -> global_id
        self.used_ids = {}        # (container, beverage) -> global_id
        self.cocktail_part1 = {}  # cocktail -> ingredient
        self.cocktail_part2 = {}  # cocktail -> ingredient
        self.dispenses_map = {}   # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktail_shots = []  # [(shot, cocktail)] from goals
        
        # Abstract task name -> id mapping
        self.task_name_map = {}
        
        # Shakers and shots
        self.shakers = []
        self.shots = []
        self.hands = []
        self.cocktails = []
        self.ingredients = []

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
        """Parse a task/operator name like '__AchieveContainsShotCocktail[shot2,cocktail1]'"""
        name = name.strip()
        # Remove leading underscores
        clean = name.lstrip('_')
        bracket_idx = clean.find("[")
        if bracket_idx > 0 and clean.endswith("]"):
            task_type = clean[:bracket_idx]
            args_str = clean[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return clean, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts into lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
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
                self.cocktail_part1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
            elif pred == "dispenses" and len(args) == 2:
                self.dispenses_map[args[1]] = args[0]

        # Collect object types from fact maps
        seen_shakers = set()
        seen_shots = set()
        seen_hands = set()
        
        for key in self.shaked_ids:
            seen_shakers.add(key)
        for key in self.unshaked_ids:
            seen_shakers.add(key)
        self.shakers = list(seen_shakers)
        
        for key in self.hand_empty_ids:
            seen_hands.add(key)
        self.hands = list(seen_hands)
        
        self.cocktails = list(self.cocktail_part1.keys())
        
        # Identify goal cocktail-shot pairs from goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                # Check if second arg is a cocktail
                if args[1] in self.cocktail_part1:
                    self.goal_cocktail_shots.append((args[0], args[1]))
        
        # Build abstract task type mapping for fast lookup
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            # Store for reference
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if a fact is true in current state. O(1) bitwise."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _unserved_cocktails_cost(self, state):
        """
        For each goal (shot, cocktail) pair not yet satisfied, estimate minimum actions.
        
        A cocktail not yet in its target shot requires at minimum:
        - If shaker doesn't contain the cocktail (shaked):
          - Clean shaker if needed: ~3 actions (grasp, empty, clean) 
          - For each of 2 ingredients: clean shot + grasp shot + fill shot + pour to shaker = ~5 actions each
          - Shake: 1 action (need shaker held + other hand empty)
          - Pour shaker to shot: grasp shaker + pour = ~2 actions
          Total: ~3 + 5 + 5 + 1 + 2 = 16 actions
        - If shaker contains cocktail (already shaked):
          - Just need to pour to clean empty shot: ~3-5 actions
        
        This is admissible because each cocktail genuinely requires these steps.
        """
        total_cost = 0
        unserved_count = 0
        
        for shot, cocktail in self.goal_cocktail_shots:
            contains_id = self.contains_ids.get((shot, cocktail))
            if contains_id is not None and self._check_fact(state, contains_id):
                continue  # Already served
            
            unserved_count += 1
            cost = 0
            
            # Check if shaker already contains this cocktail (shaked)
            shaker_has_cocktail = False
            for shaker in self.shakers:
                c_id = self.contains_ids.get((shaker, cocktail))
                s_id = self.shaked_ids.get(shaker)
                if c_id is not None and s_id is not None:
                    if self._check_fact(state, c_id) and self._check_fact(state, s_id):
                        shaker_has_cocktail = True
                        # Need: clean shot, hold shaker, pour
                        shot_clean = self.clean_ids.get(shot)
                        shot_empty = self.empty_ids.get(shot)
                        if shot_clean and self._check_fact(state, shot_clean) and shot_empty and self._check_fact(state, shot_empty):
                            cost += 1  # just pour
                        else:
                            cost += 4  # clean shot + pour
                        break
            
            if not shaker_has_cocktail:
                # Need to make the cocktail from scratch
                # Minimum per cocktail: 
                #   - Possibly clean/empty shaker: at least 1-3 actions
                #   - For each ingredient (2): get in shot, pour to shaker: ~4-6 actions each
                #   - Shake: 1 action
                #   - Pour to target shot: 1-3 actions
                
                ing1 = self.cocktail_part1.get(cocktail)
                ing2 = self.cocktail_part2.get(cocktail)
                
                # Cost for each ingredient into shaker
                for ing in [ing1, ing2]:
                    if ing is None:
                        continue
                    # Check if any shot already contains this ingredient
                    shot_has_ing = False
                    for s, i in self.contains_ids:
                        if i == ing and s != shot:  # some shot has ingredient
                            if self._check_fact(state, self.contains_ids[(s, i)]):
                                shot_has_ing = True
                                break
                    
                    if shot_has_ing:
                        # Just need to pour from shot to shaker: grasp + pour = 2
                        cost += 2
                    else:
                        # Need: clean a shot, hold it, fill from dispenser, pour to shaker
                        # grasp shot (1-2) + fill (1) + pour-to-shaker (1) = 3-4
                        cost += 4
                
                # Shake action: 1
                cost += 1
                
                # Pour shaker to target shot
                shot_clean_id = self.clean_ids.get(shot)
                shot_empty_id = self.empty_ids.get(shot)
                if shot_clean_id and self._check_fact(state, shot_clean_id) and \
                   shot_empty_id and self._check_fact(state, shot_empty_id):
                    cost += 1  # just pour
                else:
                    cost += 3  # clean shot + pour
                
                # Shaker cleanup cost for this cocktail
                for shaker in self.shakers:
                    shaker_clean = self.clean_ids.get(shaker)
                    shaker_empty = self.empty_ids.get(shaker)
                    if shaker_clean and self._check_fact(state, shaker_clean) and \
                       shaker_empty and self._check_fact(state, shaker_empty):
                        pass  # shaker ready, no extra cost
                    elif shaker_empty and self._check_fact(state, shaker_empty):
                        cost += 1  # just clean
                    else:
                        cost += 3  # empty + clean
                    break  # only one shaker typically
            
            total_cost += cost
        
        return total_cost

    def _shaker_serialization_bound(self, state):
        """
        Since there's only one shaker, each unserved cocktail requires exclusive
        shaker access. The minimum cycle per cocktail through the shaker is:
        - clean shaker (if needed): 1 action minimum
        - pour ingredient 1 to shaker: 1 action
        - pour ingredient 2 to shaker: 1 action  
        - shake: 1 action
        - pour to shot: 1 action
        Total minimum: 5 actions per cocktail (just the shaker-touching actions)
        
        This is admissible: each cocktail needs at least these shaker operations,
        and they cannot overlap.
        """
        unserved = 0
        for shot, cocktail in self.goal_cocktail_shots:
            contains_id = self.contains_ids.get((shot, cocktail))
            if contains_id is not None and self._check_fact(state, contains_id):
                continue
            unserved += 1
        
        if unserved == 0:
            return 0
        
        # First cocktail: at least 5 shaker-related actions
        # Each subsequent: at least 5 + 2 (empty + clean shaker) = 7
        # But we use a conservative bound
        # Actually minimum cycle: pour-to-clean-shaker (1) + pour-to-used-shaker (1) + shake (1) + pour-to-shot (1) = 4
        # Plus for subsequent: empty-shaker (1) + clean-shaker (1) = 2 more
        # So: first = 4, each additional = 6
        # But we also need grasp/drop actions for shots to fill ingredients
        # Conservative admissible: 4 for first, 6 for each additional
        
        # Check if shaker is clean and empty (ready for first cocktail)
        shaker_ready = False
        for shaker in self.shakers:
            sc = self.clean_ids.get(shaker)
            se = self.empty_ids.get(shaker)
            if sc and se and self._check_fact(state, sc) and self._check_fact(state, se):
                shaker_ready = True
            break
        
        if shaker_ready:
            # First cocktail uses 4 shaker actions, rest use 6
            return 4 * 1 + 6 * max(0, unserved - 1) if unserved >= 1 else 0
        else:
            # Need to prep shaker first
            return 6 * unserved

    def _task_network_cost(self, state, task_network):
        """
        State-aware task network cost estimation.
        For each task in the network, estimate its minimum decomposition cost
        based on the current state.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            if isinstance(task, Operator):
                total += 1
            elif isinstance(task, AbstractTask):
                task_type, args = self._parse_task_name(task.name)
                cost = self._estimate_abstract_task_cost(state, task_type, args)
                total += cost
        
        return total

    def _estimate_abstract_task_cost(self, state, task_type, args):
        """
        Estimate minimum actions for an abstract task given current state.
        This gives state-aware per-task costs rather than flat +1.
        """
        if task_type == "AchieveContainsShotCocktail":
            if len(args) >= 2:
                shot, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is not None and self._check_fact(state, cid):
                    return 0  # Already done
                # Full cocktail making: ~12-15 actions
                return 12
            return 12
        
        elif task_type == "AchieveContainsShakerCocktail":
            if len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shaker, cocktail))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                # Need: clean shaker + 2 ingredients + shake = ~10
                return 9
            return 9
        
        elif task_type == "AchieveContainsShakerIngredient":
            if len(args) >= 2:
                shaker, ingredient = args[0], args[1]
                cid = self.contains_ids.get((shaker, ingredient))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                # Need: fill shot with ingredient + pour to shaker = ~5
                return 4
            return 4
        
        elif task_type == "AchieveContainsShotIngredient":
            if len(args) >= 2:
                shot, ingredient = args[0], args[1]
                cid = self.contains_ids.get((shot, ingredient))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                # Need: clean shot + hold + fill = ~4
                sc = self.clean_ids.get(shot)
                se = self.empty_ids.get(shot)
                if sc and se and self._check_fact(state, sc) and self._check_fact(state, se):
                    return 2  # just grasp + fill
                return 4
            return 4
        
        elif task_type == "AchieveCleanShaker":
            if len(args) >= 1:
                shaker = args[0]
                sc = self.clean_ids.get(shaker)
                if sc and self._check_fact(state, sc):
                    return 0
                se = self.empty_ids.get(shaker)
                if se and self._check_fact(state, se):
                    return 2  # grasp + clean
                return 4  # grasp + empty + clean
            return 3
        
        elif task_type == "AchieveCleanShot":
            if len(args) >= 1:
                shot = args[0]
                sc = self.clean_ids.get(shot)
                if sc and self._check_fact(state, sc):
                    return 0
                se = self.empty_ids.get(shot)
                if se and self._check_fact(state, se):
                    return 2  # grasp + clean
                return 3  # grasp + empty + clean
            return 2
        
        elif task_type == "AchieveHolding":
            if len(args) >= 2:
                hand, container = args[0], args[1]
                hid = self.holding_ids.get((hand, container))
                if hid and self._check_fact(state, hid):
                    return 0
                # Need: possibly drop current + grasp = 1-2
                he = self.hand_empty_ids.get(hand)
                if he and self._check_fact(state, he):
                    ot = self.ontable_ids.get(container)
                    if ot and self._check_fact(state, ot):
                        return 1  # just grasp
                    return 2  # put down container + grasp
                return 2  # drop + grasp
            return 1
        
        elif task_type == "AchieveHandEmpty":
            if len(args) >= 1:
                hand = args[0]
                he = self.hand_empty_ids.get(hand)
                if he and self._check_fact(state, he):
                    return 0
                return 1  # drop
            return 1
        
        elif task_type == "AchieveOnTable":
            if len(args) >= 1:
                container = args[0]
                ot = self.ontable_ids.get(container)
                if ot and self._check_fact(state, ot):
                    return 0
                return 1  # drop
            return 1
        
        elif task_type == "DoPourShakerToShot":
            return 1  # single action
        
        # Default: at least 1 action
        return 1

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible lower bounds."""
        if not task_network and (self.model.goals & state) == self.model.goals:
            return 0
        
        # Term 1: Unserved cocktails cost (detailed state-aware estimation)
        h1 = self._unserved_cocktails_cost(state)
        
        # Term 2: Shaker serialization bound
        h2 = self._shaker_serialization_bound(state)
        
        # Term 3: Task network cost with state-aware per-task estimation
        h3 = self._task_network_cost(state, task_network)
        
        return max(h1, h2, h3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
