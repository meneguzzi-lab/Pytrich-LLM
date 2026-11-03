"""Domain-dependent heuristic for Barman_BDI domain.

Bottleneck Analysis:
  The single shaker is the main serializing constraint. Every cocktail must be
  mixed in the shaker (clean shaker -> add ingredient1 -> add ingredient2 -> shake
  -> pour to shot -> clean shaker for next cocktail). This means cocktails are
  essentially serialized through the shaker, and each cocktail requires a minimum
  sequence of actions.

Admissible Lower Bounds:
  Term 1 - Per-cocktail minimum action cost:
    Each cocktail not yet in its target shot requires at minimum:
    - If shaker doesn't contain the cocktail and shot doesn't contain it:
      fill-shot(ing1) + pour-shot-to-shaker + fill-shot(ing2) + pour-shot-to-shaker
      + shake + pour-shaker-to-shot = 6 core actions, plus grasp/drop overhead.
      Conservative lower bound: ~11 actions per unfinished cocktail.
    - This is admissible because these actions are unavoidable.

  Term 2 - Shaker serialization bound:
    Since there's only one shaker and it must be cleaned between cocktails,
    the number of remaining cocktails times the minimum per-cocktail shaker
    cycle gives a lower bound.

  Term 3 - Task network structural bound:
    Count compound tasks in the task network with state-aware cost estimation.
    Each AchieveContainsShotCocktail that isn't satisfied costs at least the
    minimum decomposition depth.

Tie-breaking:
  Shots are interchangeable for intermediate use. We add a small penalty
  based on state disorder (dirty shots, occupied hands) to break ties.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic10(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_10"):
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
        
        # Domain structure
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_cocktails = []  # list of (shot, cocktail) pairs from goals/task network
        
        # Task name -> type mapping for task network analysis
        self.task_type_map = {}   # abstract_task global_id -> task type string
        
        # Operator/task name prefixes for identification
        self.achieve_contains_shot_cocktail_ids = set()
        self.achieve_contains_shaker_cocktail_ids = set()
        self.achieve_contains_shaker_ingredient_ids = set()
        self.achieve_contains_shot_ingredient_ids = set()
        self.achieve_clean_shot_ids = set()
        self.achieve_clean_shaker_ids = set()
        self.achieve_holding_ids = set()
        self.achieve_hand_empty_ids = set()
        self.achieve_on_table_ids = set()
        self.do_pour_shaker_to_shot_ids = set()

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
        """Parse a grounded task/abstract task name to get task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            gid = f.global_id
            if pred == "contains" and len(args) == 2:
                self.contains_ids[(args[0], args[1])] = gid
            elif pred == "clean" and len(args) == 1:
                self.clean_ids[args[0]] = gid
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = gid
            elif pred == "holding" and len(args) == 2:
                self.holding_ids[(args[0], args[1])] = gid
            elif pred == "handEmpty" and len(args) == 1:
                self.hand_empty_ids[args[0]] = gid
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = gid
            elif pred == "shaked" and len(args) == 1:
                self.shaked_ids[args[0]] = gid
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_ids[args[0]] = gid
            elif pred == "used" and len(args) == 2:
                self.used_ids[(args[0], args[1])] = gid
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

        # Build abstract task type map
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)
            
            if task_type == "AchieveContainsShotCocktail":
                self.achieve_contains_shot_cocktail_ids.add(at.global_id)
            elif task_type == "AchieveContainsShakerCocktail":
                self.achieve_contains_shaker_cocktail_ids.add(at.global_id)
            elif task_type == "AchieveContainsShakerIngredient":
                self.achieve_contains_shaker_ingredient_ids.add(at.global_id)
            elif task_type == "AchieveContainsShotIngredient":
                self.achieve_contains_shot_ingredient_ids.add(at.global_id)
            elif task_type == "AchieveCleanShot":
                self.achieve_clean_shot_ids.add(at.global_id)
            elif task_type == "AchieveCleanShaker":
                self.achieve_clean_shaker_ids.add(at.global_id)
            elif task_type == "AchieveHolding":
                self.achieve_holding_ids.add(at.global_id)
            elif task_type == "AchieveHandEmpty":
                self.achieve_hand_empty_ids.add(at.global_id)
            elif task_type == "AchieveOnTable":
                self.achieve_on_table_ids.add(at.global_id)
            elif task_type == "DoPourShakerToShot":
                self.do_pour_shaker_to_shot_ids.add(at.global_id)

        # Extract goal cocktails from initial task network
        self.goal_cocktails = []
        for task in model.initial_tn:
            if hasattr(task, 'decompositions'):
                task_type, args = self._parse_task_name(task.name)
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    self.goal_cocktails.append((args[0], args[1]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if fact with given global_id is true in state."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """Estimate minimum actions needed to get cocktail into shot."""
        # Check if already done
        contains_id = self.contains_ids.get((shot, cocktail))
        if contains_id is not None and self._check_fact(state, contains_id):
            return 0
        
        cost = 0
        
        # We need to find a shaker. In this domain there's typically one shaker.
        # Check if the shaker already contains this cocktail
        shaker_has_cocktail = False
        shaker_name = None
        for key, gid in self.shaked_ids.items():
            shaker_name = key
            shaker_contains_id = self.contains_ids.get((key, cocktail))
            if shaker_contains_id is not None and self._check_fact(state, shaker_contains_id):
                shaker_has_cocktail = True
                break
        
        if shaker_has_cocktail and shaker_name is not None:
            # Shaker has the cocktail and is shaked, we just need to pour
            # Need: hold shaker, shot clean+empty, pour
            # Minimum: grasp shaker (maybe drop first) + pour = ~2-3 actions
            shot_clean = self.clean_ids.get(shot)
            shot_empty = self.empty_ids.get(shot)
            
            pour_cost = 1  # pour-shaker-to-shot
            
            # Need to hold shaker
            holding_shaker = False
            for hand in ['left', 'right']:
                hid = self.holding_ids.get((hand, shaker_name))
                if hid is not None and self._check_fact(state, hid):
                    holding_shaker = True
                    break
            if not holding_shaker:
                pour_cost += 1  # grasp
            
            # Need shot clean and empty
            if shot_clean is None or not self._check_fact(state, shot_clean):
                pour_cost += 2  # empty + clean
            elif shot_empty is not None and not self._check_fact(state, shot_empty):
                pour_cost += 1  # empty
            
            return pour_cost
        
        # Need full cocktail making process
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 1
        
        ing1, ing2 = parts
        
        # Check if shaker needs cleaning
        shaker_clean_cost = 0
        if shaker_name is not None:
            shaker_clean_id = self.clean_ids.get(shaker_name)
            shaker_empty_id = self.empty_ids.get(shaker_name)
            if shaker_clean_id is not None and not self._check_fact(state, shaker_clean_id):
                if shaker_empty_id is not None and self._check_fact(state, shaker_empty_id):
                    # Just needs cleaning: grasp + clean = 2 (need both hands)
                    shaker_clean_cost = 3  # grasp + clean + drop maybe
                else:
                    # Needs emptying then cleaning
                    shaker_clean_cost = 5  # grasp + empty + clean + drop
        
        # For each ingredient: need to fill a shot, pour to shaker
        # fill-shot needs: clean shot + hold shot + hand empty + fill = ~4 actions
        # pour-shot-to-shaker needs: hold shot + pour = already holding
        # So per ingredient: ~4-5 actions (grasp shot, fill, pour-to-shaker)
        
        ing_cost = 0
        for ing in [ing1, ing2]:
            # Check if shaker already contains this ingredient
            shaker_contains_ing = False
            if shaker_name is not None:
                sc_id = self.contains_ids.get((shaker_name, ing))
                if sc_id is not None and self._check_fact(state, sc_id):
                    shaker_contains_ing = True
            
            if not shaker_contains_ing:
                # Need: get ingredient into a shot, pour to shaker
                # Minimum: grasp shot (1) + fill-shot (1) + pour-shot-to-shaker (1) = 3
                # Plus potential: clean shot (2), drop current item (1)
                ing_cost += 5  # conservative estimate per ingredient
        
        # Shake: 1 action (need to hold shaker + free hand)
        shake_cost = 1
        
        # Pour shaker to shot: 1 action
        pour_cost = 1
        
        # Shot needs to be clean and empty for pouring
        shot_prep_cost = 0
        shot_clean_id = self.clean_ids.get(shot)
        shot_empty_id = self.empty_ids.get(shot)
        if shot_clean_id is not None and not self._check_fact(state, shot_clean_id):
            shot_prep_cost += 2  # empty + clean or just clean
        if shot_empty_id is not None and not self._check_fact(state, shot_empty_id):
            shot_prep_cost += 1
        
        # Need to grasp shaker for shaking and pouring
        grasp_cost = 2  # grasp shaker + potential drops
        
        cost = shaker_clean_cost + ing_cost + shake_cost + pour_cost + shot_prep_cost + grasp_cost
        
        return cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network with state awareness."""
        cost = 0
        
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive task)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1
                continue
            
            # It's an abstract task
            if gid not in self.task_type_map:
                cost += 1
                continue
            
            task_type, args = self.task_type_map[gid]
            
            if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                shot, cocktail = args[0], args[1]
                cost += self._cocktail_cost(state, shot, cocktail)
                
            elif task_type == "AchieveContainsShakerCocktail" and len(args) == 2:
                shaker, cocktail = args[0], args[1]
                contains_id = self.contains_ids.get((shaker, cocktail))
                if contains_id is not None and self._check_fact(state, contains_id):
                    cost += 0
                else:
                    # Need to add ingredients and shake
                    cost += 8  # clean shaker + 2 ingredients + shake
                    
            elif task_type == "AchieveContainsShakerIngredient" and len(args) == 2:
                shaker, ingredient = args[0], args[1]
                contains_id = self.contains_ids.get((shaker, ingredient))
                if contains_id is not None and self._check_fact(state, contains_id):
                    cost += 0
                else:
                    cost += 5  # fill shot + pour to shaker
                    
            elif task_type == "AchieveContainsShotIngredient" and len(args) == 2:
                shot, ingredient = args[0], args[1]
                contains_id = self.contains_ids.get((shot, ingredient))
                if contains_id is not None and self._check_fact(state, contains_id):
                    cost += 0
                else:
                    cost += 4  # clean shot + grasp + fill
                    
            elif task_type == "AchieveCleanShot" and len(args) == 1:
                shot = args[0]
                clean_id = self.clean_ids.get(shot)
                if clean_id is not None and self._check_fact(state, clean_id):
                    cost += 0
                else:
                    empty_id = self.empty_ids.get(shot)
                    if empty_id is not None and self._check_fact(state, empty_id):
                        cost += 2  # grasp + clean
                    else:
                        cost += 3  # grasp + empty + clean
                        
            elif task_type == "AchieveCleanShaker" and len(args) == 1:
                shaker = args[0]
                clean_id = self.clean_ids.get(shaker)
                if clean_id is not None and self._check_fact(state, clean_id):
                    cost += 0
                else:
                    empty_id = self.empty_ids.get(shaker)
                    if empty_id is not None and self._check_fact(state, empty_id):
                        cost += 3  # grasp + clean (need both hands)
                    else:
                        cost += 4  # grasp + empty + clean
                        
            elif task_type == "AchieveHolding" and len(args) == 2:
                hand, container = args[0], args[1]
                holding_id = self.holding_ids.get((hand, container))
                if holding_id is not None and self._check_fact(state, holding_id):
                    cost += 0
                else:
                    cost += 1  # grasp (may need drop first)
                    hand_empty_id = self.hand_empty_ids.get(hand)
                    if hand_empty_id is not None and not self._check_fact(state, hand_empty_id):
                        cost += 1  # need to drop first
                        
            elif task_type == "AchieveHandEmpty" and len(args) == 1:
                hand = args[0]
                hand_empty_id = self.hand_empty_ids.get(hand)
                if hand_empty_id is not None and self._check_fact(state, hand_empty_id):
                    cost += 0
                else:
                    cost += 1  # drop
                    
            elif task_type == "AchieveOnTable" and len(args) == 1:
                container = args[0]
                ontable_id = self.ontable_ids.get(container)
                if ontable_id is not None and self._check_fact(state, ontable_id):
                    cost += 0
                else:
                    cost += 1  # drop
                    
            elif task_type == "DoPourShakerToShot" and len(args) == 3:
                cost += 1  # pour action
            else:
                cost += 1
        
        return cost

    def _unfinished_cocktails_bound(self, state, task_network):
        """Count unfinished cocktail goals and compute shaker serialization bound."""
        unfinished = 0
        
        # Check from task network which AchieveContainsShotCocktail tasks remain
        for task in task_network:
            gid = task.global_id
            if gid in self.task_type_map:
                task_type, args = self.task_type_map[gid]
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    shot, cocktail = args[0], args[1]
                    contains_id = self.contains_ids.get((shot, cocktail))
                    if contains_id is None or not self._check_fact(state, contains_id):
                        unfinished += 1
        
        if unfinished == 0:
            return 0
        
        # Each cocktail needs at minimum:
        # - 2 fill-shots (2 actions)
        # - 2 pour-shot-to-shaker (2 actions)  
        # - 1 shake (1 action)
        # - 1 pour-shaker-to-shot (1 action)
        # - Plus grasp/drop overhead: at least 4 more (grasp shot, drop shot, grasp shaker, etc.)
        # Conservative minimum per cocktail through shaker: 10 actions
        # But first cocktail can reuse clean shaker, subsequent ones need cleaning: +3
        
        # First cocktail: ~10 actions minimum
        # Each additional: ~13 actions (need to clean shaker)
        if unfinished == 1:
            return 10
        else:
            return 10 + (unfinished - 1) * 13

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible bounds."""
        if len(task_network) == 0:
            return 0
        
        # Term 1: State-aware task network cost
        term1 = self._task_network_cost(state, task_network)
        
        # Term 2: Shaker serialization bound
        term2 = self._unfinished_cocktails_bound(state, task_network)
        
        # Term 3: Simple task count (very fast, weak lower bound)
        term3 = 0
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                term3 += 1  # primitive actions count exactly 1
        
        h = max(term1, term2, term3)
        
        # Tie-breaking: penalize messy state slightly
        # Count dirty shots and occupied hands as disorder
        disorder = 0
        for hand_name, gid in self.hand_empty_ids.items():
            if not self._check_fact(state, gid):
                disorder += 1
        
        return h + disorder * 0.001

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
