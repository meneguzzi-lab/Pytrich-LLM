# Barman BDI Domain Heuristic
# 
# BOTTLENECK: The single shaker serializes all cocktail production.
# Every cocktail must go through: clean_shaker -> fill_ingredient1_to_shot -> pour_to_shaker ->
# fill_ingredient2_to_shot -> pour_to_shaker -> shake -> pour_to_target_shot.
# The shaker must be emptied and cleaned between cocktails.
#
# ADMISSIBLE LOWER BOUNDS:
# 1. Per-cocktail action cost: Each unserved cocktail needs at minimum ~11-13 actions
#    (clean shaker, fill shot with ing1, pour to shaker, fill shot with ing2, pour to shaker,
#     shake, clean target shot, pick up shaker, pour to shot, plus grasp/drop overhead).
#    We use a conservative per-cocktail estimate based on current state.
# 2. Shaker serialization: Since cocktails are sequential through the shaker,
#    the total cost is at least the sum of per-cocktail minimum costs.
# 3. Task network depth: weighted count of remaining tasks.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic2(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_2"):
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
        self.shaker_level_ids = {}  # (shaker, level) -> global_id
        self.shaker_empty_level_ids = {}  # (shaker, level) -> global_id
        
        # Domain knowledge
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.dispenses_map = {}   # ingredient -> dispenser
        
        # Task name maps
        self.task_name_map = {}  # abstract_task global_id -> (task_type, args)
        
        # Goal cocktails: list of (shot, cocktail) pairs
        self.goal_cocktails = []
        
        # All shots, shakers, hands
        self.shots = []
        self.shakers = []
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
        """Parse a task/operator name to extract task type and arguments."""
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
        
        # Parse all facts and build lookup maps
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
            elif pred == "shakerLevel" and len(args) == 2:
                self.shaker_level_ids[(args[0], args[1])] = f.global_id
            elif pred == "shakerEmptyLevel" and len(args) == 2:
                self.shaker_empty_level_ids[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                cocktail = args[0]
                ing = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ing
            elif pred == "cocktailPart2" and len(args) == 2:
                cocktail = args[0]
                ing = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ing
            elif pred == "dispenses" and len(args) == 2:
                self.dispenses_map[args[1]] = args[0]
            elif pred == "ingredient" and len(args) == 1:
                if args[0] not in self.ingredients:
                    self.ingredients.append(args[0])

        # Collect object types from fact maps
        for key in self.ontable_ids:
            if key.startswith("shot"):
                if key not in self.shots:
                    self.shots.append(key)
            elif key.startswith("shaker"):
                if key not in self.shakers:
                    self.shakers.append(key)
        
        for key in self.hand_empty_ids:
            if key not in self.hands:
                self.hands.append(key)
        
        for key in self.cocktail_parts:
            if key not in self.cocktails:
                self.cocktails.append(key)

        # Parse abstract tasks to build task name map
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_name_map[at.global_id] = (task_type, args)

        # Extract goal cocktails from the initial task network
        self._extract_goal_cocktails(initial_node.task_network)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goal_cocktails(self, task_network):
        """Extract (shot, cocktail) pairs from top-level tasks."""
        self.goal_cocktails = []
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                task_type, args = self._parse_task_name(task.name)
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    self.goal_cocktails.append((args[0], args[1]))

    def _bit_check(self, state, fact_id):
        """Check if a fact is true in state using bitwise operation."""
        if fact_id is None:
            return False
        return (state >> fact_id) & 1

    def _estimate_cocktail_cost(self, state, shot, cocktail):
        """Estimate minimum actions needed to serve a cocktail to a shot.
        
        Full pipeline per cocktail (minimum actions):
        1. Clean shaker if dirty (3-5 actions: grasp, empty, drop other, clean, or just clean)
        2. Fill shot with ingredient1 (clean shot + grasp shot + fill = ~3-4 actions)
        3. Pour shot to shaker (~1 action, but need to be holding shot)
        4. Fill shot with ingredient2 (clean shot + grasp shot + fill = ~3-4 actions)  
        5. Pour shot to shaker (~1 action)
        6. Shake (~1 action, need to hold shaker + free hand)
        7. Clean target shot if needed (~2-3 actions)
        8. Pour shaker to target shot (~1 action)
        
        Conservative minimum: ~11 actions per cocktail
        """
        contains_key = (shot, cocktail)
        contains_id = self.contains_ids.get(contains_key)
        
        # If already served, cost is 0
        if contains_id is not None and self._bit_check(state, contains_id):
            return 0
        
        cost = 0
        
        # Check if cocktail is already in a shaker
        shaker = self.shakers[0] if self.shakers else None
        if shaker is None:
            return 0
        
        cocktail_in_shaker = False
        shaker_contains_cocktail_id = self.contains_ids.get((shaker, cocktail))
        if shaker_contains_cocktail_id is not None:
            cocktail_in_shaker = self._bit_check(state, shaker_contains_cocktail_id)
        
        shaker_shaked = self._bit_check(state, self.shaked_ids.get(shaker))
        
        if cocktail_in_shaker and shaker_shaked:
            # Just need to clean target shot and pour
            # Clean target shot if needed
            target_clean_id = self.clean_ids.get(shot)
            target_empty_id = self.empty_ids.get(shot)
            
            if target_clean_id is not None and self._bit_check(state, target_clean_id):
                if target_empty_id is not None and self._bit_check(state, target_empty_id):
                    # Shot is clean and empty, just need to pour
                    # Need to hold shaker: possibly grasp (1-2 actions) + pour (1 action)
                    cost += 2  # grasp shaker + pour
                else:
                    # Clean but not empty? Shouldn't happen if clean. 
                    cost += 4
            else:
                # Need to clean the shot first
                cost += 5  # grasp shot + empty + clean + drop + pour
            
            return cost
        
        # Need to make the cocktail from scratch
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 1
        
        ing1, ing2 = parts
        
        # Base cost for full cocktail pipeline
        # Conservative estimate:
        # - Clean shaker: 0 if clean, 3-5 if contains something
        # - Fill ing1 to auxiliary shot: grasp shot(1) + fill(1) = 2, but need clean shot first
        # - Pour to shaker: 1 (pour-shot-to-clean-shaker)
        # - Fill ing2 to shot: need clean shot again: empty(1) + clean(1) + fill(1) = 3
        # - Pour to shaker: 1 (pour-shot-to-used-shaker)  
        # - Shake: 1
        # - Clean target shot if needed
        # - Pour to target shot: 1
        # Plus grasp/drop overhead
        
        shaker_clean_id = self.clean_ids.get(shaker)
        shaker_empty_id = self.empty_ids.get(shaker)
        shaker_clean = shaker_clean_id is not None and self._bit_check(state, shaker_clean_id)
        shaker_empty = shaker_empty_id is not None and self._bit_check(state, shaker_empty_id)
        
        # Check if ingredients are already in shaker
        ing1_in_shaker = False
        ing2_in_shaker = False
        if ing1:
            ing1_shaker_id = self.contains_ids.get((shaker, ing1))
            if ing1_shaker_id is not None:
                ing1_in_shaker = self._bit_check(state, ing1_shaker_id)
        if ing2:
            ing2_shaker_id = self.contains_ids.get((shaker, ing2))
            if ing2_shaker_id is not None:
                ing2_in_shaker = self._bit_check(state, ing2_shaker_id)
        
        if ing1_in_shaker and ing2_in_shaker:
            # Both ingredients in shaker, just need shake + pour
            cost += 1  # shake
            cost += 3  # clean target shot + pour (grasp shaker + pour)
            return cost
        
        if shaker_clean and shaker_empty:
            # Shaker ready
            cost += 0
        elif shaker_empty and not shaker_clean:
            # Need to clean shaker: grasp + clean = 3 (grasp + handEmpty + clean)
            cost += 3
        elif not shaker_empty:
            # Need to empty and clean: grasp + empty + clean = 5
            cost += 5
        
        # For each ingredient not in shaker:
        ingredients_needed = 0
        if not ing1_in_shaker:
            ingredients_needed += 1
        if not ing2_in_shaker:
            ingredients_needed += 1
        
        # Each ingredient cycle:
        # Find a clean shot or clean one: ~2-3 actions (grasp + empty + clean)
        # Fill shot with ingredient: grasp shot(1) + fill(1) = 2 (need handEmpty too)
        # Pour to shaker: 1
        # Total per ingredient: ~4-6 actions
        cost += ingredients_needed * 5
        
        # Shake: 1 action
        cost += 1
        
        # Clean target shot if not clean/empty
        target_clean_id = self.clean_ids.get(shot)
        target_empty_id = self.empty_ids.get(shot)
        target_clean = target_clean_id is not None and self._bit_check(state, target_clean_id)
        target_empty = target_empty_id is not None and self._bit_check(state, target_empty_id)
        
        if target_clean and target_empty:
            cost += 0  # Ready to receive
        elif target_empty and not target_clean:
            cost += 3  # grasp + clean + handEmpty
        else:
            cost += 5  # grasp + empty + clean
        
        # Pour from shaker to shot: 1 action (already holding shaker from shake)
        cost += 1
        
        return cost

    def _estimate_task_cost(self, state, task):
        """Estimate cost for a single task in the task network."""
        if hasattr(task, 'decompositions'):  # AbstractTask
            task_type, args = self._parse_task_name(task.name)
            
            if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                return self._estimate_cocktail_cost(state, args[0], args[1])
            
            elif task_type == "AchieveContainsShakerCocktail" and len(args) == 2:
                shaker, cocktail = args[0], args[1]
                contains_id = self.contains_ids.get((shaker, cocktail))
                if contains_id is not None and self._bit_check(state, contains_id):
                    return 0
                # Need to make cocktail: clean shaker + 2 ingredient fills + shake
                return 8
            
            elif task_type == "AchieveContainsShakerIngredient" and len(args) == 2:
                shaker, ing = args[0], args[1]
                contains_id = self.contains_ids.get((shaker, ing))
                if contains_id is not None and self._bit_check(state, contains_id):
                    return 0
                return 4  # fill shot + pour to shaker
            
            elif task_type == "AchieveContainsShotIngredient" and len(args) == 2:
                shot, ing = args[0], args[1]
                contains_id = self.contains_ids.get((shot, ing))
                if contains_id is not None and self._bit_check(state, contains_id):
                    return 0
                return 3  # clean shot + grasp + fill
            
            elif task_type == "AchieveCleanShaker" and len(args) == 1:
                shaker = args[0]
                clean_id = self.clean_ids.get(shaker)
                if clean_id is not None and self._bit_check(state, clean_id):
                    return 0
                empty_id = self.empty_ids.get(shaker)
                if empty_id is not None and self._bit_check(state, empty_id):
                    return 3  # grasp + handEmpty + clean
                return 5  # grasp + empty + drop + handEmpty + clean
            
            elif task_type == "AchieveCleanShot" and len(args) == 1:
                shot = args[0]
                clean_id = self.clean_ids.get(shot)
                if clean_id is not None and self._bit_check(state, clean_id):
                    return 0
                empty_id = self.empty_ids.get(shot)
                if empty_id is not None and self._bit_check(state, empty_id):
                    return 3  # grasp + handEmpty + clean
                return 5  # grasp + empty + handEmpty + clean
            
            elif task_type == "AchieveHolding" and len(args) == 2:
                hand, container = args[0], args[1]
                holding_id = self.holding_ids.get((hand, container))
                if holding_id is not None and self._bit_check(state, holding_id):
                    return 0
                # Need to potentially drop current item + pick up
                hand_empty_id = self.hand_empty_ids.get(hand)
                if hand_empty_id is not None and self._bit_check(state, hand_empty_id):
                    ontable_id = self.ontable_ids.get(container)
                    if ontable_id is not None and self._bit_check(state, ontable_id):
                        return 1  # just grasp
                    return 2  # put down by other + grasp
                return 2  # drop + grasp
            
            elif task_type == "AchieveHandEmpty" and len(args) == 1:
                hand = args[0]
                hand_empty_id = self.hand_empty_ids.get(hand)
                if hand_empty_id is not None and self._bit_check(state, hand_empty_id):
                    return 0
                return 1  # drop
            
            elif task_type == "AchieveOnTable" and len(args) == 1:
                container = args[0]
                ontable_id = self.ontable_ids.get(container)
                if ontable_id is not None and self._bit_check(state, ontable_id):
                    return 0
                return 1  # drop
            
            elif task_type == "DoPourShakerToShot" and len(args) == 3:
                return 1  # single pour action
            
            # Default for unknown abstract tasks
            return 2
        else:
            # Primitive operator - cost 1
            return 1

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        if len(task_network) == 0:
            return 0
        
        # Term 1: Sum of individual task costs (state-aware)
        task_sum = 0
        for task in task_network:
            task_sum += self._estimate_task_cost(state, task)
        
        # Term 2: Count unserved goal cocktails and compute serialized shaker cost
        # This captures the shaker bottleneck - cocktails must be made sequentially
        unserved_cocktails = 0
        shaker_pipeline_cost = 0
        
        # Check remaining goal cocktails from the task network
        for task in task_network:
            if hasattr(task, 'decompositions'):
                task_type, args = self._parse_task_name(task.name)
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    shot, cocktail = args[0], args[1]
                    contains_id = self.contains_ids.get((shot, cocktail))
                    if contains_id is None or not self._bit_check(state, contains_id):
                        unserved_cocktails += 1
        
        # Each unserved cocktail after the first requires cleaning the shaker (minimum 3 extra actions)
        if unserved_cocktails > 1:
            # First cocktail: ~11 actions, each subsequent: ~14 (11 + 3 for shaker cleaning)
            shaker_pipeline_cost = 11 + (unserved_cocktails - 1) * 14
        elif unserved_cocktails == 1:
            shaker_pipeline_cost = 11
        
        # Term 3: Minimum remaining primitive actions estimate
        # Count remaining primitive tasks directly
        primitive_count = 0
        abstract_count = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                abstract_count += 1
            else:
                primitive_count += 1
        
        # Each abstract task expands to at least 1 primitive (unless it's a null decomposition)
        min_primitives = primitive_count + abstract_count  # very conservative
        
        # Return max of our bounds
        h = max(task_sum, shaker_pipeline_cost, min_primitives)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
