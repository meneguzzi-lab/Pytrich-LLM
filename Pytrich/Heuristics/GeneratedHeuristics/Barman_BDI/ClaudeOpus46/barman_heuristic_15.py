# Barman BDI Domain Heuristic
# 
# BOTTLENECK: Single shaker serializes all cocktail preparation. Each cocktail
# needs a full shaker cycle (clean, add 2 ingredients via shots, shake, pour to
# target shot). This cycle is ~13+ primitive actions per cocktail.
#
# ADMISSIBLE LOWER BOUNDS:
# T1 - Per-cocktail minimum cost: Each cocktail not yet in its target shot needs
#      at minimum ~13 actions (2x fill-shot, 2x pour-to-shaker, shake, pour-to-shot,
#      plus grasp/drop/clean operations). State-aware: reduces cost if intermediate
#      steps are already done.
# T2 - Task network lower bound: Count abstract tasks with state-aware minimum
#      decomposition costs. Tasks already satisfied (Null methods) cost 0.
# T3 - Shaker contention: If shaker contains wrong cocktail or is dirty, add
#      cleaning cost. Each pending cocktail adds serialized shaker cycle cost.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic15(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_15"):
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
        self.shaker_level_ids = {}  # (shaker, level) -> global_id
        
        # Domain knowledge
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.dispenser_map = {}   # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktail_shots = []  # [(shot, cocktail)] from top-level tasks
        
        # Abstract task type maps
        self.task_type_map = {}  # global_id -> (task_type_str, args)
        
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
        """Parse a task name like '__AchieveContainsShotCocktail[shot2,cocktail1]'"""
        name = name.strip()
        # Remove leading underscores
        clean = name.lstrip('_')
        bracket_idx = clean.find("[")
        if bracket_idx > 0 and clean.endswith("]"):
            task_type = clean[:bracket_idx]
            args_str = clean[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # No brackets
        return clean, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact ID maps
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
            elif pred == "shakerLevel" and len(args) == 2:
                self.shaker_level_ids[(args[0], args[1])] = f.global_id
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
                self.dispenser_map[args[1]] = args[0]
        
        # Collect object types from fact keys
        seen_shakers = set()
        seen_shots = set()
        seen_hands = set()
        seen_cocktails = set()
        seen_ingredients = set()
        
        for (container, bev) in self.contains_ids:
            if container.startswith("shaker"):
                seen_shakers.add(container)
            elif container.startswith("shot"):
                seen_shots.add(container)
        
        for c in self.clean_ids:
            if c.startswith("shaker"):
                seen_shakers.add(c)
            elif c.startswith("shot"):
                seen_shots.add(c)
        
        for h in self.hand_empty_ids:
            seen_hands.add(h)
        
        for c in self.cocktail_parts:
            seen_cocktails.add(c)
            parts = self.cocktail_parts[c]
            if parts[0]:
                seen_ingredients.add(parts[0])
            if parts[1]:
                seen_ingredients.add(parts[1])
        
        self.shakers = sorted(seen_shakers)
        self.shots = sorted(seen_shots)
        self.hands = sorted(seen_hands)
        self.cocktails = sorted(seen_cocktails)
        self.ingredients = sorted(seen_ingredients)
        
        # Parse abstract task types for quick lookup
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)
        
        # Also parse operator names
        self.op_type_map = {}
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            self.op_type_map[op.global_id] = (task_type, args)
        
        # Extract goal cocktail-shot pairs from initial task network
        self._extract_goal_pairs(initial_node.task_network)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goal_pairs(self, task_network):
        """Extract (shot, cocktail) pairs from the initial task network."""
        self.goal_cocktail_shots = []
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                task_type, args = self.task_type_map.get(task.global_id, (None, []))
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    self.goal_cocktail_shots.append((args[0], args[1]))

    def _check_fact(self, state, global_id):
        """Check if a fact is true in state using bitwise operation."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """Estimate minimum actions needed to get cocktail into shot.
        
        Full pipeline per cocktail (from scratch):
        1. Clean shaker (if needed): grasp + clean + drop = ~3 actions, or empty + clean = ~5
        2. Fill shot with ingredient1: clean_shot(~3) + grasp_shot + fill + = ~5
        3. Pour shot to shaker: (already holding shot) pour = 1
        4. Drop shot: 1
        5. Fill shot with ingredient2: clean_shot(~3) + grasp + fill = ~5  
        6. Pour shot to shaker: grasp + pour = ~2
        7. Shake: (grasp shaker if needed) shake = ~2
        8. Clean target shot: ~3
        9. Pour shaker to shot: ~1 (already holding shaker)
        Total: ~23 actions minimum per cocktail from scratch.
        
        We compute state-aware reductions.
        """
        # Check if already done
        contains_key = (shot, cocktail)
        cid = self.contains_ids.get(contains_key)
        if cid is not None and self._check_fact(state, cid):
            return 0
        
        cost = 0
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 1  # unknown cocktail
        
        ing1, ing2 = parts[0], parts[1]
        
        # Check if shaker already contains this cocktail
        shaker = self.shakers[0] if self.shakers else None
        if shaker is None:
            return 1
        
        shaker_has_cocktail = False
        shaker_is_shaked = False
        
        sc_id = self.contains_ids.get((shaker, cocktail))
        if sc_id is not None and self._check_fact(state, sc_id):
            shaker_has_cocktail = True
        
        sk_id = self.shaked_ids.get(shaker)
        if sk_id is not None and self._check_fact(state, sk_id):
            shaker_is_shaked = True
        
        if shaker_has_cocktail and shaker_is_shaked:
            # Just need to clean shot and pour
            # Clean shot: check if already clean and empty
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            
            if shot_clean and shot_empty:
                # grasp shaker (if not holding) + pour = 2-3
                cost = 3
            else:
                # Need to clean shot first: ~4-6 actions + pour
                cost = 7
            return cost
        
        # Need full cocktail preparation
        # Check shaker state
        shaker_clean = self._check_fact(state, self.clean_ids.get(shaker))
        shaker_empty = self._check_fact(state, self.empty_ids.get(shaker))
        
        # Shaker cleaning cost
        if shaker_clean and shaker_empty:
            cost += 0  # shaker ready
        elif shaker_empty and not shaker_clean:
            cost += 3  # grasp + clean-shaker + drop
        else:
            # Shaker has contents, need to empty then clean
            cost += 5  # grasp + empty + clean + ...
        
        # Check if shaker already has the ingredients
        has_ing1 = False
        has_ing2 = False
        si1_id = self.contains_ids.get((shaker, ing1))
        si2_id = self.contains_ids.get((shaker, ing2))
        if si1_id is not None and self._check_fact(state, si1_id):
            has_ing1 = True
        if si2_id is not None and self._check_fact(state, si2_id):
            has_ing2 = True
        
        # Cost to add each ingredient to shaker via a shot
        # Each ingredient: find/clean shot, grasp shot, fill from dispenser, pour to shaker
        # = clean_shot(0-4) + grasp(1) + fill(1) + pour(1) = ~3-7 actions
        if not has_ing1:
            cost += 6  # fill shot with ing1 and pour to shaker
        if not has_ing2:
            cost += 5  # fill shot with ing2 and pour to shaker (shaker already used)
        
        # Shake: grasp shaker + shake = 2
        cost += 2
        
        # Clean target shot and pour
        shot_clean = self._check_fact(state, self.clean_ids.get(shot))
        shot_empty = self._check_fact(state, self.empty_ids.get(shot))
        
        if shot_clean and shot_empty:
            cost += 1  # just pour
        elif shot_empty:
            cost += 4  # clean + pour
        else:
            cost += 6  # empty + clean + pour
        
        return cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from task network with state-aware per-task costs."""
        cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                task_type, args = self.task_type_map.get(task.global_id, (None, []))
                task_cost = self._estimate_abstract_task_cost(state, task_type, args)
                cost += task_cost
            else:
                # Primitive operator - costs 1
                cost += 1
        return cost

    def _estimate_abstract_task_cost(self, state, task_type, args):
        """State-aware cost estimation for each abstract task type."""
        if task_type is None:
            return 1
        
        if task_type == "AchieveContainsShotCocktail":
            if len(args) >= 2:
                shot, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                return self._cocktail_cost(state, shot, cocktail)
            return 13
        
        elif task_type == "AchieveContainsShakerCocktail":
            if len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shaker, cocktail))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                return 10  # full mixing cycle
            return 10
        
        elif task_type == "AchieveContainsShakerIngredient":
            if len(args) >= 2:
                shaker, ingredient = args[0], args[1]
                cid = self.contains_ids.get((shaker, ingredient))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                return 5  # fill shot + pour to shaker
            return 5
        
        elif task_type == "AchieveContainsShotIngredient":
            if len(args) >= 2:
                shot, ingredient = args[0], args[1]
                cid = self.contains_ids.get((shot, ingredient))
                if cid is not None and self._check_fact(state, cid):
                    return 0
                return 4  # clean shot + grasp + fill
            return 4
        
        elif task_type == "AchieveCleanShaker":
            if len(args) >= 1:
                shaker = args[0]
                cid = self.clean_ids.get(shaker)
                if cid is not None and self._check_fact(state, cid):
                    return 0
                eid = self.empty_ids.get(shaker)
                if eid is not None and self._check_fact(state, eid):
                    return 3  # grasp + clean + (hand empty)
                return 5  # empty + clean
            return 3
        
        elif task_type == "AchieveCleanShot":
            if len(args) >= 1:
                shot = args[0]
                cid = self.clean_ids.get(shot)
                if cid is not None and self._check_fact(state, cid):
                    return 0
                eid = self.empty_ids.get(shot)
                if eid is not None and self._check_fact(state, eid):
                    return 3  # grasp + clean
                return 4  # empty + clean
            return 3
        
        elif task_type == "AchieveHolding":
            if len(args) >= 2:
                hand, container = args[0], args[1]
                hid = self.holding_ids.get((hand, container))
                if hid is not None and self._check_fact(state, hid):
                    return 0
                return 2  # drop current + grasp
            return 2
        
        elif task_type == "AchieveHandEmpty":
            if len(args) >= 1:
                hand = args[0]
                hid = self.hand_empty_ids.get(hand)
                if hid is not None and self._check_fact(state, hid):
                    return 0
                return 1  # drop
            return 1
        
        elif task_type == "AchieveOnTable":
            if len(args) >= 1:
                container = args[0]
                oid = self.ontable_ids.get(container)
                if oid is not None and self._check_fact(state, oid):
                    return 0
                return 1  # drop
            return 1
        
        elif task_type == "DoPourShakerToShot":
            return 1  # single action
        
        return 1  # unknown task type fallback

    def _unserved_cocktails_cost(self, state):
        """Count unserved cocktails and estimate total cost.
        Each unserved cocktail requires a full shaker cycle (~13 min actions).
        Since there's one shaker, these are serialized."""
        total = 0
        for shot, cocktail in self.goal_cocktail_shots:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is None or not self._check_fact(state, cid):
                total += self._cocktail_cost(state, shot, cocktail)
        return total

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple lower bounds."""
        if len(task_network) == 0:
            return 0
        
        # Term 1: Task network decomposition lower bound
        t1 = self._task_network_cost(state, task_network)
        
        # Term 2: Unserved cocktails lower bound (goal-based)
        # Only compute if we have goal info
        t2 = 0
        if self.goal_cocktail_shots:
            t2 = self._unserved_cocktails_cost(state)
        
        # Term 3: Simple task count as minimum
        t3 = len(task_network)
        
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
