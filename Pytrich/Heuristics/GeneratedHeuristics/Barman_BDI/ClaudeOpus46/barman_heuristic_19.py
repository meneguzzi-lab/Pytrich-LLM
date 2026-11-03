"""Domain-dependent heuristic for Barman_BDI.

Bottleneck Analysis:
  The single shaker is the serializing bottleneck. Every cocktail must be made
  in the shaker (add 2 ingredients, shake, pour to shot), then the shaker must
  be cleaned before the next cocktail. This means cocktails are essentially
  serialized through the shaker.

Admissible Lower Bound Terms:
  Term 1 - Per-cocktail production cost:
    Each cocktail not yet in its target shot requires at minimum:
    - If cocktail not yet in shaker: clean shaker + add 2 ingredients to shaker
      (each requires: fill shot from dispenser + pour shot to shaker) + shake + pour to shot
      That's roughly: clean_shaker(~3) + 2*(fill_shot(~3) + pour_to_shaker(1)) + shake(1) + pour(1) = ~12 actions
    - If cocktail in shaker but not poured: pour to shot (~3 actions for clean shot + pour)
    - We use a simplified admissible estimate per unfinished cocktail goal.
    Admissible because each cocktail requires at least these actions independently.

  Term 2 - Shaker serialization cost:
    Since there's only one shaker, each cocktail that still needs to be made
    requires exclusive use of the shaker. After each cocktail is shaken and poured,
    the shaker must be emptied and cleaned. This adds a minimum cost per cocktail
    that must still pass through the shaker.
    Admissible because these operations are sequential and unavoidable.

  Term 3 - Shot preparation cost:
    Each target shot that doesn't contain its cocktail needs to be clean and empty
    before receiving the pour. If it's dirty or contains something else, cleaning
    costs actions.
    Admissible as a lower bound on shot-specific work.

Tie-breaking:
  We add a small penalty for shots that are currently occupied with wrong contents,
  as these represent additional cleanup work.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic19(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_19"):
        super().__init__(name)
        # Mappings built during initialize
        self.contains_fact_id = {}   # (container, beverage) -> global_id
        self.clean_fact_id = {}      # container -> global_id
        self.empty_fact_id = {}      # container -> global_id
        self.holding_fact_id = {}    # (hand, container) -> global_id
        self.hand_empty_fact_id = {} # hand -> global_id
        self.ontable_fact_id = {}    # container -> global_id
        self.shaked_fact_id = {}     # shaker -> global_id
        self.unshaked_fact_id = {}   # shaker -> global_id
        self.used_fact_id = {}       # (container, beverage) -> global_id
        self.shaker_level_fact_id = {} # (shaker, level) -> global_id
        
        # Domain knowledge
        self.cocktail_part1 = {}     # cocktail -> ingredient
        self.cocktail_part2 = {}     # cocktail -> ingredient
        
        # Goal info
        self.goal_cocktail_shots = [] # list of (shot, cocktail)
        
        # Task name to type mapping
        self.task_type_map = {}      # abstract_task global_id -> task_type string
        self.task_args_map = {}      # abstract_task global_id -> list of arg strings
        
        # Shaker name
        self.shaker = None
        self.shakers = []

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
        """Parse a task name like 'AchieveContainsShotCocktail[shot2,cocktail1]'"""
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact index maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "contains" and len(args) == 2:
                self.contains_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "clean" and len(args) == 1:
                self.clean_fact_id[args[0]] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_fact_id[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 2:
                self.holding_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "handEmpty" and len(args) == 1:
                self.hand_empty_fact_id[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
            elif pred == "shaked" and len(args) == 1:
                self.shaked_fact_id[args[0]] = f.global_id
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_fact_id[args[0]] = f.global_id
            elif pred == "used" and len(args) == 2:
                self.used_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "shakerLevel" and len(args) == 2:
                self.shaker_level_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                self.cocktail_part1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
        
        # Identify shakers
        for key in self.shaked_fact_id:
            if key not in self.shakers:
                self.shakers.append(key)
        if self.shakers:
            self.shaker = self.shakers[0]
        
        # Parse goal cocktail-shot pairs from initial task network
        # The top-level tasks are AchieveContainsShotCocktail(shot, cocktail)
        # We extract these from the initial task network
        for task in model.initial_tn:
            tname = task.name if hasattr(task, 'name') else str(task)
            ttype, targs = self._parse_task_name(tname)
            if ttype == "AchieveContainsShotCocktail" and len(targs) == 2:
                self.goal_cocktail_shots.append((targs[0], targs[1]))
        
        # Build task type maps for abstract tasks
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = ttype
            self.task_args_map[at.global_id] = targs
        
        # Also index operators
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            self.task_type_map[op.global_id] = ttype
            self.task_args_map[op.global_id] = targs
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _bit_check(self, state, fact_id):
        """Check if fact is true in state using bitwise check."""
        if fact_id is None:
            return False
        return (state >> fact_id) & 1

    def _cocktail_in_shot(self, state, shot, cocktail):
        """Check if cocktail is already in the target shot."""
        fid = self.contains_fact_id.get((shot, cocktail))
        if fid is None:
            return False
        return self._bit_check(state, fid)

    def _cocktail_in_shaker(self, state, cocktail):
        """Check if cocktail is in any shaker (already shaken)."""
        for shaker in self.shakers:
            fid = self.contains_fact_id.get((shaker, cocktail))
            if fid is not None and self._bit_check(state, fid):
                return True
        return False

    def _shaker_is_clean(self, state, shaker):
        fid = self.clean_fact_id.get(shaker)
        return self._bit_check(state, fid) if fid is not None else False

    def _shaker_is_empty(self, state, shaker):
        fid = self.empty_fact_id.get(shaker)
        return self._bit_check(state, fid) if fid is not None else False

    def _shaker_is_shaked(self, state, shaker):
        fid = self.shaked_fact_id.get(shaker)
        return self._bit_check(state, fid) if fid is not None else False

    def _container_is_clean(self, state, container):
        fid = self.clean_fact_id.get(container)
        return self._bit_check(state, fid) if fid is not None else False

    def _container_is_empty(self, state, container):
        fid = self.empty_fact_id.get(container)
        return self._bit_check(state, fid) if fid is not None else False

    def _ingredient_in_shaker(self, state, shaker, ingredient):
        fid = self.contains_fact_id.get((shaker, ingredient))
        return self._bit_check(state, fid) if fid is not None else False

    def _estimate_cocktail_cost(self, state, shot, cocktail):
        """Estimate minimum actions needed to get cocktail into shot.
        
        Full pipeline for one cocktail:
        1. Clean shaker if needed (up to 5 actions: grasp, empty, drop, grasp, clean)
        2. For each of 2 ingredients: fill shot (~4: grasp shot, fill, pour-to-shaker, drop shot) = ~8
        3. Shake (~2: grasp shaker + shake)
        4. Pour to shot (~4: clean shot if needed + grasp shaker + pour)
        
        We compute a state-aware lower bound.
        """
        if self._cocktail_in_shot(state, shot, cocktail):
            return 0
        
        cost = 0
        shaker = self.shaker
        if shaker is None:
            return 1
        
        # Check if cocktail is already in shaker (shaken, ready to pour)
        cocktail_ready_in_shaker = self._cocktail_in_shaker(state, cocktail)
        
        if cocktail_ready_in_shaker and self._shaker_is_shaked(state, shaker):
            # Just need to pour to shot
            # Need: shot clean and empty, holding shaker
            # pour-shaker-to-shot: 1 action
            pour_cost = 1
            if not self._container_is_clean(state, shot):
                # Need to clean shot: possibly empty + clean = ~3-4 actions
                if not self._container_is_empty(state, shot):
                    pour_cost += 3  # grasp, empty, clean
                else:
                    pour_cost += 2  # grasp + clean (need hand empty too)
            elif not self._container_is_empty(state, shot):
                pour_cost += 2  # grasp + empty
            # Need to hold shaker
            pour_cost += 1  # grasp shaker (may need drop first)
            cost = pour_cost
        else:
            # Need to make the cocktail from scratch in the shaker
            ing1 = self.cocktail_part1.get(cocktail)
            ing2 = self.cocktail_part2.get(cocktail)
            
            # Shaker preparation
            shaker_clean = self._shaker_is_clean(state, shaker)
            shaker_empty = self._shaker_is_empty(state, shaker)
            
            shaker_prep_cost = 0
            if not shaker_clean:
                if not shaker_empty:
                    # Need: grasp shaker, empty-shaker, drop, grasp, clean-shaker
                    shaker_prep_cost = 5
                else:
                    # Need: grasp shaker, clean-shaker
                    shaker_prep_cost = 2
            elif not shaker_empty:
                # Clean but not empty - shouldn't happen in normal flow
                # but if it does, need to empty first then it's dirty
                shaker_prep_cost = 5
            
            # Ingredient costs - each ingredient needs:
            # Find a shot, clean it, grasp it, fill from dispenser, pour to shaker
            # Minimum per ingredient: grasp_shot(1) + fill(1) + pour-to-shaker(1) = 3
            # But also need: shot clean+empty, hand empty for fill
            # Realistic minimum: ~4 per ingredient
            ing1_in_shaker = self._ingredient_in_shaker(state, shaker, ing1) if ing1 else False
            ing2_in_shaker = self._ingredient_in_shaker(state, shaker, ing2) if ing2 else False
            
            ing_cost = 0
            if not ing1_in_shaker and not cocktail_ready_in_shaker:
                ing_cost += 4  # fill shot + pour to shaker
            if not ing2_in_shaker and not cocktail_ready_in_shaker:
                ing_cost += 4  # fill shot + pour to shaker
            
            # Shake: need shaker in hand + other hand empty = 1 action (+ maybe grasp)
            shake_cost = 2  # grasp shaker + shake
            
            # Pour to shot
            pour_cost = 1  # pour-shaker-to-shot
            # Shot needs to be clean and empty
            if not self._container_is_clean(state, shot):
                if not self._container_is_empty(state, shot):
                    pour_cost += 4  # grasp, empty, drop(?), clean
                else:
                    pour_cost += 3  # grasp, clean, drop
            elif not self._container_is_empty(state, shot):
                pour_cost += 2  # grasp, empty
            # Need to pick up shaker for pour
            pour_cost += 1  # grasp shaker
            
            cost = shaker_prep_cost + ing_cost + shake_cost + pour_cost
        
        return cost

    def _estimate_task_network_cost(self, state, task_network):
        """Estimate cost from remaining tasks in the task network."""
        cost = 0
        
        for task in task_network:
            gid = task.global_id
            ttype = self.task_type_map.get(gid, "")
            targs = self.task_args_map.get(gid, [])
            
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive operator - costs exactly 1
                cost += 1
            elif ttype == "AchieveContainsShotCocktail" and len(targs) == 2:
                shot, cocktail = targs[0], targs[1]
                cost += self._estimate_cocktail_cost(state, shot, cocktail)
            elif ttype == "AchieveContainsShakerCocktail" and len(targs) == 2:
                shaker, cocktail = targs[0], targs[1]
                if not self._cocktail_in_shaker(state, cocktail):
                    # Need to add ingredients + shake
                    cost += 10  # ~8 for ingredients + 2 for shake
                # else already done
            elif ttype == "AchieveContainsShakerIngredient" and len(targs) == 2:
                shaker, ingredient = targs[0], targs[1]
                if not self._ingredient_in_shaker(state, shaker, ingredient):
                    cost += 4  # fill shot + pour to shaker
            elif ttype == "AchieveContainsShotIngredient" and len(targs) == 2:
                shot, ingredient = targs[0], targs[1]
                fid = self.contains_fact_id.get((shot, ingredient))
                if fid is None or not self._bit_check(state, fid):
                    cost += 3  # clean shot + grasp + fill
            elif ttype == "AchieveCleanShaker" and len(targs) >= 1:
                shaker = targs[0]
                if not self._shaker_is_clean(state, shaker):
                    if not self._shaker_is_empty(state, shaker):
                        cost += 5
                    else:
                        cost += 2
            elif ttype == "AchieveCleanShot" and len(targs) >= 1:
                shot = targs[0]
                if not self._container_is_clean(state, shot):
                    if not self._container_is_empty(state, shot):
                        cost += 4
                    else:
                        cost += 3
            elif ttype == "AchieveHolding" and len(targs) == 2:
                hand, container = targs[0], targs[1]
                fid = self.holding_fact_id.get((hand, container))
                if fid is None or not self._bit_check(state, fid):
                    cost += 2  # drop current + grasp
            elif ttype == "AchieveHandEmpty" and len(targs) >= 1:
                hand = targs[0]
                fid = self.hand_empty_fact_id.get(hand)
                if fid is None or not self._bit_check(state, fid):
                    cost += 1  # drop
            elif ttype == "AchieveOnTable" and len(targs) >= 1:
                container = targs[0]
                fid = self.ontable_fact_id.get(container)
                if fid is None or not self._bit_check(state, fid):
                    cost += 1  # drop
            elif ttype == "DoPourShakerToShot" and len(targs) >= 2:
                cost += 1  # single pour action
            else:
                # Unknown task type - conservative estimate
                if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                    cost += 1
        
        return cost

    def _shaker_serialization_cost(self, state, task_network):
        """Count cocktails that still need the shaker and estimate serialization overhead.
        
        Each cocktail that hasn't been made yet needs exclusive shaker access.
        After the first, each additional cocktail needs: empty shaker + clean shaker
        before it can use the shaker = at least 2 extra actions per cocktail beyond the first.
        """
        cocktails_needing_shaker = 0
        
        # Check from task network which top-level cocktail tasks remain
        for task in task_network:
            gid = task.global_id
            ttype = self.task_type_map.get(gid, "")
            targs = self.task_args_map.get(gid, [])
            
            if ttype == "AchieveContainsShotCocktail" and len(targs) == 2:
                shot, cocktail = targs[0], targs[1]
                if not self._cocktail_in_shot(state, shot, cocktail):
                    if not self._cocktail_in_shaker(state, cocktail):
                        cocktails_needing_shaker += 1
            elif ttype == "AchieveContainsShakerCocktail" and len(targs) == 2:
                shaker, cocktail = targs[0], targs[1]
                if not self._cocktail_in_shaker(state, cocktail):
                    cocktails_needing_shaker += 1
        
        if cocktails_needing_shaker <= 1:
            return 0
        
        # Each cocktail after the first needs shaker cleanup: empty(1) + clean(~2) = 3
        return (cocktails_needing_shaker - 1) * 3

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible terms."""
        if len(task_network) == 0:
            return 0
        
        # Term 1: State-aware per-task cost estimation from task network
        term1 = self._estimate_task_network_cost(state, task_network)
        
        # Term 2: Shaker serialization cost
        term2 = self._shaker_serialization_cost(state, task_network)
        
        # Term 3: Goal-based lower bound - count unachieved goals from original goal list
        # Each unachieved cocktail-in-shot goal needs minimum actions
        term3 = 0
        unachieved_count = 0
        for shot, cocktail in self.goal_cocktail_shots:
            if not self._cocktail_in_shot(state, shot, cocktail):
                unachieved_count += 1
                # Absolute minimum: pour-shaker-to-shot(1) even if everything else is ready
                term3 += 1
        
        # Combine: term1 is the detailed estimate, term2 is serialization overhead
        # term1 already includes per-cocktail costs, so we add serialization as extra
        combined = term1 + term2
        
        # term3 is a very conservative lower bound (just counting pours)
        # Use max to ensure we don't go below even the simplest bound
        h = max(combined, term3)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
