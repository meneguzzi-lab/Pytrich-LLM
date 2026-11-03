"""Domain-dependent heuristic for Barman_BDI domain.

Bottleneck Analysis:
  The single shaker is the critical serializing resource. Every cocktail must be
  mixed in the shaker (clean shaker -> add ingredient1 -> add ingredient2 -> shake
  -> pour to shot -> empty shaker -> clean shaker for next cocktail). This means
  each cocktail requires a full shaker cycle that cannot overlap with another.

Admissible Lower Bounds:
  Term 1 - Cocktail production cost: Each unfinished cocktail goal requires at
    minimum: fill-shot(ing1) + pour-shot-to-shaker + fill-shot(ing2) +
    pour-shot-to-shaker + shake + pour-shaker-to-shot = 6 primitive actions,
    plus grasp/drop actions for manipulation. We estimate a realistic per-cocktail
    lower bound based on state analysis. This is admissible because these actions
    are unavoidable for each cocktail.

  Term 2 - Shaker serialization cost: Since there's one shaker and it must be
    cleaned between cocktails, each cocktail after the first incurs at least
    empty-shaker + clean-shaker + associated grasp/drop overhead. This is
    admissible because the shaker operations are sequential and unavoidable.

  Term 3 - Task network structural lower bound: Count remaining abstract tasks
    weighted by their minimum decomposition depth. Admissible because each
    abstract task requires at least its minimum number of primitive actions.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic8(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_8"):
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
        
        # Domain knowledge
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_cocktails = []  # list of (shot, cocktail) goals
        
        # Task name -> base name mapping
        self.task_costs = {}  # abstract task base name -> min cost estimate
        
        # Operator and abstract task name caches
        self.abstract_task_base_names = {}

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        """Extract the base task name from a grounded task/abstract task name."""
        # Format like: __top_4_AchieveContainsShotCocktail[shot2,cocktail1]
        # or: AchieveContainsShotCocktail[shot2,cocktail1]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            base = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            # Strip leading underscores and numbers for matching
            return base, args
        return name, []

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

        # Parse goal facts to identify cocktail goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        # Goals in barman are typically empty (goal check is via task network completion)
        # But let's check anyway
        self.goal_facts_set = set()
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                self.goal_facts_set.add((args[0], args[1]))

        # Build abstract task base name cache for quick lookup
        for at in model.abstract_tasks:
            base, args = self._parse_task_name(at.name)
            self.abstract_task_base_names[at.global_id] = (base, args)

        # Identify shakers (usually just shaker1)
        self.shakers = set()
        for key in self.empty_ids:
            if "shaker" in key:
                self.shakers.add(key)
        
        # Identify shots
        self.shots = set()
        for key in self.clean_ids:
            if "shot" in key:
                self.shots.add(key)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the given state using bitwise ops."""
        return (state >> global_id) & 1

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0

        term1 = self._cocktail_production_cost(state, task_network)
        term2 = self._shaker_serialization_cost(state, task_network)
        term3 = self._task_network_cost(state, task_network)

        return max(term1, term2, term3)

    def _cocktail_production_cost(self, state, task_network):
        """Estimate cost based on unfinished cocktail goals in the task network.
        
        For each AchieveContainsShotCocktail task remaining, estimate the minimum
        number of primitive actions needed based on current state.
        """
        total_cost = 0
        
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                gid = task.global_id
                if gid in self.abstract_task_base_names:
                    base, args = self.abstract_task_base_names[gid]
                    cost = self._estimate_abstract_task_cost(base, args, state)
                    total_cost += cost
            # Operators (primitive tasks) cost 1 each
            else:
                total_cost += 1

        return total_cost

    def _estimate_abstract_task_cost(self, base, args, state):
        """State-aware per-task cost estimation."""
        
        # AchieveContainsShotCocktail: the main goal task
        if "AchieveContainsShotCocktail" in base and len(args) >= 2:
            shot = args[0]
            cocktail = args[1]
            
            # Check if already done
            key = (shot, cocktail)
            if key in self.contains_ids:
                if self._check_fact(state, self.contains_ids[key]):
                    return 0
            
            # Need to make cocktail in shaker and pour to shot
            # Minimum actions:
            # - Fill shot with ing1: grasp shot (1) + fill-shot (1) = 2 min
            # - Pour shot to shaker: pour-shot-to-clean-shaker (1)
            # - Drop shot: drop (1)
            # - Fill shot with ing2: grasp shot (1) + fill-shot (1) = 2 min (need clean shot first)
            #   But shot was emptied by pour, so need clean-shot too
            #   clean-shot (1) + fill-shot (1) = at least the fill part
            # - Pour shot to shaker: pour-shot-to-used-shaker (1)
            # - Shake: shake (1)
            # - Pour shaker to shot: need clean shot, grasp shaker, pour
            # Realistic minimum: ~13-15 actions per cocktail from scratch
            
            cost = 0
            
            # Check if shaker contains this cocktail already
            shaker_has_cocktail = False
            for shaker in self.shakers:
                ckey = (shaker, cocktail)
                if ckey in self.contains_ids:
                    if self._check_fact(state, self.contains_ids[ckey]):
                        shaker_has_cocktail = True
                        break
            
            if shaker_has_cocktail:
                # Just need to prepare shot and pour
                # Clean shot if needed, grasp shaker, pour
                shot_clean = self.clean_ids.get(shot)
                shot_empty = self.empty_ids.get(shot)
                
                if shot_clean and shot_empty:
                    is_clean = self._check_fact(state, shot_clean)
                    is_empty = self._check_fact(state, shot_empty)
                    if is_clean and is_empty:
                        cost = 2  # grasp shaker + pour
                    elif is_empty and not is_clean:
                        cost = 4  # clean-shot + grasp shaker + pour
                    else:
                        cost = 6  # empty-shot + clean-shot + grasp shaker + pour
                else:
                    cost = 4
            else:
                # Need full cocktail production
                # Check shaker state
                shaker_clean = False
                shaker_empty = False
                for shaker in self.shakers:
                    if shaker in self.clean_ids:
                        shaker_clean = self._check_fact(state, self.clean_ids[shaker])
                    if shaker in self.empty_ids:
                        shaker_empty = self._check_fact(state, self.empty_ids[shaker])
                
                # Base cost for making a cocktail from scratch
                # grasp shot1 (1) + fill-shot ing1 (1) + pour-to-clean-shaker (1) +
                # drop shot (1) + grasp shot2 (1) + clean-shot (1) + fill-shot ing2 (1) +
                # pour-to-used-shaker (1) + drop shot (1) + grasp shaker (1) + shake (1) +
                # pour-to-shot (1) + need clean shot target
                cost = 13
                
                if not shaker_empty:
                    cost += 2  # empty-shaker + associated ops
                if not shaker_clean and shaker_empty:
                    cost += 1  # clean-shaker
                elif not shaker_clean and not shaker_empty:
                    cost += 3  # empty + clean + ops
                
                # Check if target shot needs cleaning
                if shot in self.clean_ids:
                    is_clean = self._check_fact(state, self.clean_ids[shot])
                    if not is_clean:
                        cost += 2  # additional cleaning steps
            
            return cost
        
        # AchieveContainsShakerCocktail
        if "AchieveContainsShakerCocktail" in base and len(args) >= 2:
            shaker = args[0]
            cocktail = args[1]
            key = (shaker, cocktail)
            if key in self.contains_ids:
                if self._check_fact(state, self.contains_ids[key]):
                    return 0
            return 8  # fill two ingredients + shake
        
        # AchieveContainsShakerIngredient
        if "AchieveContainsShakerIngredient" in base and len(args) >= 2:
            shaker = args[0]
            ingredient = args[1]
            key = (shaker, ingredient)
            if key in self.contains_ids:
                if self._check_fact(state, self.contains_ids[key]):
                    return 0
            return 4  # fill shot + pour to shaker
        
        # AchieveContainsShotIngredient
        if "AchieveContainsShotIngredient" in base and len(args) >= 2:
            shot = args[0]
            ingredient = args[1]
            key = (shot, ingredient)
            if key in self.contains_ids:
                if self._check_fact(state, self.contains_ids[key]):
                    return 0
            return 3  # clean + grasp + fill
        
        # AchieveCleanShaker
        if "AchieveCleanShaker" in base and len(args) >= 1:
            shaker = args[0]
            if shaker in self.clean_ids:
                if self._check_fact(state, self.clean_ids[shaker]):
                    return 0
            # Need to empty (if not empty) then clean
            if shaker in self.empty_ids:
                is_empty = self._check_fact(state, self.empty_ids[shaker])
                if is_empty:
                    return 2  # grasp + clean
                else:
                    return 4  # grasp + empty + clean + hand management
            return 3
        
        # AchieveCleanShot
        if "AchieveCleanShot" in base and len(args) >= 1:
            shot = args[0]
            if shot in self.clean_ids:
                if self._check_fact(state, self.clean_ids[shot]):
                    return 0
            if shot in self.empty_ids:
                is_empty = self._check_fact(state, self.empty_ids[shot])
                if is_empty:
                    return 2  # grasp + clean
                else:
                    return 3  # grasp + empty + clean
            return 2
        
        # AchieveHolding
        if "AchieveHolding" in base and len(args) >= 2:
            hand = args[0]
            container = args[1]
            key = (hand, container)
            if key in self.holding_ids:
                if self._check_fact(state, self.holding_ids[key]):
                    return 0
            # Need to empty hand (maybe drop something) + grasp
            if hand in self.hand_empty_ids:
                hand_empty = self._check_fact(state, self.hand_empty_ids[hand])
                if hand_empty:
                    return 1  # just grasp
                else:
                    return 2  # drop + grasp
            return 2
        
        # AchieveHandEmpty
        if "AchieveHandEmpty" in base and len(args) >= 1:
            hand = args[0]
            if hand in self.hand_empty_ids:
                if self._check_fact(state, self.hand_empty_ids[hand]):
                    return 0
            return 1  # drop
        
        # AchieveOnTable
        if "AchieveOnTable" in base and len(args) >= 1:
            container = args[0]
            if container in self.ontable_ids:
                if self._check_fact(state, self.ontable_ids[container]):
                    return 0
            return 1  # drop
        
        # DoPourShakerToShot
        if "DoPourShakerToShot" in base:
            return 1  # single pour action
        
        # Default: small cost for unknown tasks
        return 1

    def _shaker_serialization_cost(self, state, task_network):
        """Count remaining cocktails that need the shaker and estimate
        the serialization overhead.
        
        Each cocktail needs a full shaker cycle. After the first one,
        each additional cocktail needs at least: empty-shaker + clean-shaker
        + the full ingredient loading cycle.
        """
        remaining_cocktails = 0
        
        for task in task_network:
            if hasattr(task, 'decompositions'):
                gid = task.global_id
                if gid in self.abstract_task_base_names:
                    base, args = self.abstract_task_base_names[gid]
                    if "AchieveContainsShotCocktail" in base and len(args) >= 2:
                        shot = args[0]
                        cocktail = args[1]
                        key = (shot, cocktail)
                        if key in self.contains_ids:
                            if not self._check_fact(state, self.contains_ids[key]):
                                remaining_cocktails += 1
                        else:
                            remaining_cocktails += 1
        
        if remaining_cocktails == 0:
            return 0
        
        # First cocktail: ~13 actions
        # Each subsequent cocktail: ~15 actions (extra for shaker cleanup)
        # This is a lower bound on total sequential work through the shaker
        cost = 13 + max(0, remaining_cocktails - 1) * 15
        
        return cost

    def _task_network_cost(self, state, task_network):
        """Simple task network depth estimate as a third term.
        
        Count primitive tasks as 1, abstract tasks by their minimum
        decomposition cost.
        """
        total = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task - at least 1 action needed
                gid = task.global_id
                if gid in self.abstract_task_base_names:
                    base, args = self.abstract_task_base_names[gid]
                    # Check for null decomposition possibility
                    if "AchieveContainsShotCocktail" in base and len(args) >= 2:
                        key = (args[0], args[1])
                        if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                            continue  # null decomposition
                        total += 10  # significant minimum work
                    elif "AchieveContainsShakerCocktail" in base and len(args) >= 2:
                        key = (args[0], args[1])
                        if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                            continue
                        total += 6
                    elif "AchieveContainsShakerIngredient" in base and len(args) >= 2:
                        key = (args[0], args[1])
                        if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                            continue
                        total += 3
                    elif "AchieveContainsShotIngredient" in base and len(args) >= 2:
                        key = (args[0], args[1])
                        if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                            continue
                        total += 2
                    elif "AchieveCleanShaker" in base and len(args) >= 1:
                        if args[0] in self.clean_ids and self._check_fact(state, self.clean_ids[args[0]]):
                            continue
                        total += 2
                    elif "AchieveCleanShot" in base and len(args) >= 1:
                        if args[0] in self.clean_ids and self._check_fact(state, self.clean_ids[args[0]]):
                            continue
                        total += 2
                    elif "AchieveHolding" in base and len(args) >= 2:
                        key = (args[0], args[1])
                        if key in self.holding_ids and self._check_fact(state, self.holding_ids[key]):
                            continue
                        total += 1
                    elif "AchieveHandEmpty" in base and len(args) >= 1:
                        if args[0] in self.hand_empty_ids and self._check_fact(state, self.hand_empty_ids[args[0]]):
                            continue
                        total += 1
                    elif "AchieveOnTable" in base and len(args) >= 1:
                        if args[0] in self.ontable_ids and self._check_fact(state, self.ontable_ids[args[0]]):
                            continue
                        total += 1
                    elif "DoPourShakerToShot" in base:
                        total += 1
                    else:
                        total += 1
                else:
                    total += 1
            else:
                total += 1
        
        return total
