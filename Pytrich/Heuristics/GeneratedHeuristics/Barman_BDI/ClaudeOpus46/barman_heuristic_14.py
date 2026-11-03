# Barman BDI Domain Heuristic
# 
# BOTTLENECK: The single shaker serializes all cocktail production. Each cocktail
# requires exclusive use of the shaker (clean, add 2 ingredients via shots, shake, pour).
# This means each unfinished cocktail costs ~15 primitive actions minimum.
#
# ADMISSIBLE LOWER BOUNDS:
# 1. Cocktail-level bound: Each cocktail not yet in its target shot requires at minimum
#    ~15 actions (clean shaker + 2x fill-shot-pour-to-shaker + shake + pour-to-shot + overhead).
#    This is admissible because actions cannot be parallelized due to single shaker.
# 2. Task network bound: Count remaining abstract tasks weighted by minimum decomposition cost
#    based on current state (state-aware). Admissible because each task needs >= computed actions.
# 3. Shot readiness bound: Each target shot that isn't clean+empty needs cleaning actions.
#    Admissible because these actions are unavoidable prerequisites.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic14(Heuristic):
    """Domain-dependent heuristic for Barman BDI domain.
    
    Exploits the single-shaker bottleneck to compute tight lower bounds
    on remaining plan length.
    """

    def __init__(self, name="barman_anthropicopus46_heuristic_14"):
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
        self.shaker_level_ids = {} # (shaker, level) -> global_id
        self.shaker_empty_level_ids = {} # (shaker, level) -> global_id
        
        # Domain structure
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.dispenser_map = {}   # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktails = []  # [(shot, cocktail), ...]
        
        # Task name maps for fast identification
        self.task_name_cache = {}
        
        # Abstract task type IDs
        self.achieve_contains_shot_cocktail_ids = set()
        self.achieve_contains_shaker_cocktail_ids = set()
        self.achieve_contains_shaker_ingredient_ids = set()
        self.achieve_contains_shot_ingredient_ids = set()
        self.achieve_clean_shaker_ids = set()
        self.achieve_clean_shot_ids = set()
        self.achieve_holding_ids = set()
        self.achieve_hand_empty_ids = set()
        self.achieve_on_table_ids = set()
        self.do_pour_shaker_to_shot_ids = set()
        
        # Operator IDs by type
        self.op_type_map = {}

    def _parse_fact_name(self, name: str):
        """Parse a grounded fact name like +predicate[arg1,arg2]."""
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
        """Parse a grounded task/operator name."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        elif bracket_idx > 0:
            task_type = name[:bracket_idx]
            return task_type, []
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Preprocess all fact mappings and goal structures."""
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
            elif pred == "dispenses" and len(args) == 2:
                self.dispenser_map[args[1]] = args[0]
            elif pred == "shakerLevel" and len(args) == 2:
                self.shaker_level_ids[(args[0], args[1])] = f.global_id
            elif pred == "shakerEmptyLevel" and len(args) == 2:
                self.shaker_empty_level_ids[(args[0], args[1])] = f.global_id
        
        # Build abstract task type maps
        for at in model.abstract_tasks:
            tname, targs = self._parse_task_name(at.name)
            if tname == "AchieveContainsShotCocktail":
                self.achieve_contains_shot_cocktail_ids.add(at.global_id)
                # Track goal cocktails
                if len(targs) >= 2:
                    self.goal_cocktails.append((targs[0], targs[1]))
            elif tname == "AchieveContainsShakerCocktail":
                self.achieve_contains_shaker_cocktail_ids.add(at.global_id)
            elif tname == "AchieveContainsShakerIngredient":
                self.achieve_contains_shaker_ingredient_ids.add(at.global_id)
            elif tname == "AchieveContainsShotIngredient":
                self.achieve_contains_shot_ingredient_ids.add(at.global_id)
            elif tname == "AchieveCleanShaker":
                self.achieve_clean_shaker_ids.add(at.global_id)
            elif tname == "AchieveCleanShot":
                self.achieve_clean_shot_ids.add(at.global_id)
            elif tname == "AchieveHolding":
                self.achieve_holding_ids.add(at.global_id)
            elif tname == "AchieveHandEmpty":
                self.achieve_hand_empty_ids.add(at.global_id)
            elif tname == "AchieveOnTable":
                self.achieve_on_table_ids.add(at.global_id)
            elif tname == "DoPourShakerToShot":
                self.do_pour_shaker_to_shot_ids.add(at.global_id)
        
        # Cache task info for all abstract tasks and operators
        for at in model.abstract_tasks:
            self.task_name_cache[at.global_id] = self._parse_task_name(at.name)
        for op in model.operators:
            self.task_name_cache[op.global_id] = self._parse_task_name(op.name)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the state using bitwise operation."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """Estimate minimum actions to get cocktail into shot from current state.
        
        Full pipeline per cocktail (when starting from scratch):
        1. Clean shaker if dirty: grasp shaker (1) + empty shaker (1) + drop something (1) + clean (1) = ~3-4
           But if shaker is clean: 0
        2. For each of 2 ingredients:
           a. Clean a shot if needed: grasp (1) + empty (1) + clean (1) + drop (1) = ~2-4
           b. Fill shot from dispenser: grasp shot (1) + fill (1) = 2
           c. Pour shot to shaker: already holding shot, pour (1) = 1
           Subtotal per ingredient: ~3-5 actions
        3. Shake: need to hold shaker + free hand: grasp shaker (1) + shake (1) = 2
           But may need to drop shot first (1)
        4. Pour shaker to shot: need clean empty shot + hold shaker
           Clean target shot if needed, hold shaker, pour (1)
        
        Minimum total when everything is clean and empty: ~13-15 actions
        """
        # Check if cocktail already in shot
        contains_key = (shot, cocktail)
        if contains_key in self.contains_ids:
            if self._check_fact(state, self.contains_ids[contains_key]):
                return 0
        
        cost = 0
        
        # Check if shaker already contains this cocktail (shaked)
        shaker_has_cocktail = False
        for shaker_key, gid in self.contains_ids.items():
            if shaker_key[1] == cocktail and "shaker" in shaker_key[0]:
                if self._check_fact(state, gid):
                    shaker_has_cocktail = True
                    break
        
        if shaker_has_cocktail:
            # Just need to pour to shot
            # Need: clean shot, empty shot, hold shaker, pour
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            
            cost = 1  # pour action
            if not shot_clean or not shot_empty:
                cost += 4  # clean shot: grasp + empty + clean + overhead
            # Need to hold shaker
            cost += 2  # grasp shaker + maybe drop something
            return cost
        
        # Need full cocktail production
        # Find which shaker to use (there's typically just one)
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 15  # fallback
        
        ingredient1, ingredient2 = parts
        
        # Cost to clean shaker
        shaker_clean_cost = 0
        for sk in self.clean_ids:
            if "shaker" in sk:
                if not self._check_fact(state, self.clean_ids[sk]):
                    shaker_clean_cost = 3  # grasp + empty-shaker + clean-shaker
                    # If shaker not empty, add emptying cost
                    if sk in self.empty_ids and not self._check_fact(state, self.empty_ids[sk]):
                        shaker_clean_cost += 1  # empty-shaker action
                break
        
        # Cost for each ingredient: fill shot + pour to shaker = ~5 actions each
        # For each ingredient: clean-shot(~2) + grasp-shot(1) + fill-shot(1) + pour-to-shaker(1) = 5
        ingredient_cost = 0
        for ing in [ingredient1, ingredient2]:
            # Check if shaker already contains this ingredient
            found_in_shaker = False
            for sk_key, gid in self.contains_ids.items():
                if sk_key[1] == ing and "shaker" in sk_key[0]:
                    if self._check_fact(state, gid):
                        found_in_shaker = True
                        break
            if found_in_shaker:
                continue
            # Need to fill a shot with ingredient and pour to shaker
            # Minimum: grasp shot (1) + fill (1) + pour-to-shaker (1) + overhead (grasp/drop) = ~5
            ingredient_cost += 5
        
        # Shake cost: 1 action (+ maybe grasp shaker + drop something = 2-3)
        shake_cost = 2
        
        # Pour to target shot cost
        shot_clean = self._check_fact(state, self.clean_ids.get(shot))
        shot_empty = self._check_fact(state, self.empty_ids.get(shot))
        pour_cost = 1
        if not shot_clean:
            pour_cost += 3  # empty + clean + overhead
        if not shot_empty:
            pour_cost += 1  # empty-shot
        
        cost = shaker_clean_cost + ingredient_cost + shake_cost + pour_cost
        return max(cost, 1)  # at least 1 if not done

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network using state-aware per-task costs."""
        cost = 0
        
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive action)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1
                continue
            
            # Abstract task - estimate based on type
            if gid in self.achieve_contains_shot_cocktail_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 2:
                    shot, cocktail = targs[0], targs[1]
                    cost += self._cocktail_cost(state, shot, cocktail)
                else:
                    cost += 14
            elif gid in self.achieve_contains_shaker_cocktail_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 2:
                    shaker, cocktail = targs[0], targs[1]
                    # Check if already done
                    key = (shaker, cocktail)
                    if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                        cost += 0
                    else:
                        cost += 10  # clean shaker + 2 ingredients + shake
                else:
                    cost += 10
            elif gid in self.achieve_contains_shaker_ingredient_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 2:
                    shaker, ingredient = targs[0], targs[1]
                    key = (shaker, ingredient)
                    if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                        cost += 0
                    else:
                        cost += 4  # fill shot + pour to shaker
                else:
                    cost += 4
            elif gid in self.achieve_contains_shot_ingredient_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 2:
                    shot, ingredient = targs[0], targs[1]
                    key = (shot, ingredient)
                    if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                        cost += 0
                    else:
                        # Clean shot + grasp + fill = ~4
                        shot_clean = self._check_fact(state, self.clean_ids.get(shot))
                        shot_empty = self._check_fact(state, self.empty_ids.get(shot))
                        c = 2  # grasp + fill minimum
                        if not shot_clean:
                            c += 2
                        if not shot_empty:
                            c += 1
                        cost += c
                else:
                    cost += 4
            elif gid in self.achieve_clean_shaker_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 1:
                    shaker = targs[0]
                    if shaker in self.clean_ids and self._check_fact(state, self.clean_ids[shaker]):
                        cost += 0
                    else:
                        # Need to empty + clean: ~3-4 actions
                        if shaker in self.empty_ids and self._check_fact(state, self.empty_ids[shaker]):
                            cost += 3  # grasp + handEmpty + clean
                        else:
                            cost += 4  # grasp + empty + handEmpty + clean
                else:
                    cost += 3
            elif gid in self.achieve_clean_shot_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 1:
                    shot = targs[0]
                    if shot in self.clean_ids and self._check_fact(state, self.clean_ids[shot]):
                        cost += 0
                    else:
                        shot_empty = self._check_fact(state, self.empty_ids.get(shot))
                        if shot_empty:
                            cost += 3  # grasp + handEmpty + clean
                        else:
                            cost += 4  # grasp + empty + handEmpty + clean
                else:
                    cost += 3
            elif gid in self.achieve_holding_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 2:
                    hand, container = targs[0], targs[1]
                    key = (hand, container)
                    if key in self.holding_ids and self._check_fact(state, self.holding_ids[key]):
                        cost += 0
                    else:
                        # Need: hand empty + container on table + grasp
                        hand_free = self._check_fact(state, self.hand_empty_ids.get(hand))
                        on_table = self._check_fact(state, self.ontable_ids.get(container))
                        c = 1  # grasp
                        if not hand_free:
                            c += 1  # drop
                        if not on_table:
                            c += 1  # other hand drops it
                        cost += c
                else:
                    cost += 2
            elif gid in self.achieve_hand_empty_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 1:
                    hand = targs[0]
                    if hand in self.hand_empty_ids and self._check_fact(state, self.hand_empty_ids[hand]):
                        cost += 0
                    else:
                        cost += 1  # drop
                else:
                    cost += 1
            elif gid in self.achieve_on_table_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 1:
                    container = targs[0]
                    if container in self.ontable_ids and self._check_fact(state, self.ontable_ids[container]):
                        cost += 0
                    else:
                        cost += 1  # drop
                else:
                    cost += 1
            elif gid in self.do_pour_shaker_to_shot_ids:
                cost += 1  # single pour action
            else:
                # Unknown abstract task - conservative estimate
                cost += 1
        
        return cost

    def _unfinished_cocktails_bound(self, state, task_network):
        """Count unfinished cocktails in the task network and compute serialization bound.
        
        Because there's only one shaker, cocktails must be made sequentially.
        Each cocktail requires a minimum set of actions through the shaker pipeline.
        """
        unfinished = 0
        
        for task in task_network:
            gid = task.global_id
            if gid in self.achieve_contains_shot_cocktail_ids:
                tname, targs = self.task_name_cache.get(gid, (None, []))
                if targs and len(targs) >= 2:
                    shot, cocktail = targs[0], targs[1]
                    key = (shot, cocktail)
                    if key in self.contains_ids and self._check_fact(state, self.contains_ids[key]):
                        continue  # already done
                    
                    # Check if shaker already has this cocktail ready
                    shaker_has_it = False
                    for sk_key, gid2 in self.contains_ids.items():
                        if sk_key[1] == cocktail and "shaker" in sk_key[0]:
                            if self._check_fact(state, gid2):
                                shaker_has_it = True
                                break
                    
                    if shaker_has_it:
                        # Still need to pour, but don't need full shaker pipeline
                        # Don't count as full unfinished cocktail
                        pass
                    else:
                        unfinished += 1
        
        # Each unfinished cocktail requires minimum ~13 actions through the shaker pipeline
        # Actions: clean_shaker(3) + ingredient1(5) + ingredient2(4) + shake(1) + pour_to_shot(1) = 14
        # But first cocktail might not need shaker cleaning if already clean
        if unfinished > 0:
            # First cocktail: check if shaker is clean
            first_cost = 12  # without cleaning
            subsequent_cost = 15  # with cleaning between cocktails
            
            # Check shaker cleanliness
            shaker_clean = False
            for sk in self.clean_ids:
                if "shaker" in sk:
                    shaker_clean = self._check_fact(state, self.clean_ids[sk])
                    break
            
            if shaker_clean:
                return first_cost + max(0, (unfinished - 1)) * subsequent_cost
            else:
                return unfinished * subsequent_cost
        
        return 0

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible bounds."""
        if len(task_network) == 0:
            return 0
        
        # Term 1: State-aware task network cost (sum of per-task minimum costs)
        tn_cost = self._task_network_cost(state, task_network)
        
        # Term 2: Serialization bound from unfinished cocktails through single shaker
        serial_bound = self._unfinished_cocktails_bound(state, task_network)
        
        return max(tn_cost, serial_bound)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Called for every node expansion."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
