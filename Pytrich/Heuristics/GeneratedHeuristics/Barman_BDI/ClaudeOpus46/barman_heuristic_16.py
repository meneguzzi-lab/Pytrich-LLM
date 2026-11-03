"""Domain-dependent heuristic for Barman_BDI HTN planning.

Bottleneck Analysis:
  The single shaker is the critical serializing resource. Every cocktail must be
  mixed in the shaker (fill with 2 ingredients, shake, pour to shot), then the
  shaker must be emptied and cleaned before the next cocktail. This means each
  cocktail requires a full shaker cycle, and these cycles are strictly sequential.

Admissible Lower Bounds:
  Term 1 (Cocktail production cost): Each cocktail not yet in its target shot
  requires at minimum: clean shaker (if dirty) + fill shot with ingredient1 +
  pour to shaker + fill shot with ingredient2 + pour to shaker + shake + pour
  shaker to shot. That's roughly 11-13 primitive actions per cocktail. We use
  a conservative per-cocktail estimate based on current state.

  Term 2 (Task network lower bound): Count remaining non-trivial abstract tasks
  in the task network, weighted by minimum decomposition depth. Each
  AchieveContainsShotCocktail that isn't satisfied needs multiple actions.

  Term 3 (Shaker bottleneck): Since there's only one shaker, cocktails that
  still need mixing must be processed sequentially. Each shaker cycle has a
  minimum cost. This is a valid lower bound because shaker cycles cannot overlap.

Tie-breaking: Shots are interchangeable for intermediate use, so we add a small
penalty for states where resources are less organized.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic16(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_16"):
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
        self.cocktail_part1 = {}  # cocktail -> ingredient
        self.cocktail_part2 = {}  # cocktail -> ingredient
        self.dispenses_map = {}   # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktail_shots = []  # list of (shot, cocktail) pairs from goals
        
        # Abstract task name mapping
        self.task_name_map = {}  # abstract_task global_id -> task_type string
        
        # Sets for type identification
        self.shots = set()
        self.shakers = set()
        self.cocktails = set()
        self.ingredients = set()
        self.hands = set()

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
        """Parse a grounded task/operator name to get the task type and arguments."""
        # Format like: AchieveContainsShotCocktail[shot2,cocktail1] or similar
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Sometimes names have underscores for grounded variants
        # Try splitting on first parenthesis or bracket
        paren_idx = name.find("(")
        if paren_idx > 0 and name.endswith(")"):
            task_type = name[:paren_idx]
            args_str = name[paren_idx+1:-1]
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
                self.cocktail_part1[args[0]] = args[1]
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
                self.ingredients.add(args[1])
            elif pred == "dispenses" and len(args) == 2:
                self.dispenses_map[args[1]] = args[0]

        # Identify objects by predicate usage
        for key in self.clean_ids:
            if key.startswith("shot"):
                self.shots.add(key)
            elif key.startswith("shaker"):
                self.shakers.add(key)
        for key in self.hand_empty_ids:
            self.hands.add(key)

        # Parse goal facts to find target (shot, cocktail) pairs
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                shot_name = args[0]
                cocktail_name = args[1]
                if shot_name.startswith("shot") and cocktail_name.startswith("cocktail"):
                    self.goal_cocktail_shots.append((shot_name, cocktail_name))

        # Build abstract task lookup by global_id for fast classification
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_name_map[at.global_id] = (task_type, args)

        # Also build operator name lookup
        self.op_name_map = {}
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            self.op_name_map[op.global_id] = (task_type, args)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the given state using bitwise check."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, cocktail, target_shot):
        """Estimate minimum actions needed to get cocktail into target_shot.
        
        Full pipeline per cocktail (when nothing is prepared):
        1. Clean shaker if dirty: ~3 actions (grasp shaker, empty, clean) or 0 if clean
        2. Fill shot with ingredient1: clean shot (~3-4), grasp shot (~1-2), fill (~1) = ~5
        3. Pour shot to shaker: ~1
        4. Fill shot with ingredient2: clean shot (~3-4), grasp shot (~1-2), fill (~1) = ~5  
        5. Pour shot to shaker: ~1
        6. Shake: ~1 (need to hold shaker + free hand)
        7. Clean target shot: ~3-4
        8. Pour shaker to shot: ~1
        
        Minimum realistic cost: ~11 actions if everything is clean and ready.
        """
        # Check if cocktail is already in the target shot
        contains_key = (target_shot, cocktail)
        if contains_key in self.contains_ids:
            if self._check_fact(state, self.contains_ids[contains_key]):
                return 0

        cost = 0
        
        # Check if any shaker already contains this cocktail (shaked)
        shaker_has_cocktail = False
        for shaker in self.shakers:
            sk = (shaker, cocktail)
            if sk in self.contains_ids and self._check_fact(state, self.contains_ids[sk]):
                if self.shaked_ids.get(shaker) is not None and self._check_fact(state, self.shaked_ids[shaker]):
                    shaker_has_cocktail = True
                    # Just need to pour to shot
                    # Need: hold shaker, shot clean & empty on table
                    cost += 1  # pour action
                    # Ensure shot is clean and empty
                    if not self._check_fact(state, self.clean_ids.get(target_shot, -1)):
                        cost += 3  # grasp, empty, clean
                    elif not self._check_fact(state, self.empty_ids.get(target_shot, -1)):
                        cost += 2  # grasp, empty
                    # Ensure holding shaker
                    found_holding = False
                    for hand in self.hands:
                        hk = (hand, shaker)
                        if hk in self.holding_ids and self._check_fact(state, self.holding_ids[hk]):
                            found_holding = True
                            break
                    if not found_holding:
                        cost += 2  # drop current + grasp shaker
                    break

        if not shaker_has_cocktail:
            # Need full cocktail production
            ing1 = self.cocktail_part1.get(cocktail)
            ing2 = self.cocktail_part2.get(cocktail)
            
            # Check if shaker needs cleaning
            shaker = None
            for s in self.shakers:
                shaker = s
                break
            
            if shaker is None:
                return 15  # fallback
            
            shaker_clean = self._check_fact(state, self.clean_ids.get(shaker, -1))
            shaker_empty = self._check_fact(state, self.empty_ids.get(shaker, -1))
            
            # Check if shaker already has the ingredients
            has_ing1 = False
            has_ing2 = False
            if ing1 and (shaker, ing1) in self.contains_ids:
                has_ing1 = self._check_fact(state, self.contains_ids[(shaker, ing1)])
            if ing2 and (shaker, ing2) in self.contains_ids:
                has_ing2 = self._check_fact(state, self.contains_ids[(shaker, ing2)])
            
            if has_ing1 and has_ing2:
                # Just need shake + pour
                cost += 1  # shake
                cost += 1  # pour
                # + overhead for holding/cleaning shot
                cost += 3
            elif has_ing1 or has_ing2:
                # Need one more ingredient + shake + pour
                cost += 5  # get remaining ingredient into shaker
                cost += 1  # shake
                cost += 1  # pour
                cost += 3  # shot overhead
            else:
                # Need everything
                if not shaker_clean or not shaker_empty:
                    if not shaker_empty:
                        cost += 3  # empty shaker (grasp + empty + maybe drop)
                    if not shaker_clean:
                        cost += 2  # clean shaker
                
                # Get ingredient 1 into shaker: fill shot + pour to shaker
                cost += 4  # grasp shot, fill, pour = ~4 minimum
                
                # Get ingredient 2 into shaker: fill shot + pour to shaker  
                cost += 4  # similar
                
                # Shake
                cost += 1
                
                # Clean target shot if needed
                if not self._check_fact(state, self.clean_ids.get(target_shot, -1)):
                    cost += 3
                elif not self._check_fact(state, self.empty_ids.get(target_shot, -1)):
                    cost += 2
                
                # Pour shaker to shot
                cost += 1

        return cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining tasks in the network."""
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            gid = task.global_id
            # Check if it's an operator (primitive task)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task - estimate based on type
                if gid in self.task_name_map:
                    task_type, args = self.task_name_map[gid]
                    cost += self._estimate_abstract_task_cost(state, task_type, args)
                else:
                    cost += 1
        return cost

    def _estimate_abstract_task_cost(self, state, task_type, args):
        """Estimate minimum cost for an abstract task based on type and current state."""
        if task_type == "AchieveContainsShotCocktail" and len(args) >= 2:
            shot, cocktail = args[0], args[1]
            ck = (shot, cocktail)
            if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                return 0
            return self._cocktail_cost(state, cocktail, shot)
        
        elif task_type == "AchieveContainsShakerCocktail" and len(args) >= 2:
            shaker, cocktail = args[0], args[1]
            ck = (shaker, cocktail)
            if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                return 0
            return 8  # Minimum: 2 ingredients to shaker (4+4) + shake (1) - some overlap
        
        elif task_type == "AchieveContainsShakerIngredient" and len(args) >= 2:
            shaker, ingredient = args[0], args[1]
            ck = (shaker, ingredient)
            if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                return 0
            return 4  # fill shot + pour to shaker + overhead
        
        elif task_type == "AchieveContainsShotIngredient" and len(args) >= 2:
            shot, ingredient = args[0], args[1]
            ck = (shot, ingredient)
            if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                return 0
            # Need: clean shot, hold shot, fill
            shot_clean = self._check_fact(state, self.clean_ids.get(shot, -1))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot, -1))
            if shot_clean and shot_empty:
                return 2  # grasp + fill
            elif shot_empty:
                return 4  # grasp + clean + fill overhead
            else:
                return 5  # empty + clean + grasp + fill
        
        elif task_type == "AchieveCleanShaker" and len(args) >= 1:
            shaker = args[0]
            if self._check_fact(state, self.clean_ids.get(shaker, -1)):
                return 0
            if self._check_fact(state, self.empty_ids.get(shaker, -1)):
                return 2  # grasp + clean
            return 4  # grasp + empty + clean + drop
        
        elif task_type == "AchieveCleanShot" and len(args) >= 1:
            shot = args[0]
            if self._check_fact(state, self.clean_ids.get(shot, -1)):
                return 0
            if self._check_fact(state, self.empty_ids.get(shot, -1)):
                return 2  # grasp + clean
            return 3  # grasp + empty + clean
        
        elif task_type == "AchieveHolding" and len(args) >= 2:
            hand, container = args[0], args[1]
            hk = (hand, container)
            if hk in self.holding_ids and self._check_fact(state, self.holding_ids[hk]):
                return 0
            # Need hand empty + container on table + grasp
            hand_free = self._check_fact(state, self.hand_empty_ids.get(hand, -1))
            on_table = self._check_fact(state, self.ontable_ids.get(container, -1))
            cost = 1  # grasp
            if not hand_free:
                cost += 1  # drop
            if not on_table:
                cost += 1  # other hand drops it
            return cost
        
        elif task_type == "AchieveHandEmpty" and len(args) >= 1:
            hand = args[0]
            if self._check_fact(state, self.hand_empty_ids.get(hand, -1)):
                return 0
            return 1  # drop
        
        elif task_type == "AchieveOnTable" and len(args) >= 1:
            container = args[0]
            if self._check_fact(state, self.ontable_ids.get(container, -1)):
                return 0
            return 1  # drop
        
        elif task_type == "DoPourShakerToShot" and len(args) >= 3:
            return 1  # single action
        
        # Default for unknown tasks
        return 1

    def _shaker_bottleneck_cost(self, state):
        """Count cocktails still needing shaker processing.
        
        Each cocktail that isn't in its target shot and isn't already shaked in
        the shaker requires a full shaker cycle. Since there's only one shaker,
        these are sequential. Minimum per cycle: ~9 actions.
        """
        unfinished = 0
        for shot, cocktail in self.goal_cocktail_shots:
            ck = (shot, cocktail)
            if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                continue
            # Check if shaker already has this cocktail shaked
            shaker_has_it = False
            for shaker in self.shakers:
                sk = (shaker, cocktail)
                if sk in self.contains_ids and self._check_fact(state, self.contains_ids[sk]):
                    if self.shaked_ids.get(shaker) is not None and self._check_fact(state, self.shaked_ids[shaker]):
                        shaker_has_it = True
                        break
            if not shaker_has_it:
                unfinished += 1
        
        # Each unfinished cocktail needs minimum shaker cycle
        # Conservative estimate: clean shaker(2) + 2*fill-and-pour(8) + shake(1) + pour(1) = 12
        # But first cocktail might not need shaker cleaning
        if unfinished == 0:
            return 0
        
        # Check current shaker state for first cocktail
        first_cost = 9  # optimistic first cycle
        subsequent_cost = 12  # need to clean between cycles
        
        shaker = None
        for s in self.shakers:
            shaker = s
            break
        
        if shaker and self._check_fact(state, self.clean_ids.get(shaker, -1)) and \
           self._check_fact(state, self.empty_ids.get(shaker, -1)):
            first_cost = 9
        else:
            first_cost = 12
        
        if unfinished == 1:
            return first_cost
        return first_cost + (unfinished - 1) * subsequent_cost

    def _compute(self, state, task_network):
        """Compute the heuristic value using max of multiple admissible bounds."""
        # Term 1: Sum of per-goal cocktail costs (not strictly admissible as sum,
        # but provides strong guidance; the shaker bottleneck is the real lower bound)
        goal_cost = 0
        for shot, cocktail in self.goal_cocktail_shots:
            goal_cost += self._cocktail_cost(state, cocktail, shot)
        
        # Term 2: Task network based cost
        tn_cost = self._task_network_cost(state, task_network)
        
        # Term 3: Shaker bottleneck (admissible - sequential constraint)
        bottleneck_cost = self._shaker_bottleneck_cost(state)
        
        # Use max of bottleneck and task network, but goal_cost captures more detail
        # The bottleneck is a true lower bound; goal_cost double-counts shaker work
        # but provides better guidance
        h = max(bottleneck_cost, tn_cost)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
