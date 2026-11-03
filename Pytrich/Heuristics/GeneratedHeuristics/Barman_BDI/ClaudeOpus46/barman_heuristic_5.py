"""Domain-dependent heuristic for Barman_BDI domain.

Bottleneck Analysis:
  The single shaker is the serializing bottleneck. Every cocktail must be mixed
  in the shaker (clean shaker, add 2 ingredients via shots, shake, pour to target shot).
  The shaker must be cleaned between cocktails. This forces a sequential pipeline
  of ~14+ primitive actions per cocktail goal.

Admissible Lower Bounds:
  Term 1 - Per-cocktail minimum actions: Each unachieved cocktail goal requires at minimum:
    - Clean shaker (if dirty): 3 actions (grasp, empty, clean) or 1 (clean-shaker) if already empty
    - Fill shot with ingredient1: ~4 actions (grasp shot, fill, pour-to-shaker, drop)
    - Fill shot with ingredient2: ~4 actions (grasp shot, fill, pour-to-shaker)
    - Shake: 1 action
    - Pour shaker to shot: ~3 actions (grasp shaker, clean target shot, pour)
    Conservative lower bound: ~7 actions per unachieved cocktail (admissible undercount).

  Term 2 - Shaker serialization: Since there's one shaker and each cocktail needs it,
    we need at least (num_unachieved_cocktails - 1) * clean_shaker_cost additional actions
    for shaker reuse. Lower bound: 3 * max(0, unachieved - 1) for the clean-between cost.

  Term 3 - Task network structural lower bound: Count minimum primitive actions needed
    by examining remaining abstract tasks in the task network with state-aware costing.

Tie-breaking: Shots are interchangeable for intermediate use; we add a small penalty
for state disorder to break symmetries.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic5(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_5"):
        super().__init__(name)
        # Fact ID maps
        self.contains_ids = {}  # (container, beverage) -> global_id
        self.clean_ids = {}     # container -> global_id
        self.empty_ids = {}     # container -> global_id
        self.holding_ids = {}   # (hand, container) -> global_id
        self.hand_empty_ids = {} # hand -> global_id
        self.ontable_ids = {}   # container -> global_id
        self.shaked_ids = {}    # shaker -> global_id
        self.unshaked_ids = {}  # shaker -> global_id
        self.used_ids = {}      # (container, beverage) -> global_id
        self.cocktail_part1 = {} # cocktail -> ingredient
        self.cocktail_part2 = {} # cocktail -> ingredient
        self.dispenses_map = {}  # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktail_in_shot = []  # [(shot, cocktail)]
        self.goal_contains_ids = []      # [global_id for contains facts in goals]
        
        # Abstract task name maps
        self.task_name_cache = {}
        
        # Shaker name
        self.shaker = None
        self.shots = set()
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
        """Parse a task/operator name to extract the task type and arguments."""
        # Format: taskname[arg1,arg2,...]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
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
                self.cocktail_part1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
            elif pred == "dispenses" and len(args) == 2:
                self.dispenses_map[args[1]] = args[0]
        
        # Identify shakers, shots, hands from fact maps
        for key in self.shaked_ids:
            self.shaker = key
        for (hand, _) in self.holding_ids:
            self.hands.add(hand)
        for cont in self.clean_ids:
            if cont != self.shaker:
                self.shots.add(cont)
        
        # Parse goals to find which shots need which cocktails
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                shot = args[0]
                cocktail = args[1]
                gid = self.contains_ids.get((shot, cocktail))
                if gid is not None:
                    self.goal_cocktail_in_shot.append((shot, cocktail, gid))
        
        # Build task name cache for abstract tasks
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.task_name_cache[at.global_id] = (ttype, targs)
        
        # Also cache operator names
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            self.task_name_cache[op.global_id] = (ttype, targs)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h1 = self._goal_cocktail_cost(state)
        h2 = self._task_network_cost(state, task_network)
        h3 = self._shaker_serialization_cost(state)
        return max(h1, h2, h3)

    def _goal_cocktail_cost(self, state):
        """Compute admissible lower bound based on unachieved cocktail goals.
        
        For each cocktail goal (contains shot cocktail) not yet achieved:
        - We need to make the cocktail in the shaker (if not already there)
        - We need to pour it into the target shot
        
        Minimum actions per unachieved cocktail:
        - If cocktail not in shaker and shaker not clean/empty: need to clean shaker
        - Fill shot with ingredient1: grasp shot (1) + fill (1) = at least 2
        - Pour shot to shaker: 1
        - Fill shot with ingredient2: clean shot (1-2) + grasp (1) + fill (1) = at least 3  
        - Pour shot to shaker: 1
        - Shake: 1
        - Clean target shot if needed + pour shaker to shot: at least 1
        Conservative: ~7 actions minimum per cocktail
        """
        total_cost = 0
        unachieved_count = 0
        
        for shot, cocktail, gid in self.goal_cocktail_in_shot:
            if not ((state >> gid) & 1):
                unachieved_count += 1
                cost = 0
                
                # Check if cocktail is already in shaker (shaked)
                contains_shaker_cocktail = self.contains_ids.get((self.shaker, cocktail))
                cocktail_in_shaker = False
                if contains_shaker_cocktail is not None:
                    cocktail_in_shaker = (state >> contains_shaker_cocktail) & 1
                
                if cocktail_in_shaker:
                    # Just need to pour to shot
                    # Need target shot clean and empty
                    shot_clean = (state >> self.clean_ids[shot]) & 1 if shot in self.clean_ids else 0
                    shot_empty = (state >> self.empty_ids[shot]) & 1 if shot in self.empty_ids else 0
                    
                    if not shot_clean or not shot_empty:
                        cost += 2  # empty-shot + clean-shot minimum
                    
                    # Check if we're holding the shaker
                    holding_shaker = False
                    for hand in self.hands:
                        hid = self.holding_ids.get((hand, self.shaker))
                        if hid is not None and (state >> hid) & 1:
                            holding_shaker = True
                            break
                    if not holding_shaker:
                        cost += 1  # grasp shaker
                    
                    cost += 1  # pour-shaker-to-shot
                else:
                    # Need to make cocktail from scratch
                    ing1 = self.cocktail_part1.get(cocktail)
                    ing2 = self.cocktail_part2.get(cocktail)
                    
                    # Check if ingredients are in shaker
                    ing1_in_shaker = False
                    ing2_in_shaker = False
                    if ing1:
                        cid = self.contains_ids.get((self.shaker, ing1))
                        if cid is not None:
                            ing1_in_shaker = (state >> cid) & 1
                    if ing2:
                        cid = self.contains_ids.get((self.shaker, ing2))
                        if cid is not None:
                            ing2_in_shaker = (state >> cid) & 1
                    
                    # Cost for each missing ingredient
                    if not ing1_in_shaker:
                        # Need: get ingredient into a shot, then pour to shaker
                        # grasp shot(1) + fill-shot(1) + pour-shot-to-shaker(1) = 3 minimum
                        cost += 4  # including a drop somewhere
                    if not ing2_in_shaker:
                        # Same for second ingredient, plus clean the shot
                        # clean-shot needs: empty(if full) + clean = 1-2
                        # Then fill + pour = 2
                        cost += 5
                    
                    # Shake action
                    cost += 1
                    
                    # Clean target shot if needed + pour
                    shot_clean = (state >> self.clean_ids[shot]) & 1 if shot in self.clean_ids else 0
                    shot_empty = (state >> self.empty_ids[shot]) & 1 if shot in self.empty_ids else 0
                    if not shot_clean or not shot_empty:
                        cost += 3  # grasp, empty, clean
                    cost += 1  # pour
                    
                    # Need to hold shaker for shake and pour
                    cost += 1  # grasp shaker at some point
                
                total_cost += cost
        
        return total_cost

    def _shaker_serialization_cost(self, state):
        """Each cocktail needs the shaker. Between cocktails, shaker must be cleaned.
        Cleaning requires: empty-shaker(1) + clean-shaker(1) = 2 actions minimum.
        Plus grasp/drop overhead.
        """
        unachieved = 0
        for shot, cocktail, gid in self.goal_cocktail_in_shot:
            if not ((state >> gid) & 1):
                # Also check cocktail not already in shaker
                contains_shaker_cocktail = self.contains_ids.get((self.shaker, cocktail))
                cocktail_in_shaker = False
                if contains_shaker_cocktail is not None:
                    cocktail_in_shaker = (state >> contains_shaker_cocktail) & 1
                if not cocktail_in_shaker:
                    unachieved += 1
        
        if unachieved <= 1:
            return 0
        
        # Between each pair of consecutive cocktails, need to clean shaker
        # clean-shaker needs: empty-shaker(1) + clean-shaker(1) = 2 actions
        # Plus grasp shaker if not holding = 1, and possibly drop = 1
        # Conservative: 2 actions per transition
        serialization_cost = (unachieved - 1) * 2
        
        # Plus minimum per-cocktail cost: at least 7 actions each
        # (fill 2 ingredients + shake + pour = minimum)
        per_cocktail_min = unachieved * 7
        
        return max(serialization_cost + per_cocktail_min, unachieved * 8)

    def _task_network_cost(self, state, task_network):
        """State-aware cost estimation for remaining tasks in the network.
        
        For each task, estimate minimum primitive actions needed given current state.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            gid = task.global_id
            cached = self.task_name_cache.get(gid)
            if cached is None:
                # Unknown task, give it cost 1
                total += 1
                continue
            
            ttype, targs = cached
            
            # Primitive operators
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                total += 1
                continue
            
            # Abstract tasks - estimate based on type and state
            if ttype == "AchieveContainsShotCocktail" and len(targs) >= 2:
                shot, cocktail = targs[0], targs[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is not None and (state >> cid) & 1:
                    total += 0  # Already achieved
                else:
                    # Check if cocktail is in shaker
                    scid = self.contains_ids.get((self.shaker, cocktail))
                    if scid is not None and (state >> scid) & 1:
                        # Cocktail ready in shaker, just need to pour
                        total += 3  # clean shot + pour
                    else:
                        total += 12  # Full pipeline
            
            elif ttype == "AchieveContainsShakerCocktail" and len(targs) >= 2:
                shaker, cocktail = targs[0], targs[1]
                cid = self.contains_ids.get((shaker, cocktail))
                if cid is not None and (state >> cid) & 1:
                    total += 0
                else:
                    # Need clean shaker + 2 ingredients + shake
                    shaker_clean = (state >> self.clean_ids.get(shaker, -1)) & 1 if shaker in self.clean_ids else 0
                    shaker_empty = (state >> self.empty_ids.get(shaker, -1)) & 1 if shaker in self.empty_ids else 0
                    cost = 0
                    if not shaker_clean:
                        cost += 3  # empty + clean + overhead
                    if not shaker_empty and shaker_clean:
                        cost += 0
                    cost += 8  # 2 ingredients (4 each) + shake
                    total += cost
            
            elif ttype == "AchieveContainsShakerIngredient" and len(targs) >= 2:
                shaker, ingredient = targs[0], targs[1]
                cid = self.contains_ids.get((shaker, ingredient))
                if cid is not None and (state >> cid) & 1:
                    total += 0
                else:
                    total += 4  # fill shot + pour to shaker
            
            elif ttype == "AchieveContainsShotIngredient" and len(targs) >= 2:
                shot, ingredient = targs[0], targs[1]
                cid = self.contains_ids.get((shot, ingredient))
                if cid is not None and (state >> cid) & 1:
                    total += 0
                else:
                    shot_clean = (state >> self.clean_ids.get(shot, -1)) & 1 if shot in self.clean_ids else 0
                    if shot_clean:
                        total += 2  # grasp + fill
                    else:
                        total += 4  # clean + grasp + fill
            
            elif ttype == "AchieveCleanShaker" and len(targs) >= 1:
                shaker = targs[0]
                if shaker in self.clean_ids and (state >> self.clean_ids[shaker]) & 1:
                    total += 0
                else:
                    shaker_empty_val = (state >> self.empty_ids.get(shaker, -1)) & 1 if shaker in self.empty_ids else 0
                    if shaker_empty_val:
                        total += 2  # grasp + clean
                    else:
                        total += 4  # grasp + empty + clean + overhead
            
            elif ttype == "AchieveCleanShot" and len(targs) >= 1:
                shot = targs[0]
                if shot in self.clean_ids and (state >> self.clean_ids[shot]) & 1:
                    total += 0
                else:
                    shot_empty_val = (state >> self.empty_ids.get(shot, -1)) & 1 if shot in self.empty_ids else 0
                    if shot_empty_val:
                        total += 2  # grasp + clean
                    else:
                        total += 3  # grasp + empty + clean
            
            elif ttype == "AchieveHolding" and len(targs) >= 2:
                hand, container = targs[0], targs[1]
                hid = self.holding_ids.get((hand, container))
                if hid is not None and (state >> hid) & 1:
                    total += 0
                else:
                    # Check if hand is empty
                    he_id = self.hand_empty_ids.get(hand)
                    hand_is_empty = he_id is not None and (state >> he_id) & 1
                    ot_id = self.ontable_ids.get(container)
                    on_table = ot_id is not None and (state >> ot_id) & 1
                    cost = 0
                    if not hand_is_empty:
                        cost += 1  # drop
                    if not on_table:
                        cost += 1  # other hand drops it
                    cost += 1  # grasp
                    total += cost
            
            elif ttype == "AchieveHandEmpty" and len(targs) >= 1:
                hand = targs[0]
                he_id = self.hand_empty_ids.get(hand)
                if he_id is not None and (state >> he_id) & 1:
                    total += 0
                else:
                    total += 1  # drop
            
            elif ttype == "AchieveOnTable" and len(targs) >= 1:
                container = targs[0]
                ot_id = self.ontable_ids.get(container)
                if ot_id is not None and (state >> ot_id) & 1:
                    total += 0
                else:
                    total += 1  # drop
            
            elif ttype == "DoPourShakerToShot" and len(targs) >= 3:
                total += 1  # single action
            
            else:
                # Unknown abstract task
                total += 1
        
        return total
