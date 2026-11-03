"""
Barman BDI Domain-Dependent Heuristic

== Bottleneck Analysis ==
The single shaker is the critical serializing resource. Every cocktail must be
made in the shaker (add 2 ingredients, shake, pour to shot), and the shaker
must be cleaned between cocktails. This means cocktails are essentially
sequential, each requiring ~13 primitive actions:
  - Clean shaker (if needed): ~3 actions (grasp, empty, clean, drop overhead)
  - Fill shot with ingredient1: ~4 actions (grasp shot, fill, pour-to-shaker, drop)
  - Fill shot with ingredient2: ~4 actions (grasp shot, fill, pour-to-shaker, drop)
  - Shake: 1 action (need shaker held + other hand empty)
  - Pour shaker to shot: need shot clean+empty, grasp shaker, pour: ~3 actions
  - Clean target shot if dirty: ~3 actions
Total per cocktail: ~13-15 actions minimum.

== Independent Admissible Lower Bounds ==
1. Unserved cocktails bound: For each cocktail goal (contains shot cocktail)
   not yet satisfied, we need at minimum:
   - If shaker doesn't contain the cocktail and shot doesn't contain it:
     ~13 actions (fill 2 ingredients into shaker via shots, shake, pour)
   - We estimate per-cocktail cost based on current state.
   This is admissible because each cocktail requires these steps independently
   of the shaker serialization.

2. Shaker serialization bound: Since there's one shaker and cocktails must be
   made sequentially, each unmade cocktail requires at least cleaning the shaker
   (if it was used) + adding ingredients + shaking + pouring. The shaker must be
   reused for each cocktail. Lower bound: number of unmade cocktails * minimum
   shaker cycle cost.

3. Task network structural bound: Count remaining abstract tasks weighted by
   minimum decomposition depth, using state-aware checks for trivial (null)
   decompositions.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic0(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_0"):
        super().__init__(name)
        # Precomputed maps
        self.contains_fact_id = {}  # (container_name, beverage_name) -> global_id
        self.clean_fact_id = {}     # container_name -> global_id
        self.empty_fact_id = {}     # container_name -> global_id
        self.holding_fact_id = {}   # (hand_name, container_name) -> global_id
        self.handEmpty_fact_id = {} # hand_name -> global_id
        self.ontable_fact_id = {}   # container_name -> global_id
        self.shaked_fact_id = {}    # shaker_name -> global_id
        self.unshaked_fact_id = {}  # shaker_name -> global_id
        self.used_fact_id = {}      # (container_name, beverage_name) -> global_id
        
        # Goal info
        self.goal_cocktail_shots = []  # list of (shot, cocktail) pairs from goals
        
        # Cocktail recipes
        self.cocktail_part1 = {}  # cocktail -> ingredient
        self.cocktail_part2 = {}  # cocktail -> ingredient
        
        # Task name mappings
        self.task_name_to_type = {}
        
        # Abstract task type -> base cost mapping
        self.abstract_task_costs = {}

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
        """Parse a task/operator name to extract the task type."""
        # Task names look like: __AchieveContainsShotCocktail[shot2,cocktail1]
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            task_type = name[:bracket_idx]
            # Remove leading underscores
            task_type = task_type.lstrip("_")
            args_str = name[bracket_idx+1:]
            if args_str.endswith("]"):
                args_str = args_str[:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name.lstrip("_"), []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
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
                self.handEmpty_fact_id[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
            elif pred == "shaked" and len(args) == 1:
                self.shaked_fact_id[args[0]] = f.global_id
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_fact_id[args[0]] = f.global_id
            elif pred == "used" and len(args) == 2:
                self.used_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                self.cocktail_part1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
        
        # Parse goal facts to find which (shot, cocktail) pairs are goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                self.goal_cocktail_shots.append((args[0], args[1]))
        
        # Identify shaker names
        self.shaker_names = list(self.shaked_fact_id.keys())
        
        # Precompute abstract task type identification
        # Build a map from abstract task global_id to its type string
        self.abt_type_map = {}
        for abt in model.abstract_tasks:
            task_type, args = self._parse_task_name(abt.name)
            self.abt_type_map[abt.global_id] = (task_type, args)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the state using bitwise check."""
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """
        Estimate the minimum number of actions to achieve contains(shot, cocktail).
        This is state-aware: checks what's already done.
        """
        # Check if already done
        cid = self.contains_fact_id.get((shot, cocktail))
        if cid is not None and self._check_fact(state, cid):
            return 0
        
        cost = 0
        
        # We need the cocktail in the shaker first (or it's already there)
        shaker = self.shaker_names[0] if self.shaker_names else None
        if shaker is None:
            return 15  # fallback
        
        shaker_has_cocktail = False
        shaker_shaked = False
        cid_shaker = self.contains_fact_id.get((shaker, cocktail))
        if cid_shaker is not None:
            shaker_has_cocktail = self._check_fact(state, cid_shaker)
        sid = self.shaked_fact_id.get(shaker)
        if sid is not None:
            shaker_shaked = self._check_fact(state, sid)
        
        if shaker_has_cocktail and shaker_shaked:
            # Shaker already has the shaked cocktail, just need to pour to shot
            # Need: shot clean+empty, hold shaker, pour
            shot_clean = False
            shot_empty = False
            cid_clean = self.clean_fact_id.get(shot)
            cid_empty = self.empty_fact_id.get(shot)
            if cid_clean is not None:
                shot_clean = self._check_fact(state, cid_clean)
            if cid_empty is not None:
                shot_empty = self._check_fact(state, cid_empty)
            
            if shot_clean and shot_empty:
                # Just need to grasp shaker and pour
                cost += 2  # grasp + pour (may need drop first)
            elif shot_empty and not shot_clean:
                # Need to clean shot (grasp, clean) then grasp shaker, pour
                cost += 5  # grasp shot, clean, drop, grasp shaker, pour
            else:
                # Shot has something: empty it, clean it, then pour
                cost += 7  # grasp shot, empty, clean, drop, grasp shaker, pour
        else:
            # Need to make the cocktail in shaker from scratch
            # Check if shaker is clean and empty
            shaker_clean = False
            shaker_empty = False
            cid_sc = self.clean_fact_id.get(shaker)
            cid_se = self.empty_fact_id.get(shaker)
            if cid_sc is not None:
                shaker_clean = self._check_fact(state, cid_sc)
            if cid_se is not None:
                shaker_empty = self._check_fact(state, cid_se)
            
            # Clean shaker cost
            if shaker_clean and shaker_empty:
                pass  # 0 cost for shaker prep
            elif shaker_empty and not shaker_clean:
                cost += 3  # grasp shaker, clean, (already held)
            else:
                # Shaker has contents: empty + clean
                cost += 5  # grasp, empty, clean, (overhead)
            
            # Add two ingredients via shots: each needs
            # grasp shot, fill from dispenser, pour to shaker, drop shot
            # For each ingredient: ~4 actions if shot is clean+empty
            # More if shot needs cleaning
            ing1 = self.cocktail_part1.get(cocktail)
            ing2 = self.cocktail_part2.get(cocktail)
            
            # Check if shaker already contains the ingredients
            has_ing1 = False
            has_ing2 = False
            if ing1:
                cid_i1 = self.contains_fact_id.get((shaker, ing1))
                if cid_i1 is not None:
                    has_ing1 = self._check_fact(state, cid_i1)
            if ing2:
                cid_i2 = self.contains_fact_id.get((shaker, ing2))
                if cid_i2 is not None:
                    has_ing2 = self._check_fact(state, cid_i2)
            
            if not has_ing1:
                cost += 4  # fill shot + pour to shaker
            if not has_ing2:
                cost += 4  # fill shot + pour to shaker
            
            # Shake: 1 action (need shaker held + hand empty)
            cost += 1
            
            # Pour shaker to shot
            # Need shot clean+empty
            shot_clean = False
            shot_empty = False
            cid_clean = self.clean_fact_id.get(shot)
            cid_empty = self.empty_fact_id.get(shot)
            if cid_clean is not None:
                shot_clean = self._check_fact(state, cid_clean)
            if cid_empty is not None:
                shot_empty = self._check_fact(state, cid_empty)
            
            if shot_clean and shot_empty:
                cost += 1  # just pour
            elif shot_empty:
                cost += 3  # clean shot then pour
            else:
                cost += 5  # empty shot, clean, pour
        
        return cost

    def _task_network_cost(self, state, task_network):
        """
        Estimate cost from the task network by examining each task.
        Uses state-aware estimation for key task types.
        """
        cost = 0
        for task in task_network:
            # Check if it's an abstract task (has decompositions)
            if hasattr(task, 'decompositions'):
                info = self.abt_type_map.get(task.global_id)
                if info is None:
                    cost += 1
                    continue
                task_type, args = info
                
                if task_type == "AchieveContainsShotCocktail" and len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    cost += self._cocktail_cost(state, shot, cocktail)
                elif task_type == "AchieveContainsShakerCocktail" and len(args) >= 2:
                    shaker, cocktail = args[0], args[1]
                    cid = self.contains_fact_id.get((shaker, cocktail))
                    if cid is not None and self._check_fact(state, cid):
                        pass  # 0 cost
                    else:
                        cost += 9  # clean shaker + 2 ingredients + shake
                elif task_type == "AchieveContainsShakerIngredient" and len(args) >= 2:
                    shaker, ingredient = args[0], args[1]
                    cid = self.contains_fact_id.get((shaker, ingredient))
                    if cid is not None and self._check_fact(state, cid):
                        pass
                    else:
                        cost += 4  # fill shot + pour to shaker
                elif task_type == "AchieveContainsShotIngredient" and len(args) >= 2:
                    shot, ingredient = args[0], args[1]
                    cid = self.contains_fact_id.get((shot, ingredient))
                    if cid is not None and self._check_fact(state, cid):
                        pass
                    else:
                        cost += 3  # clean shot + grasp + fill
                elif task_type == "AchieveCleanShaker" and len(args) >= 1:
                    shaker = args[0]
                    cid = self.clean_fact_id.get(shaker)
                    if cid is not None and self._check_fact(state, cid):
                        pass
                    else:
                        cost += 3
                elif task_type == "AchieveCleanShot" and len(args) >= 1:
                    shot = args[0]
                    cid = self.clean_fact_id.get(shot)
                    if cid is not None and self._check_fact(state, cid):
                        pass
                    else:
                        cost += 3
                elif task_type == "AchieveHolding" and len(args) >= 2:
                    hand, container = args[0], args[1]
                    hid = self.holding_fact_id.get((hand, container))
                    if hid is not None and self._check_fact(state, hid):
                        pass
                    else:
                        cost += 2  # drop current + grasp
                elif task_type == "AchieveHandEmpty" and len(args) >= 1:
                    hand = args[0]
                    hid = self.handEmpty_fact_id.get(hand)
                    if hid is not None and self._check_fact(state, hid):
                        pass
                    else:
                        cost += 1  # drop
                elif task_type == "AchieveOnTable" and len(args) >= 1:
                    container = args[0]
                    oid = self.ontable_fact_id.get(container)
                    if oid is not None and self._check_fact(state, oid):
                        pass
                    else:
                        cost += 1  # drop
                elif task_type == "DoPourShakerToShot":
                    cost += 1  # single pour action
                else:
                    cost += 1
            else:
                # Primitive operator
                cost += 1
        
        return cost

    def _unserved_goals_cost(self, state):
        """
        Count unserved cocktail goals and estimate total cost.
        Each unserved cocktail needs the shaker (serialization).
        """
        unserved = 0
        total_cocktail_cost = 0
        
        for shot, cocktail in self.goal_cocktail_shots:
            cid = self.contains_fact_id.get((shot, cocktail))
            if cid is None or not self._check_fact(state, cid):
                unserved += 1
                total_cocktail_cost += self._cocktail_cost(state, shot, cocktail)
        
        # Shaker serialization: each cocktail after the first needs at least
        # an extra shaker cleaning cycle (~3 actions: empty + clean + overhead)
        shaker_serial_cost = max(0, unserved - 1) * 3
        
        return total_cocktail_cost, shaker_serial_cost

    def _compute(self, state, task_network):
        """Compute the heuristic value using max of multiple components."""
        if self.model.goal_reached(state, task_network):
            return 0
        
        # Term 1: Task network structural cost (state-aware)
        tn_cost = self._task_network_cost(state, task_network)
        
        # Term 2: Goal-based cost (unserved cocktails)
        cocktail_cost, serial_cost = self._unserved_goals_cost(state)
        goal_cost = cocktail_cost + serial_cost
        
        # Return max of the two independent bounds
        h = max(tn_cost, goal_cost)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
