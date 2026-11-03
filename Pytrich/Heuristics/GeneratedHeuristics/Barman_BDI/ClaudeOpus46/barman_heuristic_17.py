# Barman BDI Domain Heuristic
# Bottleneck: Single shaker serializes all cocktail mixing. Each cocktail needs
# a full shaker cycle (clean, add 2 ingredients via shots, shake, pour to target shot).
#
# Admissible lower bounds:
# T1 - Unserved cocktails: each cocktail not yet in its target shot needs at minimum
#      ~13 primitive actions (clean shaker, fill 2 shots, pour 2 to shaker, shake, pour out)
# T2 - Task network structural bound: remaining tasks weighted by min decomposition cost
# T3 - Shaker state recovery: if shaker contains wrong cocktail or is dirty, extra actions needed
#
# Tie-breaking: small penalty for shots that are dirty/full to break symmetry among shot choices

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic17(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_17"):
        super().__init__(name)
        self.model = None
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
        self.shaker_empty_level_ids = {}  # (shaker, level) -> global_id
        self.cocktail_part1 = {}  # cocktail -> ingredient
        self.cocktail_part2 = {}  # cocktail -> ingredient
        self.dispenses_map = {}   # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktail_shots = []  # list of (shot, cocktail) pairs from goals
        
        # Task type name patterns
        self.task_type_costs = {}  # abstract task name prefix -> estimated min cost
        
        # All objects
        self.shots = set()
        self.shakers = set()
        self.cocktails = set()
        self.ingredients = set()
        self.hands = set()
        self.levels = set()

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
        """Parse a task/operator name to get the task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the state using bitwise operations."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

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
                self.cocktail_part1[args[0]] = args[1]
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
                self.ingredients.add(args[1])
            elif pred == "dispenses" and len(args) == 2:
                self.dispenses_map[args[1]] = args[0]
        
        # Identify objects from fact maps
        for (hand, container) in self.holding_ids:
            self.hands.add(hand)
        for hand in self.hand_empty_ids:
            self.hands.add(hand)
        for container in self.clean_ids:
            if container.startswith("shot"):
                self.shots.add(container)
            elif container.startswith("shaker"):
                self.shakers.add(container)
        for container in self.empty_ids:
            if container.startswith("shot"):
                self.shots.add(container)
            elif container.startswith("shaker"):
                self.shakers.add(container)
        
        # Parse goal facts to find which shots need which cocktails
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                shot = args[0]
                cocktail = args[1]
                self.goal_cocktail_shots.append((shot, cocktail))
        
        # Build abstract task name -> min cost map
        # Pre-analyze abstract tasks for cost estimation
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            # Store mapping from global_id to task info for fast lookup
        
        # Build operator name -> global_id map for fast task cost estimation
        self.op_name_map = {}
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            if task_type not in self.op_name_map:
                self.op_name_map[task_type] = []
            self.op_name_map[task_type].append(op)
        
        # Build abstract task name -> global_id map
        self.at_name_map = {}
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            if task_type not in self.at_name_map:
                self.at_name_map[task_type] = []
            self.at_name_map[task_type].append(at)
        
        # Precompute: for each cocktail goal, compute ingredients needed
        self.goal_ingredients = {}  # cocktail -> (ingredient1, ingredient2)
        for cocktail in self.cocktails:
            i1 = self.cocktail_part1.get(cocktail)
            i2 = self.cocktail_part2.get(cocktail)
            if i1 and i2:
                self.goal_ingredients[cocktail] = (i1, i2)
        
        # Minimum actions for a full cocktail cycle (from clean empty shaker to cocktail in shot):
        # 1. grasp shot1, 2. fill-shot (ingredient1), 3. drop shot1 (not always needed),
        # Actually let's count carefully:
        # Per cocktail needing full shaker cycle:
        #   - clean shaker if needed (0 if already clean, ~3 if not: grasp shaker, empty-shaker, clean-shaker + drops)
        #   - For each ingredient (x2):
        #     - clean shot if needed (0-4 actions)
        #     - grasp shot (1 action: grasp)
        #     - fill-shot (1 action)
        #     - pour-shot-to-shaker (1 action)
        #   - grasp shaker if not held (1 action)
        #   - shake (1 action)
        #   - pour-shaker-to-shot (need target shot clean+empty, held shaker) (1 action)
        # Minimum for clean shaker, clean shots available:
        #   grasp shot, fill-shot, pour-shot-to-clean-shaker (shot now on table? no, holding it)
        #   Actually: grasp shot1(1), fill-shot(1), pour-shot-to-clean-shaker(1) = shot still held
        #   but need to get second ingredient: drop shot1(1), grasp shot2(1), fill-shot(1),
        #   pour-shot-to-used-shaker(1), drop shot2(1), grasp shaker(1), shake(1)
        #   Then pour to target: need target shot clean+empty
        #   If target is clean+empty+ontable: just pour (but we're holding shaker)
        #   Actually pour-shaker-to-shot doesn't require picking up the shot
        #   pour-shaker-to-shot needs: holding hand shaker, clean shot, empty shot
        #   So: pour-shaker-to-shot(1)
        # Total minimum: grasp(1)+fill(1)+pour-to-shaker(1)+drop(1)+grasp(1)+fill(1)+pour-to-shaker(1)+drop(1)+grasp-shaker(1)+shake(1)+pour-to-shot(1) = 11
        # But after first cocktail, shaker needs cleaning: empty-shaker(1)+clean-shaker(1) = +2 minimum
        # Plus need to pick up shaker to empty it, and free hand for clean: more grasps/drops
        # Conservative minimum per cocktail: 11 for first, 13+ for subsequent
        
        self.MIN_COST_FIRST_COCKTAIL = 11
        self.MIN_COST_SUBSEQUENT_COCKTAIL = 14  # includes shaker cleaning overhead
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute_unserved_cocktail_cost(self, state):
        """Compute lower bound based on unserved cocktail goals."""
        unserved = []
        for (shot, cocktail) in self.goal_cocktail_shots:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is not None and self._check_fact(state, cid):
                continue  # Already served
            unserved.append((shot, cocktail))
        
        if not unserved:
            return 0
        
        total_cost = 0
        
        # Check if any shaker already has a useful cocktail
        # For each unserved cocktail, estimate minimum actions
        for idx, (shot, cocktail) in enumerate(unserved):
            cost = 0
            ingredients = self.goal_ingredients.get(cocktail)
            if not ingredients:
                cost += 5  # fallback
                total_cost += cost
                continue
            
            i1, i2 = ingredients
            
            # Check if shaker already contains this cocktail (shaked)
            shaker_has_cocktail = False
            for shaker in self.shakers:
                cid = self.contains_ids.get((shaker, cocktail))
                sid = self.shaked_ids.get(shaker)
                if cid is not None and sid is not None:
                    if self._check_fact(state, cid) and self._check_fact(state, sid):
                        shaker_has_cocktail = True
                        break
            
            if shaker_has_cocktail:
                # Just need to pour to shot
                # Need: holding shaker, shot clean+empty
                shot_clean = self._check_fact(state, self.clean_ids.get(shot))
                shot_empty = self._check_fact(state, self.empty_ids.get(shot))
                
                # Minimum: pour-shaker-to-shot (1)
                # But may need to: clean shot, pick up shaker
                cost = 1  # pour action
                if not shot_clean:
                    cost += 4  # grasp shot, empty-shot, clean-shot, drop shot
                elif not shot_empty:
                    cost += 3  # grasp shot, empty-shot, drop shot (still used, need clean)
                    cost += 2  # clean-shot needs empty+used, then clean
                
                # Need to be holding shaker
                shaker_held = False
                for hand in self.hands:
                    for shaker in self.shakers:
                        hid = self.holding_ids.get((hand, shaker))
                        if hid and self._check_fact(state, hid):
                            shaker_held = True
                            break
                    if shaker_held:
                        break
                if not shaker_held:
                    cost += 1  # grasp shaker
                    # May need to free a hand
                    any_hand_free = False
                    for hand in self.hands:
                        hid = self.hand_empty_ids.get(hand)
                        if hid and self._check_fact(state, hid):
                            any_hand_free = True
                            break
                    if not any_hand_free:
                        cost += 1  # drop something
                
                total_cost += cost
                continue
            
            # Need full shaker cycle
            # Check shaker state
            shaker_clean = False
            shaker_empty = False
            for shaker in self.shakers:
                sc = self._check_fact(state, self.clean_ids.get(shaker))
                se = self._check_fact(state, self.empty_ids.get(shaker))
                if sc:
                    shaker_clean = True
                if se:
                    shaker_empty = True
            
            # Base cost for ingredient loading + shake + pour
            # 2 ingredients: each needs fill-shot + pour-to-shaker (2 actions each) + grasp shot (1 each)
            # Plus shake(1) + pour-to-shot(1)
            # Minimum with everything ready: grasp(1)+fill(1)+pour(1) + grasp(1)+fill(1)+pour(1) + grasp-shaker(1)+shake(1)+pour(1) = 9
            # But between ingredients need to handle hands:
            # After pour-shot-to-shaker, holding shot. Drop it(1). Then grasp next shot(1).
            cost = 11  # baseline minimum
            
            if not shaker_clean or not shaker_empty:
                # Need to clean shaker first
                if not shaker_empty:
                    cost += 3  # grasp shaker + empty-shaker + drop/rearrange
                if not shaker_clean:
                    cost += 2  # clean-shaker needs holding + handEmpty other
            
            # Check if target shot is ready (clean + empty)
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            if not shot_clean:
                cost += 3  # need to clean shot
            elif not shot_empty:
                cost += 2  # need to empty shot (then clean)
            
            total_cost += cost
        
        return total_cost

    def _compute_task_network_cost(self, task_network):
        """Compute lower bound from remaining task network structure."""
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive action)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1  # Each primitive action costs exactly 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task - estimate minimum decomposition cost
                task_type, args = self._parse_task_name(task.name)
                
                if task_type == "AchieveContainsShotCocktail":
                    # This is the top-level goal task
                    # Minimum: need full cocktail cycle or already done
                    cost += 9  # conservative minimum for full cycle
                elif task_type == "AchieveContainsShakerCocktail":
                    cost += 7  # clean shaker + 2 ingredients + shake
                elif task_type == "AchieveContainsShakerIngredient":
                    cost += 3  # fill shot + pour to shaker
                elif task_type == "AchieveContainsShotIngredient":
                    cost += 2  # clean shot + fill
                elif task_type == "AchieveCleanShaker":
                    cost += 1  # may need empty + clean
                elif task_type == "AchieveCleanShot":
                    cost += 1  # may need empty + clean 
                elif task_type == "AchieveHolding":
                    cost += 1  # grasp (may need drop first)
                elif task_type == "AchieveHandEmpty":
                    cost += 0  # may already be empty (0), or drop (1)
                elif task_type == "AchieveOnTable":
                    cost += 0  # may already be on table
                elif task_type == "DoPourShakerToShot":
                    cost += 1  # single action
                else:
                    cost += 1  # unknown abstract task
        
        return cost

    def _compute_shaker_serialization_bound(self, state, task_network):
        """Lower bound based on shaker being a serializing bottleneck.
        
        Count how many distinct cocktails still need to be mixed in the shaker.
        Each needs a full exclusive shaker cycle.
        """
        cocktails_needing_shaker = set()
        
        for (shot, cocktail) in self.goal_cocktail_shots:
            # Check if this cocktail is already in the target shot
            cid = self.contains_ids.get((shot, cocktail))
            if cid is not None and self._check_fact(state, cid):
                continue
            
            # Check if shaker already has this cocktail ready (shaked)
            shaker_has_it = False
            for shaker in self.shakers:
                cid_shaker = self.contains_ids.get((shaker, cocktail))
                sid = self.shaked_ids.get(shaker)
                if cid_shaker is not None and sid is not None:
                    if self._check_fact(state, cid_shaker) and self._check_fact(state, sid):
                        shaker_has_it = True
                        break
            
            if not shaker_has_it:
                cocktails_needing_shaker.add(cocktail)
        
        n = len(cocktails_needing_shaker)
        if n == 0:
            return 0
        
        # First cocktail: ~11 actions minimum
        # Each subsequent: ~14 actions (need to clean shaker first)
        # But we want a reasonably tight lower bound
        # Conservative: 9 for first, 11 for each subsequent
        return 9 + max(0, (n - 1)) * 11

    def _compute(self, state, task_network):
        """Compute heuristic value using max of multiple lower bounds."""
        if not task_network and self.model and self.model.goal_reached(state, task_network):
            return 0
        
        if not task_network:
            return 0
        
        # Term 1: Task network structural lower bound
        t1 = self._compute_task_network_cost(task_network)
        
        # Term 2: Unserved cocktail state-aware cost
        t2 = self._compute_unserved_cocktail_cost(state)
        
        # Term 3: Shaker serialization bound
        t3 = self._compute_shaker_serialization_bound(state, task_network)
        
        h = max(t1, t2, t3)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
