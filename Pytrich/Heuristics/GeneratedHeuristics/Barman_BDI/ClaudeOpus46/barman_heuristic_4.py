"""Barman BDI Domain-Dependent Heuristic (version 4)

Bottleneck Analysis:
  The single shaker is the serializing bottleneck. Every cocktail must be
  mixed in the shaker (clean -> add ingredient1 -> add ingredient2 -> shake
  -> pour to shot -> clean again). Since there is only one shaker, cocktails
  must be made sequentially. Each cocktail cycle requires at minimum:
    - Clean shaker (if dirty): ~3 actions (grasp, empty, clean) or 0 if already clean
    - Add ingredient 1 to shaker: fill shot (~4 actions) + pour to shaker (~1)
    - Add ingredient 2 to shaker: fill shot (~4 actions) + pour to shaker (~1) 
    - Shake: 1 action
    - Pour shaker to shot: ~3 actions (grasp shaker, clean shot, pour)
  Minimum ~14 actions per cocktail.

Admissible Lower Bounds:
  Term 1 (Unserved cocktails): Count cocktails not yet in their target shots.
    Each unserved cocktail requires at least 1 action (pour-shaker-to-shot)
    to deliver, so this is a trivial lower bound. We multiply by a tighter
    per-cocktail minimum cost.

  Term 2 (Shaker serialization): Since there's one shaker, each unserved
    cocktail that hasn't been made yet requires a full make-cycle through
    the shaker. We estimate the minimum actions per cocktail cycle based
    on state analysis (is shaker clean? does it contain the right cocktail?).

  Term 3 (Task network analysis): For each abstract task remaining in the
    task network, estimate a state-aware minimum cost. Tasks that are already
    satisfied (null decomposition applicable) cost 0; others cost based on
    what state changes are needed.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic4(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_4"):
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
        
        # Domain info
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_cocktails = []  # list of (shot, cocktail) pairs from goals
        
        # Task name -> abstract task local_id mapping
        self.task_name_map = {}   # task_name_prefix -> list of abstract tasks
        
        # Shakers and shots
        self.shakers = []
        self.shots = []
        self.hands = []
        
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
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
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
        
        # Collect shakers, shots, hands from fact keys
        shaker_set = set()
        shot_set = set()
        hand_set = set()
        for key in self.shaked_ids:
            shaker_set.add(key)
        for key in self.hand_empty_ids:
            hand_set.add(key)
        for key in self.clean_ids:
            obj = key
            if obj in shaker_set:
                continue
            # Check if it's a shot by looking for contains entries
            shot_set.add(obj)
        # More robust: shots are objects that appear in contains but aren't shakers
        shot_set_refined = set()
        for (container, bev) in self.contains_ids:
            if container not in shaker_set:
                shot_set_refined.add(container)
        # Also add from clean_ids minus shakers
        for obj in self.clean_ids:
            if obj not in shaker_set:
                shot_set_refined.add(obj)
        
        self.shakers = list(shaker_set)
        self.shots = list(shot_set_refined)
        self.hands = list(hand_set)
        
        # Extract goal cocktail deliveries from the initial task network
        # The top-level tasks are AchieveContainsShotCocktail(shot, cocktail)
        self._extract_goal_cocktails(initial_node.task_network)
        
        # Build abstract task type lookup for fast classification
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            at._parsed_type = task_type
            at._parsed_args = args
        
        # Also parse operator names
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            op._parsed_type = task_type
            op._parsed_args = args
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)
    
    def _extract_goal_cocktails(self, task_network):
        """Extract (shot, cocktail) pairs from the initial task network."""
        self.goal_cocktails = []
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                task_type, args = self._parse_task_name(task.name)
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    self.goal_cocktails.append((args[0], args[1]))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _check_fact(self, state, global_id):
        """Check if a fact is true in state using bitwise operation."""
        return (state >> global_id) & 1

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        t1 = self._unserved_cocktail_cost(state, task_network)
        t2 = self._task_network_cost(state, task_network)
        t3 = self._shaker_serialization_cost(state, task_network)
        return max(t1, t2, t3)

    def _unserved_cocktail_cost(self, state, task_network):
        """Term 1: Count unserved cocktails and estimate minimum actions.
        
        Each unserved cocktail needs at minimum:
        - If cocktail not yet in shaker and not in shot: full cycle ~14 actions
        - If cocktail in shaker but not poured: ~3 actions  
        - If cocktail in shot: 0 actions
        
        Since there's one shaker, only one cocktail can be in-progress at a time.
        We use the sum of per-cocktail costs as this represents unavoidable work.
        But for admissibility with a single shaker, we note each cocktail must
        go through the shaker sequentially.
        """
        total_cost = 0
        
        for shot, cocktail in self.goal_cocktails:
            contains_key = (shot, cocktail)
            if contains_key in self.contains_ids:
                cid = self.contains_ids[contains_key]
                if self._check_fact(state, cid):
                    # Already served
                    continue
            
            # Cocktail not yet in target shot
            # Check if cocktail is in some shaker (ready to pour)
            cocktail_in_shaker = False
            shaker_is_shaked = False
            for shaker in self.shakers:
                sk = (shaker, cocktail)
                if sk in self.contains_ids:
                    sid = self.contains_ids[sk]
                    if self._check_fact(state, sid):
                        cocktail_in_shaker = True
                        if shaker in self.shaked_ids:
                            if self._check_fact(state, self.shaked_ids[shaker]):
                                shaker_is_shaked = True
                        break
            
            if cocktail_in_shaker and shaker_is_shaked:
                # Need to: clean shot if dirty, grasp shaker, pour
                shot_clean = False
                shot_empty = False
                if shot in self.clean_ids:
                    shot_clean = self._check_fact(state, self.clean_ids[shot])
                if shot in self.empty_ids:
                    shot_empty = self._check_fact(state, self.empty_ids[shot])
                
                cost = 1  # pour-shaker-to-shot
                if not shot_clean or not shot_empty:
                    cost += 3  # empty-shot + clean-shot + related grasps
                # Need to hold shaker
                shaker_held = False
                for hand in self.hands:
                    hk = (hand, self.shakers[0]) if self.shakers else None
                    if hk and hk in self.holding_ids:
                        if self._check_fact(state, self.holding_ids[hk]):
                            shaker_held = True
                            break
                if not shaker_held:
                    cost += 1  # grasp
                total_cost += cost
            elif cocktail_in_shaker and not shaker_is_shaked:
                # Need to shake + pour + clean shot
                total_cost += 5  # shake + pour + shot prep
            else:
                # Full cycle: need to make cocktail from scratch
                # Minimum: fill shot1 (4) + pour to shaker (1) + fill shot2 (4) + 
                #          pour to shaker (1) + shake (1) + pour to shot (1) +
                #          clean shaker beforehand (maybe 3) + clean shot (maybe 3)
                # Very conservative lower bound per cocktail: 10 actions
                total_cost += 10
        
        return total_cost

    def _task_network_cost(self, state, task_network):
        """Term 2: State-aware task network cost estimation.
        
        For each task in the network, estimate minimum actions needed
        based on current state.
        """
        total = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                cost = self._estimate_abstract_task_cost(state, task)
                total += cost
            else:  # Operator (primitive task)
                total += 1
        return total

    def _estimate_abstract_task_cost(self, state, task):
        """Estimate minimum cost for an abstract task given current state."""
        task_type = task._parsed_type if hasattr(task, '_parsed_type') else None
        args = task._parsed_args if hasattr(task, '_parsed_args') else []
        
        if task_type is None:
            return 1
        
        if task_type == "AchieveContainsShotCocktail":
            if len(args) >= 2:
                shot, cocktail = args[0], args[1]
                ck = (shot, cocktail)
                if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                    return 0  # Already done
                # Check if cocktail exists in shaker
                cocktail_ready = False
                for shaker in self.shakers:
                    sk = (shaker, cocktail)
                    if sk in self.contains_ids and self._check_fact(state, self.contains_ids[sk]):
                        if shaker in self.shaked_ids and self._check_fact(state, self.shaked_ids[shaker]):
                            cocktail_ready = True
                            break
                if cocktail_ready:
                    # Just need to clean shot + pour
                    shot_ok = (shot in self.clean_ids and self._check_fact(state, self.clean_ids[shot]) and
                               shot in self.empty_ids and self._check_fact(state, self.empty_ids[shot]))
                    if shot_ok:
                        return 2  # grasp shaker + pour
                    else:
                        return 6  # clean shot + grasp shaker + pour
                return 12  # Full make cycle
            return 10
        
        elif task_type == "AchieveContainsShakerCocktail":
            if len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                ck = (shaker, cocktail)
                if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                    return 0
                # Need to add 2 ingredients + shake = minimum ~9 actions
                # Check if shaker is clean
                shaker_clean = shaker in self.clean_ids and self._check_fact(state, self.clean_ids[shaker])
                shaker_empty = shaker in self.empty_ids and self._check_fact(state, self.empty_ids[shaker])
                base = 8  # 2 * (fill-shot + pour-to-shaker) + shake
                if not shaker_clean:
                    base += 3
                if not shaker_empty:
                    base += 2
                return base
            return 8
        
        elif task_type == "AchieveContainsShakerIngredient":
            if len(args) >= 2:
                shaker, ingredient = args[0], args[1]
                ck = (shaker, ingredient)
                if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                    return 0
                return 4  # fill-shot (3-4 actions) + pour to shaker (1)
            return 4
        
        elif task_type == "AchieveContainsShotIngredient":
            if len(args) >= 2:
                shot, ingredient = args[0], args[1]
                ck = (shot, ingredient)
                if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                    return 0
                # Need clean shot + hold + fill = ~4 actions
                shot_clean = shot in self.clean_ids and self._check_fact(state, self.clean_ids[shot])
                if shot_clean:
                    return 2  # grasp + fill
                return 4  # clean + grasp + fill
            return 3
        
        elif task_type == "AchieveCleanShaker":
            if len(args) >= 1:
                shaker = args[0]
                if shaker in self.clean_ids and self._check_fact(state, self.clean_ids[shaker]):
                    return 0
                shaker_empty = shaker in self.empty_ids and self._check_fact(state, self.empty_ids[shaker])
                if shaker_empty:
                    return 2  # grasp + clean
                return 4  # grasp + empty + clean
            return 3
        
        elif task_type == "AchieveCleanShot":
            if len(args) >= 1:
                shot = args[0]
                if shot in self.clean_ids and self._check_fact(state, self.clean_ids[shot]):
                    return 0
                shot_empty = shot in self.empty_ids and self._check_fact(state, self.empty_ids[shot])
                if shot_empty:
                    return 2  # grasp + clean
                return 3  # grasp + empty + clean
            return 2
        
        elif task_type == "AchieveHolding":
            if len(args) >= 2:
                hand, container = args[0], args[1]
                hk = (hand, container)
                if hk in self.holding_ids and self._check_fact(state, self.holding_ids[hk]):
                    return 0
                # Need hand empty + container on table + grasp
                hand_free = hand in self.hand_empty_ids and self._check_fact(state, self.hand_empty_ids[hand])
                on_table = container in self.ontable_ids and self._check_fact(state, self.ontable_ids[container])
                cost = 1  # grasp
                if not hand_free:
                    cost += 1  # drop
                if not on_table:
                    cost += 1  # someone else holding it, need to drop first
                return cost
            return 1
        
        elif task_type == "AchieveHandEmpty":
            if len(args) >= 1:
                hand = args[0]
                if hand in self.hand_empty_ids and self._check_fact(state, self.hand_empty_ids[hand]):
                    return 0
                return 1  # drop
            return 0
        
        elif task_type == "AchieveOnTable":
            if len(args) >= 1:
                container = args[0]
                if container in self.ontable_ids and self._check_fact(state, self.ontable_ids[container]):
                    return 0
                return 1  # drop
            return 0
        
        elif task_type == "DoPourShakerToShot":
            return 1  # single action
        
        # Unknown task type
        return 1

    def _shaker_serialization_cost(self, state, task_network):
        """Term 3: Shaker serialization cost.
        
        Count how many distinct cocktails still need to be made through the shaker.
        Each one requires a full clean-fill-fill-shake-pour cycle.
        The shaker can only process one cocktail at a time.
        
        This is admissible because these cycles cannot overlap.
        """
        # Count cocktails that need the shaker
        cocktails_needing_shaker = set()
        
        for task in task_network:
            if hasattr(task, '_parsed_type'):
                if task._parsed_type == "AchieveContainsShotCocktail":
                    args = task._parsed_args
                    if len(args) >= 2:
                        shot, cocktail = args[0], args[1]
                        ck = (shot, cocktail)
                        if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                            continue  # Already served
                        cocktails_needing_shaker.add(cocktail)
                elif task._parsed_type == "AchieveContainsShakerCocktail":
                    args = task._parsed_args
                    if len(args) >= 2:
                        shaker, cocktail = args[0], args[1]
                        ck = (shaker, cocktail)
                        if ck in self.contains_ids and self._check_fact(state, self.contains_ids[ck]):
                            continue
                        cocktails_needing_shaker.add(cocktail)
            elif hasattr(task, 'decompositions'):
                # Fallback for unparsed abstract tasks
                pass
        
        if not cocktails_needing_shaker:
            return 0
        
        n = len(cocktails_needing_shaker)
        
        # First cocktail: if shaker is clean and empty, we save cleaning cost
        # Subsequent cocktails each need: empty shaker (1) + clean shaker (1) + 
        #   fill shot with ing1 (3) + pour to shaker (1) + fill shot with ing2 (3) + 
        #   pour to shaker (1) + shake (1) + pour to shot (1) = 12 actions minimum
        # First cocktail saves empty+clean if shaker is ready = 10 actions
        
        first_shaker = self.shakers[0] if self.shakers else None
        if first_shaker is None:
            return n * 10
        
        shaker_clean = first_shaker in self.clean_ids and self._check_fact(state, self.clean_ids[first_shaker])
        shaker_empty = first_shaker in self.empty_ids and self._check_fact(state, self.empty_ids[first_shaker])
        
        # Check if shaker currently contains a needed cocktail
        current_cocktail_in_shaker = None
        for cocktail in cocktails_needing_shaker:
            sk = (first_shaker, cocktail)
            if sk in self.contains_ids and self._check_fact(state, self.contains_ids[sk]):
                current_cocktail_in_shaker = cocktail
                break
        
        if current_cocktail_in_shaker:
            # One cocktail is already in the shaker
            shaker_shaked = first_shaker in self.shaked_ids and self._check_fact(state, self.shaked_ids[first_shaker])
            if shaker_shaked:
                first_cost = 2  # Just need to pour to shot
            else:
                first_cost = 3  # shake + pour
            remaining = n - 1
            # Each remaining cocktail: clean shaker (3) + make cocktail (8) + pour (2) = ~13
            total = first_cost + remaining * 12
        else:
            if shaker_clean and shaker_empty:
                first_cost = 9  # fill+pour+fill+pour+shake+pour+grasp stuff
            elif shaker_empty:
                first_cost = 11  # clean + above
            else:
                first_cost = 12  # empty + clean + above  
            remaining = n - 1
            total = first_cost + remaining * 12
        
        return total
