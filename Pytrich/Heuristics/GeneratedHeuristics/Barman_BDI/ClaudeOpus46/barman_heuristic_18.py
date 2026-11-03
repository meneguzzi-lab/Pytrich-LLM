"""
Barman BDI Domain Heuristic

Bottleneck: The single shaker serializes all cocktail production. Every cocktail
must pass through: clean_shaker -> add_ingredient1 -> add_ingredient2 -> shake -> pour_to_shot.
This means cocktails are made one at a time, and each requires ~14+ primitive actions.

Admissible lower bounds:
1. Unserved cocktail bound: Each cocktail not yet in its target shot requires at minimum
   ~14 actions (2x fill-shot, 2x pour-to-shaker, shake, pour-shaker-to-shot, plus grasp/drop).
   Since the shaker serializes, these are additive. This is admissible because no cocktail
   can be made in fewer steps.
2. Shaker state penalty: If the shaker currently contains something or is dirty,
   additional cleanup actions are needed before the next cocktail can be made.
3. Shot preparation bound: Each target shot that isn't clean+empty needs cleaning actions.

We return max of individual terms plus additive sequential shaker cost.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic18(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_18"):
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
        self.cocktail_part1 = {}  # cocktail -> ingredient
        self.cocktail_part2 = {}  # cocktail -> ingredient
        self.dispenses_map = {}   # ingredient -> dispenser
        
        # Goal tracking
        self.goal_cocktail_shots = []  # List of (shot, cocktail) pairs from goals
        
        # Task type mappings
        self.task_type_map = {}   # abstract_task global_id -> task_type string
        
        # All shakers and shots
        self.shakers = set()
        self.shots = set()
        self.hands = set()
        self.cocktails = set()
        self.ingredients = set()
        
        # Minimum actions per cocktail production (sequential through shaker)
        # clean_shaker(if needed) + fill_shot_ing1(~4: clean_shot+grasp+fill) +
        # grasp_shot + pour_to_shaker + fill_shot_ing2(~4) + grasp_shot + pour_to_shaker +
        # grasp_shaker + shake + pour_shaker_to_shot + various drops
        # Conservative minimum: ~14 actions per cocktail
        self.MIN_ACTIONS_PER_COCKTAIL = 14
        # If shaker needs cleaning first (already contains something)
        self.SHAKER_CLEAN_COST = 4  # grasp + empty + drop(other hand free) + clean
        # Cost to clean a dirty/used shot
        self.SHOT_CLEAN_COST = 3  # grasp + empty(if full) + clean

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
        """Parse a task name like __AchieveContainsShotCocktail[shot2,cocktail1]"""
        name = name.strip()
        # Remove leading underscores
        clean_name = name.lstrip('_')
        bracket_idx = clean_name.find("[")
        if bracket_idx > 0 and clean_name.endswith("]"):
            task_type = clean_name[:bracket_idx]
            args_str = clean_name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return clean_name, []

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
                self.shakers.add(args[0])
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_ids[args[0]] = f.global_id
                self.shakers.add(args[0])
            elif pred == "used" and len(args) == 2:
                self.used_ids[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                self.cocktail_part1[args[0]] = args[1]
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
            elif pred == "dispenses" and len(args) == 2:
                self.dispenses_map[args[1]] = args[0]

        # Collect hands and shots from holding facts
        for (hand, container) in self.holding_ids:
            self.hands.add(hand)
            if container.startswith("shot"):
                self.shots.add(container)
        for container in self.ontable_ids:
            if container.startswith("shot"):
                self.shots.add(container)
        for container in self.clean_ids:
            if container.startswith("shot"):
                self.shots.add(container)
        for hand in self.hand_empty_ids:
            self.hands.add(hand)

        # Build abstract task type map
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)

        # Extract goal cocktail-shot pairs from the initial task network
        # The top-level tasks are AchieveContainsShotCocktail tasks
        self._extract_goal_pairs(initial_node.task_network)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goal_pairs(self, task_network):
        """Extract (shot, cocktail) goal pairs from the task network."""
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                task_type, args = self.task_type_map.get(task.global_id, (None, []))
                if task_type and 'AchieveContainsShotCocktail' in task_type and len(args) >= 2:
                    self.goal_cocktail_shots.append((args[0], args[1]))

    def _check_fact(self, state, global_id):
        """Fast bitwise check if fact is true."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _count_unserved_cocktails(self, state):
        """Count cocktails not yet in their target shots."""
        count = 0
        for shot, cocktail in self.goal_cocktail_shots:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is None or not self._check_fact(state, cid):
                count += 1
        return count

    def _shaker_state_cost(self, state):
        """Estimate additional cost due to current shaker state."""
        cost = 0
        for shaker in self.shakers:
            # If shaker is not clean and not empty, it needs emptying + cleaning
            clean_id = self.clean_ids.get(shaker)
            empty_id = self.empty_ids.get(shaker)
            shaked_id = self.shaked_ids.get(shaker)
            
            is_clean = self._check_fact(state, clean_id) if clean_id is not None else False
            is_empty = self._check_fact(state, empty_id) if empty_id is not None else False
            is_shaked = self._check_fact(state, shaked_id) if shaked_id is not None else False
            
            if not is_clean:
                if not is_empty:
                    # Need to empty then clean: grasp + empty + hand_empty + clean = 4
                    if is_shaked:
                        cost += 4  # empty-shaker + clean-shaker + grasp/drop overhead
                    else:
                        # Unshaked with contents - unusual but possible
                        cost += 4
                elif is_empty:
                    # Empty but not clean: grasp + clean = 3 (grasp + handEmpty other + clean)
                    cost += 3
        return cost

    def _shot_preparation_cost(self, state):
        """Estimate cost to prepare target shots (clean + empty them if needed)."""
        cost = 0
        for shot, cocktail in self.goal_cocktail_shots:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is not None and self._check_fact(state, cid):
                continue  # Already served
            
            clean_id = self.clean_ids.get(shot)
            empty_id = self.empty_ids.get(shot)
            
            is_clean = self._check_fact(state, clean_id) if clean_id is not None else False
            is_empty = self._check_fact(state, empty_id) if empty_id is not None else False
            
            if not is_clean:
                if not is_empty:
                    # Shot has contents and is dirty: empty + clean = ~4 actions
                    cost += 4
                else:
                    # Empty but used/dirty: clean = ~3 actions  
                    cost += 3
        return cost

    def _task_network_cost(self, task_network):
        """Estimate minimum cost from remaining task network structure."""
        cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                gid = task.global_id
                task_info = self.task_type_map.get(gid, (None, []))
                task_type = task_info[0] if task_info else None
                
                if task_type is None:
                    cost += 1
                elif 'AchieveContainsShotCocktail' in task_type:
                    # This is a top-level goal task - major cost
                    cost += self.MIN_ACTIONS_PER_COCKTAIL
                elif 'AchieveContainsShakerCocktail' in task_type:
                    # Making a cocktail in shaker: ~10 actions
                    cost += 10
                elif 'AchieveContainsShakerIngredient' in task_type:
                    # Adding ingredient to shaker: ~5 actions
                    cost += 5
                elif 'AchieveContainsShotIngredient' in task_type:
                    # Filling a shot with ingredient: ~4 actions
                    cost += 4
                elif 'AchieveCleanShaker' in task_type:
                    cost += 3
                elif 'AchieveCleanShot' in task_type:
                    cost += 2
                elif 'AchieveHolding' in task_type:
                    cost += 2
                elif 'AchieveHandEmpty' in task_type:
                    cost += 1
                elif 'AchieveOnTable' in task_type:
                    cost += 1
                elif 'DoPourShakerToShot' in task_type:
                    cost += 1
                else:
                    cost += 1
            else:
                # Primitive operator - costs exactly 1
                cost += 1
        return cost

    def _state_aware_task_cost(self, state, task_network):
        """Compute state-aware cost for each task in the network."""
        cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):  # AbstractTask
                gid = task.global_id
                task_info = self.task_type_map.get(gid, (None, []))
                task_type = task_info[0] if task_info else None
                args = task_info[1] if task_info else []
                
                if task_type is None:
                    cost += 1
                    continue
                
                if 'AchieveContainsShotCocktail' in task_type and len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    cid = self.contains_ids.get((shot, cocktail))
                    if cid is not None and self._check_fact(state, cid):
                        # Already done - null decomposition
                        cost += 0
                    else:
                        # Need to make the cocktail
                        # Check if shaker already has this cocktail
                        shaker_has = False
                        for shaker in self.shakers:
                            scid = self.contains_ids.get((shaker, cocktail))
                            if scid is not None and self._check_fact(state, scid):
                                shaker_has = True
                                break
                        
                        if shaker_has:
                            # Just need to pour: clean_shot + hold_shaker + pour = ~5
                            shot_clean = self.clean_ids.get(shot)
                            shot_empty = self.empty_ids.get(shot)
                            is_clean = self._check_fact(state, shot_clean) if shot_clean else False
                            is_empty = self._check_fact(state, shot_empty) if shot_empty else False
                            
                            base = 1  # pour action
                            if not is_clean:
                                base += 3 if is_empty else 4
                            if not is_empty and is_clean:
                                base += 2  # shouldn't happen but safety
                            base += 2  # grasp shaker overhead
                            cost += base
                        else:
                            # Full pipeline needed
                            # Check shaker state for cleanup cost
                            pipeline_cost = self.MIN_ACTIONS_PER_COCKTAIL
                            
                            # Add shaker cleanup if needed
                            for shaker in self.shakers:
                                s_clean = self.clean_ids.get(shaker)
                                s_empty = self.empty_ids.get(shaker)
                                is_s_clean = self._check_fact(state, s_clean) if s_clean else False
                                is_s_empty = self._check_fact(state, s_empty) if s_empty else False
                                if not is_s_clean or not is_s_empty:
                                    pipeline_cost += 2
                                break  # Only one shaker typically
                            
                            cost += pipeline_cost
                
                elif 'AchieveContainsShakerCocktail' in task_type and len(args) >= 2:
                    shaker, cocktail = args[0], args[1]
                    cid = self.contains_ids.get((shaker, cocktail))
                    if cid is not None and self._check_fact(state, cid):
                        cost += 0
                    else:
                        cost += 10
                
                elif 'AchieveContainsShakerIngredient' in task_type and len(args) >= 2:
                    shaker, ingredient = args[0], args[1]
                    cid = self.contains_ids.get((shaker, ingredient))
                    if cid is not None and self._check_fact(state, cid):
                        cost += 0
                    else:
                        cost += 5
                
                elif 'AchieveContainsShotIngredient' in task_type and len(args) >= 2:
                    shot, ingredient = args[0], args[1]
                    cid = self.contains_ids.get((shot, ingredient))
                    if cid is not None and self._check_fact(state, cid):
                        cost += 0
                    else:
                        cost += 4
                
                elif 'AchieveCleanShaker' in task_type and len(args) >= 1:
                    shaker = args[0]
                    cid = self.clean_ids.get(shaker)
                    if cid is not None and self._check_fact(state, cid):
                        cost += 0
                    else:
                        eid = self.empty_ids.get(shaker)
                        is_empty = self._check_fact(state, eid) if eid else False
                        cost += 3 if is_empty else 5
                
                elif 'AchieveCleanShot' in task_type and len(args) >= 1:
                    shot = args[0]
                    cid = self.clean_ids.get(shot)
                    if cid is not None and self._check_fact(state, cid):
                        cost += 0
                    else:
                        eid = self.empty_ids.get(shot)
                        is_empty = self._check_fact(state, eid) if eid else False
                        cost += 3 if is_empty else 4
                
                elif 'AchieveHolding' in task_type and len(args) >= 2:
                    hand, container = args[0], args[1]
                    hid = self.holding_ids.get((hand, container))
                    if hid is not None and self._check_fact(state, hid):
                        cost += 0
                    else:
                        # Need to: empty hand (maybe) + put container on table (maybe) + grasp
                        he_id = self.hand_empty_ids.get(hand)
                        is_hand_empty = self._check_fact(state, he_id) if he_id else False
                        ot_id = self.ontable_ids.get(container)
                        is_ontable = self._check_fact(state, ot_id) if ot_id else False
                        
                        c = 1  # grasp
                        if not is_hand_empty:
                            c += 1  # drop
                        if not is_ontable:
                            c += 1  # someone else needs to drop it
                        cost += c
                
                elif 'AchieveHandEmpty' in task_type and len(args) >= 1:
                    hand = args[0]
                    he_id = self.hand_empty_ids.get(hand)
                    if he_id is not None and self._check_fact(state, he_id):
                        cost += 0
                    else:
                        cost += 1  # drop
                
                elif 'AchieveOnTable' in task_type and len(args) >= 1:
                    container = args[0]
                    ot_id = self.ontable_ids.get(container)
                    if ot_id is not None and self._check_fact(state, ot_id):
                        cost += 0
                    else:
                        cost += 1  # drop
                
                elif 'DoPourShakerToShot' in task_type:
                    cost += 1  # single action
                
                else:
                    cost += 1
            else:
                # Primitive operator
                cost += 1
        return cost

    def _compute(self, state, task_network):
        """Compute the heuristic value."""
        if len(task_network) == 0:
            return 0
        
        # Term 1: State-aware task network cost (most informative)
        term1 = self._state_aware_task_cost(state, task_network)
        
        # Term 2: Unserved cocktails * minimum sequential cost through shaker
        # This captures the serialization bottleneck
        unserved = self._count_unserved_cocktails(state)
        # Each unserved cocktail needs at minimum the full shaker pipeline
        term2 = unserved * self.MIN_ACTIONS_PER_COCKTAIL
        
        # Term 3: Current shaker overhead + shot preparation
        term3 = 0
        if unserved > 0:
            term3 = self._shaker_state_cost(state) + self._shot_preparation_cost(state)
            # Add the sequential cocktail cost minus the overhead we already counted
            term3 += (unserved - 1) * self.MIN_ACTIONS_PER_COCKTAIL
        
        # Return max of the terms for admissibility
        h = max(term1, term2, term3)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
