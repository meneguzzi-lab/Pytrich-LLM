"""Domain-Dependent Heuristic for Barman_BDI HTN Planning

Bottleneck Analysis:
  The single shaker is the serializing bottleneck. Every cocktail must be mixed
  in the shaker (clean shaker -> add ingredient1 -> add ingredient2 -> shake -> pour).
  The shaker can only hold one cocktail at a time, so cocktails are fully serialized.
  Each cocktail requires: clean shaker + fill 2 shots with ingredients + pour each
  to shaker + shake + pour to target shot + clean shaker for next cocktail.

Admissible Lower Bounds:
  Term 1 - Remaining cocktails to serve: Each unserved cocktail requires at minimum
    ~14 primitive actions (grasp shot, fill, pour to shaker x2, shake, pour to shot,
    clean shaker). We use a conservative per-cocktail cost.
  Term 2 - Shaker usage count: Since there's one shaker and each cocktail needs it,
    the number of distinct cocktails still needing the shaker * min shaker cycle cost
    is a lower bound.
  Term 3 - Task network structural lower bound: Count remaining abstract tasks
    weighted by their minimum decomposition depth/cost using state-aware estimation.

Tie-breaking: Shots are interchangeable for intermediate use; we don't add symmetry
penalties since the HTN structure already binds specific shots to cocktails.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic11(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_11"):
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
        
        # Domain structure
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_cocktails = {}  # shot -> cocktail (from goal tasks)
        
        # Task name -> type mapping
        self.task_type_map = {}   # abstract_task global_id -> task_type string
        
        # Operator/AbstractTask identification
        self.abstract_task_ids = set()
        self.operator_ids = set()

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
        # No brackets
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

        # Build abstract task type map
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)
            self.abstract_task_ids.add(at.global_id)
        
        for op in model.operators:
            self.operator_ids.add(op.global_id)

        # Identify all shakers (usually just shaker1)
        self.shakers = set()
        for key in self.shaked_ids:
            self.shakers.add(key)
        
        # Identify all shots
        self.shots = set()
        for key in self.clean_ids:
            if key.startswith("shot"):
                self.shots.add(key)
        
        # Identify hands
        self.hands = []
        for key in self.hand_empty_ids:
            self.hands.append(key)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _fact_true(self, state, global_id):
        return (state >> global_id) & 1

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0
        
        # Term 1: State-aware per-task cost estimation
        term1 = self._task_network_cost(state, task_network)
        
        # Term 2: Unserved cocktails lower bound
        term2 = self._unserved_cocktails_cost(state, task_network)
        
        # Term 3: Structural task count with weights
        term3 = self._weighted_task_count(state, task_network)
        
        return max(term1, term2, term3)

    def _task_network_cost(self, state, task_network):
        """State-aware per-task cost estimation.
        For each task in the network, estimate minimum actions needed."""
        total_cost = 0
        
        for task in task_network:
            gid = task.global_id
            
            if gid in self.operator_ids:
                # Primitive task = 1 action
                total_cost += 1
            elif gid in self.abstract_task_ids:
                info = self.task_type_map.get(gid)
                if info is None:
                    total_cost += 1
                    continue
                task_type, args = info
                cost = self._estimate_abstract_task_cost(state, task_type, args)
                total_cost += cost
        
        return total_cost

    def _estimate_abstract_task_cost(self, state, task_type, args):
        """Estimate minimum cost for an abstract task given current state."""
        
        if task_type == "AchieveContainsShotCocktail":
            if len(args) < 2:
                return 1
            shot, cocktail = args[0], args[1]
            # Check if already contains
            cid = self.contains_ids.get((shot, cocktail))
            if cid is not None and self._fact_true(state, cid):
                return 0
            # Need to: make cocktail in shaker + clean shot + pour
            # Making cocktail: clean shaker(~3) + 2x(fill shot with ingredient(~4) + pour to shaker(~3)) + shake(1) = ~18
            # Clean shot: 0-4 depending on state
            # Pour shaker to shot: ~1 (already holding shaker)
            # Total minimum: ~14 actions for a fresh cocktail
            cost = 0
            
            # Cost to make cocktail in shaker
            parts = self.cocktail_parts.get(cocktail)
            if parts:
                # Check if shaker already contains this cocktail
                shaker = list(self.shakers)[0] if self.shakers else None
                if shaker:
                    shaker_has_cocktail = False
                    scid = self.contains_ids.get((shaker, cocktail))
                    if scid is not None and self._fact_true(state, scid):
                        shaker_has_cocktail = True
                    
                    if shaker_has_cocktail:
                        # Just need to pour to shot
                        cost += 1  # pour-shaker-to-shot
                        # May need to clean shot
                        cost += self._clean_shot_cost(state, shot)
                        # May need to pick up shaker
                        cost += self._holding_cost(state, shaker)
                    else:
                        # Need full cocktail making process
                        # Clean shaker cost
                        cost += self._clean_shaker_cost(state, shaker)
                        # Add 2 ingredients to shaker (each needs: fill shot + pour to shaker)
                        for ing in parts:
                            if ing is not None:
                                cost += self._add_ingredient_to_shaker_cost(state, shaker, ing)
                        # Shake: need to hold shaker + have other hand empty + 1 action
                        cost += 2  # grasp shaker + shake
                        # Pour to shot
                        cost += self._clean_shot_cost(state, shot)
                        cost += 1  # pour-shaker-to-shot
                else:
                    cost = 14
            else:
                cost = 14
            return max(cost, 1)

        elif task_type == "AchieveContainsShakerCocktail":
            if len(args) < 2:
                return 1
            shaker, cocktail = args[0], args[1]
            cid = self.contains_ids.get((shaker, cocktail))
            if cid is not None and self._fact_true(state, cid):
                return 0
            # Need: clean shaker + add 2 ingredients + shake
            cost = self._clean_shaker_cost(state, shaker)
            parts = self.cocktail_parts.get(cocktail)
            if parts:
                for ing in parts:
                    if ing is not None:
                        cost += self._add_ingredient_to_shaker_cost(state, shaker, ing)
            cost += 2  # hold shaker + shake
            return max(cost, 1)

        elif task_type == "AchieveContainsShakerIngredient":
            if len(args) < 2:
                return 1
            shaker, ingredient = args[0], args[1]
            cid = self.contains_ids.get((shaker, ingredient))
            if cid is not None and self._fact_true(state, cid):
                return 0
            return self._add_ingredient_to_shaker_cost(state, shaker, ingredient)

        elif task_type == "AchieveContainsShotIngredient":
            if len(args) < 2:
                return 1
            shot, ingredient = args[0], args[1]
            cid = self.contains_ids.get((shot, ingredient))
            if cid is not None and self._fact_true(state, cid):
                return 0
            # Need: clean shot + hold shot + fill
            cost = self._clean_shot_cost(state, shot)
            cost += 2  # grasp + fill-shot
            return cost

        elif task_type == "AchieveCleanShot":
            if len(args) < 1:
                return 1
            shot = args[0]
            return self._clean_shot_cost(state, shot)

        elif task_type == "AchieveCleanShaker":
            if len(args) < 1:
                return 1
            shaker = args[0]
            return self._clean_shaker_cost(state, shaker)

        elif task_type == "AchieveHolding":
            if len(args) < 2:
                return 1
            hand, container = args[0], args[1]
            hid = self.holding_ids.get((hand, container))
            if hid is not None and self._fact_true(state, hid):
                return 0
            return self._holding_cost_hand(state, hand, container)

        elif task_type == "AchieveHandEmpty":
            if len(args) < 1:
                return 1
            hand = args[0]
            heid = self.hand_empty_ids.get(hand)
            if heid is not None and self._fact_true(state, heid):
                return 0
            return 1  # drop

        elif task_type == "AchieveOnTable":
            if len(args) < 1:
                return 1
            container = args[0]
            otid = self.ontable_ids.get(container)
            if otid is not None and self._fact_true(state, otid):
                return 0
            return 1  # drop

        elif task_type == "DoPourShakerToShot":
            return 1  # pour-shaker-to-shot action

        return 1

    def _clean_shot_cost(self, state, shot):
        """Estimate cost to clean a shot glass."""
        cid = self.clean_ids.get(shot)
        if cid is not None and self._fact_true(state, cid):
            return 0
        
        eid = self.empty_ids.get(shot)
        is_empty = eid is not None and self._fact_true(state, eid)
        
        if is_empty:
            # Empty but dirty: grasp + clean-shot = 2 (need other hand empty too)
            return 2
        else:
            # Full: grasp + empty-shot + clean-shot = 3
            return 3

    def _clean_shaker_cost(self, state, shaker):
        """Estimate cost to clean a shaker."""
        cid = self.clean_ids.get(shaker)
        if cid is not None and self._fact_true(state, cid):
            return 0
        
        eid = self.empty_ids.get(shaker)
        is_empty = eid is not None and self._fact_true(state, eid)
        
        if is_empty:
            # Empty but dirty: grasp + clean-shaker = 2
            return 2
        else:
            # Full: grasp + empty-shaker + clean-shaker = 3
            return 3

    def _add_ingredient_to_shaker_cost(self, state, shaker, ingredient):
        """Cost to add an ingredient to the shaker."""
        cid = self.contains_ids.get((shaker, ingredient))
        if cid is not None and self._fact_true(state, cid):
            return 0
        # Need: fill a shot with ingredient (clean shot + grasp + fill = ~3-4)
        # + grasp shot + pour to shaker = ~1
        # Minimum: find a clean empty shot = 0 cost, grasp(1) + fill(1) + pour(1) = 3
        # If no clean shot available, add cleaning cost
        min_shot_cost = 3  # best case: grasp clean empty shot, fill, pour
        
        # Check if any shot is clean and empty
        found_clean_empty = False
        for shot in self.shots:
            clean_id = self.clean_ids.get(shot)
            empty_id = self.empty_ids.get(shot)
            if (clean_id is not None and self._fact_true(state, clean_id) and
                empty_id is not None and self._fact_true(state, empty_id)):
                found_clean_empty = True
                break
        
        if not found_clean_empty:
            min_shot_cost += 2  # Need to clean a shot first
        
        return min_shot_cost

    def _holding_cost(self, state, container):
        """Cost to hold a container in any hand."""
        for hand in self.hands:
            hid = self.holding_ids.get((hand, container))
            if hid is not None and self._fact_true(state, hid):
                return 0
        # Need: ensure hand empty (0-1) + ensure on table (0-1) + grasp (1)
        # Best case: hand empty, on table = 1
        otid = self.ontable_ids.get(container)
        on_table = otid is not None and self._fact_true(state, otid)
        
        any_hand_empty = False
        for hand in self.hands:
            heid = self.hand_empty_ids.get(hand)
            if heid is not None and self._fact_true(state, heid):
                any_hand_empty = True
                break
        
        cost = 1  # grasp
        if not on_table:
            cost += 1  # drop from whoever holds it first... but someone else holds it
            # Actually if not on table, it's being held by some hand
            # We need a different hand to be free, or drop it and re-grasp
            # For simplicity: it's being held, so 0 extra if we want that hand to hold it
            # But we might want a specific hand. Conservative: 1
        if not any_hand_empty:
            cost += 1  # drop something
        
        return cost

    def _holding_cost_hand(self, state, hand, container):
        """Cost to have a specific hand hold a specific container."""
        hid = self.holding_ids.get((hand, container))
        if hid is not None and self._fact_true(state, hid):
            return 0
        
        cost = 1  # grasp
        
        # Check if hand is empty
        heid = self.hand_empty_ids.get(hand)
        hand_empty = heid is not None and self._fact_true(state, heid)
        if not hand_empty:
            cost += 1  # drop
        
        # Check if container is on table
        otid = self.ontable_ids.get(container)
        on_table = otid is not None and self._fact_true(state, otid)
        if not on_table:
            cost += 1  # need to put it down first (someone else is holding it)
        
        return cost

    def _unserved_cocktails_cost(self, state, task_network):
        """Count unserved cocktails and multiply by minimum per-cocktail cost.
        Each cocktail that hasn't been served requires at minimum:
        - If shaker needs cleaning + making cocktail: ~12 actions minimum
        - Conservative estimate: 10 actions per unserved cocktail
        """
        unserved = 0
        # Count AchieveContainsShotCocktail tasks that aren't already satisfied
        for task in task_network:
            gid = task.global_id
            if gid not in self.abstract_task_ids:
                continue
            info = self.task_type_map.get(gid)
            if info is None:
                continue
            task_type, args = info
            if task_type == "AchieveContainsShotCocktail" and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is None or not self._fact_true(state, cid):
                    unserved += 1
        
        if unserved == 0:
            return 0
        
        # Each cocktail needs the shaker. After the first, the shaker must be
        # cleaned (3 actions) before making the next cocktail.
        # First cocktail: ~11 actions (fill 2 shots, pour 2 to shaker, shake, pour to shot)
        # Each subsequent: ~14 actions (clean shaker + same)
        # Conservative lower bound:
        if unserved == 1:
            return 8
        else:
            return 8 + (unserved - 1) * 11

    def _weighted_task_count(self, state, task_network):
        """Weighted count of remaining tasks."""
        cost = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.operator_ids:
                cost += 1
            elif gid in self.abstract_task_ids:
                info = self.task_type_map.get(gid)
                if info is None:
                    cost += 1
                    continue
                task_type, args = info
                # Minimum actions for each task type (conservative)
                if task_type == "AchieveContainsShotCocktail":
                    if len(args) >= 2:
                        cid = self.contains_ids.get((args[0], args[1]))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    cost += 10
                elif task_type == "AchieveContainsShakerCocktail":
                    if len(args) >= 2:
                        cid = self.contains_ids.get((args[0], args[1]))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    cost += 7
                elif task_type == "AchieveContainsShakerIngredient":
                    if len(args) >= 2:
                        cid = self.contains_ids.get((args[0], args[1]))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    cost += 3
                elif task_type == "AchieveContainsShotIngredient":
                    if len(args) >= 2:
                        cid = self.contains_ids.get((args[0], args[1]))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    cost += 3
                elif task_type == "AchieveCleanShot":
                    if len(args) >= 1:
                        c = self._clean_shot_cost(state, args[0])
                        cost += c
                    else:
                        cost += 2
                elif task_type == "AchieveCleanShaker":
                    if len(args) >= 1:
                        c = self._clean_shaker_cost(state, args[0])
                        cost += c
                    else:
                        cost += 2
                elif task_type == "AchieveHolding":
                    cost += 1
                elif task_type == "AchieveHandEmpty":
                    if len(args) >= 1:
                        heid = self.hand_empty_ids.get(args[0])
                        if heid is not None and self._fact_true(state, heid):
                            continue
                    cost += 1
                elif task_type == "AchieveOnTable":
                    if len(args) >= 1:
                        otid = self.ontable_ids.get(args[0])
                        if otid is not None and self._fact_true(state, otid):
                            continue
                    cost += 1
                elif task_type == "DoPourShakerToShot":
                    cost += 1
                else:
                    cost += 1
        return cost
