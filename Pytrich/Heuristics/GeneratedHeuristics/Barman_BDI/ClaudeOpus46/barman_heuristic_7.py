"""Domain-dependent heuristic for Barman_BDI domain.

Bottleneck Analysis:
  The single shaker is the critical serializing resource. Every cocktail must be
  made in the shaker (clean shaker -> add ingredient1 -> add ingredient2 -> shake
  -> pour to shot -> clean shaker for next cocktail). This means each cocktail
  requires a full shaker cycle, and cocktails must be made sequentially through
  the shaker.

Admissible Lower Bounds:
  Term 1 - Per-cocktail production cost:
    Each unfinished cocktail goal (shot doesn't contain the cocktail) requires at
    minimum: fill-shot with ingredient1 (needs clean shot + grasp + fill = ~3-4 ops),
    pour-shot-to-shaker (~1 op), fill-shot with ingredient2 (~3-4 ops),
    pour-shot-to-shaker (~1 op), shake (~1 op), pour-shaker-to-shot (~1 op).
    Conservative lower bound: ~7 actions per unfinished cocktail for the shaker
    cycle alone. We use a tighter per-cocktail estimate based on current state.

  Term 2 - Shaker serialization bound:
    Since there's only one shaker and it must be cleaned between cocktails,
    each cocktail after the first requires at least: empty-shaker + clean-shaker
    + the full production cycle. This gives a sequential lower bound.

  Term 3 - Task network depth estimate:
    Count remaining abstract tasks weighted by their minimum decomposition depth,
    adjusted by current state to avoid flat +1 costing.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic7(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_7"):
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
        self.goal_cocktails = []  # list of (shot, cocktail) goals
        self.shots = set()
        self.shakers = set()
        self.hands = set()
        self.ingredients = set()
        self.cocktails = set()

        # Abstract task name -> id mapping
        self.abstract_task_map = {}

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
        """Parse a grounded task/operator name to get the base name and arguments."""
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            base = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return base, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            gid = f.global_id

            if pred == "contains" and len(args) == 2:
                self.contains_ids[(args[0], args[1])] = gid
            elif pred == "clean" and len(args) == 1:
                self.clean_ids[args[0]] = gid
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = gid
            elif pred == "holding" and len(args) == 2:
                self.holding_ids[(args[0], args[1])] = gid
            elif pred == "handEmpty" and len(args) == 1:
                self.hand_empty_ids[args[0]] = gid
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = gid
            elif pred == "shaked" and len(args) == 1:
                self.shaked_ids[args[0]] = gid
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_ids[args[0]] = gid
            elif pred == "used" and len(args) == 2:
                self.used_ids[(args[0], args[1])] = gid
            elif pred == "cocktailPart1" and len(args) == 2:
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ing
                self.cocktails.add(cocktail)
                self.ingredients.add(ing)
            elif pred == "cocktailPart2" and len(args) == 2:
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ing
                self.cocktails.add(cocktail)
                self.ingredients.add(ing)
            elif pred == "dispenses" and len(args) == 2:
                self.dispenser_map[args[1]] = args[0]
            elif pred == "shakerLevel" and len(args) == 2:
                self.shaker_level_ids[(args[0], args[1])] = gid
            elif pred == "shakerEmptyLevel" and len(args) == 2:
                self.shaker_empty_level_ids[(args[0], args[1])] = gid

        # Identify objects by type from fact patterns
        for (hand, container) in self.holding_ids:
            self.hands.add(hand)
        for hand in self.hand_empty_ids:
            self.hands.add(hand)
        for shaker in self.shaked_ids:
            self.shakers.add(shaker)
        for container in self.clean_ids:
            if container.startswith("shot"):
                self.shots.add(container)
            elif container.startswith("shaker"):
                self.shakers.add(container)

        # Parse goal facts to find cocktail goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "contains" and len(args) == 2:
                shot, cocktail = args[0], args[1]
                self.goal_cocktails.append((shot, cocktail))

        # Map abstract tasks by name prefix for quick identification
        for at in model.abstract_tasks:
            base, args = self._parse_task_name(at.name)
            self.abstract_task_map[at.global_id] = (base, args)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _bit_check(self, state, gid):
        """Check if fact with global_id gid is true in state."""
        if gid is None:
            return False
        return (state >> gid) & 1

    def _estimate_cocktail_cost(self, state, shot, cocktail):
        """Estimate the minimum number of actions to achieve contains(shot, cocktail)
        given the current state. This accounts for:
        - Whether the cocktail is already in the shaker
        - Whether the shaker needs cleaning
        - Whether shots need cleaning for intermediate ingredient holding
        - Whether hands are free
        """
        # Already done
        cid = self.contains_ids.get((shot, cocktail))
        if cid is not None and self._bit_check(state, cid):
            return 0

        cost = 0
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 0

        ing1, ing2 = parts[0], parts[1]

        # Check if cocktail is already in any shaker (shaked and contains)
        cocktail_in_shaker = False
        for shaker in self.shakers:
            sid = self.contains_ids.get((shaker, cocktail))
            shaked_id = self.shaked_ids.get(shaker)
            if sid is not None and shaked_id is not None:
                if self._bit_check(state, sid) and self._bit_check(state, shaked_id):
                    cocktail_in_shaker = True
                    break

        if cocktail_in_shaker:
            # Need: clean shot, pour shaker to shot
            # Clean shot cost
            shot_clean = self.clean_ids.get(shot)
            shot_empty = self.empty_ids.get(shot)
            if shot_clean is not None and self._bit_check(state, shot_clean):
                if shot_empty is not None and self._bit_check(state, shot_empty):
                    # Shot is clean and empty, just need to pour
                    cost += 1  # pour-shaker-to-shot
                else:
                    # Shot is clean but not empty - empty it first
                    cost += 2  # grasp + empty-shot + pour
            else:
                # Shot needs cleaning
                shot_is_empty = shot_empty is not None and self._bit_check(state, shot_empty)
                if shot_is_empty:
                    # empty but dirty: grasp + clean-shot + pour
                    cost += 3
                else:
                    # full and dirty: grasp + empty-shot + clean-shot + pour
                    cost += 4
            return cost

        # Full cocktail production cycle
        # Need to: prepare shaker, add 2 ingredients, shake, pour to shot

        # Step 1: Clean shaker if needed
        for shaker in self.shakers:
            shaker_clean = self.clean_ids.get(shaker)
            shaker_empty = self.empty_ids.get(shaker)
            if shaker_clean is not None and self._bit_check(state, shaker_clean):
                if shaker_empty is not None and self._bit_check(state, shaker_empty):
                    pass  # Shaker ready, no cost
                else:
                    cost += 2  # Need to empty then clean
            else:
                shaker_is_empty = shaker_empty is not None and self._bit_check(state, shaker_empty)
                if shaker_is_empty:
                    cost += 2  # grasp + clean-shaker
                else:
                    cost += 4  # grasp + empty-shaker + clean-shaker (+ hand management)

        # Step 2: Add ingredient 1 to shaker via a shot
        # Need: clean shot, fill shot with ingredient, pour to shaker
        # For each ingredient: fill-shot (1) + pour-shot-to-shaker (1) = 2 minimum
        # But also need: grasp shot (1), potentially clean shot (2), fill (1), pour (1)
        # Minimum per ingredient: grasp(1) + fill(1) + pour(1) = 3
        # But shot may need cleaning: +2
        cost += 3  # ingredient 1: grasp + fill + pour-to-shaker
        cost += 3  # ingredient 2: grasp + fill + pour-to-shaker

        # Step 3: Shake
        cost += 1  # shake action

        # Step 4: Pour shaker to shot
        # Need clean and empty target shot
        shot_clean = self.clean_ids.get(shot)
        shot_empty = self.empty_ids.get(shot)
        is_clean = shot_clean is not None and self._bit_check(state, shot_clean)
        is_empty = shot_empty is not None and self._bit_check(state, shot_empty)

        if is_clean and is_empty:
            cost += 1  # just pour
        elif is_empty and not is_clean:
            cost += 3  # grasp + clean + pour
        elif not is_empty:
            cost += 4  # grasp + empty + clean + pour
        else:
            cost += 1

        # Hand management overhead (drops and grasps between steps)
        cost += 2  # conservative estimate for hand management

        return cost

    def _task_network_estimate(self, state, task_network):
        """Estimate cost from the task network with state-aware per-task costing."""
        if not task_network:
            return 0

        cost = 0
        for task in task_network:
            tid = task.global_id

            # Check if this is an operator (primitive action)
            if tid >= self.model.iop_init and tid <= self.model.iop_end:
                cost += 1
                continue

            # Abstract task - estimate based on type
            if tid >= self.model.iabt_init and tid <= self.model.iabt_end:
                info = self.abstract_task_map.get(tid)
                if info is None:
                    cost += 1
                    continue
                base, args = info

                if base == "AchieveContainsShotCocktail" and len(args) == 2:
                    shot, cocktail = args[0], args[1]
                    cid = self.contains_ids.get((shot, cocktail))
                    if cid is not None and self._bit_check(state, cid):
                        pass  # Already achieved, no cost
                    else:
                        cost += self._estimate_cocktail_cost(state, shot, cocktail)

                elif base == "AchieveContainsShakerCocktail" and len(args) == 2:
                    shaker, cocktail = args[0], args[1]
                    cid = self.contains_ids.get((shaker, cocktail))
                    if cid is not None and self._bit_check(state, cid):
                        pass
                    else:
                        # Need: clean shaker + 2 ingredients + shake = ~9
                        cost += 9

                elif base == "AchieveContainsShakerIngredient" and len(args) == 2:
                    shaker, ing = args[0], args[1]
                    cid = self.contains_ids.get((shaker, ing))
                    if cid is not None and self._bit_check(state, cid):
                        pass
                    else:
                        cost += 4  # fill shot + pour to shaker

                elif base == "AchieveContainsShotIngredient" and len(args) == 2:
                    shot, ing = args[0], args[1]
                    cid = self.contains_ids.get((shot, ing))
                    if cid is not None and self._bit_check(state, cid):
                        pass
                    else:
                        cost += 3  # clean + grasp + fill

                elif base == "AchieveCleanShaker" and len(args) == 1:
                    shaker = args[0]
                    clean_id = self.clean_ids.get(shaker)
                    if clean_id is not None and self._bit_check(state, clean_id):
                        pass
                    else:
                        empty_id = self.empty_ids.get(shaker)
                        if empty_id is not None and self._bit_check(state, empty_id):
                            cost += 2  # grasp + clean
                        else:
                            cost += 4  # grasp + empty + clean

                elif base == "AchieveCleanShot" and len(args) == 1:
                    shot = args[0]
                    clean_id = self.clean_ids.get(shot)
                    if clean_id is not None and self._bit_check(state, clean_id):
                        pass
                    else:
                        empty_id = self.empty_ids.get(shot)
                        if empty_id is not None and self._bit_check(state, empty_id):
                            cost += 2  # grasp + clean
                        else:
                            cost += 3  # grasp + empty + clean

                elif base == "AchieveHolding" and len(args) == 2:
                    hand, container = args[0], args[1]
                    hid = self.holding_ids.get((hand, container))
                    if hid is not None and self._bit_check(state, hid):
                        pass
                    else:
                        cost += 1  # grasp (may need drop first, but covered elsewhere)
                        # Check if hand is occupied
                        he_id = self.hand_empty_ids.get(hand)
                        if he_id is not None and not self._bit_check(state, he_id):
                            cost += 1  # drop first

                elif base == "AchieveHandEmpty" and len(args) == 1:
                    hand = args[0]
                    he_id = self.hand_empty_ids.get(hand)
                    if he_id is not None and self._bit_check(state, he_id):
                        pass
                    else:
                        cost += 1  # drop

                elif base == "AchieveOnTable" and len(args) == 1:
                    container = args[0]
                    ot_id = self.ontable_ids.get(container)
                    if ot_id is not None and self._bit_check(state, ot_id):
                        pass
                    else:
                        cost += 1  # drop

                elif base == "DoPourShakerToShot" and len(args) == 3:
                    cost += 1  # pour action

                else:
                    cost += 1  # fallback
            else:
                cost += 1

        return cost

    def _shaker_serialization_bound(self, state):
        """Lower bound based on shaker being a bottleneck resource.
        Each unfinished cocktail requires exclusive use of the shaker.
        Between cocktails: empty-shaker + clean-shaker = 2 actions minimum.
        Per cocktail in shaker: 2 pour-to-shaker + shake = 3 actions minimum.
        Plus pour-shaker-to-shot = 1 action.
        Total per cocktail: at least 6 actions (after first: +2 for clean cycle = 8).
        """
        unfinished = 0
        for shot, cocktail in self.goal_cocktails:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is None or not self._bit_check(state, cid):
                unfinished += 1

        if unfinished == 0:
            return 0

        # Check if any cocktail is currently in the shaker (partially done)
        cocktail_in_shaker = False
        for shaker in self.shakers:
            for _, cocktail in self.goal_cocktails:
                sid = self.contains_ids.get((shaker, cocktail))
                if sid is not None and self._bit_check(state, sid):
                    cocktail_in_shaker = True
                    break
            if cocktail_in_shaker:
                break

        # Minimum cost per cocktail through the shaker:
        # fill-shot1(1) + pour-to-shaker(1) + fill-shot2(1) + pour-to-shaker(1) + shake(1) + pour-to-shot(1) = 6
        # Plus grasps: grasp-shot(1) before each pour = +2 more grasps
        # Plus fills need grasp too
        # Conservative: 8 actions per cocktail
        # Between cocktails: empty-shaker(1) + clean-shaker(1) = 2
        base_per_cocktail = 8
        shaker_switch_cost = 2

        if cocktail_in_shaker:
            # One cocktail is partially done, reduce its cost
            total = 1  # at least pour-to-shot for the in-progress one
            total += (unfinished - 1) * base_per_cocktail
            if unfinished > 1:
                total += (unfinished - 1) * shaker_switch_cost
        else:
            total = unfinished * base_per_cocktail
            if unfinished > 1:
                total += (unfinished - 1) * shaker_switch_cost

        return total

    def _goal_completion_bound(self, state):
        """Simple count of unfinished goals weighted by estimated completion cost."""
        cost = 0
        for shot, cocktail in self.goal_cocktails:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is None or not self._bit_check(state, cid):
                cost += self._estimate_cocktail_cost(state, shot, cocktail)
        return cost

    def _compute(self, state, task_network):
        """Compute the heuristic value as max of multiple admissible bounds."""
        # Term 1: Task network based estimate (state-aware)
        t1 = self._task_network_estimate(state, task_network)

        # Term 2: Shaker serialization lower bound
        t2 = self._shaker_serialization_bound(state)

        # Term 3: Goal completion bound
        t3 = self._goal_completion_bound(state)

        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
