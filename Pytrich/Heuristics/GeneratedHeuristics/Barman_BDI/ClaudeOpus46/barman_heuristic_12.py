"""Domain-dependent heuristic for Barman_BDI domain.

Bottleneck Analysis:
  The single shaker is the serializing bottleneck. Every cocktail must be mixed
  in the shaker (clean shaker, add 2 ingredients via shots, shake, pour to target
  shot). The shaker must be cleaned between cocktails. This means cocktails are
  essentially serialized through the shaker.

Admissible Lower Bounds:
  Term 1 - Per-cocktail production cost:
    Each cocktail not yet in its target shot requires at minimum:
    - If cocktail not yet in shaker: clean shaker + fill 2 shots with ingredients
      + pour 2 shots to shaker + shake + pour shaker to shot = ~13 actions minimum.
    - If cocktail in shaker but not poured: pour to shot = ~3-4 actions.
    - We estimate based on current state how much work remains per cocktail goal.
    This is admissible because each cocktail truly requires these steps.

  Term 2 - Shaker reuse cost:
    The shaker must be cleaned between each cocktail. For N remaining cocktails,
    we need at least N-1 shaker cleanings (each ~3 actions: empty + clean + hand ops).
    This is admissible as a lower bound on unavoidable shaker cycling.

  Term 3 - Task network structural lower bound:
    Count remaining primitive-equivalent work in the task network using
    state-aware per-task minimum cost estimation.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic12(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_12"):
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
        
        # Domain knowledge
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_cocktails = []  # list of (shot, cocktail) pairs from goals
        self.all_shots = set()
        self.all_shakers = set()
        self.all_cocktails = set()
        self.all_ingredients = set()
        
        # Task name -> abstract task mapping
        self.task_name_map = {}

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
        """Parse a task/operator name like 'taskName[arg1,arg2]' or '__taskName[...]'"""
        name = name.strip()
        # Remove leading underscores
        clean = name.lstrip('_')
        bracket_idx = clean.find("[")
        if bracket_idx > 0 and clean.endswith("]"):
            task_type = clean[:bracket_idx]
            args_str = clean[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return clean, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
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
                cocktail = args[0]
                ing = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ing
                self.all_cocktails.add(cocktail)
                self.all_ingredients.add(ing)
            elif pred == "cocktailPart2" and len(args) == 2:
                cocktail = args[0]
                ing = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ing
                self.all_cocktails.add(cocktail)
                self.all_ingredients.add(ing)

        # Identify objects from fact maps
        for (container, bev), _ in self.contains_ids.items():
            if container.startswith("shot"):
                self.all_shots.add(container)
            elif container.startswith("shaker"):
                self.all_shakers.add(container)
        for container in self.clean_ids:
            if container.startswith("shot"):
                self.all_shots.add(container)
            elif container.startswith("shaker"):
                self.all_shakers.add(container)
        for container in self.empty_ids:
            if container.startswith("shot"):
                self.all_shots.add(container)
            elif container.startswith("shaker"):
                self.all_shakers.add(container)

        # Extract goal cocktails from initial task network
        self._extract_goal_cocktails(initial_node.task_network)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goal_cocktails(self, task_network):
        """Extract (shot, cocktail) goals from the task network."""
        from Pytrich.model import Model as M
        for task in task_network:
            task_type, args = self._parse_task_name(task.name)
            if "AchieveContainsShotCocktail" in task_type and len(args) >= 2:
                shot = args[0]
                cocktail = args[1]
                self.goal_cocktails.append((shot, cocktail))

    def _bit_check(self, state, global_id):
        """Check if a fact is true in the state."""
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """Estimate minimum actions to get cocktail into shot from current state."""
        # Check if already done
        cid = self.contains_ids.get((shot, cocktail))
        if cid is not None and self._bit_check(state, cid):
            return 0
        
        cost = 0
        parts = self.cocktail_parts.get(cocktail)
        if parts is None:
            return 1  # Unknown cocktail, minimal cost
        
        ing1, ing2 = parts[0], parts[1]
        
        # Check if any shaker already contains this cocktail (shaked)
        shaker_has_cocktail = False
        shaker_ready = False  # shaked and contains cocktail
        
        for shaker in self.all_shakers:
            c_id = self.contains_ids.get((shaker, cocktail))
            s_id = self.shaked_ids.get(shaker)
            if c_id is not None and s_id is not None:
                if self._bit_check(state, c_id) and self._bit_check(state, s_id):
                    shaker_ready = True
                    break
                elif c_id is not None and self._bit_check(state, c_id):
                    shaker_has_cocktail = True
        
        if shaker_ready:
            # Just need to pour from shaker to shot
            # Need: clean shot, empty shot, hold shaker
            # Minimum: clean shot (0-4 actions) + ensure holding shaker + pour (1)
            shot_clean_id = self.clean_ids.get(shot)
            shot_empty_id = self.empty_ids.get(shot)
            shot_cost = 0
            if shot_clean_id is not None and not self._bit_check(state, shot_clean_id):
                shot_cost += 3  # empty-shot + clean-shot + hand management
            if shot_empty_id is not None and not self._bit_check(state, shot_empty_id):
                shot_cost += 1  # empty-shot
            # hold shaker + pour
            cost = shot_cost + 2  # grasp shaker + pour
            return cost
        
        # Need to make the cocktail from scratch in shaker
        # Steps: clean shaker + fill ingredient shots + pour to shaker + shake + pour to target shot
        
        # Shaker cleaning cost
        shaker_cost = 0
        for shaker in self.all_shakers:
            s_clean = self.clean_ids.get(shaker)
            s_empty = self.empty_ids.get(shaker)
            if s_clean is not None and self._bit_check(state, s_clean):
                if s_empty is not None and self._bit_check(state, s_empty):
                    shaker_cost = 0  # Already clean and empty
                    break
            else:
                shaker_cost = 5  # empty + clean + hand ops
        
        # For each ingredient: fill shot + pour to shaker
        # fill-shot: grasp shot + fill (need clean empty shot + hand empty) = ~4 actions
        # pour-shot-to-shaker: 1 action (but shot already held)
        # Total per ingredient: ~4 actions
        ing_cost = 0
        for ing in [ing1, ing2]:
            if ing is None:
                continue
            # Check if any shaker already has this ingredient
            found_in_shaker = False
            for shaker in self.all_shakers:
                ci = self.contains_ids.get((shaker, ing))
                if ci is not None and self._bit_check(state, ci):
                    found_in_shaker = True
                    break
            if found_in_shaker:
                continue
            # Need to fill a shot with ingredient and pour to shaker
            # find shot + clean it + grasp + fill + pour = ~5 actions
            ing_cost += 5
        
        # Shake: 1 action (need shaker held + other hand empty)
        shake_cost = 1
        
        # Pour to target shot: need clean empty target shot + pour
        target_cost = 0
        shot_clean_id = self.clean_ids.get(shot)
        shot_empty_id = self.empty_ids.get(shot)
        if shot_clean_id is not None and not self._bit_check(state, shot_clean_id):
            target_cost += 3
        if shot_empty_id is not None and not self._bit_check(state, shot_empty_id):
            target_cost += 1
        target_cost += 1  # pour action
        
        cost = shaker_cost + ing_cost + shake_cost + target_cost
        return cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network with state awareness."""
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            # Check if it's an operator (primitive)
            if task.global_id >= self.model.iop_init and task.global_id <= self.model.iop_end:
                cost += 1
                continue
            
            # It's an abstract task
            task_type, args = self._parse_task_name(task.name)
            
            if "AchieveContainsShotCocktail" in task_type and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is not None and self._bit_check(state, cid):
                    continue  # Already satisfied
                cost += self._cocktail_cost(state, shot, cocktail)
            elif "AchieveContainsShakerCocktail" in task_type and len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shaker, cocktail))
                if cid is not None and self._bit_check(state, cid):
                    continue
                cost += 8  # clean shaker + 2 ingredients + shake
            elif "AchieveContainsShakerIngredient" in task_type and len(args) >= 2:
                shaker, ing = args[0], args[1]
                cid = self.contains_ids.get((shaker, ing))
                if cid is not None and self._bit_check(state, cid):
                    continue
                cost += 5  # fill shot + pour to shaker
            elif "AchieveContainsShotIngredient" in task_type and len(args) >= 2:
                shot, ing = args[0], args[1]
                cid = self.contains_ids.get((shot, ing))
                if cid is not None and self._bit_check(state, cid):
                    continue
                cost += 4  # clean shot + grasp + fill
            elif "AchieveCleanShaker" in task_type and len(args) >= 1:
                shaker = args[0]
                sid = self.clean_ids.get(shaker)
                if sid is not None and self._bit_check(state, sid):
                    continue
                cost += 4  # empty + clean + hand management
            elif "AchieveCleanShot" in task_type and len(args) >= 1:
                shot = args[0]
                sid = self.clean_ids.get(shot)
                if sid is not None and self._bit_check(state, sid):
                    continue
                cost += 3  # empty + clean + hand management
            elif "AchieveHolding" in task_type and len(args) >= 2:
                hand, container = args[0], args[1]
                hid = self.holding_ids.get((hand, container))
                if hid is not None and self._bit_check(state, hid):
                    continue
                cost += 2  # drop current + grasp
            elif "AchieveHandEmpty" in task_type and len(args) >= 1:
                hand = args[0]
                hid = self.hand_empty_ids.get(hand)
                if hid is not None and self._bit_check(state, hid):
                    continue
                cost += 1  # drop
            elif "AchieveOnTable" in task_type and len(args) >= 1:
                container = args[0]
                oid = self.ontable_ids.get(container)
                if oid is not None and self._bit_check(state, oid):
                    continue
                cost += 1  # drop
            elif "DoPourShakerToShot" in task_type:
                cost += 1  # pour action
            else:
                # Unknown task, assign minimal cost
                cost += 1
        
        return cost

    def _remaining_cocktails_cost(self, state, task_network):
        """Count remaining cocktail goals and estimate serialized shaker cost."""
        remaining = 0
        total_cocktail_cost = 0
        
        for task in task_network:
            task_type, args = self._parse_task_name(task.name)
            if "AchieveContainsShotCocktail" in task_type and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shot, cocktail))
                if cid is None or not self._bit_check(state, cid):
                    remaining += 1
                    total_cocktail_cost += self._cocktail_cost(state, shot, cocktail)
        
        # Shaker serialization: after the first cocktail, each subsequent one
        # needs to clean the shaker (empty + clean = ~3 actions)
        shaker_reuse_cost = max(0, remaining - 1) * 3
        
        return max(total_cocktail_cost, shaker_reuse_cost)

    def _compute(self, state, task_network):
        """Compute the heuristic value."""
        if not task_network:
            return 0
        
        # Term 1: Task network structural estimate
        t1 = self._task_network_cost(state, task_network)
        
        # Term 2: Remaining cocktails with shaker serialization
        t2 = self._remaining_cocktails_cost(state, task_network)
        
        return max(t1, t2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
