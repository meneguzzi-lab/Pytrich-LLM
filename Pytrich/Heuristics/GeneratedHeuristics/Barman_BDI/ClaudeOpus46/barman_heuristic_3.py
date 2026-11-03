"""
Barman BDI Domain-Dependent Heuristic (Version 3)

Bottleneck Analysis:
  The single shaker is the critical serializing resource. Every cocktail must be
  mixed in the shaker (clean shaker -> add ingredient1 -> add ingredient2 -> shake
  -> pour to shot -> clean shaker for next cocktail). This means cocktails are
  essentially serialized through the shaker, and each cocktail requires a minimum
  sequence of actions to produce.

Admissible Lower Bound Terms:

  Term 1 - Per-cocktail production cost:
    Each cocktail not yet in its target shot requires at minimum:
    - If shaker doesn't contain the cocktail and cocktail not yet made:
      * Clean shaker (if dirty): ~3 actions (grasp, empty, clean) or 0 if clean
      * Fill shot with ingredient1: ~4 actions (grasp shot, fill, pour to shaker, drop)
      * Fill shot with ingredient2: ~4 actions (same)
      * Shake: 1 action
      * Pour shaker to shot: ~2 actions (grasp shaker if needed, pour)
      * Clean shot if needed: ~3 actions
    We conservatively estimate the minimum actions per unfinished cocktail.
    This is admissible because each cocktail independently requires these steps.

  Term 2 - Shaker serialization cost:
    Since there's only one shaker, cocktails that still need to be made must
    be produced sequentially. Each cocktail production through the shaker requires
    at minimum: clean_shaker + 2*pour_ingredient + shake + pour_out = ~7 actions
    for the shaker-related steps alone. We count remaining cocktails needing
    the shaker and multiply by a conservative per-cocktail shaker cost.

  Term 3 - Shot preparation cost:
    Each target shot that doesn't contain its cocktail needs to be clean and empty
    before receiving the cocktail. If the shot is dirty or contains something else,
    additional actions are needed.

The heuristic returns max(term1, term2, term3) plus small tie-breaking.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic3(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_3"):
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
        self.shaker_level_ids = {}  # (shaker, level) -> global_id
        self.shaker_empty_level_ids = {}  # (shaker, level) -> global_id
        
        # Domain structure
        self.cocktail_parts = {}  # cocktail -> (ingredient1, ingredient2)
        self.goal_shots = {}      # shot -> cocktail (from goal tasks)
        
        # Task type maps
        self.task_type_map = {}   # abstract_task global_id -> task_type string
        self.task_args_map = {}   # abstract_task global_id -> list of arg names
        
        # Shakers list
        self.shakers = set()
        self.shots = set()
        self.cocktails = set()
        self.ingredients = set()

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
        """Parse a task/abstract_task name to extract task type and arguments."""
        # Format like: AchieveContainsShotCocktail[shot2,cocktail1]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Could also be just a name without brackets
        paren_idx = name.find("(")
        if paren_idx > 0:
            task_type = name[:paren_idx]
            return task_type, []
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
                self.shakers.add(args[0])
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_ids[args[0]] = f.global_id
                self.shakers.add(args[0])
            elif pred == "used" and len(args) == 2:
                self.used_ids[(args[0], args[1])] = f.global_id
            elif pred == "shakerLevel" and len(args) == 2:
                self.shaker_level_ids[(args[0], args[1])] = f.global_id
            elif pred == "shakerEmptyLevel" and len(args) == 2:
                self.shaker_empty_level_ids[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                cocktail = args[0]
                ing = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ing
                self.cocktails.add(cocktail)
                self.ingredients.add(ing)
            elif pred == "cocktailPart2" and len(args) == 2:
                cocktail = args[0]
                ing = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ing
                self.cocktails.add(cocktail)
                self.ingredients.add(ing)
            elif pred == "dispenses":
                pass  # Static, not needed for heuristic
            elif pred == "next":
                pass  # Static
            elif pred == "ingredient":
                if len(args) == 1:
                    self.ingredients.add(args[0])
        
        # Collect shots from contains_ids keys
        for (container, bev) in self.contains_ids:
            if container.startswith("shot"):
                self.shots.add(container)
        for container in self.clean_ids:
            if container.startswith("shot"):
                self.shots.add(container)
        
        # Parse abstract tasks to identify task types
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = task_type
            self.task_args_map[at.global_id] = args
        
        # Extract goal cocktail-shot assignments from initial task network
        self._extract_goals_from_tn(initial_node.task_network)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goals_from_tn(self, task_network):
        """Extract shot->cocktail goals from the initial task network."""
        for task in task_network:
            gid = task.global_id
            if gid in self.task_type_map:
                ttype = self.task_type_map[gid]
                args = self.task_args_map[gid]
                if ttype == "AchieveContainsShotCocktail" and len(args) == 2:
                    shot = args[0]
                    cocktail = args[1]
                    self.goal_shots[shot] = cocktail

    def _fact_true(self, state, global_id):
        """Check if a fact is true using bitwise operation."""
        return (state >> global_id) & 1

    def _compute(self, state, task_network):
        """Compute the heuristic value."""
        # Count remaining unfinished cocktail goals
        unfinished_goals = []
        for task in task_network:
            gid = task.global_id
            if gid in self.task_type_map:
                ttype = self.task_type_map[gid]
                args = self.task_args_map[gid]
                if ttype == "AchieveContainsShotCocktail" and len(args) == 2:
                    shot = args[0]
                    cocktail = args[1]
                    cid = self.contains_ids.get((shot, cocktail))
                    if cid is not None and self._fact_true(state, cid):
                        continue  # Already done
                    unfinished_goals.append((shot, cocktail))
        
        if not unfinished_goals:
            # Check if there are still tasks remaining
            return len(task_network)
        
        # Term 1: Per-cocktail minimum action cost (state-aware)
        term1 = 0
        cocktails_needing_shaker = 0
        
        for (shot, cocktail) in unfinished_goals:
            cost = 0
            
            # Check if shaker already contains this cocktail (shaked)
            shaker_has_cocktail = False
            for shaker in self.shakers:
                cid = self.contains_ids.get((shaker, cocktail))
                sid = self.shaked_ids.get(shaker)
                if cid is not None and sid is not None:
                    if self._fact_true(state, cid) and self._fact_true(state, sid):
                        shaker_has_cocktail = True
                        break
            
            if shaker_has_cocktail:
                # Just need to pour to shot: clean shot + pour
                # Check if shot is clean and empty
                shot_clean_id = self.clean_ids.get(shot)
                shot_empty_id = self.empty_ids.get(shot)
                shot_is_clean = shot_clean_id is not None and self._fact_true(state, shot_clean_id)
                shot_is_empty = shot_empty_id is not None and self._fact_true(state, shot_empty_id)
                
                if shot_is_clean and shot_is_empty:
                    # Need to hold shaker and pour: grasp shaker (2: empty hand + grasp) + pour (1) = 3
                    cost = 3
                elif shot_is_empty and not shot_is_clean:
                    # Clean shot: grasp shot (2) + clean (1) + drop (1) = 4, then pour = 3
                    cost = 7
                else:
                    # Empty shot + clean + pour: ~8-10
                    cost = 10
            else:
                # Need to make cocktail in shaker then pour
                cocktails_needing_shaker += 1
                
                parts = self.cocktail_parts.get(cocktail)
                if parts is None:
                    cost = 15
                else:
                    ing1, ing2 = parts[0], parts[1]
                    
                    # Check if shaker contains ingredients already
                    shaker_cost = 0
                    for shaker in self.shakers:
                        # Check if shaker is clean and empty
                        s_clean = self.clean_ids.get(shaker)
                        s_empty = self.empty_ids.get(shaker)
                        is_clean = s_clean is not None and self._fact_true(state, s_clean)
                        is_empty = s_empty is not None and self._fact_true(state, s_empty)
                        
                        has_ing1 = False
                        has_ing2 = False
                        if ing1:
                            cid1 = self.contains_ids.get((shaker, ing1))
                            has_ing1 = cid1 is not None and self._fact_true(state, cid1)
                        if ing2:
                            cid2 = self.contains_ids.get((shaker, ing2))
                            has_ing2 = cid2 is not None and self._fact_true(state, cid2)
                        
                        if has_ing1 and has_ing2:
                            # Just need to shake + pour
                            shaker_cost = 1  # shake
                        elif is_clean and is_empty:
                            # Need to add both ingredients + shake
                            # Each ingredient: fill shot(4) + pour to shaker(1) = 5 per ingredient
                            shaker_cost = 5 + 5 + 1  # 11
                        elif is_empty and not is_clean:
                            # Clean shaker (1) + add ingredients + shake
                            shaker_cost = 1 + 5 + 5 + 1  # 12
                        elif has_ing1 and not has_ing2:
                            shaker_cost = 5 + 1  # add ing2 + shake
                        elif has_ing2 and not has_ing1:
                            shaker_cost = 5 + 1  # add ing1 + shake
                        else:
                            # Shaker has something else: empty(1) + clean(1) + add(5+5) + shake(1) = 13
                            shaker_cost = 13
                        break  # Only one shaker typically
                    
                    if not self.shakers:
                        shaker_cost = 13
                    
                    # Shot preparation cost
                    shot_clean_id = self.clean_ids.get(shot)
                    shot_empty_id = self.empty_ids.get(shot)
                    shot_is_clean = shot_clean_id is not None and self._fact_true(state, shot_clean_id)
                    shot_is_empty = shot_empty_id is not None and self._fact_true(state, shot_empty_id)
                    
                    shot_prep = 0
                    if not shot_is_clean or not shot_is_empty:
                        if not shot_is_empty:
                            shot_prep += 4  # grasp, empty, clean
                        elif not shot_is_clean:
                            shot_prep += 3  # grasp, clean
                    
                    # Pour from shaker to shot: ~2 actions (already holding shaker ideally, pour)
                    pour_cost = 2
                    
                    cost = shaker_cost + shot_prep + pour_cost
            
            term1 += cost
        
        # Term 2: Shaker serialization lower bound
        # Each cocktail that needs the shaker requires at minimum cleaning the shaker
        # between uses. The minimum shaker cycle is:
        # clean(~3 if dirty) + pour_ing1_to_shaker(~5) + pour_ing2_to_shaker(~5) + shake(1) + pour_out(1) = ~15
        # But we use a conservative estimate per additional cocktail after the first
        if cocktails_needing_shaker > 0:
            # First cocktail: ~12 actions through shaker
            # Each subsequent: ~15 actions (includes clean shaker from previous)
            # Conservative: 10 for first, 13 for each additional
            term2 = 10 + max(0, (cocktails_needing_shaker - 1)) * 13
        else:
            term2 = 0
        
        # Term 3: Count remaining abstract tasks with state-aware weighting
        term3 = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.task_type_map:
                ttype = self.task_type_map[gid]
                args = self.task_args_map[gid]
                
                if ttype == "AchieveContainsShotCocktail":
                    # Already counted in term1
                    pass
                elif ttype == "AchieveContainsShakerCocktail":
                    if len(args) >= 2:
                        shaker, cocktail = args[0], args[1]
                        cid = self.contains_ids.get((shaker, cocktail))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    term3 += 8
                elif ttype == "AchieveContainsShakerIngredient":
                    if len(args) >= 2:
                        shaker, ing = args[0], args[1]
                        cid = self.contains_ids.get((shaker, ing))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    term3 += 5
                elif ttype == "AchieveContainsShotIngredient":
                    if len(args) >= 2:
                        shot_name, ing = args[0], args[1]
                        cid = self.contains_ids.get((shot_name, ing))
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    term3 += 4
                elif ttype == "AchieveCleanShaker":
                    if len(args) >= 1:
                        shaker = args[0]
                        cid = self.clean_ids.get(shaker)
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    term3 += 3
                elif ttype == "AchieveCleanShot":
                    if len(args) >= 1:
                        shot_name = args[0]
                        cid = self.clean_ids.get(shot_name)
                        if cid is not None and self._fact_true(state, cid):
                            continue
                    term3 += 3
                elif ttype == "AchieveHolding":
                    if len(args) >= 2:
                        hand, container = args[0], args[1]
                        hid = self.holding_ids.get((hand, container))
                        if hid is not None and self._fact_true(state, hid):
                            continue
                    term3 += 2
                elif ttype == "AchieveHandEmpty":
                    if len(args) >= 1:
                        hand = args[0]
                        hid = self.hand_empty_ids.get(hand)
                        if hid is not None and self._fact_true(state, hid):
                            continue
                    term3 += 1
                elif ttype == "AchieveOnTable":
                    if len(args) >= 1:
                        container = args[0]
                        oid = self.ontable_ids.get(container)
                        if oid is not None and self._fact_true(state, oid):
                            continue
                    term3 += 1
                elif ttype == "DoPourShakerToShot":
                    term3 += 1
                else:
                    term3 += 1
            else:
                # It's an operator (primitive task), count as 1
                term3 += 1
        
        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
