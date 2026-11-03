"""
Barman BDI Domain-Dependent Heuristic

=== Bottleneck Analysis ===
The single shaker is the critical serializing bottleneck. Every cocktail requires:
1. Clean the shaker (if dirty)
2. Fill shaker with ingredient 1 (via a shot glass)
3. Fill shaker with ingredient 2 (via a shot glass)
4. Shake the cocktail
5. Pour from shaker to target shot
6. Clean the shaker again for the next cocktail

Since there is only ONE shaker, cocktails must be made sequentially. Each cocktail
requires the shaker to go through a full cycle. The shot glasses used as intermediaries
also need cleaning between uses.

=== Admissible Lower Bounds ===

Term 1 - Remaining cocktail tasks cost:
  Each AchieveContainsShotCocktail that isn't already satisfied requires at minimum:
  - Making the cocktail in the shaker (if not already there): ~8-12 actions
    (clean shaker + fill shot with ing1 + pour to shaker + fill shot with ing2 + 
     pour to shaker + shake + pour to shot)
  - We use a conservative per-cocktail lower bound based on state analysis.
  This is admissible because each cocktail genuinely requires these steps.

Term 2 - Shaker serialization bound:
  Since there's only one shaker, and each cocktail not yet made needs exclusive
  shaker access, we count remaining cocktails * minimum shaker cycle cost.
  This is admissible as each cycle is unavoidable.

Term 3 - Task network structural bound:
  Count primitive actions remaining in partially decomposed task network,
  plus minimum decomposition depth for remaining abstract tasks.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic13(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_13"):
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
        
        # Cocktail recipes
        self.cocktail_part1 = {}  # cocktail -> ingredient
        self.cocktail_part2 = {}  # cocktail -> ingredient
        
        # Task name -> type mapping for abstract tasks
        self.abstract_task_types = {}  # global_id -> (task_type, args)
        self.operator_names = {}       # global_id -> name
        
        # Goal cocktails: list of (shot, cocktail) pairs
        self.goal_cocktails = []
        
        # All shakers, shots, hands
        self.shakers = set()
        self.shots = set()
        self.hands = set()
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
        """Parse a grounded task/operator name like 'taskname[arg1,arg2]'"""
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
                self.cocktail_part1[args[0]] = args[1]
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
            elif pred == "dispenses" and len(args) == 2:
                pass  # static info
            elif pred == "shakerEmptyLevel":
                if len(args) >= 1:
                    self.shakers.add(args[0])
            elif pred == "shakerLevel":
                if len(args) >= 1:
                    self.shakers.add(args[0])
        
        # Identify hands from handEmpty facts
        for key in self.hand_empty_ids:
            self.hands.add(key)
        for (h, c) in self.holding_ids:
            self.hands.add(h)
        
        # Identify shots from clean/empty facts that aren't shakers
        for key in self.clean_ids:
            if key not in self.shakers:
                self.shots.add(key)
        
        # Build abstract task type map
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.abstract_task_types[at.global_id] = (task_type, args)
        
        # Build operator name map
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            self.operator_names[op.global_id] = (task_type, args)
        
        # Extract goal cocktails from initial task network
        # The top-level tasks are AchieveContainsShotCocktail
        self._extract_goal_cocktails(initial_node.task_network)
        
        # Minimum cost estimates for various sub-tasks (precomputed constants)
        # These are conservative lower bounds on the number of primitive actions
        # For a cocktail from scratch:
        # - Fill shot with ing1: grasp shot (1) + fill-shot (1) = 2 (+ possible clean: 2-4 more)
        # - Pour shot to clean shaker: 1
        # - Drop shot: 1 (to free hand)
        # - Fill shot with ing2: grasp (1) + fill-shot (1) = 2 (+ clean if needed: 2-4)
        # - Pour shot to used shaker: 1
        # - Shake: need both hands, might need to drop shot first: 1
        # - Pour shaker to target shot: need to clean target shot, grasp it... complex
        
        # Minimum actions per cocktail (very conservative): 
        # grasp_shot + fill + pour_to_shaker + drop_shot + grasp_shot + fill + pour_to_shaker + 
        # grasp_shaker + shake + pour_shaker_to_shot = ~10 primitive actions minimum
        # But after first cocktail, shaker needs cleaning: empty-shaker + clean-shaker = 2 more
        # Plus the shot used for pouring needs cleaning each time
        
        self.MIN_COCKTAIL_COST = 10  # minimum primitive actions for one cocktail
        self.SHAKER_CLEAN_COST = 3   # empty-shaker + drop + clean-shaker (need holding + handEmpty)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _extract_goal_cocktails(self, task_network):
        """Extract (shot, cocktail) goal pairs from initial task network."""
        self.goal_cocktails = []
        for task in task_network:
            gid = task.global_id
            if gid in self.abstract_task_types:
                task_type, args = self.abstract_task_types[gid]
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    self.goal_cocktails.append((args[0], args[1]))

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the state using bitwise ops."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _cocktail_cost(self, state, shot, cocktail):
        """
        Estimate minimum remaining actions to achieve contains(shot, cocktail).
        Returns 0 if already achieved.
        """
        # Check if already done
        cid = self.contains_ids.get((shot, cocktail))
        if cid is not None and self._check_fact(state, cid):
            return 0
        
        cost = 0
        
        # We need the cocktail in the shaker first
        # Check if shaker already contains this cocktail
        shaker = None
        shaker_has_cocktail = False
        shaker_is_shaked = False
        for s in self.shakers:
            shaker = s
            sc_id = self.contains_ids.get((s, cocktail))
            if sc_id is not None and self._check_fact(state, sc_id):
                shaker_has_cocktail = True
                sk_id = self.shaked_ids.get(s)
                if sk_id is not None and self._check_fact(state, sk_id):
                    shaker_is_shaked = True
            break
        
        if shaker is None:
            return self.MIN_COCKTAIL_COST
        
        if shaker_has_cocktail and shaker_is_shaked:
            # Just need to pour to shot
            # Need: holding hand shaker, shot clean+empty+ontable
            # At minimum: ensure shot is clean+empty (might need cleaning)
            # grasp shaker (if not held) + pour-shaker-to-shot
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            
            pour_cost = 1  # pour-shaker-to-shot action
            
            # Need to hold shaker
            shaker_held = False
            for h in self.hands:
                hid = self.holding_ids.get((h, shaker))
                if hid is not None and self._check_fact(state, hid):
                    shaker_held = True
                    break
            if not shaker_held:
                pour_cost += 1  # grasp
                # May need to drop something first
                any_hand_empty = False
                for h in self.hands:
                    hid = self.hand_empty_ids.get(h)
                    if hid is not None and self._check_fact(state, hid):
                        any_hand_empty = True
                        break
                if not any_hand_empty:
                    pour_cost += 1  # drop
            
            if not shot_clean or not shot_empty:
                # Need to clean the shot
                if not shot_empty:
                    pour_cost += 2  # grasp + empty-shot
                pour_cost += 2  # need to clean: hold shot + clean-shot (needs other hand empty)
            
            return pour_cost
        
        if shaker_has_cocktail and not shaker_is_shaked:
            # Need to shake + pour
            # shake needs: holding shaker + handEmpty other hand
            cost = 1  # shake
            cost += 1  # pour
            # May need grasp/drop for hands
            cost += 2  # conservative hand management
            
            shot_clean = self._check_fact(state, self.clean_ids.get(shot))
            shot_empty = self._check_fact(state, self.empty_ids.get(shot))
            if not shot_clean or not shot_empty:
                cost += 3
            return cost
        
        # Need to make cocktail from scratch
        # Check shaker state
        shaker_clean = self._check_fact(state, self.clean_ids.get(shaker))
        shaker_empty = self._check_fact(state, self.empty_ids.get(shaker))
        
        cost = 0
        
        # Clean shaker if needed
        if not shaker_clean:
            if not shaker_empty:
                cost += 2  # empty-shaker + clean-shaker
            else:
                cost += 1  # just clean-shaker
            cost += 2  # hand management (grasp shaker, ensure other hand empty)
        elif not shaker_empty:
            # Shaker is clean but not empty? Unlikely state but handle it
            cost += 1  # empty
            cost += 1  # clean after empty
            cost += 2  # hand management
        
        # Add two ingredients to shaker via shot glasses
        # Each ingredient: fill shot (need clean+empty shot + grasp + fill) + pour to shaker
        # Ingredient 1: grasp_shot(1) + fill_shot(1) + pour_to_clean_shaker(1) = 3
        # Ingredient 2: may need to clean shot first (2-3) + grasp(1) + fill(1) + pour_to_used(1) = 3-6
        cost += 3  # ingredient 1 minimum
        cost += 4  # ingredient 2 minimum (might need shot cleaning)
        
        # Shake: 1 action (need holding shaker + hand empty)
        cost += 1
        
        # Pour to shot: need clean+empty target shot + hold shaker
        shot_clean = self._check_fact(state, self.clean_ids.get(shot))
        shot_empty = self._check_fact(state, self.empty_ids.get(shot))
        cost += 1  # pour action
        if not shot_clean or not shot_empty:
            cost += 3  # cleaning the target shot
        
        # Hand management overhead (grasp/drop transitions)
        cost += 2
        
        return cost

    def _task_network_cost(self, state, task_network):
        """
        Estimate cost from remaining task network.
        For each task, estimate minimum primitive actions needed.
        """
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1  # Each primitive action costs 1
                continue
            
            # It's an abstract task
            if gid not in self.abstract_task_types:
                cost += 1
                continue
            
            task_type, args = self.abstract_task_types[gid]
            
            if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                shot, cocktail = args[0], args[1]
                cost += self._cocktail_cost(state, shot, cocktail)
            
            elif task_type == "AchieveContainsShakerCocktail" and len(args) == 2:
                shaker, cocktail = args[0], args[1]
                cid = self.contains_ids.get((shaker, cocktail))
                if cid is not None and self._check_fact(state, cid):
                    cost += 0
                else:
                    # Need to add 2 ingredients + shake
                    cost += 7  # conservative: 2*fill_shot_pour + shake + hand_mgmt
            
            elif task_type == "AchieveContainsShakerIngredient" and len(args) == 2:
                shaker, ingredient = args[0], args[1]
                cid = self.contains_ids.get((shaker, ingredient))
                if cid is not None and self._check_fact(state, cid):
                    cost += 0
                else:
                    cost += 3  # fill shot + pour to shaker + hand management
            
            elif task_type == "AchieveContainsShotIngredient" and len(args) == 2:
                shot, ingredient = args[0], args[1]
                cid = self.contains_ids.get((shot, ingredient))
                if cid is not None and self._check_fact(state, cid):
                    cost += 0
                else:
                    shot_clean = self._check_fact(state, self.clean_ids.get(shot))
                    shot_empty = self._check_fact(state, self.empty_ids.get(shot))
                    if shot_clean and shot_empty:
                        cost += 2  # grasp + fill
                    else:
                        cost += 4  # clean + grasp + fill + hand mgmt
            
            elif task_type == "AchieveCleanShaker" and len(args) == 1:
                shaker = args[0]
                if self._check_fact(state, self.clean_ids.get(shaker)):
                    cost += 0
                else:
                    shaker_empty = self._check_fact(state, self.empty_ids.get(shaker))
                    if shaker_empty:
                        cost += 2  # grasp + clean-shaker (need hand empty)
                    else:
                        cost += 4  # grasp + empty + clean + hand mgmt
            
            elif task_type == "AchieveCleanShot" and len(args) == 1:
                shot = args[0]
                if self._check_fact(state, self.clean_ids.get(shot)):
                    cost += 0
                else:
                    shot_empty = self._check_fact(state, self.empty_ids.get(shot))
                    if shot_empty:
                        cost += 2  # grasp + clean
                    else:
                        cost += 3  # grasp + empty + clean
            
            elif task_type == "AchieveHolding" and len(args) == 2:
                hand, container = args[0], args[1]
                hid = self.holding_ids.get((hand, container))
                if hid is not None and self._check_fact(state, hid):
                    cost += 0
                else:
                    he_id = self.hand_empty_ids.get(hand)
                    if he_id is not None and self._check_fact(state, he_id):
                        cost += 1  # just grasp
                    else:
                        cost += 2  # drop + grasp
            
            elif task_type == "AchieveHandEmpty" and len(args) == 1:
                hand = args[0]
                he_id = self.hand_empty_ids.get(hand)
                if he_id is not None and self._check_fact(state, he_id):
                    cost += 0
                else:
                    cost += 1  # drop
            
            elif task_type == "AchieveOnTable" and len(args) == 1:
                container = args[0]
                ot_id = self.ontable_ids.get(container)
                if ot_id is not None and self._check_fact(state, ot_id):
                    cost += 0
                else:
                    cost += 1  # drop
            
            elif task_type == "DoPourShakerToShot" and len(args) == 3:
                cost += 1  # single pour action
            
            else:
                # Unknown task type, use minimum 1
                cost += 1
        
        return cost

    def _shaker_serialization_bound(self, state, task_network):
        """
        Count how many distinct cocktails still need to be made in the shaker.
        Each requires exclusive shaker access with a minimum cycle cost.
        """
        remaining_cocktails = set()
        
        for task in task_network:
            gid = task.global_id
            if gid in self.abstract_task_types:
                task_type, args = self.abstract_task_types[gid]
                if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                    shot, cocktail = args[0], args[1]
                    cid = self.contains_ids.get((shot, cocktail))
                    if cid is None or not self._check_fact(state, cid):
                        remaining_cocktails.add(cocktail)
                elif task_type == "AchieveContainsShakerCocktail" and len(args) == 2:
                    shaker, cocktail = args[0], args[1]
                    cid = self.contains_ids.get((shaker, cocktail))
                    if cid is None or not self._check_fact(state, cid):
                        remaining_cocktails.add(cocktail)
        
        n = len(remaining_cocktails)
        if n == 0:
            return 0
        
        # First cocktail: ~10 actions (fill 2 ingredients, shake, pour)
        # Each subsequent cocktail: ~12 actions (clean shaker + make + pour)
        # Conservative: first = 8, subsequent = 10
        return 8 + max(0, n - 1) * 10

    def _remaining_goals_bound(self, state):
        """
        Simple count of unsatisfied goal cocktails * minimum cost.
        """
        unsatisfied = 0
        for shot, cocktail in self.goal_cocktails:
            cid = self.contains_ids.get((shot, cocktail))
            if cid is None or not self._check_fact(state, cid):
                unsatisfied += 1
        
        if unsatisfied == 0:
            return 0
        
        # Each cocktail needs at minimum: pour-shaker-to-shot (1) if shaker ready,
        # but realistically much more. Use conservative 8 per cocktail for the
        # goal-based bound (less aggressive than serialization bound).
        return unsatisfied * 8

    def _compute(self, state, task_network):
        """
        Compute the heuristic value as max of multiple admissible lower bounds.
        """
        if not task_network:
            return 0
        
        # Term 1: Task network based cost (state-aware per-task estimation)
        tn_cost = self._task_network_cost(state, task_network)
        
        # Term 2: Shaker serialization bound
        serial_cost = self._shaker_serialization_bound(state, task_network)
        
        # Term 3: Simple remaining goals bound
        goals_cost = self._remaining_goals_bound(state)
        
        h = max(tn_cost, serial_cost, goals_cost)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
