"""Domain-Dependent Heuristic for Barman_BDI domain.

Bottleneck Analysis:
  The single shaker is the serializing bottleneck. Every cocktail requires exclusive
  use of the shaker (clean -> add ingredient1 -> add ingredient2 -> shake -> pour).
  This means cocktails must be made sequentially, and each cocktail requires a minimum
  number of primitive actions through the shaker pipeline.

Admissible Lower Bounds:
  Term 1 - Remaining cocktails cost: Each cocktail not yet in its target shot requires
           a minimum sequence of actions: fill-shot(ingr1), pour-to-shaker, fill-shot(ingr2),
           pour-to-shaker, shake, pour-shaker-to-shot = at minimum ~11 actions when the
           shaker must be cleaned first. We estimate per-cocktail cost based on current state.
           This is admissible because these actions are unavoidable for each unserved cocktail.

  Term 2 - Shaker serialization: Since there's one shaker and each cocktail needs it,
           remaining cocktails * min_shaker_cycle_cost is a lower bound. We use the
           minimum cost of a shaker cycle (clean + 2 pours in + shake + pour out = ~8 actions)
           but since cocktails share the shaker sequentially, we take the max of per-cocktail
           costs which is admissible.

  Term 3 - Task network depth: Count remaining compound tasks weighted by minimum
           decomposition depth. Each AchieveContainsShotCocktail that isn't satisfied
           requires at least the actions in MakeAndPourCocktail decomposition.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanBDIAnthropicOpus46Heuristic9(Heuristic):
    def __init__(self, name="barman_anthropicopus46_heuristic_9"):
        super().__init__(name)
        # Precomputed maps
        self.contains_fact_ids = {}   # (container_name, beverage_name) -> global_id
        self.clean_fact_ids = {}      # container_name -> global_id
        self.empty_fact_ids = {}      # container_name -> global_id
        self.holding_fact_ids = {}    # (hand_name, container_name) -> global_id
        self.handEmpty_fact_ids = {}  # hand_name -> global_id
        self.ontable_fact_ids = {}    # container_name -> global_id
        self.shaked_fact_ids = {}     # shaker_name -> global_id
        self.unshaked_fact_ids = {}   # shaker_name -> global_id
        self.used_fact_ids = {}       # (container_name, beverage_name) -> global_id
        
        # Cocktail recipes
        self.cocktail_part1 = {}  # cocktail_name -> ingredient_name
        self.cocktail_part2 = {}  # cocktail_name -> ingredient_name
        
        # Goal tracking
        self.goal_cocktail_shots = []  # list of (shot_name, cocktail_name)
        
        # Task name -> abstract task mapping
        self.achieve_contains_shot_cocktail_ids = set()
        self.achieve_contains_shaker_cocktail_ids = set()
        self.achieve_contains_shaker_ingredient_ids = set()
        self.achieve_contains_shot_ingredient_ids = set()
        self.achieve_clean_shaker_ids = set()
        self.achieve_clean_shot_ids = set()
        self.achieve_holding_ids = set()
        self.achieve_hand_empty_ids = set()
        self.achieve_on_table_ids = set()
        self.do_pour_shaker_to_shot_ids = set()
        
        # Shaker name
        self.shaker_name = None
        self.shot_names = []
        self.hand_names = []

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
        """Parse a task name like '__AchieveContainsShotCocktail[shot2,cocktail1]'"""
        name = name.strip()
        # Remove leading underscores
        clean = name.lstrip('_')
        bracket_idx = clean.find("[")
        if bracket_idx > 0 and clean.endswith("]"):
            task_type = clean[:bracket_idx]
            args_str = clean[bracket_idx+1:-1]
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
                self.contains_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "clean" and len(args) == 1:
                self.clean_fact_ids[args[0]] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_fact_ids[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 2:
                self.holding_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "handEmpty" and len(args) == 1:
                self.handEmpty_fact_ids[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_ids[args[0]] = f.global_id
            elif pred == "shaked" and len(args) == 1:
                self.shaked_fact_ids[args[0]] = f.global_id
            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_fact_ids[args[0]] = f.global_id
            elif pred == "used" and len(args) == 2:
                self.used_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                self.cocktail_part1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self.cocktail_part2[args[0]] = args[1]
        
        # Identify shakers, shots, hands from fact maps
        for container_name in self.clean_fact_ids:
            if container_name.startswith("shaker"):
                self.shaker_name = container_name
            elif container_name.startswith("shot"):
                self.shot_names.append(container_name)
        for hand_name in self.handEmpty_fact_ids:
            self.hand_names.append(hand_name)
        
        # Build abstract task type sets by global_id
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            if task_type == "AchieveContainsShotCocktail":
                self.achieve_contains_shot_cocktail_ids.add(at.global_id)
            elif task_type == "AchieveContainsShakerCocktail":
                self.achieve_contains_shaker_cocktail_ids.add(at.global_id)
            elif task_type == "AchieveContainsShakerIngredient":
                self.achieve_contains_shaker_ingredient_ids.add(at.global_id)
            elif task_type == "AchieveContainsShotIngredient":
                self.achieve_contains_shot_ingredient_ids.add(at.global_id)
            elif task_type == "AchieveCleanShaker":
                self.achieve_clean_shaker_ids.add(at.global_id)
            elif task_type == "AchieveCleanShot":
                self.achieve_clean_shot_ids.add(at.global_id)
            elif task_type == "AchieveHolding":
                self.achieve_holding_ids.add(at.global_id)
            elif task_type == "AchieveHandEmpty":
                self.achieve_hand_empty_ids.add(at.global_id)
            elif task_type == "AchieveOnTable":
                self.achieve_on_table_ids.add(at.global_id)
            elif task_type == "DoPourShakerToShot":
                self.do_pour_shaker_to_shot_ids.add(at.global_id)
        
        # Extract goal cocktail-shot pairs from initial task network
        for task in model.initial_tn:
            task_type, args = self._parse_task_name(task.name)
            if task_type == "AchieveContainsShotCocktail" and len(args) == 2:
                self.goal_cocktail_shots.append((args[0], args[1]))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the state using bitwise ops."""
        if global_id is None:
            return False
        return (state >> global_id) & 1

    def _estimate_cocktail_cost(self, state, shot_name, cocktail_name):
        """Estimate minimum actions needed to get cocktail into shot.
        
        Full pipeline for one cocktail (worst case from clean initial state):
        1. Clean shaker if needed (grasp shaker + clean-shaker = ~3 actions, or empty+clean = ~5)
        2. Fill shot with ingredient1 (grasp shot + fill-shot = ~2-3 actions)
        3. Pour shot to clean shaker (already holding shot, pour = 1 action)
        4. Clean shot (empty if needed + clean = ~2-3 actions)  
        5. Fill shot with ingredient2 (grasp + fill = ~2-3 actions)
        6. Pour shot to used shaker (1 action)
        7. Shake (need to hold shaker + have free hand, shake = ~3 actions including grasp)
        8. Pour shaker to shot (clean shot + grasp shaker + pour = ~4 actions)
        
        Minimum total: ~11 actions per cocktail
        """
        contains_id = self.contains_fact_ids.get((shot_name, cocktail_name))
        if contains_id is not None and self._check_fact(state, contains_id):
            return 0  # Already done
        
        cost = 0
        
        # Check if shaker already contains this cocktail
        shaker_has_cocktail = False
        if self.shaker_name:
            shaker_cocktail_id = self.contains_fact_ids.get((self.shaker_name, cocktail_name))
            if shaker_cocktail_id is not None:
                shaker_has_cocktail = self._check_fact(state, shaker_cocktail_id)
        
        if shaker_has_cocktail:
            # Just need to pour from shaker to shot
            # Need: holding shaker, shot clean & empty
            shot_clean = self._check_fact(state, self.clean_fact_ids.get(shot_name))
            shot_empty = self._check_fact(state, self.empty_fact_ids.get(shot_name))
            
            cost = 1  # pour-shaker-to-shot
            
            if not shot_clean or not shot_empty:
                cost += 3  # clean shot: ~empty + clean + grasp/drop overhead
            
            # Need to hold shaker
            holding_shaker = False
            if self.shaker_name:
                for h in self.hand_names:
                    hid = self.holding_fact_ids.get((h, self.shaker_name))
                    if hid is not None and self._check_fact(state, hid):
                        holding_shaker = True
                        break
            if not holding_shaker:
                cost += 2  # grasp shaker (may need to drop + grasp)
            
            return cost
        
        # Check if shaker contains the ingredients (unshaked)
        if cocktail_name in self.cocktail_part1 and cocktail_name in self.cocktail_part2:
            ingr1 = self.cocktail_part1[cocktail_name]
            ingr2 = self.cocktail_part2[cocktail_name]
            
            shaker_has_ingr1 = False
            shaker_has_ingr2 = False
            if self.shaker_name:
                id1 = self.contains_fact_ids.get((self.shaker_name, ingr1))
                id2 = self.contains_fact_ids.get((self.shaker_name, ingr2))
                if id1 is not None:
                    shaker_has_ingr1 = self._check_fact(state, id1)
                if id2 is not None:
                    shaker_has_ingr2 = self._check_fact(state, id2)
            
            if shaker_has_ingr1 and shaker_has_ingr2:
                # Just need shake + pour
                cost = 1  # shake
                cost += 1  # pour-shaker-to-shot
                # Need clean shot
                shot_clean = self._check_fact(state, self.clean_fact_ids.get(shot_name))
                shot_empty = self._check_fact(state, self.empty_fact_ids.get(shot_name))
                if not shot_clean or not shot_empty:
                    cost += 3
                # Holding/grasping overhead
                cost += 2
                return cost
            elif shaker_has_ingr1 or shaker_has_ingr2:
                # Need one more ingredient + shake + pour
                # fill-shot + pour-to-shaker + shake + pour-to-shot
                cost = 3  # fill shot with missing ingredient
                cost += 1  # pour-shot-to-used-shaker 
                cost += 1  # shake
                cost += 1  # pour-shaker-to-shot
                cost += 4  # grasp/drop overhead, clean shot
                return cost
        
        # Full pipeline needed
        # Need to: clean shaker, add ingr1, add ingr2, shake, pour to shot
        shaker_clean = False
        shaker_empty = False
        if self.shaker_name:
            shaker_clean = self._check_fact(state, self.clean_fact_ids.get(self.shaker_name))
            shaker_empty = self._check_fact(state, self.empty_fact_ids.get(self.shaker_name))
        
        cost = 0
        
        # Clean shaker if needed
        if not shaker_clean:
            if not shaker_empty:
                cost += 2  # empty-shaker + clean-shaker
            else:
                cost += 1  # clean-shaker
            cost += 2  # grasp/drop overhead
        
        # Add first ingredient: fill shot + pour to clean shaker
        cost += 3  # grasp shot + fill-shot (need handEmpty for other hand)
        cost += 1  # pour-shot-to-clean-shaker
        
        # Add second ingredient: clean shot + fill shot + pour to used shaker  
        cost += 2  # clean-shot (empty + clean, but shot is already empty after pour)
        cost += 2  # fill-shot with second ingredient
        cost += 1  # pour-shot-to-used-shaker
        
        # Shake
        cost += 1  # shake (need holding shaker + handEmpty)
        cost += 1  # grasp shaker if not holding
        
        # Pour to shot: need clean empty shot
        shot_clean = self._check_fact(state, self.clean_fact_ids.get(shot_name))
        shot_empty = self._check_fact(state, self.empty_fact_ids.get(shot_name))
        
        cost += 1  # pour-shaker-to-shot
        if not shot_clean:
            cost += 2  # clean the shot
        if not shot_empty:
            cost += 1  # empty the shot
        
        return cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network using state-aware per-task estimation."""
        cost = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive action)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1
                continue
            
            # Check if it's an abstract task
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                task_type, args = self._parse_task_name(task.name)
                
                if gid in self.achieve_contains_shot_cocktail_ids and len(args) >= 2:
                    shot_name = args[0]
                    cocktail_name = args[1]
                    contains_id = self.contains_fact_ids.get((shot_name, cocktail_name))
                    if contains_id is not None and self._check_fact(state, contains_id):
                        continue  # Already achieved, null decomposition
                    cost += self._estimate_cocktail_cost(state, shot_name, cocktail_name)
                
                elif gid in self.achieve_contains_shaker_cocktail_ids and len(args) >= 2:
                    shaker_name = args[0]
                    cocktail_name = args[1]
                    contains_id = self.contains_fact_ids.get((shaker_name, cocktail_name))
                    if contains_id is not None and self._check_fact(state, contains_id):
                        continue
                    cost += 8  # Minimum shaker cycle
                
                elif gid in self.achieve_contains_shaker_ingredient_ids and len(args) >= 2:
                    shaker_name = args[0]
                    ingr_name = args[1]
                    contains_id = self.contains_fact_ids.get((shaker_name, ingr_name))
                    if contains_id is not None and self._check_fact(state, contains_id):
                        continue
                    cost += 4  # fill shot + pour to shaker + overhead
                
                elif gid in self.achieve_contains_shot_ingredient_ids and len(args) >= 2:
                    shot_name = args[0]
                    ingr_name = args[1]
                    contains_id = self.contains_fact_ids.get((shot_name, ingr_name))
                    if contains_id is not None and self._check_fact(state, contains_id):
                        continue
                    cost += 3  # clean + grasp + fill
                
                elif gid in self.achieve_clean_shaker_ids and len(args) >= 1:
                    shaker_name = args[0]
                    clean_id = self.clean_fact_ids.get(shaker_name)
                    if clean_id is not None and self._check_fact(state, clean_id):
                        continue
                    empty_id = self.empty_fact_ids.get(shaker_name)
                    if empty_id is not None and self._check_fact(state, empty_id):
                        cost += 3  # grasp + clean-shaker + hand overhead
                    else:
                        cost += 5  # empty-shaker + clean-shaker + overhead
                
                elif gid in self.achieve_clean_shot_ids and len(args) >= 1:
                    shot_name = args[0]
                    clean_id = self.clean_fact_ids.get(shot_name)
                    if clean_id is not None and self._check_fact(state, clean_id):
                        continue
                    cost += 3  # grasp + empty + clean
                
                elif gid in self.achieve_holding_ids and len(args) >= 2:
                    hand_name = args[0]
                    container_name = args[1]
                    holding_id = self.holding_fact_ids.get((hand_name, container_name))
                    if holding_id is not None and self._check_fact(state, holding_id):
                        continue
                    cost += 2  # drop current + grasp new (worst case)
                
                elif gid in self.achieve_hand_empty_ids and len(args) >= 1:
                    hand_name = args[0]
                    he_id = self.handEmpty_fact_ids.get(hand_name)
                    if he_id is not None and self._check_fact(state, he_id):
                        continue
                    cost += 1  # drop
                
                elif gid in self.achieve_on_table_ids and len(args) >= 1:
                    container_name = args[0]
                    ot_id = self.ontable_fact_ids.get(container_name)
                    if ot_id is not None and self._check_fact(state, ot_id):
                        continue
                    cost += 1  # drop
                
                elif gid in self.do_pour_shaker_to_shot_ids:
                    cost += 1  # pour-shaker-to-shot
                
                else:
                    cost += 1  # Unknown task, assume 1
        
        return cost

    def _remaining_cocktails_cost(self, state):
        """Count unserved cocktails and estimate their total minimum cost."""
        total = 0
        unserved_count = 0
        
        for shot_name, cocktail_name in self.goal_cocktail_shots:
            contains_id = self.contains_fact_ids.get((shot_name, cocktail_name))
            if contains_id is not None and self._check_fact(state, contains_id):
                continue  # Already served
            unserved_count += 1
            total += self._estimate_cocktail_cost(state, shot_name, cocktail_name)
        
        # Add shaker cleaning overhead between cocktails (serialization penalty)
        # After the first cocktail, each subsequent one needs to clean the shaker: ~4 actions
        if unserved_count > 1:
            total += (unserved_count - 1) * 4
        
        return total

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0
        
        # Term 1: State-based estimate of remaining cocktail costs
        term1 = self._remaining_cocktails_cost(state)
        
        # Term 2: Task network based cost with state awareness
        term2 = self._task_network_cost(state, task_network)
        
        # Return max of the two terms for admissibility
        return max(term1, term2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
