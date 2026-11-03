from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for Barman_BDI.
    
    Bottleneck Analysis:
    The main bottleneck is the parameter-binding symmetry involving shots. Since any cocktail requires multiple 
    ingredients gathered via shots and then mixed in a single shaker, the planner often wastes cycles 
    trying every possible shot for every intermediate ingredient step. 

    Admissible Sketches:
    1. Task Network Cost: Every task in the current network must eventually be decomposed or executed. 
       A minimum lower bound is 1 per operator and a higher estimate (e.g., 5-10) for abstract tasks.
    2. Goal Stability: Every unmet goal fact (cocktail in shot) requires a complex sequence of tasks 
       (clean shaker, fill shot, pour to shaker, shake, pour to shot). Each unmet goal contributes a minimum cost.

    Symmetry-Breaking (Non-admissible but highly effective):
    - Goal-shot penalty: High penalty (500) for using a shot designated for a final cocktail as an intermediate vessel.
    - Working-shot ordering: Sequential penalties (0, 100, 200...) for non-goal shots to force a canonical choice.
    - Hand preference: Penalty (50) for using the non-preferred hand to reduce branching.
    """

    def __init__(self, barman_gemini3flash_heuristic_10="barman_gemini3flash_heuristic_10"):
        super().__init__(name=barman_gemini3flash_heuristic_10)
        self.goal_shots = set()
        self.shot_names = []
        self.working_shot_penalties = {}
        self.preferred_hand = None
        self.non_preferred_hand = None
        self.goal_fact_ids = []

    def _parse_item_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify Goal Facts and Goal Shots
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "contains" and len(args) == 2:
                # args[0] is shot, args[1] is cocktail
                self.goal_shots.add(args[0])
            
            # Map goal fact to global ID for bitwise checking
            for f in model.facts:
                if f.name == gn:
                    self.goal_fact_ids.append(f.global_id)
                    break

        # 2. Identify all shots and setup working-shot ordering
        all_shots = set()
        hands = []
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "ontable" or pred == "clean" or pred == "empty":
                # This is likely a container
                for arg in args:
                    if "shot" in arg:
                        all_shots.add(arg)
            if pred == "handEmpty":
                for arg in args:
                    if arg not in hands:
                        hands.append(arg)
        
        self.shot_names = sorted(list(all_shots))
        working_shots = [s for s in self.shot_names if s not in self.goal_shots]
        for i, ws in enumerate(working_shots):
            self.working_shot_penalties[ws] = i * 100
            
        # 3. Setup Hand Preference
        if hands:
            self.preferred_hand = hands[0]
            if len(hands) > 1:
                self.non_preferred_hand = hands[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        h = 0
        
        # Term 1: Task Network Cost + Symmetry Penalties
        for task in task_network:
            t_name, t_args = self._parse_item_name(task.name)
            
            # Admissible base cost
            if t_name.startswith("Achieve"):
                h += 10
            else:
                h += 1
            
            # Symmetry Penalties
            if t_args:
                # Grounded operators often have shot at args[0]
                first_arg = t_args[0]
                if "shot" in first_arg:
                    if first_arg in self.goal_shots:
                        # Penalty for using goal shots for anything except final cocktail goals
                        if not (t_name == "AchieveContainsShotCocktail" or t_name == "DoPourShakerToShot" or t_name == "pour-shaker-to-shot"):
                            h += 500
                    elif first_arg in self.working_shot_penalties:
                        # Penalty for working-shot symmetry breaking
                        h += self.working_shot_penalties[first_arg]
                
                # Hand preference penalty
                if self.non_preferred_hand:
                    for arg in t_args:
                        if arg == self.non_preferred_hand:
                            h += 50
                            break

        # Term 2: Goal Progress (Unmet cocktail goals)
        for g_id in self.goal_fact_ids:
            if not ((state >> g_id) & 1):
                h += 30

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
