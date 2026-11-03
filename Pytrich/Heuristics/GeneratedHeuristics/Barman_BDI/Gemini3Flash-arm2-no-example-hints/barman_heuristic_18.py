from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic18(Heuristic):
    """
    Barman BDI Heuristic - Gemini 3 Flash Version 18
    
    Bottleneck Analysis:
    The primary bottleneck in the Barman domain is the efficient management of shots and hands. 
    Because all shots are interchangeable as transfer vessels for ingredients, the planner 
    faces a huge branching factor. The goal shots (those that must contain the final cocktail) 
    should not be used as intermediaries to avoid unnecessary cleaning and contamination cycles.

    Admissible Lower Bounds:
    1. Unsatisfied Goals: Each missing goal fact (contains shot cocktail) represents at least 2 actions (shake, pour).
    2. Task Network Complexity: Each task in the network requires at least one action to resolve or is an action itself.

    Guidance Features:
    - Goal-shot penalty: Large penalty (10000) for using a shot intended for a final cocktail as an intermediary ingredient carrier.
    - Symmetry breaking: Increasing penalties (0, 100, 200...) assigned to working shots to enforce a canonical choice order.
    - Hand preference: Small penalty (50) for using the non-preferred hand to reduce branching by half.
    """

    def __init__(self, barman_gemini3flash_heuristic_18="barman_gemini3flash_heuristic_18"):
        super().__init__(barman_gemini3flash_heuristic_18)
        self.goal_mask = 0
        self.goal_shots = set()
        self.working_shot_penalties = {}
        self.non_preferred_hand = None
        self.shot_names = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            bracket_idx = name.find("[")
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.goal_mask = model.goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        
        # Extract shots and identify which are goal shots
        all_objects = set()
        for f in model.facts:
            _, args = self._parse_fact_name(f.name)
            all_objects.update(args)
        
        self.shot_names = sorted([obj for obj in all_objects if "shot" in obj.lower()])
        hands = sorted([obj for obj in all_objects if "hand" in obj.lower() or obj in ["left", "right"]])
        if len(hands) > 1:
            self.non_preferred_hand = hands[1] # Prefer hands[0] (e.g., 'left')

        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains" and args and "shot" in args[0].lower():
                self.goal_shots.add(args[0])

        working_shots = [s for s in self.shot_names if s not in self.goal_shots]
        for i, s in enumerate(working_shots):
            self.working_shot_penalties[s] = i * 100

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # 1. State-aware Goal term: count missing bits in goal mask
        missing_goals = bin(self.goal_mask & ~state).count('1')
        h = missing_goals * 100

        # 2. Task network term with guidance
        for task in task_network:
            h += 10  # Base task cost
            t_name, t_args = self._parse_fact_name(task.name)

            # Hand Preference (Halve branching)
            if self.non_preferred_hand in t_args:
                h += 50

            # Shot management
            for arg in t_args:
                if arg in self.shot_names:
                    # Symmetry breaking for working shots
                    if arg in self.working_shot_penalties:
                        h += self.working_shot_penalties[arg]
                    
                    # Penalty for using goal-shots as intermediaries
                    # Grounding Warning: In Barman BDI, AchieveContainsShotIngredient is the bottleneck intermediary task
                    if arg in self.goal_shots:
                        if t_name.startswith("AchieveContainsShotIngredient") or t_name.startswith("fill-shot"):
                            h += 10000
                    break
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
