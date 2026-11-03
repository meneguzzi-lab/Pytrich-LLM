from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic9(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.

    BOTTLENECK: The sequential dependency of hand availability and shot use. 
    Ingredients must pass through shots into shakers, then cocktails must 
    be poured back into shots. This creates significant parameter symmetry 
    in shot selection and hand assignment.

    ADMISSIBLE BOUNDS:
    1. Task count: Every task in the task network requires at least 1 action (often more).
    2. Cocktail Progress: If a goal cocktail is not in its destination shot, 
       at least 1-6 actions (pour, shake, fill) are required.

    SYMMETRY BREAKING (Hints applied):
    - Goal-shot penalty: Using a shot destined for a goal as an intermediary costs 500.
    - Hand preference: Using the non-preferred (right) hand costs 50.
    - Working-shot ordering: Assigning increasing penalties (0, 100, 200...) to identical working shots.
    """

    def __init__(self, barman_gemini3flash_heuristic_9="barman_gemini3flash_heuristic_9"):
        super().__init__(name=barman_gemini3flash_heuristic_9)
        self.goal_shots = set()
        self.goal_contains_ids = []
        self.task_penalties = {}
        self.working_shot_penalties = {}
        self.hand_preference = {}

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

    def _parse_task_args(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            args_str = name[bracket_idx+1:-1]
            return args_str.split(",")
        return []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify goal shots and goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains" and len(args) == 2:
                # args[0] is usually the container (shot/shaker)
                if "shot" in args[0]:
                    self.goal_shots.add(args[0])

        # 2. Map facts for progress tracking
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "contains" and f.name in goal_fact_names:
                self.goal_contains_ids.append(f.global_id)

        # 3. Setup symmetry breaking for working shots
        all_shots = sorted([f.name.split('[')[1].split(',')[0].split(']')[0] for f in model.facts if "shot" in f.name and "+ontable" in f.name])
        all_shots = sorted(list(set(all_shots)))
        working_shot_count = 0
        for s in all_shots:
            if s not in self.goal_shots:
                self.working_shot_penalties[s] = working_shot_count * 100
                working_shot_count += 1

        # 4. Precompute penalties for all Operators and Abstract Tasks
        for task_list in [model.operators, model.abstract_tasks]:
            for t in task_list:
                penalty = 0
                args = self._parse_task_args(t.name)
                if args:
                    # Rule: Penalty for using goal shots for ingredients/cleaning
                    if args[0] in self.goal_shots:
                        if "Ingredient" in t.name or "fill" in t.name or "clean" in t.name or "pour-shot" in t.name:
                            # Don't penalize if the task is the final pour
                            if "AchieveContainsShotCocktail" not in t.name and "DoPour" not in t.name:
                                penalty += 500
                    
                    # Rule: Hand preference
                    if "right" in args:
                        penalty += 50
                    
                    # Rule: Working shot symmetry
                    if args[0] in self.working_shot_penalties:
                        penalty += self.working_shot_penalties[args[0]]
                
                self.task_penalties[t.global_id] = penalty

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        # Base cost: task count
        h = len(task_network) * 10
        
        # Add precomputed symmetry-breaking and task-specific penalties
        for task in task_network:
            h += self.task_penalties.get(task.global_id, 0)
            
        # Add progress cost for unfulfilled goal facts
        for g_id in self.goal_contains_ids:
            is_true = (state >> g_id) & 1
            if not is_true:
                h += 20  # Admissible lower bound for getting a cocktail into a shot
                
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
