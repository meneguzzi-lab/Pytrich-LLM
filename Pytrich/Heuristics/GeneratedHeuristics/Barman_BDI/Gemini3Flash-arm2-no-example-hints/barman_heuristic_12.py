from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic12(Heuristic):
    """
    Heuristic for the Barman BDI domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the sequential nature of the shaker and hand management. 
    Each cocktail requires a specific chain of cleaning, filling, pouring, and shaking 
    that can be easily disrupted by symmetry (many identical shots) and inefficient 
    vessel choices (using goal-destination shots for intermediate ingredient transport).

    Admissibility Sketches:
    - Term 1 (Task Network Sum): The task network represents the remaining work to reach a terminal state. 
      Assigning conservative lower-bound costs to each task (e.g., 10 for a full cocktail, 3 for an ingredient transfer) 
      provides an admissible estimate of the remaining sequence length, as these tasks must be decomposed into at least 
      that many primitive actions.
    - Term 2 (Goal Progress): Each unsatisfied 'contains(shot, cocktail)' goal requires at least 10 actions 
      (fill-shot x2, pour x2, shake, cleanings, etc.). Summing these for goals not currently true provides 
      a conservative lower bound.

    Symmetry Breaking:
    - Goal-shot penalty: Using a shot designated for a final cocktail as an intermediary for ingredients is penalized (+500).
    - Hand preference: Non-preferred hand (left) use is penalized (+50).
    - Working shot ordering: Lexicographical ordering of shots to reduce N-way branching (+100 * index).
    """

    def __init__(self, barman_gemini3flash_heuristic_12="BarmanBDIGemini3FlashHeuristic12"):
        super().__init__(name=barman_gemini3flash_heuristic_12)
        self.contains_ids = {} # (container, beverage) -> global_id
        self.clean_ids = {}    # container -> global_id
        
        self.goal_shots = set()
        self.goal_contains_ids = {} # (shot, cocktail) -> global_id
        self.shot_order = {}
        self.preferred_hand = "right"

    def _parse_name(self, name: str):
        """Helper to parse grounded fact/task/op names like +pred[arg1,arg2]"""
        name = name.strip()
        if "[" not in name:
            return name.replace("+", "").replace("-", ""), []
        parts = name.split("[")
        pred = parts[0].replace("+", "").replace("-", "")
        args_str = parts[1].rstrip("]")
        args = args_str.split(",") if args_str else []
        return pred, args

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates for fast lookup
        for fact in model.facts:
            pred, args = self._parse_name(fact.name)
            if pred == "contains":
                self.contains_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clean":
                self.clean_ids[args[0]] = fact.global_id

        # 2. Identify Goal Shots and Goal Facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g in goal_fact_names:
            pred, args = self._parse_name(g)
            if pred == "contains" and len(args) == 2:
                self.goal_shots.add(args[0])
                # Map goal contains specifically
                if (args[0], args[1]) in self.contains_ids:
                    self.goal_contains_ids[(args[0], args[1])] = self.contains_ids[(args[0], args[1])]

        # 3. Establish Shot Ordering for Symmetry Breaking
        all_shots = set()
        for fact in model.facts:
            pred, args = self._parse_name(fact.name)
            if pred == "clean" and "shot" in args[0].lower():
                all_shots.add(args[0])
        sorted_shots = sorted(list(all_shots))
        self.shot_order = {name: i for i, name in enumerate(sorted_shots)}

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state, task_network):
        h_tasks_base = 0
        h_tasks_sym = 0

        for task in task_network:
            pred, args = self._parse_name(task.name)
            
            # Symmetry: Non-preferred hand penalty
            if any(self.preferred_hand not in arg.lower() and "hand" in arg.lower() for arg in args):
                h_tasks_sym += 50
            if "left" in task.name.lower():
                h_tasks_sym += 50

            # Task Costs and Symmetry
            if pred == "AchieveContainsShotCocktail":
                # Args: [shot, cocktail]
                shot, cocktail = args[0], args[1]
                fid = self.contains_ids.get((shot, cocktail))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks_base += 10
            
            elif pred == "AchieveContainsShakerCocktail":
                shaker, cocktail = args[0], args[1]
                fid = self.contains_ids.get((shaker, cocktail))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks_base += 6
            
            elif pred == "AchieveContainsShakerIngredient":
                shaker, ingredient = args[0], args[1]
                fid = self.contains_ids.get((shaker, ingredient))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks_base += 3
            
            elif pred == "AchieveContainsShotIngredient":
                # Args: [shot, ingredient]
                shot_name = args[0]
                fid = self.contains_ids.get((shot_name, args[1]))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks_base += 2
                    # Symmetry: Penalize using goal shot as intermediary
                    if shot_name in self.goal_shots:
                        h_tasks_sym += 500
                    else:
                        h_tasks_sym += self.shot_order.get(shot_name, 0) * 100
            
            elif pred == "AchieveCleanShot":
                fid = self.clean_ids.get(args[0])
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks_base += 2
            
            elif pred == "AchieveCleanShaker":
                fid = self.clean_ids.get(args[0])
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks_base += 2
            
            elif pred in ["AchieveHolding", "AchieveHandEmpty", "AchieveOnTable"]:
                h_tasks_base += 1

        # Term 2: Unsatisfied goals cost
        h_goals = 0
        for (shot, cocktail), fid in self.goal_contains_ids.items():
            if not ((state >> fid) & 1):
                h_goals += 10

        return max(h_tasks_base, h_goals) + h_tasks_sym

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
