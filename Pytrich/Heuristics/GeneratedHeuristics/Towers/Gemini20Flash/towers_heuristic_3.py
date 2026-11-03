import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic3(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_3=1, name="TowersGemini20FlashHeuristic3"):
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_3 = towers_gemini20flash_heuristic_3
        self.preprocessing_time = 0
        self.method_scores = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        self.method_scores = {}
        for method in self.model.decompositions:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = {
                'num_subtasks': len(method.task_network) if hasattr(method, 'task_network') else 0,
                'is_move_abstract': 'move_abstract' in method_name,
                'is_exchange': 'exchange' in method_name,
                'is_rotate': 'rotateTower' in method_name,
                'is_select_direction': 'selectDirection' in method_name,
                'is_shift_tower': 'shiftTower' in method_name
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method and the current state.
        """
        if not hasattr(node, 'method') or not node.method:
            # If there's no method, it's likely the initial node, return an estimate based on initial state.
            return self._estimate_initial_state_heuristic()

        return self._score_method(node.method, node)

    def _estimate_initial_state_heuristic(self):
        """
        Estimate the initial heuristic value based on the initial state of the problem.
        This involves counting the number of rings that are not on the goal tower.
        """
        goal_tower = None
        rings = []
        initial_tower = None
        for obj in self.model.facts:
          obj_name = obj.name if hasattr(obj, 'name') else str(obj)
          if 't3' in obj_name and 'TOWER' in obj_name:
            goal_tower = obj
          if 'r' in obj_name and 'RING' in obj_name:
            rings.append(obj)
          if 't1' in obj_name and 'TOWER' in obj_name:
            initial_tower = obj
            
        
        misplaced_rings = 0
        for ring in rings:
            # Check if the ring is on the initial tower in the initial state.
            on_initial = False
            goal_achieved = False
            for fact_id in self.model.initial_state:
                fact = self.model.facts[fact_id]
                fact_name = fact.name if hasattr(fact, 'name') else str(fact)
                if f'(on {ring.name} {initial_tower.name})' in fact_name:
                  on_initial = True
                if f'(goal_on {ring.name} {goal_tower.name})' in fact_name:
                  goal_achieved = True
            
            if on_initial and not goal_achieved:
                misplaced_rings += 1


        # Each misplaced ring needs to be moved at least once.
        return misplaced_rings * self.towers_gemini20flash_heuristic_3

    def _score_method(self, method, node):
        """
        Score a method based on various factors related to the Towers of Hanoi domain.
        Lower scores indicate more promising methods.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            method_info = self.method_scores[method_name]
            # Prioritize methods that directly contribute to moving rings.
            if method_info['is_move_abstract']:
                score += 1
            elif method_info['is_exchange']:
                score += 2
            # Slightly penalize methods involved in setting up moves, like rotating or selecting direction.
            elif method_info['is_rotate']:
                score += 3
            elif method_info['is_select_direction']:
                score += 4
            elif method_info['is_shift_tower']:
              score += 5
            # Add a small penalty based on the number of subtasks, to favor simpler methods.
            score += method_info['num_subtasks'] * 0.1
        else:
            # If method info is not available, assign a higher penalty.
            score += 10  # Default penalty for unknown methods

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic3(towers_gemini20flash_heuristic_3={self.towers_gemini20flash_heuristic_3})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic3(towers_gemini20flash_heuristic_3={self.towers_gemini20flash_heuristic_3})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 3 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
