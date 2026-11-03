import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic4(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_4="RoverGemini20FlashHeuristic4"):  # init parameter name all lower case `rover_gemini20flash_heuristic_4`
        super().__init__(name=rover_gemini20flash_heuristic_4)
        self.method_scores = {}
        self.waypoint_connections = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self._precompute_waypoint_connections()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        self.method_scores = {}
        for method in self.model.decompositions:
            method_name = method.name
            self.method_scores[method_name] = {
                'num_subtasks': len(method.task_network),
                'is_nav': 'navigate' in method_name.lower(),
                'is_sample': 'sample' in method_name.lower(),
                'is_communicate': 'communicate' in method_name.lower(),
                'is_calibrate': 'calibrate' in method_name.lower(),
                'is_empty_store': 'empty_store' in method_name.lower(),
                'is_do_navigate1': 'do_navigate1' in method_name.lower(),
                'is_do_navigate2': 'do_navigate2' in method_name.lower(),
                'is_get': 'get' in method_name.lower(),
                'is_send': 'send' in method_name.lower()
            }

    def _precompute_waypoint_connections(self):
        self.waypoint_connections = {}
        for op in self.model.operators:
            if op.name == 'navigate':
                p1, p2 = op.name.split(" ")[1], op.name.split(" ")[2]
                self.waypoint_connections[(p1, p2)] = op.cost


    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0
        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name
        info = self.method_scores.get(method_name, {})

        if info.get('is_do_navigate1'):
           score += 1

        if info.get('is_do_navigate2'):
            score += 2

        if info.get('is_get'):
            score += 3
        if info.get('is_send'):
            score += 4

        if info.get('is_sample'):
            score += 5

        return score


    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic4()"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic4()"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 4 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
