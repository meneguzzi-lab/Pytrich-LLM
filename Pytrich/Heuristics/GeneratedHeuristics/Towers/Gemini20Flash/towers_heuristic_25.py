import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini20FlashHeuristic25(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_25:str, misplaced_ring_penalty=1, blocking_ring_penalty=2, name="towers_gemini20flash_heuristic_25"):
        super().__init__(name=name)
        self.misplaced_ring_penalty = misplaced_ring_penalty
        self.blocking_ring_penalty = blocking_ring_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            move_abstract_count = 0
            exchange_count = 0
            rotate_count = 0
            select_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'move_abstract' in subtask_lower:
                        move_abstract_count += 1
                    elif 'exchange' in subtask_lower:
                        exchange_count += 1
                    elif 'rotate' in subtask_lower:
                        rotate_count += 1
                    elif 'selectdirection' in subtask_lower:
                        select_count += 1

            self.method_scores[method_name] = {
                'move_abstract_count': move_abstract_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'select_count': select_count
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'state') or not node.state:
            return 0

        state = node.state
        h_value = 0

        on_facts = {}
        goal_on_facts = {}
        tower_top_facts = {}

        for fact_id in range(len(self.model.facts)):
          if (state & (1 << fact_id)):
            fact = self.model.facts[fact_id]
            if fact.name.startswith('on '):
              parts = fact.name.split()
              on_facts[parts[1]] = parts[2]
            elif fact.name.startswith('goal_on '):
              parts = fact.name.split()
              goal_on_facts[parts[1]] = parts[2]
            elif fact.name.startswith('towerTop '):
              parts = fact.name.split()
              tower_top_facts[parts[1]] = parts[2]
        
        # Calculate misplaced rings
        for ring, obj in on_facts.items():
            if ring in goal_on_facts and obj != goal_on_facts[ring]:
                h_value += self.misplaced_ring_penalty

        # Calculate blocking rings
        for ring, obj in on_facts.items():
          if obj not in ['t1', 't2', 't3']:
            continue

          current_tower = obj
          
          
          # Find the ring that should be on this object according to goal state
          for goal_ring, goal_obj in goal_on_facts.items():
            if goal_obj == current_tower:
              # If current on-fact does not correspond to goal fact, then blockig
              if ring != goal_ring:
                h_value += self.blocking_ring_penalty
        
        if hasattr(node, 'method') and node.method and node.method.name in self.method_scores:
            method_name = node.method.name
            method_score = self.method_scores[method_name]

            h_value += method_score['move_abstract_count'] * 0.1
            h_value += method_score['exchange_count'] * 0.2
            h_value += method_score['rotate_count'] * 0.1
            h_value += method_score['select_count'] * 0.05

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic25(misplaced={self.misplaced_ring_penalty}, blocking={self.blocking_ring_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic25(misplaced={self.misplaced_ring_penalty}, blocking={self.blocking_ring_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 25 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced ring penalty: {self.misplaced_ring_penalty}\n"
            f"\tBlocking ring penalty: {self.blocking_ring_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
