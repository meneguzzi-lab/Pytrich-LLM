import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic22(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_22=1, name="BlocksworldGemini20FlashHeuristic22"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0


    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.facts = model.facts
        self.op_dict = {op.name: op for op in model.operators}
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        '''
        Calculates the heuristic value of a given node by considering the number of blocks that are not on their goal positions.
        Also considers the number of 'puton' actions present in the task network
        '''

        state = node.state
        goal_state = self.model.goals
        misplaced_blocks = 0
        puton_actions = 0

        # Count misplaced blocks
        for goal in goal_state:
            if goal not in state:
                misplaced_blocks += 1

        # Count 'puton' actions in the task network
        if hasattr(node, 'task_network') and node.task_network:
            for task in node.task_network:
                if hasattr(task, 'name') and 'puton' in task.name:
                    puton_actions += 1


        # Combining the heuristics (can tune weights)
        heuristic_value = misplaced_blocks + puton_actions

        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}()"

    def __str__(self):
        return f"{self.name}()"

    def __output__(self):
        return (
            f"{self.name} info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
