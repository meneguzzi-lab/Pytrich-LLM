import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic29(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_29, misplaced_block_penalty=1, ontable_penalty=0.5, name="blocksworld_gemini20flash_heuristic_29"):
        super().__init__(name=name)
        self.misplaced_block_penalty = misplaced_block_penalty
        self.ontable_penalty = ontable_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
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

            self.method_scores[method_name] = {}


    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on the number of misplaced blocks.
        """
        state = node.state
        goal = self.model.goals
        misplaced_blocks = 0
        ontable_misplaced = 0

        # Iterate over the goal facts to check if they are in the current state
        for goal_fact_id in goal:
            goal_fact_name = self.model.get_component(goal_fact_id).name
            if goal_fact_name.startswith('on'):
                # If 'on' fact is not in the state, increment misplaced_blocks
                if goal_fact_id not in state:
                    misplaced_blocks += 1
            elif goal_fact_name.startswith('ontable'):
                #If 'ontable' fact not in state, increment ontable_misplaced
                if goal_fact_id not in state:
                    ontable_misplaced +=1

        # The heuristic value is the number of misplaced blocks
        return (misplaced_blocks * self.misplaced_block_penalty) + (ontable_misplaced * self.ontable_penalty)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic29(misplaced_block_penalty={self.misplaced_block_penalty}, ontable_penalty={self.ontable_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic29(misplaced_block_penalty={self.misplaced_block_penalty}, ontable_penalty={self.ontable_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 29 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
