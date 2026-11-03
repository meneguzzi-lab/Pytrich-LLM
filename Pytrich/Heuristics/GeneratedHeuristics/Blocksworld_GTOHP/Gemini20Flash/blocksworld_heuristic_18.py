import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic18(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_18: str = "blocksworld_gemini20flash_heuristic_18"):
        super().__init__(name=blocksworld_gemini20flash_heuristic_18)
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates a heuristic value for the given node based on the Blocksworld domain.

        This heuristic estimates the number of 'on' goals that are not yet satisfied
        in the current state. It iterates through the goal state and checks if each
        'on' predicate is present in the current state or can be achieved through
        a simple stacking operation. If a goal cannot be trivially satisfied, it's
        considered an unsatisfied goal, and the heuristic value is incremented.
        """
        unsatisfied_goals = 0

        if self.model is None:
            return 0

        goal_state = self.model.goals  # Get the goal state from the model.

        current_state = self._get_state_from_node(node)
        if current_state is None:
            return 0

        for goal in goal_state:
            goal_str = str(self.model.get_component(goal))
            if goal_str.startswith('on'):
                # Check if the goal is already satisfied in the current state.
                if goal in current_state:
                    continue

                # If not, increment the count of unsatisfied goals.
                unsatisfied_goals += 1

        # Return the number of unsatisfied goals as the heuristic estimate.
        return unsatisfied_goals

    def _get_state_from_node(self, node):
        """
        Extracts the state (as a set of fact IDs) from the given node.

        Handles both HTNNode (for hierarchical search) and potentially other
        node types. Returns None if the node type is not supported.
        """
        if isinstance(node, HTNNode):
            return node.state
        else:
            return None # Or raise an exception for unsupported node types.

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(name={self.name})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(name={self.name})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
