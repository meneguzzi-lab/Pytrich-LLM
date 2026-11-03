import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain.
class BlocksworldGemini20FlashHeuristic6(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_6="blocksworld_gemini20flash_heuristic_6", name="BlocksworldGemini20FlashHeuristic6"):
        super().__init__(name=name)
        self.ontable_weight = 1
        self.on_weight = 1.5
        self.clear_weight = 0.8
        self.handempty_weight = 0.5
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and performs any necessary preprocessing.
        """
        start_time = time.time()
        self.model = model
        self.goal_state = model.goals #Store the goal state facts for heuristic computation
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        """
        Calculates the heuristic value for a given node based on the Blocksworld domain.
        """
        state = node.state
        h = 0

        # Goal information to iterate through and find matches.
        goal_facts = self.goal_state
        # Iterate through the facts in the goal state.
        for goal_fact_id in goal_facts:
            goal_fact = self.model.get_component(goal_fact_id)
            goal_fact_name = goal_fact.name

            # Check for 'on' relationships.
            if goal_fact_name.startswith('on'):
                if goal_fact_id in state:
                    h += self.on_weight
                else:
                    h += self.on_weight * 2  # Higher penalty if 'on' is not met

            # Check for 'ontable' relationships.
            elif goal_fact_name.startswith('ontable'):
                if goal_fact_id in state:
                    h += self.ontable_weight
                else:
                    h += self.ontable_weight * 2  # Higher penalty if 'ontable' is not met

            # Check for 'clear' relationships.
            elif goal_fact_name.startswith('clear'):
                if goal_fact_id in state:
                    h += self.clear_weight
                else:
                    h += self.clear_weight * 2  # Higher penalty if 'clear' is not met

            # Check for 'handempty' relationships.
            elif goal_fact_name == 'handempty':
                if goal_fact_id in state:
                    h += self.handempty_weight
                else:
                    h += self.handempty_weight * 2  # Higher penalty if 'handempty' is not met

        return h

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic6(ontable={self.ontable_weight}, on={self.on_weight}, clear={self.clear_weight}, handempty={self.handempty_weight})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic6(ontable={self.ontable_weight}, on={self.on_weight}, clear={self.clear_weight}, handempty={self.handempty_weight})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 6 info:\n"
            f"\tName: {self.name}\n"
            f"\tOntable Weight: {self.ontable_weight}\n"
            f"\tOn Weight: {self.on_weight}\n"
            f"\tClear Weight: {self.clear_weight}\n"
            f"\tHandempty Weight: {self.handempty_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
