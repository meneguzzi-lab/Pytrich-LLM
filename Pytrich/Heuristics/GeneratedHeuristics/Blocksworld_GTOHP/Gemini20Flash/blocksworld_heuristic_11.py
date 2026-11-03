import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic11(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_11=1, name="BlocksworldGemini20FlashHeuristic11"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0
        self.ontable_penalty = 1
        self.onground_penalty = 2
        self.wrong_order_penalty = 3
        self.handempty_reward = -1

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model.
        """
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        initial_h_value = self._calculate_heuristic(initial_node)

        return super().initialize(model, initial_h_value)

    def _calculate_heuristic(self, node):
        """
        Calculate the heuristic value for a given node.
        """
        state = node.state
        goal_facts = self.model.goals
        h_value = 0

        # Check if handempty
        handempty = False
        for fact in self.model.facts:
          if fact.name == "handempty" and fact.global_id in state:
            handempty = True
            break
        if handempty: h_value += self.handempty_reward

        # Count the number of blocks that are not on the table but should be
        # or blocks that are in the wrong order
        for goal_fact in goal_facts:
            if "ontable" in goal_fact.name:
              if goal_fact.global_id not in state:
                h_value += self.ontable_penalty
            elif "on" in goal_fact.name:
              if goal_fact.global_id not in state:
                  h_value += self.onground_penalty

        # Check for blocks in wrong order in the initial state
        # Iterate through goal facts and check if they are in the initial state
        wrong_order_count = 0
        for goal_fact in goal_facts:
            if "on" in goal_fact.name:
                # Extract the two blocks from the fact name
                parts = goal_fact.name.split(" ")
                if len(parts) == 3:
                    block1 = parts[1]
                    block2 = parts[2]

                    # Check if the blocks are in the correct order in the current state
                    correct_order = False

                    # Iterate through state facts to find the relationships
                    for state_fact_id in state:
                      state_fact = self.model.get_component(state_fact_id)
                      if "on" in state_fact.name:
                        state_parts = state_fact.name.split(" ")
                        if len(state_parts) == 3 and state_parts[1] == block1 and state_parts[2] == block2:
                          correct_order = True
                          break

                    if not correct_order:
                      wrong_order_count += 1

        # Apply penalty for the wrong order
        h_value += self.wrong_order_penalty * wrong_order_count

        return h_value


    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
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
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
