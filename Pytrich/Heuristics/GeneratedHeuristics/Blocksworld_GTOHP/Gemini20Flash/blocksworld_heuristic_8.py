import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic8(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_8 = "BlocksworldGemini20FlashHeuristic8", misplaced_block_penalty=1, incorrect_order_penalty=0.5):
        super().__init__(name=blocksworld_gemini20flash_heuristic_8)
        self.misplaced_block_penalty = misplaced_block_penalty
        self.incorrect_order_penalty = incorrect_order_penalty
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
            self.method_scores[method_name] = {}

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'state') or not node.state:
            return 0

        state = node.state
        goal_state = self.model.goals

        misplaced_blocks = 0
        incorrect_orderings = 0

        # Convert state and goal_state bitwise representations to explicit lists of facts
        current_facts = self.model.state_explicit_repr(state)
        goal_facts = self.model.state_explicit_repr(goal_state)

        # Simplified goal parsing (assuming 'on' predicates indicate stack order)
        goal_stacks = {}
        for fact in goal_facts:
            if fact.startswith('(on'):
                parts = fact.split()
                block1 = parts[1]
                block2 = parts[2].replace(')', '')
                goal_stacks[block1] = block2

        # Count misplaced blocks and incorrect orderings
        for fact in current_facts:
            if fact.startswith('(on'):
                parts = fact.split()
                block1 = parts[1]
                block2 = parts[2].replace(')', '')

                # Check if this 'on' fact is in the goal
                if fact not in goal_facts:
                    misplaced_blocks += 1
                else:
                    # Check if the order is correct
                    if block1 in goal_stacks and goal_stacks[block1] != block2:
                        incorrect_orderings += 1
            elif fact.startswith('(ontable'):
                block = fact.split()[1].replace(')', '')
                # Check if block should be on the table in the goal
                goal_ontable = f'(ontable {block})'
                if goal_ontable not in goal_facts:
                    misplaced_blocks += 1
            elif fact.startswith('(clear'):
                block = fact.split()[1].replace(')', '')
                goal_clear = f'(clear {block})'
                if goal_clear not in goal_facts:
                    misplaced_blocks +=1
            elif fact.startswith('(handempty)'):
                goal_handempty = '(handempty)'
                if goal_handempty not in goal_facts:
                    misplaced_blocks +=1

        # Heuristic value calculation
        h_value = (misplaced_blocks * self.misplaced_block_penalty) + \
                  (incorrect_orderings * self.incorrect_order_penalty)

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic8(misplaced={self.misplaced_block_penalty}, order={self.incorrect_order_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic8(misplaced={self.misplaced_block_penalty}, order={self.incorrect_order_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 8 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tIncorrect order penalty: {self.incorrect_order_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
