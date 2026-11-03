import time
from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini25ProHeuristic5(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost to reach the goal state by counting the number of "misplaced" blocks.
    A block is considered misplaced if it is not resting on the correct final support (another block or the table),
    or if its support, while correct, is itself misplaced. This creates a recursive definition that correctly
    identifies blocks that are part of "correct towers" versus those that need to be moved.

    Heuristic value is calculated as:
    h(n) = (number of misplaced blocks * penalty_per_block) + (penalty_if_hand_not_empty)

    - Each misplaced block requires at least two actions (e.g., unstack and stack).
    - A block held in the hand requires at least one action (e.g., stack or putdown).
    This heuristic is admissible and provides a good estimate of the remaining work.
    """

    def __init__(self, blocksworld_gemini25pro_heuristic_5={}, misplaced_block_penalty=2, hand_penalty=1):
        """
        Initializes the BlocksworldGemini25ProHeuristic.
        
        :param blocksworld_gemini25pro_heuristic_5: A dictionary of parameters (as per user request).
        :param misplaced_block_penalty: The cost assigned to each misplaced block. Should be >= 2 for admissibility.
        :param hand_penalty: The cost assigned if the robot's hand is not empty. Should be >= 1 for admissibility.
        """
        super().__init__(name="blocksworld_gemini25pro_heuristic_5")
        # Allow overriding defaults from the parameter dictionary
        self.misplaced_block_penalty = blocksworld_gemini25pro_heuristic_5.get('misplaced_block_penalty', misplaced_block_penalty)
        self.hand_penalty = blocksworld_gemini25pro_heuristic_5.get('hand_penalty', hand_penalty)
        
        self.goal_config = {}
        self.all_blocks = set()
        self.model = None
        self.preprocessing_time = 0

    def _parse_config_from_facts(self, facts):
        """
        Parses a set of facts to create a configuration dictionary.
        Maps each block to its support (another block or 'table').
        
        :param facts: A set of Fact objects.
        :return: A dictionary representing the block configuration.
        """
        config = {}
        for fact in facts:
            fact_name = fact.name.strip()
            if fact_name.startswith('(on '):
                parts = fact_name.replace('(', '').replace(')', '').split()
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    config[block_on_top] = block_below
            elif fact_name.startswith('(ontable '):
                parts = fact_name.replace('(', '').replace(')', '').split()
                if len(parts) == 2:
                    block = parts[1]
                    config[block] = 'table'
        return config

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and pre-computes goal information.
        """
        start_time = time.time()
        self.model = model
        
        # Parse goal configuration once and store it.
        self.goal_config = self._parse_config_from_facts(model.goals)
        
        # Discover all unique block names from the domain facts.
        for fact in model.facts:
            parts = fact.name.replace('(', '').replace(')', '').split()
            if not parts:
                continue
            predicate = parts[0].lower()
            if predicate in ['on', 'ontable', 'clear']:
                for obj in parts[1:]:
                    self.all_blocks.add(obj)

        self.preprocessing_time = time.time() - start_time
        
        # Calculate heuristic for the initial state.
        h_value = self._calculate_heuristic(initial_node.state_facts)
        return super().initialize(model, h_value)

    def _is_misplaced(self, block, current_config, memo):
        """
        Recursively determines if a block is misplaced with memoization.
        A block is misplaced if its support is not its goal support, or if its
        support is itself misplaced.
        
        :param block: The block to check.
        :param current_config: The current configuration of all blocks.
        :param memo: A dictionary for memoization to avoid re-computation.
        :return: True if the block is misplaced, False otherwise.
        """
        if block in memo:
            return memo[block]

        if block not in self.goal_config:
            memo[block] = False
            return False
        
        goal_support = self.goal_config.get(block)
        current_support = current_config.get(block)

        if goal_support != current_support:
            memo[block] = True
            return True
    
        if goal_support == 'table':
            memo[block] = False
            return False
    
        result = self._is_misplaced(goal_support, current_config, memo)
        memo[block] = result
        return result

    def _calculate_heuristic(self, current_state_facts):
        """
        Calculates the heuristic value for a given state.
        
        :param current_state_facts: A set of Fact objects for the current state.
        :return: The calculated integer heuristic value.
        """
        current_config = self._parse_config_from_facts(current_state_facts)
        memo = {}  # Memoization for the recursive check
        misplaced_count = 0

        for block in self.all_blocks:
            if self._is_misplaced(block, current_config, memo):
                misplaced_count += 1
            
        h_value = misplaced_count * self.misplaced_block_penalty
    
        is_hand_empty = any(fact.name.strip() == '(handempty)' for fact in current_state_facts)
        if not is_hand_empty:
            h_value += self.hand_penalty
            
        return h_value

    def __call__(self, parent_node, node):
        """
        The main entry point for the planner to get a heuristic value for a node.
        
        :param parent_node: The parent node in the search tree.
        :param node: The current node to evaluate.
        :return: The heuristic value.
        """
        h_value = self._calculate_heuristic(node.state_facts)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic5(penalty={self.misplaced_block_penalty}, hand_penalty={self.hand_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 5 info:\n"
            f"\tName: {self.name}\n"
            f"\tBlocks tracked: {len(self.all_blocks)}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tHand not empty penalty: {self.hand_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
