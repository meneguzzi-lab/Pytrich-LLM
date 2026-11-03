import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Fact

class BlocksworldGemini25ProHeuristic15(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic calculates the number of "unsettled" blocks. A block is
    considered settled if it is in its final goal position and every block
    supporting it is also settled. This forms a recursive definition that
    checks if the tower supporting a block is correct from the table upwards.

    The heuristic value is the count of unsettled blocks multiplied by a penalty.
    Each unsettled block must be moved at least once to solve the problem,
    making this an admissible and informative heuristic.
    """

    def __init__(self, blocksworld_gemini25pro_heuristic_15, name="blocksworld_gemini25pro_heuristic_15"):
        """
        Initializes the heuristic.
        :param blocksworld_gemini25pro_heuristic_15: A dictionary containing heuristic parameters.
                                         Expected keys: 'unsettled_penalty'.
        :param name: The name of the heuristic.
        """
        super().__init__(name=name)
        self.unsettled_penalty = blocksworld_gemini25pro_heuristic_15.get("unsettled_penalty", 1.0)
        self.preprocessing_time = 0
        self.goal_on = {}
        self.goal_ontable = set()
        self.blocks = set()
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic with the problem model and initial node.
        Parses the goal state to build a representation of the goal towers
        and collects all block objects present in the problem.
        """
        start_time = time.time()
        self.model = model
        self._parse_goal_state(model.goals)
        self._collect_blocks(model)
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _parse_fact_string(self, fact_string: str):
        """Helper to parse a PDDL fact string like '(on b1 b2)' into a predicate and arguments."""
        clean_str = fact_string.strip().strip('()')
        parts = clean_str.split()
        if not parts:
            return None, []
        return parts[0], parts[1:]

    def _parse_facts_to_state(self, facts: set[Fact]):
        """Converts a set of Fact objects into structured on_map and ontable_set."""
        on_map = {}
        ontable_set = set()
        for fact in facts:
            pred, args = self._parse_fact_string(fact.name)
            if pred == 'on' and len(args) == 2:
                on_map[args[0]] = args[1]
            elif pred == 'ontable' and len(args) == 1:
                ontable_set.add(args[0])
        return on_map, ontable_set

    def _parse_goal_state(self, goal_facts: set[Fact]):
        """Parses the problem's goal facts to establish the target configuration."""
        self.goal_on, self.goal_ontable = self._parse_facts_to_state(goal_facts)

    def _collect_blocks(self, model: Model):
        """Infers all unique block names from the initial state and goal facts."""
        all_facts = model.initial_state.union(model.goals)
        for fact in all_facts:
            _, args = self._parse_fact_string(fact.name)
            for arg in args:
                self.blocks.add(arg)

    def _get_current_state_maps(self, node: HTNNode):
        """Decodes the bitwise state of a node into on_map and ontable_set."""
        current_facts = set()
        state_bitwise = node.state
        for i in range(self.model.ifacts_end + 1):
            if (state_bitwise >> i) & 1:
                current_facts.add(self.model.facts[i])
        return self._parse_facts_to_state(current_facts)

    def _calculate_node_heuristic(self, node: HTNNode):
        """Calculates the heuristic value for a given search node."""
        if not self.blocks:
            return 0

        current_on, current_ontable = self._get_current_state_maps(node)
        unsettled_count = 0
        settled_status = {}  # Memoization for settled status

        for block in self.blocks:
            if not self._is_block_settled(block, current_on, current_ontable, settled_status):
                unsettled_count += 1

        return unsettled_count * self.unsettled_penalty

    def _is_block_settled(self, block: str, current_on: dict, current_ontable: set, settled_status: dict):
        """ 
        Recursively determines if a block is settled.
        A block is settled if its support is its goal support, and its support is also settled.
        Uses memoization to avoid re-calculating for blocks in the same tower.
        """
        if block in settled_status:
            return settled_status[block]

        goal_support = self.goal_on.get(block)
        if goal_support is None:
            goal_support = "TABLE" if block in self.goal_ontable else None

        if goal_support is None:
            settled_status[block] = True
            return True

        current_support = current_on.get(block)
        if current_support is None:
            current_support = "TABLE" if block in current_ontable else None

        if current_support != goal_support:
            settled_status[block] = False
            return False

        if current_support == "TABLE":
            settled_status[block] = True
            return True

        is_settled = self._is_block_settled(current_support, current_on, current_ontable, settled_status)
        settled_status[block] = is_settled
        return is_settled

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """Calculates and returns the heuristic value for the given node."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic15(unsettled_penalty={self.unsettled_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 15 Info:\n"
            f"\tName: {self.name}\n"
            f"\tBlocks Tracked: {len(self.blocks)}\n"
            f"\tUnsettled Block Penalty: {self.unsettled_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s"
        )
