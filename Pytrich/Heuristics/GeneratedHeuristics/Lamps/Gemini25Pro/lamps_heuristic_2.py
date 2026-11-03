import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsGemini25ProHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Lamps (Lights Out) domain.

    This heuristic calculates the number of lamps that are not in their goal state.

    Bottleneck:
    The primary challenge is the cascading effect of the 'flip' action. A single 'turn' can alter many lamps non-locally, making it difficult to predict the outcome and choose the optimal initial flip to correct a specific mismatch.

    Admissibility Sketch (h_mismatch):
    1.  The heuristic counts the number of lamps whose current state (Y/N) differs from their goal state.
    2.  Each mismatched lamp must be flipped an odd number of times to reach its goal state, while every matched lamp must be flipped an even number of times.
    3.  Therefore, to correct a single mismatch, the corresponding lamp must be affected by at least one 'flip' action over the course of the entire plan.
    4.  Since each primitive 'flip' action has a cost of 1, the total cost of the plan must be at least the number of mismatched lamps. This makes the count an admissible lower bound on the remaining cost.
    """

    def __init__(self, lamps_gemini25pro_heuristic_2=None, name="LampsGemini25ProHeuristic2"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Preprocessed data
        self.goal_config = {}  # Maps (x, y) tuple -> goal status ('Y' or 'N')
        self.on_facts_map = {} # Maps (x, y, status) tuple -> fact global_id
        self.coords = set()    # Set of all (x, y) coordinates present in the problem

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model to build fast lookup tables.
        This is called once before the search begins.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into a (predicate, args) tuple.
        Handles the grounded format: "+predicate[arg1,arg2]"
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Extracts goal configuration and a map of all 'on' facts from the model.
        This populates self.goal_config, self.on_facts_map, and self.coords.
        """
        # 1. Get goal configuration from model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            predicate, args = self._parse_fact_name(name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord_tuple = (x, y)
                self.goal_config[coord_tuple] = status
                self.coords.add(coord_tuple)

        # 2. Map all possible 'on' facts for quick lookup of their global_id
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord_tuple = (x, y)
                self.on_facts_map[(x, y, status)] = fact.global_id
                self.coords.add(coord_tuple)

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Calculate the heuristic value for the given state by counting mismatched lamps.
        This method must be very fast.
        """
        mismatches = 0
        for coord in self.coords:
            goal_status = self.goal_config.get(coord)
            if goal_status is None:
                # This coordinate is not part of the goal, so we don't care about its state.
                # In this domain, all coords are expected to be in the goal.
                continue

            # Determine the current status of the lamp at 'coord'
            current_status = None
            # We check the 'Y' status. If it's not on, it must be 'N'.
            # This relies on the domain having exactly two states, Y and N.
            on_y_gid = self.on_facts_map.get((coord[0], coord[1], 'Y'))
            if on_y_gid is not None:
                if (state_bitwise >> on_y_gid) & 1:
                    current_status = 'Y'
                else:
                    current_status = 'N'
            
            if current_status is not None and current_status != goal_status:
                mismatches += 1
        
        return mismatches

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate heuristic value for a node during search.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini25ProHeuristic2(grid_size={len(self.coords)})"

    def __str__(self):
        return f"LampsGemini25ProHeuristic2 for a grid with {len(self.coords)} lamps"

    def __output__(self):
        return (
            f"Lamps Gemini 2.5 Pro Heuristic 2 Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
        )
