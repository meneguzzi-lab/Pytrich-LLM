"""
Domain-dependent heuristic for the Rover domain.

This is a WORKING EXAMPLE heuristic demonstrating:
1. How to correctly parse grounded fact names
2. How to preprocess static domain information
3. How to efficiently check state using bitwise operations
4. How to implement a domain-specific cost estimation for science missions

This heuristic estimates the cost based on:
- Uncommunicated soil data objectives
- Uncommunicated rock data objectives
- Uncommunicated image data objectives
- Whether samples have been collected but not yet communicated
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverExampleHeuristic(Heuristic):
    """
    Heuristic for Rover domain based on science mission objectives.

    Key operations in Rover:
    - navigate: Move rover between waypoints (1 action)
    - sample_soil: Collect soil sample (1 action)
    - sample_rock: Collect rock sample (1 action)
    - drop: Drop sample from store (1 action)
    - calibrate: Calibrate camera (1 action)
    - take_image: Take picture (1 action)
    - communicate_*_data: Send data to lander (1 action)

    Science objectives typically require:
    - Soil data: navigate + sample + navigate + communicate (~4 actions)
    - Rock data: navigate + sample + navigate + communicate (~4 actions)
    - Image data: calibrate + navigate + take_image + navigate + communicate (~5 actions)
    """

    def __init__(self, rover_example_heuristic=None, name="RoverExampleHeuristic"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_soil_data = set()        # waypoints needing soil data communicated
        self.goal_rock_data = set()        # waypoints needing rock data communicated
        self.goal_image_data = set()       # (objective, mode) pairs needing communication

        # Fact lookup tables
        self.communicated_soil_facts = {}   # waypoint -> fact
        self.communicated_rock_facts = {}   # waypoint -> fact
        self.communicated_image_facts = {}  # (objective, mode) -> fact
        self.have_soil_facts = {}           # (rover, waypoint) -> fact
        self.have_rock_facts = {}           # (rover, waypoint) -> fact
        self.have_image_facts = {}          # (rover, objective, mode) -> fact

        self.num_goals = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model.

        IMPORTANT: This is called ONCE before search starts.
        Do all expensive preprocessing here, not in __call__.
        """
        start_time = time.time()
        self.model = model

        # Preprocess static domain information
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.

        Handles TWO formats:
        1. Grounded format: "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]"
        2. HDDL format: "(predicate arg1 arg2)"

        Returns: (predicate, [arg1, arg2, ...]) or (None, []) if parsing fails
        """
        name = name.strip()

        # Format 1: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            # Remove the +/- prefix
            name = name[1:]
            # Find the bracket
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]  # Remove [ and ]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Format 2: (predicate arg1 arg2)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        """
        Extract domain structure from grounded facts.

        Key predicates in Rover:
        - communicated_soil_data[waypoint]: soil data sent for waypoint
        - communicated_rock_data[waypoint]: rock data sent for waypoint
        - communicated_image_data[objective,mode]: image data sent
        - have_soil_analysis[rover,waypoint]: rover has collected soil sample
        - have_rock_analysis[rover,waypoint]: rover has collected rock sample
        - have_image[rover,objective,mode]: rover has taken image
        - at[rover,waypoint]: rover location
        - empty/full[store]: rover storage state

        Goals are communicated_* facts.
        """
        # First, extract goal facts from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)

                if predicate == "communicated_soil_data" and len(args) >= 1:
                    waypoint = args[0]
                    self.goal_soil_data.add(waypoint)

                elif predicate == "communicated_rock_data" and len(args) >= 1:
                    waypoint = args[0]
                    self.goal_rock_data.add(waypoint)

                elif predicate == "communicated_image_data" and len(args) >= 2:
                    objective = args[0]
                    mode = args[1]
                    self.goal_image_data.add((objective, mode))

        self.num_goals = len(self.goal_soil_data) + len(self.goal_rock_data) + len(self.goal_image_data)

        # Build fact lookup tables
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None:
                continue

            if predicate == "communicated_soil_data" and len(args) >= 1:
                waypoint = args[0]
                self.communicated_soil_facts[waypoint] = fact

            elif predicate == "communicated_rock_data" and len(args) >= 1:
                waypoint = args[0]
                self.communicated_rock_facts[waypoint] = fact

            elif predicate == "communicated_image_data" and len(args) >= 2:
                objective = args[0]
                mode = args[1]
                self.communicated_image_facts[(objective, mode)] = fact

            elif predicate == "have_soil_analysis" and len(args) >= 2:
                rover = args[0]
                waypoint = args[1]
                self.have_soil_facts[(rover, waypoint)] = fact

            elif predicate == "have_rock_analysis" and len(args) >= 2:
                rover = args[0]
                waypoint = args[1]
                self.have_rock_facts[(rover, waypoint)] = fact

            elif predicate == "have_image" and len(args) >= 3:
                rover = args[0]
                objective = args[1]
                mode = args[2]
                self.have_image_facts[(rover, objective, mode)] = fact

    def _is_fact_true(self, state_bitwise: int, fact) -> bool:
        """Check if a fact is true in the current state using bitwise operations."""
        if fact is None:
            return False
        return (state_bitwise >> fact.global_id) & 1 == 1

    def _has_soil_sample(self, state_bitwise: int, waypoint: str):
        """Check if any rover has soil analysis for this waypoint."""
        for (rover, wp), fact in self.have_soil_facts.items():
            if wp == waypoint and self._is_fact_true(state_bitwise, fact):
                return True
        return False

    def _has_rock_sample(self, state_bitwise: int, waypoint: str):
        """Check if any rover has rock analysis for this waypoint."""
        for (rover, wp), fact in self.have_rock_facts.items():
            if wp == waypoint and self._is_fact_true(state_bitwise, fact):
                return True
        return False

    def _has_image(self, state_bitwise: int, objective: str, mode: str):
        """Check if any rover has image for this objective+mode."""
        for (rover, obj, m), fact in self.have_image_facts.items():
            if obj == objective and m == mode and self._is_fact_true(state_bitwise, fact):
                return True
        return False

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value for the given state.

        For each science objective:
        - Soil data: If communicated: 0
                     If have sample: 2 (navigate + communicate)
                     Otherwise: 4 (navigate + sample + navigate + communicate)

        - Rock data: Same as soil

        - Image data: If communicated: 0
                      If have image: 2 (navigate + communicate)
                      Otherwise: 5 (calibrate + navigate + take + navigate + communicate)

        This is admissible as it underestimates navigation and preparation costs.
        """
        total_cost = 0

        # Soil data objectives
        for waypoint in self.goal_soil_data:
            comm_fact = self.communicated_soil_facts.get(waypoint)
            if self._is_fact_true(state_bitwise, comm_fact):
                continue  # Already communicated

            if self._has_soil_sample(state_bitwise, waypoint):
                total_cost += 2  # Navigate to lander + communicate
            else:
                total_cost += 4  # Navigate + sample + navigate + communicate

        # Rock data objectives
        for waypoint in self.goal_rock_data:
            comm_fact = self.communicated_rock_facts.get(waypoint)
            if self._is_fact_true(state_bitwise, comm_fact):
                continue  # Already communicated

            if self._has_rock_sample(state_bitwise, waypoint):
                total_cost += 2  # Navigate to lander + communicate
            else:
                total_cost += 4  # Navigate + sample + navigate + communicate

        # Image data objectives
        for objective, mode in self.goal_image_data:
            comm_fact = self.communicated_image_facts.get((objective, mode))
            if self._is_fact_true(state_bitwise, comm_fact):
                continue  # Already communicated

            if self._has_image(state_bitwise, objective, mode):
                total_cost += 2  # Navigate to lander + communicate
            else:
                total_cost += 5  # Calibrate + navigate + take_image + navigate + communicate

        return total_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic value during search.

        This is called for every node expansion, so it must be FAST.
        All preprocessing was done in initialize().
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverExampleHeuristic(goals={self.num_goals})"

    def __str__(self):
        return f"RoverExampleHeuristic with {self.num_goals} science objectives"

    def __output__(self):
        return (
            f"Rover Example Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tSoil Data Goals: {self.goal_soil_data}\n"
            f"\tRock Data Goals: {self.goal_rock_data}\n"
            f"\tImage Data Goals: {self.goal_image_data}\n"
            f"\tTotal Goals: {self.num_goals}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
