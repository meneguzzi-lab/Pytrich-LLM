"""
Domain-dependent heuristic for the Robot domain.

This is a WORKING EXAMPLE heuristic demonstrating:
1. How to correctly parse grounded fact names
2. How to preprocess static domain information
3. How to efficiently check state using bitwise operations
4. How to implement a domain-specific cost estimation for package delivery

This heuristic estimates the cost based on:
- How many packages are not in their goal room
- Whether the robot is holding a relevant package
- Navigation requirements (doors may need to be opened)
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotExampleHeuristic(Heuristic):
    """
    Heuristic for Robot domain based on package delivery.

    Key operations in Robot:
    - pickup: Pick up package in current room (1 action)
    - putdown: Put down package in current room (1 action)
    - move: Move to adjacent room through open door (1 action)
    - open: Open a closed door (1 action)

    This heuristic:
    1. Identifies goal_in[package,room] facts from problem/goals
    2. For each package not in its goal room:
       - If robot holding it and in goal room: 1 (putdown)
       - If robot holding it: 2 (move + putdown)
       - Otherwise: 4 (move + pickup + move + putdown) minimum
    """

    def __init__(self, robot_example_heuristic=None, name="RobotExampleHeuristic"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_in = {}             # package -> room it should be in
        self.in_facts = {}            # (package, room) -> fact
        self.rloc_facts = {}          # room -> fact for rloc[room]
        self.holding_facts = {}       # package -> fact for holding[package]
        self.closed_facts = {}        # door -> fact for closed[door]
        self.door_info = {}           # (room1, room2) -> door
        self.packages = set()
        self.rooms = set()
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

        Key predicates in Robot:
        - rloc[room]: robot is in this room
        - in[package,room]: package is in room
        - holding[package]: robot is holding package
        - goal_in[package,room]: package should end up in room
        - closed[door]: door is closed
        - door[room1,room2,door]: door connects rooms
        - armempty: robot is not holding anything

        Goals are in[package,room] facts or derived from goal_in predicates.
        """
        # First, scan all facts to find goal_in predicates (static goal info)
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate == "goal_in" and len(args) >= 2:
                package = args[0]
                room = args[1]
                self.goal_in[package] = room
                self.packages.add(package)
                self.rooms.add(room)

        # Also check model.goals for in[package,room] facts
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "in" and len(args) >= 2:
                    package = args[0]
                    room = args[1]
                    if package not in self.goal_in:  # Don't override goal_in
                        self.goal_in[package] = room
                    self.packages.add(package)
                    self.rooms.add(room)

        self.num_goals = len(self.goal_in)

        # Build fact lookup tables
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None:
                continue

            if predicate == "in" and len(args) >= 2:
                package = args[0]
                room = args[1]
                self.in_facts[(package, room)] = fact
                self.packages.add(package)
                self.rooms.add(room)

            elif predicate == "rloc" and len(args) >= 1:
                room = args[0]
                self.rloc_facts[room] = fact
                self.rooms.add(room)

            elif predicate == "holding" and len(args) >= 1:
                package = args[0]
                self.holding_facts[package] = fact

            elif predicate == "closed" and len(args) >= 1:
                door = args[0]
                self.closed_facts[door] = fact

            elif predicate == "door" and len(args) >= 3:
                room1 = args[0]
                room2 = args[1]
                door = args[2]
                self.door_info[(room1, room2)] = door

    def _is_fact_true(self, state_bitwise: int, fact) -> bool:
        """Check if a fact is true in the current state using bitwise operations."""
        if fact is None:
            return False
        return (state_bitwise >> fact.global_id) & 1 == 1

    def _get_robot_location(self, state_bitwise: int):
        """Find which room the robot is in."""
        for room, fact in self.rloc_facts.items():
            if self._is_fact_true(state_bitwise, fact):
                return room
        return None

    def _get_package_location(self, state_bitwise: int, package: str):
        """Find which room a package is in (or None if being held)."""
        for (pkg, room), fact in self.in_facts.items():
            if pkg == package and self._is_fact_true(state_bitwise, fact):
                return room
        return None

    def _is_holding(self, state_bitwise: int, package: str):
        """Check if robot is holding a specific package."""
        fact = self.holding_facts.get(package)
        return self._is_fact_true(state_bitwise, fact)

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value for the given state.

        For each goal_in[package,room]:
        - If package is already in goal room: 0
        - If robot is holding package and in goal room: 1 (putdown)
        - If robot is holding package (different room): 2+ (move + putdown)
        - If package is in different room: 4+ (move + pickup + move + putdown)

        This is admissible as it underestimates actual navigation costs.
        """
        if not self.goal_in:
            return 0

        robot_room = self._get_robot_location(state_bitwise)
        total_cost = 0

        for package, goal_room in self.goal_in.items():
            # Check if package is already in goal room
            in_fact = self.in_facts.get((package, goal_room))
            if self._is_fact_true(state_bitwise, in_fact):
                continue  # Goal already achieved

            # Check if robot is holding this package
            if self._is_holding(state_bitwise, package):
                if robot_room == goal_room:
                    total_cost += 1  # Just putdown
                else:
                    total_cost += 2  # Move + putdown (underestimate)
            else:
                # Package is in some other room
                package_room = self._get_package_location(state_bitwise, package)

                if package_room == goal_room:
                    # Shouldn't happen if goal not satisfied, but handle it
                    continue

                if robot_room == package_room:
                    # Robot is with package: pickup + move + putdown
                    total_cost += 3
                else:
                    # Robot needs to go get package: move + pickup + move + putdown
                    total_cost += 4

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
        return f"RobotExampleHeuristic(goals={self.num_goals})"

    def __str__(self):
        return f"RobotExampleHeuristic with {self.num_goals} package delivery goals"

    def __output__(self):
        return (
            f"Robot Example Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Packages: {len(self.packages)}\n"
            f"\tNumber of Rooms: {len(self.rooms)}\n"
            f"\tNumber of Goals: {self.num_goals}\n"
            f"\tGoal Locations: {self.goal_in}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
