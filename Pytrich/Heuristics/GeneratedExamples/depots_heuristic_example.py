"""
Domain-dependent heuristic for the Depots domain.

This is a WORKING EXAMPLE heuristic demonstrating:
1. How to correctly parse grounded fact names
2. How to preprocess static domain information
3. How to efficiently check state using bitwise operations
4. How to implement a domain-specific cost estimation for logistics

This heuristic estimates the cost based on:
- How many crates are not on their goal surface
- Whether transportation between locations is needed
- The state of crates (on surface, in truck, being lifted)
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsExampleHeuristic(Heuristic):
    """
    Heuristic for Depots domain based on crate positioning.

    Key operations in Depots:
    - Lift: Pick up crate from surface (1 action)
    - Drop: Put crate on surface (1 action)
    - Load: Load crate onto truck (1 action)
    - Unload: Unload crate from truck (1 action)
    - Drive: Move truck between places (1 action)

    This heuristic:
    1. Identifies goal on[crate,surface] relationships from model.goals
    2. For each crate not on its goal surface:
       - If at same location: 2 actions (lift + drop)
       - If at different location: 5 actions (lift + load + drive + unload + drop)
    """

    def __init__(self, depots_example_heuristic=None, name="DepotsExampleHeuristic"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_on = {}             # crate -> surface it should be on
        self.on_facts = {}            # (crate, surface) -> fact
        self.at_facts = {}            # (object, place) -> fact
        self.in_facts = {}            # (crate, truck) -> fact
        self.lifting_facts = {}       # (hoist, crate) -> fact
        self.crates = set()
        self.surfaces = set()
        self.places = set()
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

        Key predicates in Depots:
        - on[crate,surface]: crate is on surface (pallet or another crate)
        - at[object,place]: object is at a place (depot/distributor)
        - in[crate,truck]: crate is loaded in truck
        - lifting[hoist,crate]: hoist is holding crate
        - clear[surface]: nothing on top of surface
        - available[hoist]: hoist is free

        Goals are on[crate,surface] facts indicating desired positions.
        """
        # First, extract goal facts from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    crate = args[0]
                    surface = args[1]
                    self.goal_on[crate] = surface
                    self.crates.add(crate)
                    self.surfaces.add(surface)

        self.num_goals = len(self.goal_on)

        # Build fact lookup tables
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None:
                continue

            if predicate == "on" and len(args) >= 2:
                crate = args[0]
                surface = args[1]
                self.on_facts[(crate, surface)] = fact
                self.crates.add(crate)
                self.surfaces.add(surface)

            elif predicate == "at" and len(args) >= 2:
                obj = args[0]
                place = args[1]
                self.at_facts[(obj, place)] = fact
                self.places.add(place)

            elif predicate == "in" and len(args) >= 2:
                crate = args[0]
                truck = args[1]
                self.in_facts[(crate, truck)] = fact

            elif predicate == "lifting" and len(args) >= 2:
                hoist = args[0]
                crate = args[1]
                self.lifting_facts[(hoist, crate)] = fact

    def _is_fact_true(self, state_bitwise: int, fact) -> bool:
        """Check if a fact is true in the current state using bitwise operations."""
        if fact is None:
            return False
        return (state_bitwise >> fact.global_id) & 1 == 1

    def _get_object_location(self, state_bitwise: int, obj: str):
        """Find where an object is located."""
        for (o, place), fact in self.at_facts.items():
            if o == obj and self._is_fact_true(state_bitwise, fact):
                return place
        return None

    def _is_crate_in_truck(self, state_bitwise: int, crate: str):
        """Check if crate is loaded in any truck."""
        for (c, truck), fact in self.in_facts.items():
            if c == crate and self._is_fact_true(state_bitwise, fact):
                return True
        return False

    def _is_crate_being_lifted(self, state_bitwise: int, crate: str):
        """Check if crate is being held by any hoist."""
        for (hoist, c), fact in self.lifting_facts.items():
            if c == crate and self._is_fact_true(state_bitwise, fact):
                return True
        return False

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value for the given state.

        For each goal on[crate,surface]:
        - If crate is already on goal surface: 0
        - If crate is being lifted: 1 (just drop)
        - If crate is in truck: 3 (unload + maybe drive + drop)
        - If crate is at same location as goal: 2 (lift + drop)
        - If crate is at different location: 5 (lift + load + drive + unload + drop)

        This is admissible as it underestimates actual costs.
        """
        if not self.goal_on:
            return 0

        total_cost = 0

        for crate, goal_surface in self.goal_on.items():
            # Check if already on goal surface
            on_fact = self.on_facts.get((crate, goal_surface))
            if self._is_fact_true(state_bitwise, on_fact):
                continue  # Goal already achieved

            # Check if crate is being lifted
            if self._is_crate_being_lifted(state_bitwise, crate):
                total_cost += 1  # Just need to drop
                continue

            # Check if crate is in a truck
            if self._is_crate_in_truck(state_bitwise, crate):
                total_cost += 3  # Unload + (maybe drive) + drop
                continue

            # Crate is on some surface, need to move it
            crate_location = self._get_object_location(state_bitwise, crate)
            goal_surface_location = self._get_object_location(state_bitwise, goal_surface)

            if crate_location == goal_surface_location:
                # Same location: lift + drop
                total_cost += 2
            else:
                # Different location: lift + load + drive + unload + drop
                total_cost += 5

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
        return f"DepotsExampleHeuristic(goals={self.num_goals})"

    def __str__(self):
        return f"DepotsExampleHeuristic with {self.num_goals} crate placement goals"

    def __output__(self):
        return (
            f"Depots Example Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Crates: {len(self.crates)}\n"
            f"\tNumber of Places: {len(self.places)}\n"
            f"\tNumber of Goals: {self.num_goals}\n"
            f"\tGoal Configurations: {self.goal_on}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
