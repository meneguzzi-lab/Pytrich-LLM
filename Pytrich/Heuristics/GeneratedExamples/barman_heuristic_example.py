"""
Domain-dependent heuristic for the Barman domain.

This is a WORKING EXAMPLE heuristic demonstrating:
1. How to correctly parse grounded fact names
2. How to preprocess static domain information
3. How to efficiently check state using bitwise operations
4. How to implement a domain-specific cost estimation for cocktail preparation

This heuristic estimates the cost to achieve all required cocktails based on:
- Whether cocktails are already served
- Whether the shaker contains the cocktail (already shaked)
- How many ingredients are still needed
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanExampleHeuristic(Heuristic):
    """
    Heuristic for Barman domain based on cocktail preparation steps.

    Making a cocktail requires roughly these steps:
    1. Fill shot with ingredient 1 (2 actions: fill + pour to shaker)
    2. Fill shot with ingredient 2 (2 actions: fill + pour to shaker)
    3. Shake the shaker (1 action)
    4. Pour to shot glass (1 action)
    Total: ~6 actions per cocktail minimum

    This heuristic:
    1. Identifies goal cocktails from the task network (contains[shot,cocktail])
    2. For each cocktail not yet served, estimates remaining work
    3. Gives credit for partial progress (ingredients in shaker, shaked state)
    """

    def __init__(self, barman_example_heuristic=None, name="BarmanExampleHeuristic"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_cocktails = {}      # shot -> cocktail that should be in it
        self.cocktail_parts = {}      # cocktail -> (ingredient1, ingredient2)
        self.contains_facts = {}      # (container, beverage) -> fact
        self.shaked_facts = {}        # shaker -> fact for shaked predicate
        self.clean_facts = {}         # container -> fact for clean predicate
        self.empty_facts = {}         # container -> fact for empty predicate
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

        Key predicates in Barman:
        - contains[container,beverage]: What's in each container
        - shaked[shaker]: The shaker has been shaked (cocktail is ready)
        - clean[container]: Container is clean
        - empty[container]: Container is empty
        - cocktailPart1/cocktailPart2[cocktail,ingredient]: Recipe

        Goals are typically contains[shot,cocktail] facts.
        """
        # First, extract goal facts from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "contains" and len(args) >= 2:
                    container = args[0]
                    beverage = args[1]
                    # If container is a shot and beverage is a cocktail
                    if container.startswith("shot") and beverage.startswith("cocktail"):
                        self.goal_cocktails[container] = beverage

        self.num_goals = len(self.goal_cocktails)

        # Build fact lookup tables
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None:
                continue

            if predicate == "contains" and len(args) >= 2:
                container = args[0]
                beverage = args[1]
                self.contains_facts[(container, beverage)] = fact

            elif predicate == "shaked" and len(args) >= 1:
                shaker = args[0]
                self.shaked_facts[shaker] = fact

            elif predicate == "clean" and len(args) >= 1:
                container = args[0]
                self.clean_facts[container] = fact

            elif predicate == "empty" and len(args) >= 1:
                container = args[0]
                self.empty_facts[container] = fact

            elif predicate == "cocktailPart1" and len(args) >= 2:
                cocktail = args[0]
                ingredient = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ingredient

            elif predicate == "cocktailPart2" and len(args) >= 2:
                cocktail = args[0]
                ingredient = args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ingredient

    def _is_fact_true(self, state_bitwise: int, fact) -> bool:
        """Check if a fact is true in the current state using bitwise operations."""
        if fact is None:
            return False
        return (state_bitwise >> fact.global_id) & 1 == 1

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value for the given state.

        For each goal cocktail (shot should contain cocktail):
        - If shot already contains cocktail: 0
        - If shaker contains cocktail (shaked): ~2 (pour + maybe clean shot)
        - Otherwise: ~6 (fill x2, pour x2, shake, pour)

        This is admissible as it underestimates the true cost.
        """
        if not self.goal_cocktails:
            return 0

        total_cost = 0

        for shot, cocktail in self.goal_cocktails.items():
            # Check if shot already contains the cocktail (goal satisfied)
            contains_fact = self.contains_facts.get((shot, cocktail))
            if self._is_fact_true(state_bitwise, contains_fact):
                continue  # Goal already achieved

            # Check if any shaker contains the cocktail (partially done)
            shaker_has_cocktail = False
            shaker_is_shaked = False
            for (container, beverage), fact in self.contains_facts.items():
                if beverage == cocktail and container.startswith("shaker"):
                    if self._is_fact_true(state_bitwise, fact):
                        shaker_has_cocktail = True
                        # Check if this shaker is shaked
                        shaked_fact = self.shaked_facts.get(container)
                        if self._is_fact_true(state_bitwise, shaked_fact):
                            shaker_is_shaked = True
                        break

            if shaker_is_shaked:
                # Cocktail is ready in shaker, just need to pour
                # Need: clean shot (maybe), pour shaker to shot
                shot_clean = self.clean_facts.get(shot)
                if self._is_fact_true(state_bitwise, shot_clean):
                    total_cost += 1  # Just pour
                else:
                    total_cost += 3  # Empty, clean, pour
            elif shaker_has_cocktail:
                # Shaker has cocktail but not shaked yet
                total_cost += 4  # Shake + pour (+ maybe clean shot)
            else:
                # Need full preparation
                # Fill shot with ing1, pour to shaker, fill shot with ing2,
                # pour to shaker, shake, pour to shot
                total_cost += 6

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
        return f"BarmanExampleHeuristic(goals={self.num_goals})"

    def __str__(self):
        return f"BarmanExampleHeuristic with {self.num_goals} cocktail goals"

    def __output__(self):
        return (
            f"Barman Example Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Goal Cocktails: {self.num_goals}\n"
            f"\tGoal Cocktails: {self.goal_cocktails}\n"
            f"\tCocktail Recipes: {self.cocktail_parts}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
