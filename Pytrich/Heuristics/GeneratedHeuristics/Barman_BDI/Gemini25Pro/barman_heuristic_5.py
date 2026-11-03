import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, AbstractTask

class BarmanGemini25ProHeuristic5(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    Rationale:
    The Barman planning problem is heavily constrained by resource contention and
    sequential dependencies, primarily centered around the use of a single shaker
    and the two hands of the barman. This heuristic estimates the remaining cost
    to achieve the goal by focusing on the most significant bottleneck: the
    serial process of making multiple cocktails.

    Heuristic Calculation Strategy:
    1.  Goal Counting: The heuristic first identifies the number of top-level goals
        (AchieveContainsShotCocktail) that are not yet satisfied in the current state.

    2.  Cost-per-Goal Estimation: For each unsatisfied goal, a minimum number of actions
        is estimated. This cost is broken down into two main components:
        - Making Cost: The primitive actions required to mix a cocktail (filling shots
          with ingredients, pouring them into the shaker, and shaking). A lower bound of
          6 actions is used (fill, pour, fill, pour, shake, pour).
        - Cleaning Cost: Since there is only one shaker, it must be cleaned after each
          cocktail is made to prepare the next one. This adds a cost for emptying and
          washing the shaker, estimated at a lower bound of 2 actions.

    3.  Sequential Bottleneck Modeling: The total cost is calculated by summing the costs
        for all remaining cocktails. The calculation explicitly models the sequential
        nature of the task: an initial cleaning cost (if the shaker starts dirty), the
        making cost for each cocktail, and the cleaning cost between each cocktail.

    4.  Admissibility: The heuristic is designed to be admissible (h(n) <= h*(n)). It only
        counts a subset of the necessary primitive actions (mixing and cleaning the shaker)
        while ignoring others like grasping, dropping, and cleaning shot glasses. This ensures
        the estimate is a consistent lower bound on the true remaining cost.

    5.  Efficiency: The heuristic is computationally efficient. It performs a one-time
        preprocessing step (`initialize`) to cache information about the shaker and relevant
        facts. The per-node calculation (`__call__`) involves a simple loop over the
        current task network and basic arithmetic, avoiding expensive state lookups.
    """

    def __init__(self, barman_gemini25pro_heuristic_5, cost_making=6, cost_cleaning=2):
        """
        Initializes the Barman heuristic.

        Args:
            barman_gemini25pro_heuristic_5 (str): The name for this heuristic instance, as per task requirement.
            cost_making (int): The estimated minimum action cost to make one cocktail.
            cost_cleaning (int): The estimated minimum action cost to clean the shaker.
        """
        super().__init__(name=barman_gemini25pro_heuristic_5)
        self.cost_making = cost_making
        self.cost_cleaning = cost_cleaning
        self.model = None
        self.fact_name_to_id = {}
        self.shaker_name = None
        self.clean_shaker_fact_id = -1
        self.empty_shaker_fact_id = -1
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by preprocessing the model to find key information
        like the shaker's name and IDs of relevant state facts.
        """
        start_time = time.time()
        self.model = model

        # Precompute fact name to ID mapping for quick lookups during evaluation.
        self.fact_name_to_id = {fact.name: fact.global_id for fact in self.model.facts}

        # Dynamically find the shaker's name and its relevant fact IDs.
        self._find_shaker_info()

        self.preprocessing_time = time.time() - start_time

        # Calculate the initial heuristic value for the root node.
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _find_shaker_info(self):
        """
        Finds the shaker object name and the global IDs for its 'clean' and 'empty'
        predicates by inspecting the model's facts. This makes the heuristic robust
        to different object names in problem files.
        """
        for fact_name in self.fact_name_to_id.keys():
            if fact_name.startswith("(shakerlevel "):
                self.shaker_name = fact_name.split()[1]
                break

        if self.shaker_name:
            clean_fact_str = f"(clean {self.shaker_name})"
            empty_fact_str = f"(empty {self.shaker_name})"
            self.clean_shaker_fact_id = self.fact_name_to_id.get(clean_fact_str, -1)
            self.empty_shaker_fact_id = self.fact_name_to_id.get(empty_fact_str, -1)

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for a given search node.
        The value is an estimate of the minimum number of actions to complete the plan.
        """
        state_bitmask = node.state

        # 1. Count unsatisfied 'AchieveContainsShotCocktail' goals in the current task network.
        num_remaining_goals = 0
        for task in node.task_network:
            if isinstance(task, AbstractTask) and task.name.startswith("(achievecontainsshotcocktail"):
                try:
                    parts = task.name.replace("(", "").replace(")", "").split()
                    shot, cocktail = parts[1], parts[2]
                    goal_fact_str = f"(contains {shot} {cocktail})"
                    goal_fact_id = self.fact_name_to_id.get(goal_fact_str)

                    if goal_fact_id is not None and not ((state_bitmask >> goal_fact_id) & 1):
                        num_remaining_goals += 1
                except (IndexError, KeyError):
                    continue

        if num_remaining_goals == 0:
            super().update_info(0)
            return 0

        # 2. Calculate heuristic value based on the number of cocktails to prepare.
        h_value = 0

        # Add the cost for making all remaining cocktails.
        h_value += num_remaining_goals * self.cost_making

        # Add the cost for cleaning the shaker between each cocktail (N-1 times for N cocktails).
        if num_remaining_goals > 1:
            h_value += (num_remaining_goals - 1) * self.cost_cleaning

        # 3. Add a one-time cost for the initial cleaning if the shaker is not ready.
        if self.shaker_name and self.clean_shaker_fact_id != -1 and self.empty_shaker_fact_id != -1:
            is_shaker_clean = (state_bitmask >> self.clean_shaker_fact_id) & 1
            is_shaker_empty = (state_bitmask >> self.empty_shaker_fact_id) & 1

            if not is_shaker_clean or not is_shaker_empty:
                h_value += self.cost_cleaning

        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic5(name={self.name})"

    def __str__(self):
        return f"BarmanGemini25ProHeuristic5(making_cost={self.cost_making}, cleaning_cost={self.cost_cleaning})"

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 5 Info:\n"
            f"\tName: {self.name}\n"
            f"\tCost per Cocktail (Making): {self.cost_making}\n"
            f"\tCost per Cocktail (Cleaning): {self.cost_cleaning}\n"
            f"\tDetected Shaker: {self.shaker_name}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )