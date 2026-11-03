import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini20FlashHeuristic24(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_24=1.0, name="RobotGemini20FlashHeuristic24"):
        super().__init__(name=name)
        self.reach_goal_penalty = robot_gemini20flash_heuristic_24
        self.preprocessing_time = 0


    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        """
        Calculates a heuristic value for a given node based on the current state and goal conditions.

        This heuristic estimates the number of steps needed to achieve the goals by considering:
        - The number of packages that are not in their goal locations.
        - Whether the robot's arm is empty.
        - Number of closed doors

        Each of these factors contributes to the overall heuristic value, providing a basis for the planner
        to prioritize actions that lead towards satisfying the goal conditions efficiently.
        """

        if not self.model:
            return 0

        state = node.state if hasattr(node, 'state') else set()
        goal_facts = self.model.goals if hasattr(self.model, 'goals') else set()

        packages_in_wrong_locations = 0
        arm_empty = True
        closed_doors = 0

        # Get Fact name from ID in state and goal
        state_fact_names = [self.model.facts[fact_id].name for fact_id in range(len(self.model.facts)) if state & 1 << fact_id]
        goal_fact_names  = [self.model.facts[fact_id].name for fact_id in range(len(self.model.facts)) if goal_facts & 1 << fact_id]

        # Count packages not in goal location
        for fact_name in state_fact_names:
            if 'in' in fact_name and fact_name not in goal_fact_names:
                 packages_in_wrong_locations += 1

        # Determine if the arm is empty
        if any('holding' in fact for fact in state_fact_names):
            arm_empty = False

        # Count closed doors
        closed_doors = sum('closed' in fact for fact in state_fact_names)

        # Assign penalties
        heuristic_value = (
            packages_in_wrong_locations * self.reach_goal_penalty +
            (1 if not arm_empty else 0) * self.reach_goal_penalty +
            closed_doors * self.reach_goal_penalty
        )

        return heuristic_value


    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic24(reach_goal_penalty={self.reach_goal_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic24(reach_goal_penalty={self.reach_goal_penalty})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 24 info:\n"
            f"\tName: {self.name}\n"
            f"\tReach Goal Penalty: {self.reach_goal_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
