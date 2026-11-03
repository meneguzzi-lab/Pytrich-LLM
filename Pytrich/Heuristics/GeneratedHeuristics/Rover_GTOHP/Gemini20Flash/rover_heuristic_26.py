import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic26(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_26 = 1.0):
        super().__init__(name='RoverGemini20FlashHeuristic26')
        self.land_target_penalty = rover_gemini20flash_heuristic_26
        self.known_locations = set()
        self.calibration_bonus = 0.5
        self.rock_sample_bonus = 0.8
        self.soil_sample_bonus = 0.7
        self.image_bonus = 0.6


    def initialize(self, model, initial_node):
        self.model = model
        self.goal_facts = set()

        for goal_fact in model.goals:
            self.goal_facts.add(goal_fact.name)

        initial_heuristic = self._calculate_heuristic(initial_node)

        return super().initialize(model, initial_heuristic)

    def _calculate_heuristic(self, node):
        h_value = 0.0

        # Penalize unvisited locations related to goals
        needed_locations = self._get_needed_locations()
        if hasattr(node, 'state'):
            state = node.state
        else:
            state = set()

        # Base heuristic on the number of tasks remaining
        if hasattr(node, 'uncompleted_tasks'):
            h_value += len(node.uncompleted_tasks) * 0.1  # Small penalty per task

        for loc in needed_locations:
            visited = False
            location_str = str(loc)

            if any(f'at rover' in fact_name and location_str in fact_name for fact_name in state):
                visited = True

            if not visited:
                h_value += self.land_target_penalty

        # Apply a domain-specific bonus based on actions in the plan
        if hasattr(node, 'applied_op') and node.applied_op:
            if 'sample_rock' in node.applied_op.name:
                h_value -= self.rock_sample_bonus
            elif 'sample_soil' in node.applied_op.name:
                h_value -= self.soil_sample_bonus
            elif 'take_image' in node.applied_op.name:
                h_value -= self.image_bonus
            elif 'calibrate' in node.applied_op.name:
                h_value -= self.calibration_bonus

        # Apply bonuses for communicated data
        for fact_name in self.goal_facts:
            if 'communicated' in fact_name and fact_name in state:
                h_value -= 0.3  # A small bonus for each communicated fact

        return max(h_value, 0.0)

    def _get_needed_locations(self):
        locations = set()

        # Extract waypoints from soil data goals
        soil_data_goals = [g for g in self.goal_facts if 'communicated_soil_data' in g]
        for goal in soil_data_goals:
            try:
                waypoint = goal.split()[1]
                locations.add(waypoint)
            except:
                None

        rock_data_goals = [g for g in self.goal_facts if 'communicated_rock_data' in g]
        for goal in rock_data_goals:
            try:
                waypoint = goal.split()[1]
                locations.add(waypoint)
            except:
                None

        image_data_goals = [g for g in self.goal_facts if 'communicated_image_data' in g]
        for goal in image_data_goals:
            try:
                # Extract the objective
                objective = goal.split()[1]
                # Find the waypoint where the objective is visible from

                # Iterate through all visibility conditions and find the waypoint
                # where the current objective is visible
                for fact in self.model.facts:
                    if 'visible_from' in fact.name and objective in fact.name:
                        waypoint = fact.name.split()[2]
                        locations.add(waypoint)
            except:
                None
        return locations



    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic26(land_target_penalty={self.land_target_penalty})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic26(land_target_penalty={self.land_target_penalty})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 26 info:\n"
            f"\tPenalty per land target: {self.land_target_penalty}\n"
            f"\tCalibration Bonus: {self.calibration_bonus}\n"
            f"\tRock Sample Bonus: {self.rock_sample_bonus}\n"
            f"\tSoil Sample Bonus: {self.soil_sample_bonus}\n"
            f"\tImage Bonus: {self.image_bonus}\n"
            f"\tKnown locations: {len(self.known_locations)}\n"
        )
