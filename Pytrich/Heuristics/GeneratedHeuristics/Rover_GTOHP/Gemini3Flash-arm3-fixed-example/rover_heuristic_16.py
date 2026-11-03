"""
Domain-dependent heuristic for the Rover_GTOHP planning domain.

Bottleneck Analysis:
The primary bottleneck in the Rover domain is the serialized nature of data acquisition and communication. 
A rover must (1) navigate to a waypoint, (2) sample/image data, (3) navigate to a location visible to a lander, 
and (4) communicate the data. Navigation is the most expensive part of this chain due to waypoint graph constraints.

Admissible Lower Bounds:
1. h_goals: Each unmet communication goal requires at least a 'communicate' action (cost 1). 
   If the data has not been collected yet by any rover, it also requires at least a 'sample' or 'take_image' action (cost +1).
   This is a lower bound because it ignores travel time entirely. (Goal-based bound)

2. h_tasks: Each task in the network represents a future sequence of actions. 
   Tasks like 'get_soil_data' or 'get_image_data' require at least 2 or 3 actions respectively to reach a primitive state.
   This is a lower bound based on minimum decomposition depth. (Structure-based bound)

Heuristic is max(h_goals, h_tasks).
"""

import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic16(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_16=None, name="RoverGTOHPGemini3FlashHeuristic16"):
        super().__init__(name=name)
        self.goal_info = []
        self.task_min_costs = {
            "get_soil_data": 2,
            "get_rock_data": 2,
            "get_image_data": 3,
            "send_soil_data": 1,
            "send_rock_data": 1,
            "send_image_data": 1,
            "do_navigate1": 1,
            "do_navigate2": 1,
            "empty_store": 1,
            "do_calibrate": 1,
            "navigate": 1,
            "sample_soil": 1,
            "sample_rock": 1,
            "drop": 1,
            "calibrate": 1,
            "take_image": 1,
            "communicate_soil_data1": 1,
            "communicate_soil_data2": 1,
            "communicate_rock_data1": 1,
            "communicate_rock_data2": 1,
            "communicate_image_data": 1,
            "visit": 0,
            "unvisit": 0,
            "nop": 0
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Index Goal Fact IDs
        goal_soil_wp = {}
        goal_rock_wp = {}
        goal_image_obj_mode = {}
        
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        goal_ids = []
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "communicated_soil_data":
                goal_soil_wp[args[0]] = None # Placeholder for fact ID
            elif pred == "communicated_rock_data":
                goal_rock_wp[args[0]] = None
            elif pred == "communicated_image_data":
                goal_image_obj_mode[(args[0], args[1])] = None
        
        # 2. Map goals to 'have_analysis' predicates across all rovers
        have_soil_ids = {}
        have_rock_ids = {}
        have_image_ids = {}
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "communicated_soil_data":
                goal_soil_wp[args[0]] = fact.global_id
            elif pred == "communicated_rock_data":
                goal_rock_wp[args[0]] = fact.global_id
            elif pred == "communicated_image_data":
                goal_image_obj_mode[(args[0], args[1])] = fact.global_id
            
            elif pred == "have_soil_analysis":
                # args: [rover, waypoint]
                wp = args[1]
                if wp not in have_soil_ids: have_soil_ids[wp] = []
                have_soil_ids[wp].append(fact.global_id)
            elif pred == "have_rock_analysis":
                # args: [rover, waypoint]
                wp = args[1]
                if wp not in have_rock_ids: have_rock_ids[wp] = []
                have_rock_ids[wp].append(fact.global_id)
            elif pred == "have_image":
                # args: [rover, objective, mode]
                obj, mode = args[1], args[2]
                if (obj, mode) not in have_image_ids: have_image_ids[(obj, mode)] = []
                have_image_ids[(obj, mode)].append(fact.global_id)

        # 3. Create goal_info structures for O(1) checks
        self.goal_info = []
        for wp, gid in goal_soil_wp.items():
            if gid is not None: self.goal_info.append((gid, have_soil_ids.get(wp, [])))
        for wp, gid in goal_rock_wp.items():
            if gid is not None: self.goal_info.append((gid, have_rock_ids.get(wp, [])))
        for key, gid in goal_image_obj_mode.items():
            if gid is not None: self.goal_info.append((gid, have_image_ids.get(key, [])))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _compute(self, state, task_network):
        # Term 1: Goal-based
        h_goals = 0
        for g_id, have_ids in self.goal_info:
            if not ((state >> g_id) & 1):
                h_goals += 1
                # If no rover currently has this data collected
                collected = False
                for h_id in have_ids:
                    if (state >> h_id) & 1:
                        collected = True
                        break
                if not collected:
                    h_goals += 1

        # Term 2: Task-based
        h_tasks = 0
        for t in task_network:
            base_name = t.name.split('[')[0] if '[' in t.name else t.name
            h_tasks += self.task_min_costs.get(base_name, 1)

        # Tie-breaker to favor smaller task networks when costs are equal
        return max(h_goals, h_tasks) + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
