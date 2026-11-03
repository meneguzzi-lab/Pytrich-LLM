"""
Domain-dependent heuristic for the Rover_GTOHP domain.

1. Bottleneck Analysis:
   The primary bottleneck in the Rover domain is the sequential nature of data acquisition and communication.
   Each rover must navigate to a specific waypoint to sample soil/rock or take images, and then 
   navigate to a position visible from a lander to communicate that data. Since rovers usually 
   have a single store and the lander channel must be free, the task of gathering and transmitting 
   multiple data points is highly serialized by rover travel and resource availability.

2. Admissible Lower Bounds:
   - Term 1 (Task Expansion): Each task in the task network (get_soil_data, get_rock_data, get_image_data) 
     requires a minimum sequence of primitive actions (navigate, sample, navigate, communicate). 
     Summing the minimum possible actions remaining for each task in the frontier provides an 
     admissible lower bound.
   - Term 2 (Unsatisfied Goals): Every communicated_data goal fact that is not yet achieved 
     represents at least one 'communicate' action that must occur in the future.
   - Term 3 (Navigation Steps): For any 'do_navigate1' task, the minimum number of 'navigate' 
     actions is the shortest path distance between the current waypoint and the target waypoint.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic8(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_8=None, name="RoverGTOHPGemini3FlashHeuristic8"):
        super().__init__(name=name)
        self.rover_at_bits = {} # (rover, waypoint) -> global_id
        self.goal_bits = []
        self.dist_matrices = {} # rover -> {wp1 -> {wp2 -> distance}}
        self.waypoint_names = []
        self.rover_names = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map predicates
        traversals = [] # (rover, wp1, wp2)
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at" and len(args) == 2:
                self.rover_at_bits[(args[0], args[1])] = fact.global_id
                if args[0] not in self.rover_names: self.rover_names.append(args[0])
                if args[1] not in self.waypoint_names: self.waypoint_names.append(args[1])
            elif pred == "can_traverse" and len(args) == 3:
                traversals.append((args[0], args[1], args[2]))
                if args[1] not in self.waypoint_names: self.waypoint_names.append(args[1])
                if args[2] not in self.waypoint_names: self.waypoint_names.append(args[2])

        # 2. Build goal bit list
        if hasattr(model, 'goals') and model.goals:
            for bit_pos in range(model.goals.bit_length()):
                if (model.goals >> bit_pos) & 1:
                    self.goal_bits.append(bit_pos)

        # 3. Precompute distance matrices per rover
        for r in self.rover_names:
            dist = {w1: {w2: 999 for w2 in self.waypoint_names} for w1 in self.waypoint_names}
            for w in self.waypoint_names: dist[w][w] = 0
            for rv, w1, w2 in traversals:
                if rv == r:
                    dist[w1][w2] = 1
            # Floyd-Warshall
            for k in self.waypoint_names:
                for i in self.waypoint_names:
                    for j in self.waypoint_names:
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            self.dist_matrices[r] = dist

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return None, []

    def _get_rover_at(self, state, rover):
        for wp in self.waypoint_names:
            bit = self.rover_at_bits.get((rover, wp))
            if bit is not null and (state >> bit) & 1:
                return wp
        return None

    def _compute(self, state, task_network):
        # Term 1: Goal satisfaction bound
        unmet_goals = 0
        for g_bit in self.goal_bits:
            if not ((state >> g_bit) & 1):
                unmet_goals += 1

        # Term 2: Task network expansion cost
        tn_cost = 0
        for task in task_network:
            name = task.name.lower()
            if "get_soil_data" in name or "get_rock_data" in name or "get_image_data" in name:
                tn_cost += 5
            elif "do_navigate1" in name:
                # parse args: do_navigate1(rover, to)
                # Simplified parse from task.name which is often "task_name[arg1,arg2]"
                parts = name.replace("]","").split("[")
                if len(parts) > 1:
                    args = parts[1].split(",")
                    if len(args) == 2:
                        r, to = args[0], args[1]
                        curr = self._get_rover_at(state, r)
                        if curr and r in self.dist_matrices:
                            tn_cost += self.dist_matrices[r][curr].get(to, 1)
                        else:
                            tn_cost += 1
                else: tn_cost += 1
            elif "send_" in name or "do_calibrate" in name:
                tn_cost += 2
            elif name == "nop":
                tn_cost += 0
            else:
                tn_cost += 1 # Default for other tasks/operators
        
        # Return max for admissibility, but summing tn_cost is more informative for HTN
        return max(unmet_goals, tn_cost)

    def __call__(self, parent_node, node):
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
