"""
Domain-dependent heuristic for the Rover_GTOHP domain.

Bottleneck Analysis:
The primary bottleneck in the Rover domain is navigation. Every high-level task (soil, rock, image)
requires a sequence of movements between waypoints: moving to a sampling/imaging site, 
possibly moving to a calibration site, and finally moving to a communication point visible to the lander.
Travel cost (number of 'navigate' actions) dominates the plan length.

Admissibility Sketches:
1. Task Decomposition Bound: Each task in the task network (get_soil_data, get_rock_data, get_image_data) 
   requires a minimum number of atomic actions (navigate, sample/take_image, communicate). 
   This is a lower bound because we use the shortest path distance in the traversal graph.
2. Goal Progress Bound: Goals (communicated data) that are not yet true in the state must 
   eventually result in at least one 'communicate' action and the preceding sampling/imaging sequence. 
   Counting unachieved goals provides a disjoint lower bound on future actions.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic6(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_6=None, name="RoverGTOHPGemini3FlashHeuristic6"):
        super().__init__(name=name)
        self.dist_matrix = {}
        self.waypoints = []
        self.lander_visible_waypoints = set()
        self.goal_facts = []
        self.at_fact_map = {}  # rover -> waypoint -> fact_id
        self.have_soil_map = {} # rover -> waypoint -> fact_id
        self.have_rock_map = {} # rover -> waypoint -> fact_id
        self.have_image_map = {} # rover -> objective -> mode -> fact_id
        self.comm_soil_id = {} # waypoint -> fact_id
        self.comm_rock_id = {} # waypoint -> fact_id
        self.comm_image_id = {} # (obj, mode) -> fact_id

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute_h(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if bracket_idx < len(name)-2 else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        traversal_edges = []
        waypoints_set = set()
        visible_pairs = []
        lander_locations = []

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "at":
                if args[0] not in self.at_fact_map: self.at_fact_map[args[0]] = {}
                self.at_fact_map[args[0]][args[1]] = fact.global_id
            elif pred == "can_traverse":
                traversal_edges.append((args[1], args[2]))
                waypoints_set.add(args[1])
                waypoints_set.add(args[2])
            elif pred == "visible":
                visible_pairs.append((args[0], args[1]))
            elif pred == "at_lander":
                lander_locations.append(args[1])
            elif pred == "have_soil_analysis":
                if args[0] not in self.have_soil_map: self.have_soil_map[args[0]] = {}
                self.have_soil_map[args[0]][args[1]] = fact.global_id
            elif pred == "have_rock_analysis":
                if args[0] not in self.have_rock_map: self.have_rock_map[args[0]] = {}
                self.have_rock_map[args[0]][args[1]] = fact.global_id
            elif pred == "have_image":
                key = (args[1], args[2])
                if args[0] not in self.have_image_map: self.have_image_map[args[0]] = {}
                self.have_image_map[args[0]][key] = fact.global_id
            elif pred == "communicated_soil_data":
                self.comm_soil_id[args[0]] = fact.global_id
            elif pred == "communicated_rock_data":
                self.comm_rock_id[args[0]] = fact.global_id
            elif pred == "communicated_image_data":
                self.comm_image_id[(args[0], args[1])] = fact.global_id

        self.waypoints = list(waypoints_set)
        for w1 in self.waypoints:
            self.dist_matrix[w1] = {w2: float('inf') for w2 in self.waypoints}
            self.dist_matrix[w1][w1] = 0
        
        for u, v in traversal_edges:
            self.dist_matrix[u][v] = 1
            self.dist_matrix[v][u] = 1

        # Floyd-Warshall
        for k in self.waypoints:
            for i in self.waypoints:
                for j in self.waypoints:
                    if self.dist_matrix[i][j] > self.dist_matrix[i][k] + self.dist_matrix[k][j]:
                        self.dist_matrix[i][j] = self.dist_matrix[i][k] + self.dist_matrix[k][j]

        # Comms check
        for w_rover in self.waypoints:
            for w_lander in lander_locations:
                if (w_rover, w_lander) in visible_pairs or (w_lander, w_rover) in visible_pairs:
                    self.lander_visible_waypoints.add(w_rover)
                    break

    def _get_rover_at(self, state, rover):
        if rover not in self.at_fact_map: return None
        for w, fid in self.at_fact_map[rover].items():
            if (state >> fid) & 1: return w
        return None

    def _compute_h(self, state, task_network):
        if not task_network: return 0
        h = 0
        # We iterate over tasks to estimate work remaining
        for task in task_network:
            pred, args = self._parse_fact_name(task.name)
            if not pred: continue
            
            if pred == "get_soil_data":
                target_w = args[0]
                if not ((state >> self.comm_soil_id.get(target_w, 0)) & 1):
                    h += 3 # nav to site, sample, nav to comm, communicate
            elif pred == "get_rock_data":
                target_w = args[0]
                if not ((state >> self.comm_rock_id.get(target_w, 0)) & 1):
                    h += 3
            elif pred == "get_image_data":
                if not ((state >> self.comm_image_id.get((args[0], args[1]), 0)) & 1):
                    h += 4 # calibrate, nav to view, take, nav to comm, comm
            elif pred == "do_navigate1":
                # This is a subtask, it's roughly the distance
                rover, target = args[0], args[1]
                curr = self._get_rover_at(state, rover)
                if curr and curr != target:
                    d = self.dist_matrix.get(curr, {}).get(target, 1)
                    h += d

        # Penalty for unachieved goals not explicitly in the logic above
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            gpred, gargs = self._parse_fact_name(gname)
            if gpred == "communicated_soil_data":
                if not ((state >> self.comm_soil_id.get(gargs[0], 0)) & 1): h += 0.1
            elif gpred == "communicated_rock_data":
                if not ((state >> self.comm_rock_id.get(gargs[0], 0)) & 1): h += 0.1
            elif gpred == "communicated_image_data":
                if not ((state >> self.comm_image_id.get((gargs[0], gargs[1]), 0)) & 1): h += 0.1

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute_h(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
