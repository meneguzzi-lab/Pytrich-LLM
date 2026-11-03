"""
Domain-dependent heuristic for the Rover_GTOHP domain.

Bottleneck:
    The bottleneck in this domain is rover travel between waypoints. Each high-level task 
    (soil, rock, or image collection) requires a specific rover to navigate to a source 
    waypoint, perform one or more actions (sampling/imaging), and eventually navigate 
    to a position where it can transmit data to a lander.

Admissible Lower Bounds:
    1. Task-Specific Action Count: Every 'get' task in the task network requires a 
       minimum sequence of primitive actions (e.g., sample + communicate) regardless 
       of location. 
       - get_soil_data: 2 (sample_soil + communicate)
       - get_rock_data: 2 (sample_rock + communicate)
       - get_image_data: 3 (calibrate + take_image + communicate)
    2. Navigation Distance: For tasks that are already grounded to a specific rover 
       (like do_navigate1), the shortest path distance (number of navigate actions) 
       between the current location and the target waypoint is an unavoidable cost.
    3. Unfulfilled Goals: Any goal in the state not yet communicated requires at least 
       one communication action.

Guidance:
    Summing the minimum decomposition costs of tasks in the current task network 
    provides a strong lower bound and excellent guidance for HTN planning.
"""

import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic18(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_18=None, name="RoverGTOHPGemini3FlashHeuristic18"):
        super().__init__(name=name)
        self.at_id = {}  # (rover, waypoint) -> global_id
        self.rovers = []
        self.waypoints = []
        self.dist_matrix = {} # rover -> { (w1, w2) -> distance }
        self.equipped_soil = set() # rovers
        self.equipped_rock = set() # rovers
        self.equipped_imaging = set() # rovers
        self.visibility_map = {} # objective -> list of waypoints
        self.goal_bits = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse domain objects and facts
        rovers_set = set()
        waypoints_set = set()
        can_traverse_data = [] # (rover, w1, w2)
        visible_from_data = [] # (obj, waypoint)

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "at":
                self.at_id[(args[0], args[1])] = fact.global_id
                rovers_set.add(args[0])
                waypoints_set.add(args[1])
            elif pred == "can_traverse":
                can_traverse_data.append((args[0], args[1], args[2]))
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])
            elif pred == "visible_from":
                if args[0] not in self.visibility_map: self.visibility_map[args[0]] = []
                self.visibility_map[args[0]].append(args[1])

        self.rovers = list(rovers_set)
        self.waypoints = list(waypoints_set)

        # 2. Build Distance Matrices (BFS per rover)
        for r in self.rovers:
            self.dist_matrix[r] = {}
            adj = {w: [] for w in self.waypoints}
            for rov, w1, w2 in can_traverse_data:
                if rov == r:
                    adj[w1].append(w2)
            
            for start_w in self.waypoints:
                distances = {start_w: 0}
                queue = deque([start_w])
                while queue:
                    curr = queue.popleft()
                    for neighbor in adj[curr]:
                        if neighbor not in distances:
                            distances[neighbor] = distances[curr] + 1
                            queue.append(neighbor)
                for end_w, d in distances.items():
                    self.dist_matrix[r][(start_w, end_w)] = d

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

    def _get_rover_pos(self, state_bitwise, rover):
        for w in self.waypoints:
            if (rover, w) in self.at_id:
                if (state_bitwise >> self.at_id[(rover, w)]) & 1:
                    return w
        return None

    def _get_min_dist(self, state_bitwise, target_waypoint, required_capability=None):
        min_d = 999
        for r in self.rovers:
            if required_capability == "soil" and r not in self.equipped_soil: continue
            if required_capability == "rock" and r not in self.equipped_rock: continue
            if required_capability == "imaging" and r not in self.equipped_imaging: continue
            
            pos = self._get_rover_pos(state_bitwise, r)
            if pos and (pos, target_waypoint) in self.dist_matrix[r]:
                min_d = min(min_d, self.dist_matrix[r][(pos, target_waypoint)])
        return min_d if min_d != 999 else 0

    def _compute(self, state, task_network):
        h = 0
        if not task_network:
            return 0

        for task in task_network:
            # task can be an Operator or AbstractTask
            pred, args = self._parse_fact_name(task.name)
            if not pred: 
                # Primitive actions from model might not have '+' prefix if from model.operators
                name = task.name.replace("(", "").replace(")", "")
                parts = name.split()
                pred = parts[0]
                args = parts[1:]

            if pred == "get_soil_data":
                h += 2 + self._get_min_dist(state, args[0], "soil")
            elif pred == "get_rock_data":
                h += 2 + self._get_min_dist(state, args[0], "rock")
            elif pred == "get_image_data":
                # args[0] is objective, args[1] is mode
                vis_waypoint = self.visibility_map.get(args[0], [None])[0]
                dist = self._get_min_dist(state, vis_waypoint, "imaging") if vis_waypoint else 0
                h += 3 + dist
            elif pred == "do_navigate1":
                # args[0] is rover, args[1] is waypoint
                pos = self._get_rover_pos(state, args[0])
                if pos and pos != args[1]:
                    h += self.dist_matrix.get(args[0], {}).get((pos, args[1]), 1)
            elif pred == "do_navigate2":
                h += self.dist_matrix.get(args[0], {}).get((args[1], args[2]), 1)
            elif pred in ["send_soil_data", "send_rock_data", "send_image_data", "do_calibrate"]:
                h += 1
            elif pred == "nop":
                h += 0
            else:
                # Standard primitive action cost
                h += 1

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
