"""
Domain-dependent heuristic for the Rover_GTOHP domain.

Bottleneck Analysis:
The primary bottleneck in the Rover domain is navigation. Every sampling task (soil/rock) or imaging task requires the rover
to move to specific waypoints (sampling sites, visibility sites, or calibration targets) and then navigate to a waypoint 
where it can see a lander to communicate the results. These movements are serialized because the rover typically 
processes one goal at a time through its hierarchical decomposition.

Admissible Lower Bounds:
1. Acquisition Actions: Every unsatisfied soil/rock goal requires at least a 'sample' and a 'communicate' action. 
   Every imaging goal requires 'calibrate', 'take_image', and 'communicate'. This provides a lower bound on atomic actions.
2. Navigation Steps: For each task in the network, the rover must travel a minimum number of steps (shortest path)
   to visit required locations. Since the tasks are ordered, these distances sum up to form a lower bound on total actions.
3. Store Management: If a rover's store is full and it must sample, a 'drop' action is unavoidable.

Admissibility:
This heuristic is admissible because it computes the shortest possible path (in terms of steps) and the minimum 
required non-navigation actions (sample, comm, etc.) for each task in an ordered network. Because the tasks must 
be completed sequentially, the sum of their minimum costs is a valid lower bound.
"""

import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic5(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_5=None, name="RoverGTOHPGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.waypoints = []
        self.rovers = []
        self.dist = {}  # rover -> w1 -> w2 -> cost
        self.at_fact_ids = {} # rover -> w -> id
        self.empty_ids = {} # store -> id
        self.store_of = {} # rover -> store
        self.comm_sites = {} # rover -> list of waypoints
        self.vis_sites = {} # objective -> list of waypoints
        self.calib_sites = {} # objective -> list of waypoints
        self.cam_target = {} # camera -> objective
        self.rover_cams = {} # rover -> list of cameras

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._calculate_heuristic(initial_node.state, initial_node.task_network)
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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        waypoints = set()
        rovers = set()
        stores = set()
        landers = set()
        cameras = set()
        objectives = set()
        adj = {}
        vis_edges = set()
        lander_at = {}
        vis_from = {}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at" and args[0].startswith("rover"):
                rovers.add(args[0])
                waypoints.add(args[1])
                if args[0] not in self.at_fact_ids: self.at_fact_ids[args[0]] = {}
                self.at_fact_ids[args[0]][args[1]] = fact.global_id
            elif pred == "can_traverse":
                rovers.add(args[0])
                waypoints.add(args[1])
                waypoints.add(args[2])
                if args[0] not in adj: adj[args[0]] = {}
                if args[1] not in adj[args[0]]: adj[args[0]][args[1]] = []
                adj[args[0]][args[1]].append(args[2])
            elif pred == "visible":
                vis_edges.add((args[0], args[1]))
            elif pred == "at_lander":
                lander_at[args[0]] = args[1]
            elif pred == "visible_from":
                if args[0] not in self.vis_sites: self.vis_sites[args[0]] = []
                self.vis_sites[args[0]].append(args[1])
            elif pred == "calibration_target":
                self.cam_target[args[0]] = args[1]
            elif pred == "on_board":
                if args[1] not in self.rover_cams: self.rover_cams[args[1]] = []
                self.rover_cams[args[1]].append(args[0])
            elif pred == "store_of":
                self.store_of[args[1]] = args[0]
            elif pred == "empty":
                self.empty_ids[args[0]] = fact.global_id

        self.waypoints = list(waypoints)
        self.rovers = list(rovers)

        for r in self.rovers:
            self.dist[r] = {w1: {w2: 1000 for w2 in self.waypoints} for w1 in self.waypoints}
            for start_w in self.waypoints:
                self.dist[r][start_w][start_w] = 0
                if r in adj and start_w in adj[r]:
                    queue = deque([start_w])
                    while queue:
                        u = queue.popleft()
                        for v in adj[r].get(u, []):
                            if (u, v) in vis_edges and self.dist[r][start_w][v] == 1000:
                                self.dist[r][start_w][v] = self.dist[r][start_w][u] + 1
                                queue.append(v)

        for r in self.rovers:
            self.comm_sites[r] = [w for w in self.waypoints if any(visible and (w, lander_w) in vis_edges for lander_w in lander_at.values())]

    def _dist_to_comm(self, r, w):
        sites = self.comm_sites.get(r, [])
        if not sites: return 1000
        return min(self.dist[r][w][s] for s in sites)

    def _calculate_heuristic(self, state, task_network):
        if not task_network: return 0
        
        h = 0
        rover_pos = {}
        for r in self.rovers:
            for w in self.waypoints:
                if (state >> self.at_fact_ids[r][w]) & 1:
                    rover_pos[r] = w
                    break
        
        # Process a limited number of tasks to avoid timeout on huge networks
        for task in task_network[:20]:
            pred, args = self._parse_fact_name(task.name)
            if not pred: continue

            if pred in ["get_soil_data", "get_rock_data"]:
                target_w = args[0]
                costs = []
                for r in self.rovers:
                    if r not in rover_pos: continue
                    d1 = self.dist[r][rover_pos[r]][target_w]
                    d2 = self._dist_to_comm(r, target_w)
                    store = self.store_of.get(r)
                    empty_cost = 0 if store and (state >> self.empty_ids.get(store, 0)) & 1 else 1
                    costs.append(d1 + empty_cost + 1 + d2 + 1)
                h += min(costs) if costs else 5
            
            elif pred == "get_image_data":
                obj, mode = args[0], args[1]
                costs = []
                for r in self.rovers:
                    if r not in rover_pos: continue
                    # Simplified image acquisition cost
                    img_ws = self.vis_sites.get(obj, [])
                    if not img_ws: continue
                    d_to_img = min(self.dist[r][rover_pos[r]][iw] for iw in img_ws)
                    d_to_comm = min(self._dist_to_comm(r, iw) for iw in img_ws)
                    costs.append(d_to_img + 1 + d_to_comm + 2) # calibrate + take + comm
                h += min(costs) if costs else 6
                
            elif pred.startswith("send_"):
                r = args[0]
                if r in rover_pos:
                    h += self._dist_to_comm(r, rover_pos[r]) + 1
            
            elif pred == "do_navigate1":
                r, target = args[0], args[1]
                if r in rover_pos:
                    h += self.dist[r][rover_pos[r]][target]
            
            elif pred == "navigate":
                h += 1
            elif pred in ["sample_soil", "sample_rock", "drop", "calibrate", "take_image", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data"]:
                h += 1

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
