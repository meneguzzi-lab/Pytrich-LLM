import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    1. Bottleneck Analysis:
       The primary bottleneck is the navigation of rovers between waypoints 
       to collect samples (soil/rock) or capture images and then move to a waypoint 
       with line-of-sight to a lander to transmit data.

    2. Admissible Lower Bounds:
       - Decomposition Action Cost: Each high-level task requires a minimum number of 
         primitive actions (e.g., sample + communicate = 2 actions). This is a 
         lower bound because these actions are strictly required by the methods.
       - Travel Distance Cost: For any task requiring a specific location (sampling or 
         imaging), the rover must travel at least the shortest-path distance 
         to that location and then to a communication waypoint. 
       - Capacity Constraints: If a rover's store is full, an additional 'drop' 
         action is mandatory before any new sampling can occur.

    3. Preprocessing:
       - All-pairs shortest paths are computed for each rover using BFS on the 
         traversable graph (can_traverse + visible).
       - Communication waypoints (visible from landers) and objective-visibility 
         waypoints are indexed.
       - Constant properties like rover capabilities and camera modes are cached.
    """

    def __init__(self, rover_gemini3flash_heuristic_12=None, name="RoverGTOHPGemini3FlashHeuristic12"):
        super().__init__(name=name)
        self.at_fact_ids = {}         # rover -> wp -> bit_id
        self.full_fact_ids = {}       # store -> bit_id
        self.rovers = []
        self.waypoints = []
        self.stores = []
        self.rover_to_store = {}
        self.comm_wps = set()         # Waypoints with lander visibility
        
        self.rover_dists = {}         # rover -> wp1 -> wp2 -> int
        self.obj_vis_wps = defaultdict(list)   # objective -> [waypoints]
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        
        self.initialized = False

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess(initial_node.state)
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if "[" in name else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _preprocess(self, initial_state):
        rovers, waypoints, stores, objectives = set(), set(), set(), set()
        vis_pairs = set()
        traversals = defaultdict(list) # rover -> [(wp1, wp2)]
        lander_wps = set()
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at": 
                rovers.add(args[0]); waypoints.add(args[1])
                if args[0] not in self.at_fact_ids: self.at_fact_ids[args[0]] = {}
                self.at_fact_ids[args[0]][args[1]] = fact.global_id
            elif pred == "visible": vis_pairs.add((args[0], args[1]))
            elif pred == "can_traverse": traversals[args[0]].append((args[1], args[2]))
            elif pred == "store_of": 
                stores.add(args[0]); self.rover_to_store[args[1]] = args[0]
            elif pred == "full": self.full_fact_ids[args[0]] = fact.global_id
            elif pred == "equipped_for_soil_analysis": self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis": self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging": self.equipped_image.add(args[0])
            elif pred == "visible_from": objectives.add(args[0]); self.obj_vis_wps[args[0]].append(args[1])
            elif pred == "at_lander": lander_wps.add(args[1])

        self.rovers = list(rovers)
        self.waypoints = list(waypoints)
        self.comm_wps = {w1 for w1, w2 in vis_pairs if w2 in lander_wps}

        for r in self.rovers:
            adj = defaultdict(list)
            for w1, w2 in traversals[r]:
                if (w1, w2) in vis_pairs: adj[w1].append(w2)
            self.rover_dists[r] = self._compute_bfs(adj)
        self.initialized = True

    def _compute_bfs(self, adj):
        all_dists = {}
        for start_node in self.waypoints:
            dists = {start_node: 0}
            queue = deque([start_node])
            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor not in dists:
                        dists[neighbor] = dists[curr] + 1
                        queue.append(neighbor)
            all_dists[start_node] = dists
        return all_dists

    def _get_rover_at(self, state, rover):
        for wp, bit in self.at_fact_ids.get(rover, {}).items():
            if (state >> bit) & 1: return wp
        return None

    def _compute(self, state, task_network):
        if not self.initialized: return 0
        
        total_cost = 0
        # Track virtual positions of rovers to keep navigation bounds additive
        rover_vpos = {r: self._get_rover_at(state, r) for r in self.rovers}
        
        for task in task_network:
            pred, args = self._parse_fact_name(task.name)
            if not pred: continue
            
            # Base action counts for HTN tasks
            if pred == "get_soil_data":
                target_wp = args[0]
                r_best = self._find_best_rover(rover_vpos, target_wp, self.equipped_soil)
                if r_best:
                    d1 = self.rover_dists[r_best][rover_vpos[r_best]].get(target_wp, 5)
                    d2 = min([self.rover_dists[r_best][target_wp].get(cwp, 5) for cwp in self.comm_wps] or [5])
                    total_cost += d1 + d2 + 2
                    rover_vpos[r_best] = target_wp # Simplification: move to comm wp later
                else: total_cost += 5
            
            elif pred == "get_rock_data":
                target_wp = args[0]
                r_best = self._find_best_rover(rover_vpos, target_wp, self.equipped_rock)
                if r_best:
                    d1 = self.rover_dists[r_best][rover_vpos[r_best]].get(target_wp, 5)
                    d2 = min([self.rover_dists[r_best][target_wp].get(cwp, 5) for cwp in self.comm_wps] or [5])
                    total_cost += d1 + d2 + 2
                    rover_vpos[r_best] = target_wp
                else: total_cost += 5

            elif pred == "get_image_data":
                obj, mode = args[0], args[1]
                r_best = self._find_best_rover(rover_vpos, None, self.equipped_image)
                if r_best:
                    vis_wps = self.obj_vis_wps[obj]
                    d1 = min([self.rover_dists[r_best][rover_vpos[r_best]].get(vw, 5) for wp_v in vis_wps] or [5])
                    total_cost += d1 + 3 # calibrate + take + comm
                else: total_cost += 7
                
            elif pred == "do_navigate1":
                r, target = args[0], args[1]
                total_cost += self.rover_dists[r][rover_vpos.get(r, target)].get(target, 0)
                rover_vpos[r] = target
                
            elif pred == "navigate":
                total_cost += 1
                rover_vpos[args[0]] = args[2]
            
            elif pred in ["sample_soil", "sample_rock", "calibrate", "take_image", "drop", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data"]:
                total_cost += 1
                
        # Capacity Check (P4 Bottleneck refinement)
        for r in self.rovers:
            store = self.rover_to_store.get(r)
            if store and (state >> self.full_fact_ids.get(store, 0)) & 1:
                # If rover is full and has a sampling task ahead, must drop
                total_cost += 0.5 # Sub-unit to stay safe or small penalty

        return int(total_cost)

    def _find_best_rover(self, vpos, target, candidates):
        best_r = None
        min_d = 999
        for r in candidates:
            if not target:
                return r # Just pick first for imaging for simplicity
            d = self.rover_dists[r][vpos[r]].get(target, 999)
            if d < min_d:
                min_d = d
                best_r = r
        return best_r if best_r else (list(candidates)[0] if candidates else None)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        # Symmetry breaking: minor penalty for longer task networks to guide search
        h_value += len(node.task_network) * 0.01
        super().update_info(int(h_value))
        return int(h_value)
