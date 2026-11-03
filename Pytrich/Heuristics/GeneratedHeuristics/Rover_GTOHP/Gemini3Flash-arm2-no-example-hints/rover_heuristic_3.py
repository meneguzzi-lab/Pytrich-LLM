from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask
import math
from collections import defaultdict

class RoverGTOHPGemini3FlashHeuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.

    BOTTLENECK:
    The main bottleneck is the sequential navigation effort required for each rover to reach a sampling 
    site and then relocate to a waypoint visible to a lander for data transmission.

    ADMISSIBLE LOWER BOUNDS:
    1. Task Network Decomposition Cost (Summation): Every abstract task in the network has a minimum number 
       of required primitive actions (sample, communicate, drop, calibrate) plus a navigation lower bound. 
       Summing these is admissible as they represent work items that must be executed to empty the network.
    2. Navigation Lower Bound: The minimum path distance (using Floyd-Warshall) from current rover positions 
       through required sampling waypoints to communication waypoints (dist * 3, accounting for visit/unvisit).
    """

    def __init__(self, rover_gemini3flash_heuristic_3="rover_gemini3flash_heuristic_3"):
        super().__init__(name=rover_gemini3flash_heuristic_3)
        self.at_fact_ids = {}  # rover -> wp -> global_id
        self.full_fact_ids = {} # store -> global_id
        self.dists = {} # rover -> wp1 -> wp2 -> int
        self.vl_sites = {} # rover -> set(wp)
        self.vo_sites = defaultdict(set) # obj -> set(wp)
        self.vc_sites = defaultdict(set) # cam -> set(wp)
        self.dist_to_lander = {} # rover -> wp -> min_dist
        self.dist_to_vc = {} # rover -> cam -> wp -> min_dist
        self.dist_vc_to_vo = {} # cam -> obj -> min_dist
        self.dist_vo_to_vl = {} # rover -> obj -> min_dist
        self.rover_stores = {} # rover -> store_name
        self.rover_soil_capable = set()
        self.rover_rock_capable = set()
        self.rover_image_capable = defaultdict(set) # (mode) -> set(rovers)
        self.camera_on_rover = {} # cam -> rover
        self.camera_modes = defaultdict(set) # cam -> set(mode)
        self.waypoints = set()
        self.rovers = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse entities and static capabilities from operators (grounder strips static facts)
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if not pred: continue
            
            if pred == "navigate":
                # args: [rover, from, to]
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                self.waypoints.add(args[2])
            elif pred == "sample_soil":
                # args: [rover, store, wp]
                self.rover_soil_capable.add(args[0])
                self.rover_stores[args[0]] = args[1]
            elif pred == "sample_rock":
                self.rover_rock_capable.add(args[0])
                self.rover_stores[args[0]] = args[1]
            elif pred == "take_image":
                # args: [rover, p, obj, cam, mode]
                self.camera_on_rover[args[3]] = args[0]
                self.camera_modes[args[3]].add(args[4])
                self.rover_image_capable[args[4]].add(args[0])
                self.vo_sites[args[2]].add(args[1])
            elif pred == "calibrate":
                # args: [rover, cam, obj, wp]
                self.vc_sites[args[1]].add(args[3])
            elif pred.startswith("communicate"):
                # communicate_soil_data1[rover, lander, p1, p2, p3] -> p2 is rover pos
                # communicate_soil_data2[rover, lander, p1, p2] -> p1 is rover pos
                # communicate_image_data[rover, lander, o, m, p1, p2] -> p1 is rover pos
                self.vl_sites.setdefault(args[0], set()).add(args[-2])

        # 2. Build Fact Maps
        at_map = defaultdict(dict)
        for f in model.facts:
            p, a = self._parse_fact_name(f.name)
            if p == "at": at_map[a[0]][a[1]] = f.global_id
            if p == "full": self.full_fact_ids[a[0]] = f.global_id
        self.at_fact_ids = at_map

        # 3. Floyd-Warshall for all-pairs distances per rover map
        self.waypoints = sorted(list(self.waypoints))
        self.rovers = sorted(list(self.rovers))
        rover_maps = {}
        for r in self.rovers:
            adj = {w: {w: 0} for w in self.waypoints}
            for op in model.operators:
                p, a = self._parse_fact_name(op.name)
                if p == "navigate" and a[0] == r:
                    adj[a[1]][a[2]] = 1
            
            map_key = tuple(sorted([(k, tuple(sorted(v.keys()))) for k,v in adj.items()]))
            if map_key not in rover_maps:
                rover_maps[map_key] = self._compute_fw(adj)
            self.dists[r] = rover_maps[map_key]

        # 4. Precompute combined distances
        for r in self.rovers:
            self.dist_to_lander[r] = {}
            for w in self.waypoints:
                vl = self.vl_sites.get(r, [])
                self.dist_to_lander[r][w] = min([self.dists[r][w][v] for v in vl] + [100])

            self.dist_VO_to_VL[r] = {}
            for obj in self.vo_sites:
                vos = self.vo_sites[obj]
                vl = self.vl_sites.get(r, [])
                self.dist_VO_to_VL[r][obj] = min([self.dists[r][wi][wl] for wi in vos for wl in vl] + [100])

        self.dist_VC_to_VO = defaultdict(dict)
        for obj in self.vo_sites:
            vos = self.vo_sites[obj]
            for cam in self.vc_sites:
                vcs = self.vc_sites[cam]
                self.dist_VC_to_VO[obj][cam] = min([self.dists[self.camera_on_rover[cam]][wc][wi] for wc in vcs for wi in vos] + [100])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute_fw(self, adj):
        dist = {w1: {w2: 100 for w2 in self.waypoints} for w1 in self.waypoints}
        for w1 in self.waypoints: dist[w1][w1] = 0
        for w1, neighbors in adj.items():
            for w2, cost in neighbors.items(): dist[w1][w2] = cost
        for k in self.waypoints:
            for i in self.waypoints:
                for j in self.waypoints:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def _parse_fact_name(self, name):
        name = name.strip('+').strip('-')
        parts = name.split('[')
        if len(parts) < 2: return None, []
        pred = parts[0]
        args = parts[1].rstrip(']').split(',')
        return pred, args

    def _get_rover_wp(self, rover, state):
        if rover not in self.at_fact_ids: return None
        for wp, fid in self.at_fact_ids[rover].items():
            if (state >> fid) & 1: return wp
        return None

    def _compute(self, state, task_network):
        if not task_network:
            return 0 if (state & self.model.goals) == self.model.goals else 100

        total_h = 0
        rover_curr_wp = {r: self._get_rover_wp(r, state) for r in self.rovers}
        
        for task in task_network:
            pred, args = self._parse_fact_name(task.name)
            if not pred: continue

            # --- Abstract Task Estimation ---
            if pred == "get_soil_data":
                wp = args[0]
                best_cost = 100
                target_r = None
                for r in self.rover_soil_capable:
                    d1 = self.dists[r].get(rover_curr_wp[r], {}).get(wp, 100)
                    d2 = self.dist_to_lander[r].get(wp, 100)
                    cost = (d1 + d2) * 3 + 2
                    if cost < best_cost:
                        best_cost = cost
                        target_r = r
                total_h += best_cost
                if target_r: rover_curr_wp[target_r] = "unknown" # Approximate
            
            elif pred == "get_rock_data":
                wp = args[0]
                best_cost = 100
                target_r = None
                for r in self.rover_rock_capable:
                    d1 = self.dists[r].get(rover_curr_wp[r], {}).get(wp, 100)
                    d2 = self.dist_to_lander[r].get(wp, 100)
                    cost = (d1 + d2) * 3 + 2
                    if cost < best_cost:
                        best_cost = cost
                        target_r = r
                total_h += best_cost

            elif pred == "get_image_data":
                obj, mode = args[0], args[1]
                best_cost = 100
                for r in self.rover_image_capable.get(mode, []):
                    for cam in self.rover_cameras_for_mode(r, mode):
                        d_vc = min([self.dists[r].get(rover_curr_wp[r], {}).get(wc, 100) for wc in self.vc_sites[cam]] + [100])
                        d_vo = self.dist_VC_to_VO[obj].get(cam, 100)
                        d_vl = self.dist_VO_to_VL[r].get(obj, 100)
                        cost = (d_vc + d_vo + d_vl) * 3 + 3
                        best_cost = min(best_cost, cost)
                total_h += best_cost

            elif pred == "do_navigate1":
                r, to_wp = args[0], args[1]
                d = self.dists[r].get(rover_curr_wp[r], {}).get(to_wp, 100)
                total_h += d * 3
                rover_curr_wp[r] = to_wp

            elif pred == "empty_store":
                r = args[1]
                store = self.rover_stores.get(r)
                if store and (state >> self.full_fact_ids.get(store, 0)) & 1: total_h += 1

            # --- Primitive Action Costs ---
            elif isinstance(task, Operator):
                total_h += task.cost
            else:
                total_h += 1 # Default for generic abstract tasks

        return total_h

    def rover_cameras_for_mode(self, rover, mode):
        res = []
        for cam, r in self.camera_on_rover.items():
            if r == rover and mode in self.camera_modes[cam]:
                res.append(cam)
        return res

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
