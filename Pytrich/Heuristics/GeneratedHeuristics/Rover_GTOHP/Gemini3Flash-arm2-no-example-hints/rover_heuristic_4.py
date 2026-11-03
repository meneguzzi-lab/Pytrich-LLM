from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic4(Heuristic):
    '''
    DOMAIN BOTTLENECK: 
    The primary bottleneck in the Rover domain is the serialized navigation required 
    to visit sampling locations, calibration targets, and communication spots. 
    Each movement (hop) between waypoints has a heavy cost of 3 actions (visit, navigate, unvisit).

    ADMISSIBLE LOWER BOUNDS:
    1. Navigation Cost: A lower bound on the number of primitive actions needed to move a rover 
       between required waypoints. Each hop is estimated at 3 actions. Admissible because 
       every hop in the methods m1/m4 involves visit, navigate, and unvisit.
    2. Intrinsic Action Cost: Every sampling, imaging, or communication task requires 
       at least 1 primitive action (sample_*, take_image, calibrate, communicate_*) regardless 
       of position. Admissible because these are unavoidable terminal operations.
    3. Task Network Summation: Since tasks are ordered in this domain, the total cost 
       is the sum of the remaining work for all tasks in the current network.
    '''

    def __init__(self, rover_gemini3flash_heuristic_4="rover_gemini3flash_heuristic_4"):
        super().__init__(rover_gemini3flash_heuristic_4)
        self.rover_at_fids = {}  # rover -> {waypoint: global_id}
        self.empty_fids = {}     # rover -> global_id
        self.dists = {}          # rover -> {wp1: {wp2: dist}}
        
        self.task_info = {}      # tid -> (pred, args)
        self.intrinsic_costs = {}# tid -> cost
        self.task_target_wp = {} # tid -> waypoint
        self.obj_vis_wps = {}    # objective -> set of waypoints
        self.min_dist_to_comm = {} # waypoint -> min_dist_to_any_comm_spot
        
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()

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

    def _bfs(self, start_wp, adj):
        dists = {start_wp: 0}
        queue = collections.deque([start_wp])
        while queue:
            u = queue.popleft()
            for v in adj.get(u, []):
                if v not in dists:
                    dists[v] = dists[u] + 1
                    queue.append(v)
        return dists

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify objects and fact maps
        rover_adj = {}
        comm_spots = set()
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                r, wp = args[0], args[1]
                self.rovers.add(r)
                self.waypoints.add(wp)
                if r not in self.rover_at_fids: self.rover_at_fids[r] = {}
                self.rover_at_fids[r][wp] = f.global_id
            elif pred == "empty":
                # args[0] is store, but we need it per rover. 
                # store_of is static, so we'll map rover directly later via actions.
                pass
            elif pred == "at_soil_sample" or pred == "at_rock_sample":
                self.waypoints.add(args[0])

        # 2. Extract static capabilities from operators
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == "navigate":
                r, fr, to = args[0], args[1], args[2]
                if r not in rover_adj: rover_adj[r] = {}
                if fr not in rover_adj[r]: rover_adj[r][fr] = []
                rover_adj[r][fr].append(to)
            elif "communicate_soil_data1" in pred or "communicate_rock_data1" in pred:
                comm_spots.add(args[3])
            elif "communicate_soil_data2" in pred or "communicate_rock_data2" in pred:
                comm_spots.add(args[2])
            elif "communicate_image_data" in pred:
                comm_spots.add(args[4])
            elif pred == "take_image":
                obj, wp = args[2], args[1]
                self.objectives.add(obj)
                if obj not in self.obj_vis_wps: self.obj_vis_wps[obj] = set()
                self.obj_vis_wps[obj].add(wp)
            elif pred == "calibrate":
                obj, wp = args[2], args[3]
                self.objectives.add(obj)
                if obj not in self.obj_vis_wps: self.obj_vis_wps[obj] = set()
                self.obj_vis_wps[obj].add(wp)
            elif pred == "sample_soil" or pred == "sample_rock":
                # Map store 'empty' fact id to rover
                # sample_soil ?x - rover ?s - store ?p - waypoint
                pass

        # 3. Precompute distances for each rover
        for r in self.rovers:
            self.dists[r] = {}
            adj = rover_adj.get(r, {})
            for wp in self.waypoints:
                self.dists[r][wp] = self._bfs(wp, adj)

        # 4. Precompute waypoint connectivity to communication
        for wp in self.waypoints:
            m = 999
            for spot in comm_spots:
                for r in self.rovers:
                    d = self.dists[r][wp].get(spot, 999)
                    if d < m: m = d
            self.min_dist_to_comm[wp] = m

        # 5. Pre-parse task costs and targets
        all_tasks = model.abstract_tasks + model.operators
        for t in all_tasks:
            pred, args = self._parse_fact_name(t.name)
            if not pred: 
                pred = t.name.split('[')[0] if '[' in t.name else t.name
                args = t.name.split('[')[1][:-1].split(',') if '[' in t.name else []
            self.task_info[t.global_id] = (pred, args)
            
            if pred == "get_soil_data" or pred == "get_rock_data":
                self.task_target_wp[t.global_id] = args[0]
                self.intrinsic_costs[t.global_id] = 2 + 3 * self.min_dist_to_comm.get(args[0], 0)
            elif pred == "get_image_data":
                # Objective vis waypoint is target. We'll handle navigation specially in compute.
                self.intrinsic_costs[t.global_id] = 3 
            elif pred == "do_navigate1" or pred == "do_navigate2":
                self.task_target_wp[t.global_id] = args[-1]
                self.intrinsic_costs[t.global_id] = 0
            elif pred == "navigate":
                self.task_target_wp[t.global_id] = args[2]
                self.intrinsic_costs[t.global_id] = 1
            elif pred in ["sample_soil", "sample_rock", "take_image", "calibrate", "drop"]:
                self.intrinsic_costs[t.global_id] = 1
            elif pred.startswith("communicate_"):
                self.intrinsic_costs[t.global_id] = 1
            else:
                self.intrinsic_costs[t.global_id] = 0

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state, task_network):
        if not task_network:
            return 0

        # Current rover positions
        rover_pos = {}
        for r in self.rovers:
            for wp, fid in self.rover_at_fids.get(r, {}).items():
                if (state >> fid) & 1:
                    rover_pos[r] = wp
                    break

        h = 0
        first_task = True
        for task_obj in task_network:
            tid = task_obj.global_id
            if tid not in self.task_info: continue
            
            pred, args = self.task_info[tid]
            h += self.intrinsic_costs.get(tid, 1)

            # For the immediate next task, calculate travel cost from current state
            if first_task:
                target_wp = self.task_target_wp.get(tid)
                if not target_wp and pred == "get_image_data":
                    # Target objective vis waypoints
                    obj = args[0]
                    vis_wps = self.obj_vis_wps.get(obj, [])
                    min_nav = 999
                    for r in self.rovers:
                        curr = rover_pos.get(r)
                        if curr:
                            for vwp in vis_wps:
                                d = 3 * self.dists[r][curr].get(vwp, 20)
                                if d < min_nav: min_nav = d
                    if min_nav < 999: h += min_nav
                elif target_wp:
                    if pred in ["do_navigate1", "do_navigate2", "navigate"]:
                        r = args[0]
                        curr = rover_pos.get(r)
                        if curr: h += 3 * self.dists[r][curr].get(target_wp, 20)
                    else:
                        min_nav = 999
                        for r in self.rovers:
                            curr = rover_pos.get(r)
                            if curr:
                                d = 3 * self.dists[r][curr].get(target_wp, 20)
                                if d < min_nav: min_nav = d
                        if min_nav < 999: h += min_nav
                first_task = False
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
