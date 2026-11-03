from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic5(Heuristic):
    """
    Domain-Dependent Heuristic for Rover_GTOHP.

    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is navigation and store management. 
    Every goal (soil data, rock data, or image) requires the rover to be at a specific waypoint, 
    perform a sequence of actions (potentially emptying the store, calibrating a camera, 
    and taking the sample), then navigating to a waypoint visible to a lander to transmit. 
    The distance-based travel cost (3 actions per waypoint hop) dominates the plan length.

    Admissible Lower Bounds:
    1. Goal-Specific Minimum Completion Cost: For each uncompleted data goal, we calculate the shortest path 
       for any rover to reach the source location, perform the action, and reach a communication waypoint.
       Term = max over all uncompleted goals.
    2. Communication Lower Bound: Every piece of data on board that hasn't been communicated yet 
       requires at least one communication action and navigation to a comm-point.
    """

    def __init__(self, rover_gemini3flash_heuristic_5="rover_gemini3flash_heuristic_5"):
        super().__init__(rover_gemini3flash_heuristic_5)
        self.rover_ids = []
        self.waypoint_ids = []
        self.objective_ids = []
        self.mode_ids = []
        
        self.dist_matrix = {} # rover -> (wp1, wp2) -> int
        self.comm_waypoints = {} # rover -> set(waypoint_ids)
        
        self.goals_info = [] # List of (goal_type, target_arg, goal_fact_id)
        
        # Fact Maps
        self.at_map = {} # (rover, wp) -> fact_id
        self.have_soil_map = {} # (rover, wp) -> fact_id
        self.have_rock_map = {} # (rover, wp) -> fact_id
        self.have_image_map = {} # (rover, obj, mode) -> fact_id
        self.comm_soil_id = {} # wp -> fact_id
        self.comm_rock_id = {} # wp -> fact_id
        self.comm_image_id = {} # (obj, mode) -> fact_id
        
        # Sampling/Imaging Map
        self.sample_wp = {} # ("soil"/"rock", target) -> wp_id
        self.image_objs = {} # (obj) -> [waypoint_ids] (visible_from)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify static components and build fact maps
        adjacency = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_map[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis":
                self.have_soil_map[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis":
                self.have_rock_map[(args[0], args[1])] = f.global_id
            elif pred == "have_image":
                self.have_image_map[(args[0], args[1], args[2])] = f.global_id
            elif pred == "communicated_soil_data":
                self.comm_soil_id[args[0]] = f.global_id
            elif pred == "communicated_rock_data":
                self.comm_rock_id[args[0]] = f.global_id
            elif pred == "communicated_image_data":
                self.comm_image_id[(args[0], args[1])] = f.global_id

        # Extract navigation graph and lander comms from operators
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == "navigate":
                rover, w1, w2 = args[0], args[1], args[2]
                adjacency[rover][w1].append(w2)
                if rover not in self.rover_ids: self.rover_ids.append(rover)
            elif pred.startswith("communicate_soil_data"):
                rover, wp_comms = args[0], args[4] if "1" in pred else args[3]
                if rover not in self.comm_waypoints: self.comm_waypoints[rover] = set()
                self.comm_waypoints[rover].add(wp_comms)
            elif pred.startswith("communicate_rock_data"):
                rover, wp_comms = args[0], args[4] if "1" in pred else args[3]
                if rover not in self.comm_waypoints: self.comm_waypoints[rover] = set()
                self.comm_waypoints[rover].add(wp_comms)
            elif pred == "communicate_image_data":
                rover, wp_comms = args[0], args[5]
                if rover not in self.comm_waypoints: self.comm_waypoints[rover] = set()
                self.comm_waypoints[rover].add(wp_comms)
            elif pred == "sample_soil":
                self.sample_wp[("soil", args[2])] = args[2]
            elif pred == "sample_rock":
                self.sample_wp[("rock", args[2])] = args[2]
            elif pred == "take_image":
                obj = args[2]
                if obj not in self.image_objs: self.image_objs[obj] = set()
                self.image_objs[obj].add(args[1])

        # 2. Precompute All-Pairs Shortest Paths for each rover
        for rover in self.rover_ids:
            self.dist_matrix[rover] = {}
            wps = list(adjacency[rover].keys())
            for start in wps:
                dists = {start: 0}
                queue = collections.deque([start])
                while queue:
                    curr = queue.popleft()
                    for neighbor in adjacency[rover][curr]:
                        if neighbor not in dists:
                            dists[neighbor] = dists[curr] + 1
                            queue.append(neighbor)
                for end, d in dists.items():
                    self.dist_matrix[rover][(start, end)] = d

        # 3. Analyze Goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "communicated_soil_data":
                self.goals_info.append(("soil", args[0], self.comm_soil_id[args[0]]))
            elif pred == "communicated_rock_data":
                self.goals_info.append(("rock", args[1] if len(args)>1 else args[0], self.comm_rock_id[args[1] if len(args)>1 else args[0]]))
            elif pred == "communicated_image_data":
                self.goals_info.append(("image", (args[0], args[1]), self.comm_image_id[(args[0], args[1])]))

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _compute(self, state):
        goal_costs = []
        
        # Find current rover positions
        rover_locs = {}
        for r in self.rover_ids:
            for wp in self.at_map:
                if wp[0] == r and (state >> self.at_map[wp]) & 1:
                    rover_locs[r] = wp[1]
                    break

        for g_type, target, g_id in self.goals_info:
            if (state >> g_id) & 1:
                continue
                
            min_g_cost = 999999
            
            for r in self.rover_ids:
                curr_wp = rover_locs.get(r, None)
                if not curr_wp: continue
                
                # Check if data is already on board this specific rover
                on_board = False
                if g_type == "soil":
                    on_board = (state >> self.have_soil_map.get((r, target), 0)) & 1
                elif g_type == "rock":
                    on_board = (state >> self.have_rock_map.get((r, target), 0)) & 1
                elif g_type == "image":
                    on_board = (state >> self.have_image_map.get((r, target[0], target[1]), 0)) & 1
                
                rover_dist_matrix = self.dist_matrix[r]
                r_comms = self.comm_waypoints.get(r, [])
                
                if on_board:
                    # Path to any comm-point
                    d = min([rover_dist_matrix.get((curr_wp, cw), 999) for cw in r_comms] or [999])
                    cost = d * 3 + 1
                else:
                    # Path to sample-point then to comm-point
                    if g_type == "image":
                        s_wps = self.image_objs.get(target[0], [])
                    else:
                        s_wps = [self.sample_wp.get((g_type, target), None)]
                    
                    costs_to_sample = []
                    for swp in s_wps:
                        if swp and (curr_wp, swp) in rover_dist_matrix:
                            d1 = rover_dist_matrix[(curr_wp, swp)]
                            d2 = min([rover_dist_matrix.get((swp, cw), 999) for cw in r_comms] or [999])
                            costs_to_sample.append((d1 + d2) * 3 + 2)
                    
                    cost = min(costs_to_sample) if costs_to_sample else 999
                
                min_g_cost = min(min_g_cost, cost)
            
            if min_g_cost < 999:
                goal_costs.append(min_g_cost)

        if not goal_costs:
            return 0
            
        # Admissible lower bound: max cost of any single unavoidable goal journey
        return max(goal_costs)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
