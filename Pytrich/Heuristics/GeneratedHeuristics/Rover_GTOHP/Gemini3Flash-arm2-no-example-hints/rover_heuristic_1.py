from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque, defaultdict

class RoverGTOHPGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    1. Bottleneck Analysis:
       The primary bottleneck in this domain is navigation. A rover must sequentially travel between 
       waypoints to sample soil/rock, capture images (often requiring calibration at a target waypoint 
       first), and finally move to a waypoint visible from a lander to transmit the data. Since moving 
       between waypoints in this HDDL domain requires multiple actions (visit + navigate + unvisit), 
       the navigation costs dominate the plan length.

    2. Admissible Lower Bounds:
       - Term 1 (Action Cost): Every unsatisfied data goal (soil, rock, image) requires a minimum number 
         of atomic actions (2 for samples, 3 for images). This is a lower bound on unavoidable work.
       - Term 2 (Navigation Cost): For each rover to reach a target waypoint and then a communication 
         waypoint, it must traverse a shortest path. We use a cost of 3 per hop (visit + navigate + unvisit).
       - Term 3 (Goal count): Unsatisfied goal facts provide a basic lower bound for satisficing search.

    3. Design Strategy:
       - Precompute all-pairs shortest paths for every rover using BFS during initialization.
       - Identify all valid communication points and objective visibility points from grounded operators.
       - Estimate cost per goal based on current rover positions and remaining tasks in the network.
    """

    def __init__(self, rover_gemini3flash_heuristic_1="blind"):
        super().__init__(rover_gemini3flash_heuristic_1)
        self.at_fact_ids = defaultdict(dict)  # rover -> wp -> fid
        self.have_soil_ids = defaultdict(dict) # rover -> wp -> fid
        self.have_rock_ids = defaultdict(dict) # rover -> wp -> fid
        self.have_image_ids = defaultdict(lambda: defaultdict(dict)) # rover -> obj -> mode -> fid
        self.full_ids = {} # rover -> fid

        self.dist = defaultdict(lambda: defaultdict(dict)) # rover -> wp1 -> {wp2: steps}
        self.comm_wps = defaultdict(set) # rover -> set of waypoints
        self.obj_visible_wps = defaultdict(set) # objective -> set of waypoints
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.rovers = set()
        self.waypoints = set()
        self.goal_info = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for args in args_str.split(",") for a in args.split(" ") if a]
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse and build fact maps
        for fact in model.facts:
            p, args = self._parse_fact_name(fact.name)
            if not args: continue
            if p == "at": 
                self.at_fact_ids[args[0]][args[1]] = fact.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif p == "have_soil_analysis": self.have_soil_ids[args[0]][args[1]] = fact.global_id
            elif p == "have_rock_analysis": self.have_rock_analysis_ids[args[0]][args[1]] = fact.global_id
            elif p == "have_image": self.have_image_ids[args[0]][args[1]][args[2]] = fact.global_id
            elif p == "full": self.full_ids[args[0]] = fact.global_id

        # 2. Extract static properties from grounded operators
        adj = defaultdict(lambda: defaultdict(set))
        for op in model.operators:
            p, args = self._parse_fact_name(op.name)
            if p == "navigate":
                # navigate[rover, from, to]
                adj[args[0]][args[1]].add(args[2])
            elif p.startswith("communicate_soil") or p.startswith("communicate_rock"):
                # communicate_soil_data1[rover, lander, from_wp, at_wp, lander_wp]
                if "data1" in p: self.comm_wps[args[0]].add(args[3])
                else: self.comm_wps[args[0]].add(args[2])
            elif p == "communicate_image_data":
                # communicate_image_data[rover, lander, obj, mode, at_wp, lander_wp]
                self.comm_wps[args[0]].add(args[4])
            elif p == "take_image":
                # take_image[rover, wp, obj, camera, mode]
                self.obj_visible_wps[args[2]].add(args[1])
            elif p == "calibrate":
                # calibrate[rover, camera, obj, wp]
                self.obj_visible_wps[args[2]].add(args[3])
            
            # Capability sets
            if p == "sample_soil": self.equipped_soil.add(args[0])
            elif p == "sample_rock": self.equipped_rock.add(args[0])
            elif p == "take_image": self.equipped_image.add(args[0])

        # 3. BFS for all-pairs shortest paths per rover
        for r in self.rovers:
            for start_wp in self.waypoints:
                self.dist[r][start_wp] = self._bfs(adj[r], start_wp)

        # 4. Goal analysis
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            p, args = self._parse_fact_name(name)
            # Use global_id from model facts set
            fid = -1
            for f in model.facts:
                if f.name == name:
                    fid = f.global_id
                    break
            if fid != -1:
                self.goal_info.append((p, args, fid))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _bfs(self, adj_map, start):
        dists = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_map[u]:
                if v not in dists:
                    dists[v] = dists[u] + 1
                    q.append(v)
        return dists

    def _compute(self, state, task_network):
        # Find current rover positions
        rover_pos = {}
        for r, wps in self.at_fact_ids.items():
            for wp, fid in wps.items():
                if (state >> fid) & 1:
                    rover_pos[r] = wp
                    break

        h = 0
        for p, args, g_id in self.goal_info:
            # Goal already satisfied
            if (state >> g_id) & 1:
                continue

            target = args[0]
            min_goal_cost = 1000

            # DATA TYPE: SOIL OR ROCK
            if p == "communicated_soil_data" or p == "communicated_rock_data":
                relevant_equipped = self.equipped_soil if "soil" in p else self.equipped_rock
                for r in relevant_equipped:
                    if r not in rover_pos: continue
                    curr_wp = rover_pos[r]
                    # Cost = Travel to sampling point + Sample (1) + Travel to comm + Comm (1)
                    # Use 3 actions per hop
                    travel_to_data = self.dist[r][curr_wp].get(target, 100)
                    
                    # Shortest travel from data point to ANY comm point
                    travel_to_comm = min([self.dist[r][target].get(cw, 100) for cw in self.comm_wps[r]] or [100])
                    
                    total_cost = 3 * (travel_to_data + travel_to_comm) + 2
                    # Check if rover already has the analysis
                    if p == "communicated_soil_data":
                        if (state >> self.have_soil_ids[r].get(target, 0)) & 1:
                             total_cost = 3 * min([self.dist[r][curr_wp].get(cw, 100) for cw in self.comm_wps[r]] or [100]) + 1
                    
                    min_goal_cost = min(min_goal_cost, total_cost)

            # DATA TYPE: IMAGE
            elif p == "communicated_image_data":
                # args = [objective, mode]
                obj = args[0]
                mode = args[1]
                for r in self.equipped_image:
                    if r not in rover_pos: continue
                    curr_wp = rover_pos[r]
                    # Image cost: 3 actions (calibrate, take, comm) + 3*hops
                    # Estimate hops: dist to image point + dist to comm point
                    best_img_travel = 100
                    for vwp in self.obj_visible_wps[obj]:
                        dist_to_vwp = self.dist[r][curr_wp].get(vwp, 100)
                        dist_vwp_to_comm = min([self.dist[r][vwp].get(cw, 100) for cw in self.comm_wps[r]] or [100])
                        best_img_travel = min(best_img_travel, dist_to_vwp + dist_vwp_to_comm)
                    
                    total_cost = 3 * best_img_travel + 3
                    
                    # Check if rover already has image
                    if (state >> self.have_image_ids[r][obj].get(mode, 0)) & 1:
                        total_cost = 3 * min([self.dist[r][curr_wp].get(cw, 100) for cw in self.comm_wps[r]] or [100]) + 1
                    
                    min_goal_cost = min(min_goal_cost, total_cost)
            
            if min_goal_cost < 1000:
                h += min_goal_cost
            else:
                h += 10 # Default for obscure goals

        # HTN tie-breaking: prefer completing decomposition
        h += 0.01 * len(task_network)
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
