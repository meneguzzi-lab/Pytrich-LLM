from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class RoverGTOHPGemini3FlashHeuristic19(Heuristic):
    """
    Domain-Dependent Heuristic for Rover_GTOHP.
    
    1. Bottleneck: Rover navigation between sampling waypoints and communication points visible to landers.
    2. Admissible Bounds:
       - Navigation requires at least 3 actions per hop (visit + navigate + unvisit) based on the domain's 
         recursive movement methods, or 1 action for a direct hop (3d is a robust estimate).
       - Every uncompleted data goal requires at least one sampling/imaging action and one communication action (+2).
       - Imaging requires an additional calibration action (+1).
       - Store management (drop) adds unavoidable actions if sampling is needed while the store is full.
    """
    
    def __init__(self, rover_gemini3flash_heuristic_19="RoverGTOHPGemini3FlashHeuristic19"):
        super().__init__(rover_gemini3flash_heuristic_19)
        self.rovers = set()
        self.waypoints = set()
        self.adj = defaultdict(lambda: defaultdict(set))
        self.dist = {} # rover -> wp1 -> wp2 -> int
        
        self.capable_soil = defaultdict(set) # wp -> {rovers}
        self.capable_rock = defaultdict(set) # wp -> {rovers}
        self.capable_image = defaultdict(set) # (obj, mode) -> {(rover, camera)}
        
        self.W_comm = defaultdict(set) # rover -> {waypoints}
        self.W_cal = defaultdict(set) # camera -> {waypoints}
        self.W_img = defaultdict(set) # objective -> {waypoints}
        
        self.camera_on_rover = {} # camera -> rover
        self.camera_targets = {} # camera -> objective
        self.rover_stores = {} # rover -> store

        self.at_ids = defaultdict(dict) # rover -> wp -> id
        self.have_soil_ids = defaultdict(dict) # rover -> wp -> id
        self.have_rock_ids = defaultdict(dict) # rover -> wp -> id
        self.have_image_ids = defaultdict(dict) # rover -> (obj, mode) -> id
        self.calibrated_ids = defaultdict(dict) # rover -> camera -> id
        self.empty_ids = {} # rover -> bit_id
        
        self.goal_bits = [] # list of (type, args, bit_id)

        self.min_dist_to_comm = {} # rover -> wp -> min_dist
        self.d_img_logic = {} # (rover, camera, obj) -> min(dist_to_img + dist_to_comm)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    def floyd_warshall(self, adj, nodes):
        dist = {n1: {n2: 999 for n2 in nodes} for n1 in nodes}
        for n in nodes: dist[n][n] = 0
        for u, vs in adj.items():
            for v in vs:
                dist[u][v] = 1
        for k in nodes:
            for i in nodes:
                if dist[i][k] < 999:
                    for j in nodes:
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Scrape static info from grounded operators
        for op in model.operators:
            p, args = self._parse_fact_name(op.name)
            if not p: continue
            if p == "navigate":
                r, w1, w2 = args
                self.rovers.add(r)
                self.waypoints.update([w1, w2])
                self.adj[r][w1].add(w2)
            elif p == "sample_soil":
                r, s, wp = args
                self.capable_soil[wp].add(r)
                self.rover_stores[r] = s
            elif p == "sample_rock":
                r, s, wp = args
                self.capable_rock[wp].add(r)
                self.rover_stores[r] = s
            elif p == "calibrate":
                r, c, o, wp = args
                self.W_cal[c].add(wp)
                self.camera_on_rover[c] = r
                self.camera_targets[c] = o
            elif p == "take_image":
                r, wp, o, c, m = args
                self.W_img[o].add(wp)
                self.capable_image[(o, m)].add((r, c))
            elif p.startswith("communicate"):
                r = args[0]
                if p == "communicate_soil_data1" or p == "communicate_rock_data1":
                    self.W_comm[r].add(args[3])
                elif p == "communicate_soil_data2" or p == "communicate_rock_data2":
                    self.W_comm[r].add(args[2])
                elif p == "communicate_image_data":
                    self.W_comm[r].add(args[4])

        # 2. Distance Matrix Calculation with unique graph optimization
        graph_to_dist = {}
        for r in self.rovers:
            edges = tuple(sorted((u, tuple(sorted(vs))) for u, vs in self.adj[r].items()))
            if edges not in graph_to_dist:
                graph_to_dist[edges] = self.floyd_warshall(self.adj[r], self.waypoints)
            self.dist[r] = graph_to_dist[edges]

        # 3. Precompute Visibility/Travel sets
        for r in self.rovers:
            self.min_dist_to_comm[r] = {}
            for w in self.waypoints:
                self.min_dist_to_comm[r][w] = min([self.dist[r][w][cp] for cp in self.W_comm[r]], default=999)
        
        for r, c_list in self.capable_image.items():
            for r_name, c_name in c_list:
                target = self.camera_targets.get(c_name)
                obj = r[0]
                if not target: continue
                best_logic = 999
                for w_cal in self.W_cal[c_name]:
                    for w_img in self.W_img[obj]:
                        path = self.dist[r_name][w_cal][w_img] + self.min_dist_to_comm[r_name][w_img]
                        if path < best_logic: best_logic = path
                self.d_img_logic[(r_name, c_name, obj)] = best_logic

        # 4. Map Facts to Bit IDs
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if not p: continue
            if p == "at": self.at_ids[args[0]][args[1]] = f.global_id
            elif p == "have_soil_analysis": self.have_soil_ids[args[0]][args[1]] = f.global_id
            elif p == "have_rock_analysis": self.have_rock_ids[args[0]][args[1]] = f.global_id
            elif p == "have_image": self.have_image_ids[args[0]][(args[1], args[2])] = f.global_id
            elif p == "calibrated": self.calibrated_ids[args[1]][args[0]] = f.global_id
            elif p == "empty":
                for r, s in self.rover_stores.items():
                    if s == args[0]: self.empty_ids[r] = f.global_id

        # 5. Index Goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            p, args = self._parse_fact_name(gn)
            for f in model.facts:
                if f.name == gn:
                    if p == "communicated_soil_data": self.goal_bits.append(("soil", args[0], f.global_id))
                    elif p == "communicated_rock_data": self.goal_bits.append(("rock", args[0], f.global_id))
                    elif p == "communicated_image_data": self.goal_bits.append(("image", (args[0], args[1]), f.global_id))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        total_h = 0.0
        
        rover_pos = {}
        for r in self.rovers:
            for wp, bid in self.at_ids[r].items():
                if (state >> bid) & 1:
                    rover_pos[r] = wp
                    break
        
        for gtype, target, bid in self.goal_bits:
            if (state >> bid) & 1: continue
            
            min_g_cost = 10000
            if gtype == "soil":
                for r in self.capable_soil[target]:
                    curr_pos = rover_pos.get(r)
                    if not curr_pos: continue
                    if (state >> self.have_soil_ids[r].get(target, 0)) & 1:
                        cost = 3 * self.min_dist_to_comm[r][curr_pos] + 1
                    else:
                        store_p = 0 if (state >> self.empty_ids.get(r, 0)) & 1 else 1
                        cost = 3 * (self.dist[r][curr_pos][target] + self.min_dist_to_comm[r][target]) + 2 + store_p
                    if cost < min_g_cost: min_g_cost = cost
            elif gtype == "rock":
                for r in self.capable_rock[target]:
                    curr_pos = rover_pos.get(r)
                    if not curr_pos: continue
                    if (state >> self.have_rock_ids[r].get(target, 0)) & 1:
                        cost = 3 * self.min_dist_to_comm[r][curr_pos] + 1
                    else:
                        store_p = 0 if (state >> self.empty_ids.get(r, 0)) & 1 else 1
                        cost = 3 * (self.dist[r][curr_pos][target] + self.min_dist_to_comm[r][target]) + 2 + store_p
                    if cost < min_g_cost: min_g_cost = cost
            elif gtype == "image":
                for r, c in self.capable_image[target]:
                    curr_pos = rover_pos.get(r)
                    if not curr_pos: continue
                    if (state >> self.have_image_ids[r].get(target, 0)) & 1:
                        cost = 3 * self.min_dist_to_comm[r][curr_pos] + 1
                    elif (state >> self.calibrated_ids[r].get(c, 0)) & 1:
                        best_img_wp_dist = min([self.dist[r][curr_pos][w] + self.min_dist_to_comm[r][w] for w in self.W_img[target[0]]], default=999)
                        cost = 3 * best_img_wp_dist + 2
                    else:
                        base = self.d_img_logic.get((r, c, target[0]), 999)
                        d_to_cal = min([self.dist[r][curr_pos][w] for w in self.W_cal[c]], default=999)
                        cost = 3 * (d_to_cal) + base + 3
                    if cost < min_g_cost: min_g_cost = cost
            
            total_h += min_g_cost
            
        return int(total_h + 0.001 * len(task_network))
