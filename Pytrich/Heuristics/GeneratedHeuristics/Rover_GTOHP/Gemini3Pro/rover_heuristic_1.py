import sys
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic1(Heuristic):
    def __init__(self, rover_gemini3pro_heuristic_0, model):
        super().__init__(rover_gemini3pro_heuristic_0, model)
        self.INF = 99999
        self._initialize_static_data()

    def _parse_name_args(self, name):
        # Parses strings like "+at[rover0,waypoint1]" or "task[arg1]"
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        args = []
        if "[" in name and name.endswith("]"):
            idx = name.find("[")
            pred = name[:idx]
            args_str = name[idx+1:-1]
            if args_str:
                args = [a.strip() for a in args_str.split(",")]
            return pred, args
        return name, []

    def _initialize_static_data(self):
        # 1. Identify objects and static predicates
        self.rovers = set()
        self.waypoints = set()
        
        # Static Maps
        self.can_traverse = defaultdict(set) # (rover, from) -> {to}
        self.visible = defaultdict(set)      # from -> {to}
        self.visible_from = defaultdict(set) # objective -> {waypoint}
        self.at_lander = defaultdict(set)    # lander -> waypoint
        self.calibration_target = {}         # camera -> objective
        self.on_board = defaultdict(set)     # rover -> {camera}
        self.supports = defaultdict(set)     # camera -> {mode}
        self.rover_store_map = {}
        
        # Rover Capabilities
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_img = set()
        
        # Parse all ground facts to build static models
        init_state = self.model.initial_state
        
        for fact in self.model.facts:
            name, args = self._parse_name_args(fact.name)
            
            # Static Relations (usually present in initial state or inferred)
            if (init_state >> fact.global_id) & 1:
                if name == "can_traverse":
                    self.can_traverse[(args[0], args[1])].add(args[2])
                    self.rovers.add(args[0])
                    self.waypoints.add(args[1])
                    self.waypoints.add(args[2])
                elif name == "visible":
                    self.visible[args[0]].add(args[1])
                elif name == "visible_from":
                    self.visible_from[args[0]].add(args[1])
                elif name == "at_lander":
                    self.at_lander[args[0]] = args[1]
                elif name == "calibration_target":
                    self.calibration_target[args[0]] = args[1]
                elif name == "on_board":
                    self.on_board[args[1]].add(args[0])
                elif name == "supports":
                    self.supports[args[0]].add(args[1])
                elif name == "equipped_for_soil_analysis":
                    self.equipped_soil.add(args[0])
                elif name == "equipped_for_rock_analysis":
                    self.equipped_rock.add(args[0])
                elif name == "equipped_for_imaging":
                    self.equipped_img.add(args[0])
                elif name == "store_of":
                    self.rover_store_map[args[1]] = args[0]

        # 2. Compute Distances (APSP) for each rover using BFS
        # Distance matrix: distances[rover][from_wp][to_wp]
        self.distances = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.INF)))
        
        waypoints_list = list(self.waypoints)
        for r in self.rovers:
            # Build adjacency list for this rover
            adj = defaultdict(set)
            for (rover, f), t_set in self.can_traverse.items():
                if rover == r:
                    for t in t_set:
                        adj[f].add(t)
            
            for start_node in waypoints_list:
                self.distances[r][start_node][start_node] = 0
                q = deque([start_node])
                visited = {start_node}
                while q:
                    u = q.popleft()
                    d = self.distances[r][start_node][u]
                    for v in adj[u]:
                        if v not in visited:
                            self.distances[r][start_node][v] = d + 1
                            visited.add(v)
                            q.append(v)

        # 3. Precompute Communication Spots
        # A waypoint is a comm spot if it is visible to a lander
        self.comm_spots = set()
        for l, l_pos in self.at_lander.items():
            # Assuming lander location itself is good if visible reflexivity holds, 
            # but strictly need visible(w, l_pos)
            for w in self.waypoints:
                if l_pos in self.visible[w]:
                    self.comm_spots.add(w)
        
        # 4. Precompute min distance to any comm spot
        # min_dist_to_comm[r][w]
        self.min_dist_to_comm = defaultdict(lambda: defaultdict(lambda: self.INF))
        for r in self.rovers:
            for w in waypoints_list:
                best = self.INF
                for cs in self.comm_spots:
                    d = self.distances[r][w][cs]
                    if d < best:
                        best = d
                self.min_dist_to_comm[r][w] = best

        # 5. Lookup Optimizations
        self.rover_at_lookup = defaultdict(list)
        self.store_empty_lookup = {}
        
        for fact in self.model.facts:
            name, args = self._parse_name_args(fact.name)
            if name == "at":
                self.rover_at_lookup[args[0]].append((args[1], fact.global_id))
            elif name == "empty":
                self.store_empty_lookup[args[0]] = fact.global_id

    def _get_rover_loc(self, rover, state):
        for wp, fid in self.rover_at_lookup[rover]:
            if (state >> fid) & 1:
                return wp
        return None

    def h(self, task_network, state):
        cost = 0
        rover_locs_cache = {}

        for task in task_network:
            name_str = task.name
            name, args = self._parse_name_args(name_str)
            
            # --- Primitive Actions ---
            # Check if it is an operator (primitive)
            if isinstance(task, self.model.operators[0].__class__):
                # Discount bookkeeping actions to keep heuristic admissible/tight
                if name in ["nop", "visit", "unvisit"]:
                    continue
                cost += 1
                continue
            
            # --- Abstract Tasks ---
            
            # Navigation
            if name in ["do_navigate1", "do_navigate2"]:
                rover, to = args[0], args[1]
                if rover not in rover_locs_cache:
                    rover_locs_cache[rover] = self._get_rover_loc(rover, state)
                curr = rover_locs_cache[rover]
                
                if curr:
                    d = self.distances[rover][curr][to]
                    if d != self.INF:
                        cost += d
                    else:
                        return self.INF # Dead end
            
            # Soil Analysis Pipeline
            elif name == "get_soil_data":
                wp = args[0]
                best_r_cost = self.INF
                for r in self.equipped_soil:
                    if r not in rover_locs_cache:
                        rover_locs_cache[r] = self._get_rover_loc(r, state)
                    curr = rover_locs_cache[r]
                    if not curr: continue
                    
                    d_go = self.distances[r][curr][wp]
                    d_comm = self.min_dist_to_comm[r][wp]
                    
                    # Check store status
                    store = self.rover_store_map.get(r)
                    store_full_cost = 0
                    if store:
                        fid = self.store_empty_lookup.get(store)
                        if fid is not None and not ((state >> fid) & 1):
                            store_full_cost = 1 # Need to drop
                    
                    if d_go != self.INF and d_comm != self.INF:
                        # go + (drop) + sample(1) + go_comm + send(1)
                        total = d_go + store_full_cost + 1 + d_comm + 1
                        if total < best_r_cost:
                            best_r_cost = total
                
                if best_r_cost != self.INF:
                    cost += best_r_cost
                else:
                    return self.INF

            # Rock Analysis Pipeline
            elif name == "get_rock_data":
                wp = args[0]
                best_r_cost = self.INF
                for r in self.equipped_rock:
                    if r not in rover_locs_cache:
                        rover_locs_cache[r] = self._get_rover_loc(r, state)
                    curr = rover_locs_cache[r]
                    if not curr: continue
                    
                    d_go = self.distances[r][curr][wp]
                    d_comm = self.min_dist_to_comm[r][wp]
                    
                    store = self.rover_store_map.get(r)
                    store_full_cost = 0
                    if store:
                        fid = self.store_empty_lookup.get(store)
                        if fid is not None and not ((state >> fid) & 1):
                            store_full_cost = 1
                    
                    if d_go != self.INF and d_comm != self.INF:
                        total = d_go + store_full_cost + 1 + d_comm + 1
                        if total < best_r_cost:
                            best_r_cost = total
                
                if best_r_cost != self.INF:
                    cost += best_r_cost
                else:
                    return self.INF

            # Image Analysis Pipeline
            elif name == "get_image_data":
                obj, mode = args[0], args[1]
                best_r_cost = self.INF
                
                view_wps = self.visible_from.get(obj, set())
                if not view_wps: return self.INF
                
                for r in self.equipped_img:
                    # Check capability for mode
                    capable = False
                    for cam in self.on_board[r]:
                        if mode in self.supports[cam]:
                            capable = True
                            break
                    if not capable: continue
                    
                    if r not in rover_locs_cache:
                        rover_locs_cache[r] = self._get_rover_loc(r, state)
                    curr = rover_locs_cache[r]
                    if not curr: continue
                    
                    # Find best viewing spot
                    best_view_cost = self.INF
                    best_w = None
                    for w in view_wps:
                        d = self.distances[r][curr][w]
                        if d < best_view_cost:
                            best_view_cost = d
                            best_w = w
                    
                    if best_view_cost == self.INF: continue
                    
                    d_comm = self.min_dist_to_comm[r][best_w]
                    
                    # Lower bound: navigate to view + take_image(1) + navigate to comm + send(1)
                    # Ignoring calibration cost for admissibility/performance
                    if d_comm != self.INF:
                        total = best_view_cost + 1 + d_comm + 1
                        if total < best_r_cost:
                            best_r_cost = total
                
                if best_r_cost != self.INF:
                    cost += best_r_cost
                else:
                    return self.INF
            
            # Helper tasks
            elif name == "empty_store":
                store = args[0]
                fid = self.store_empty_lookup.get(store)
                if fid is not None and not ((state >> fid) & 1):
                    cost += 1
            
            elif name in ["send_soil_data", "send_rock_data", "send_image_data"]:
                r = args[0]
                if r not in rover_locs_cache:
                    rover_locs_cache[r] = self._get_rover_loc(r, state)
                curr = rover_locs_cache[r]
                if curr:
                    d = self.min_dist_to_comm[r][curr]
                    if d != self.INF:
                        cost += d + 1
                    else:
                        return self.INF
            
            elif name == "do_calibrate":
                # r, cam
                cost += 1 # At least calibrate action

        return cost
