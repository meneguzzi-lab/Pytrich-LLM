import sys
import heapq
from collections import defaultdict

# Assumed imports from Pytrich context - ensuring compatibility with framework
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic3(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_2=None):
        super().__init__(model)
        self.inf = float('inf')
        
        # --- 1. Parse Static Domain Info from Facts & Initial State ---
        # We need to parse the static graph structure (map connectivity),
        # rover capabilities, and object locations to support the heuristic.
        
        self.rovers = []
        self.waypoints = []
        self.lander_locs = set()
        
        # Capability & Configuration Maps
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_img = set()
        self.camera_on_board = defaultdict(list)
        self.camera_supports = defaultdict(set)
        self.camera_targets = {}
        self.visible_from = defaultdict(set)
        
        # Connectivity Graph: rover -> w1 -> [(w2, edge_name)]
        # Note: 'can_traverse' is rover-specific in this domain.
        self.graph = defaultdict(list)
        self.visible = defaultdict(set)
        self.at_lander = {}
        
        # Map global_id to parsed (predicate, args) for fast lookup in h()
        self.fact_id_map = {}
        self.gid_cache = defaultdict(dict)

        # First pass: Parse names to identify objects and structure
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_id_map[fact.global_id] = (pred, args)
                self.gid_cache[pred][tuple(args)] = fact.global_id
                
                if pred == 'can_traverse':
                    r, w1, w2 = args
                    if r not in self.rovers: self.rovers.append(r)
                    if w1 not in self.waypoints: self.waypoints.append(w1)
                    if w2 not in self.waypoints: self.waypoints.append(w2)
                elif pred == 'at' and 'waypoint' in args[1]:
                    # Infer waypoints from 'at' facts if not seen in traverse
                    if args[1] not in self.waypoints: self.waypoints.append(args[1])

        self.rovers = list(set(self.rovers))
        self.waypoints = list(set(self.waypoints))
        
        # Second pass: Build Static Graph from Initial State Bitmask
        # We check self.model.initial_state to see which static predicates are true.
        init_bits = self.model.initial_state
        
        for fid, (pred, args) in self.fact_id_map.items():
            # Check if bit is set in initial state
            if (init_bits >> fid) & 1:
                if pred == 'can_traverse':
                    r, w1, w2 = args
                    self.graph[r].append(w2) # Adjacency list for rover r at w1
                    # Storing as tuple list for graph construction later
                elif pred == 'visible':
                    w1, w2 = args
                    self.visible[w1].add(w2)
                    self.visible[w2].add(w1)
                elif pred == 'at_lander':
                    l, w = args
                    self.at_lander[l] = w
                    self.lander_locs.add(w)
                elif pred == 'equipped_for_soil_analysis':
                    self.equipped_soil.add(args[0])
                elif pred == 'equipped_for_rock_analysis':
                    self.equipped_rock.add(args[0])
                elif pred == 'equipped_for_imaging':
                    self.equipped_img.add(args[0])
                elif pred == 'on_board':
                    cam, r = args
                    self.camera_on_board[r].append(cam)
                elif pred == 'supports':
                    cam, m = args
                    self.camera_supports[cam].add(m)
                elif pred == 'calibration_target':
                    cam, obj = args
                    self.camera_targets[cam] = obj
                elif pred == 'visible_from':
                    obj, w = args
                    self.visible_from[obj].add(w)

        # Refine Graph Structure for APSP
        # self.graph needs to be structured: rover -> w1 -> list of neighbors
        # The previous loop iterated facts, so we need to reorganize.
        rover_adj = {r: defaultdict(list) for r in self.rovers}
        for fid, (pred, args) in self.fact_id_map.items():
            if (init_bits >> fid) & 1 and pred == 'can_traverse':
                r, w1, w2 = args
                rover_adj[r][w1].append(w2)
        
        # --- 2. Precompute All-Pairs Shortest Paths (APSP) ---
        # Since traversability depends on the rover, we compute distances per rover.
        # This allows O(1) distance lookups during heuristic evaluation.
        self.dist = {r: defaultdict(lambda: defaultdict(lambda: self.inf)) for r in self.rovers}
        
        for r in self.rovers:
            # BFS from every node for this rover
            # Note: waypoints list contains all waypoints, even those unreachable by this rover
            for start_node in self.waypoints:
                self.dist[r][start_node][start_node] = 0
                q = [start_node]
                visited = {start_node}
                d = 0
                
                curr_level = [start_node]
                while curr_level:
                    d += 1
                    next_level = []
                    for u in curr_level:
                        for v in rover_adj[r][u]:
                            if v not in visited:
                                visited.add(v)
                                self.dist[r][start_node][v] = d
                                next_level.append(v)
                    curr_level = next_level

        # --- 3. Precompute Communication Spots ---
        # A waypoint is a comm spot if it is visible from a lander location.
        self.comm_spots = set()
        for w in self.waypoints:
            for l_loc in self.lander_locs:
                if l_loc in self.visible[w] or l_loc == w:
                    self.comm_spots.add(w)
                    break
        
        # Precompute distance to nearest comm spot for every rover/waypoint pair
        self.dist_to_comm = {r: defaultdict(lambda: self.inf) for r in self.rovers}
        for r in self.rovers:
            for w in self.waypoints:
                min_d = self.inf
                for c in self.comm_spots:
                    d = self.dist[r][w][c]
                    if d < min_d:
                        min_d = d
                self.dist_to_comm[r][w] = min_d

    def _parse_fact_name(self, name):
        """Parses +predicate[arg1,arg2] into ('predicate', ['arg1', 'arg2'])."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        idx = name.find("[")
        if idx != -1 and name.endswith("]"):
            pred = name[:idx]
            args = name[idx+1:-1].split(",")
            return pred, args
        return None, []

    def get_state_fact(self, state, pred, *args):
        """Fast bitwise check for fact existence."""
        gid = self.gid_cache.get(pred, {}).get(tuple(args))
        if gid is None: return False
        return (state >> gid) & 1

    def h(self, task_network, state_bitwise, **kwargs):
        """
        Domain-dependent heuristic for Rover.
        Estimates cost by analyzing the current location of rovers relative to 
        task targets (soil, rock, image) and communication points.
        """
        cost = 0
        
        # 1. Update Rover Locations from Dynamic State
        # We iterate known rovers and find their 'at' fact.
        rover_locs = {}
        for r in self.rovers:
            # Optimization: check 'at' for this rover. 
            # Iterate waypoints is costly but necessary if we don't track state changes externally.
            # However, typically rovers don't jump. 
            # In a heuristic, we can iterate all waypoints or assume consistency.
            # For P30 (130 waypoints, 25 rovers), 3250 checks is acceptable.
            for w in self.waypoints:
                if self.get_state_fact(state_bitwise, 'at', r, w):
                    rover_locs[r] = w
                    break
        
        # 2. Iterate Tasks and Estimate Cost
        for task in task_network:
            # Parse task name. Format: "task_name arg1 arg2 ..."
            if hasattr(task, 'name'):
                parts = task.name.split()
                t_name = parts[0]
                args = parts[1:]
            else:
                # Operator object
                cost += getattr(task, 'cost', 1)
                continue

            if t_name in ['do_navigate1', 'do_navigate2', 'navigate']:
                # Navigation Cost: BFS distance
                # Args: rover, [from], to
                r = args[0]
                to = args[-1]
                if r in rover_locs:
                    d = self.dist[r][rover_locs[r]][to]
                    if d != self.inf:
                        cost += d
                    else:
                        cost += 100 # Penalty for unreachability
            
            elif t_name in ['get_soil_data', 'get_rock_data']:
                w = args[0]
                is_soil = (t_name == 'get_soil_data')
                
                # Check if goal already achieved
                comm_pred = 'communicated_soil_data' if is_soil else 'communicated_rock_data'
                if self.get_state_fact(state_bitwise, comm_pred, w):
                    continue
                
                # Estimate Cost: Find BEST rover for the job
                # Cost = Drive to sample + Sample(1) + Drive to Comm + Comm(1)
                # If data already held, just Drive to Comm + Comm(1)
                
                min_task_cost = self.inf
                capable_rovers = self.equipped_soil if is_soil else self.equipped_rock
                
                have_pred = 'have_soil_analysis' if is_soil else 'have_rock_analysis'
                
                for r in capable_rovers:
                    if r not in rover_locs: continue
                    curr_w = rover_locs[r]
                    
                    if self.get_state_fact(state_bitwise, have_pred, r, w):
                        # Already has sample
                        d = self.dist_to_comm[r][curr_w]
                        c = d + 1 # +1 to communicate
                    else:
                        # Fetch and Carry
                        d1 = self.dist[r][curr_w][w]
                        d2 = self.dist_to_comm[r][w]
                        if d1 != self.inf and d2 != self.inf:
                            c = d1 + 1 + d2 + 1
                        else:
                            c = self.inf
                    
                    if c < min_task_cost:
                        min_task_cost = c
                
                if min_task_cost != self.inf:
                    cost += min_task_cost
                else:
                    cost += 50 # Penalty if no rover can do it

            elif t_name == 'get_image_data':
                obj = args[0]
                mode = args[1]
                
                if self.get_state_fact(state_bitwise, 'communicated_image_data', obj, mode):
                    continue
                
                min_task_cost = self.inf
                
                for r in self.equipped_img:
                    if r not in rover_locs: continue
                    curr_w = rover_locs[r]
                    
                    # Check holding
                    if self.get_state_fact(state_bitwise, 'have_image', r, obj, mode):
                        d = self.dist_to_comm[r][curr_w]
                        c = d + 1
                        if c < min_task_cost: min_task_cost = c
                        continue
                    
                    # Check Cameras on this rover
                    for cam in self.camera_on_board[r]:
                        if mode in self.camera_supports[cam]:
                            # Cost: Calibrate + Move to View + Take + Move to Comm + Send
                            
                            # Calibration Cost
                            cost_cal = 0
                            if not self.get_state_fact(state_bitwise, 'calibrated', cam, r):
                                tgt = self.camera_targets.get(cam)
                                if tgt:
                                    # Find closest view for calibration target
                                    min_d_cal = self.inf
                                    for w_cal in self.visible_from[tgt]:
                                        d = self.dist[r][curr_w][w_cal]
                                        if d < min_d_cal: min_d_cal = d
                                    if min_d_cal == self.inf:
                                        cost_cal = self.inf
                                    else:
                                        cost_cal = min_d_cal + 1
                                else:
                                    cost_cal = 1 # Assume 1 if target static logic fails
                            
                            if cost_cal == self.inf: continue

                            # Move to View Cost
                            # Note: if calibrated, we moved. This is a lower bound approx.
                            min_d_view = self.inf
                            for w_view in self.visible_from[obj]:
                                d = self.dist[r][curr_w][w_view]
                                if d < min_d_view: min_d_view = d
                            
                            if min_d_view == self.inf: continue
                            
                            # Comm Cost (from view point)
                            # We approximate comm from w_view
                            d_comm = self.dist_to_comm[r][w_view]
                            
                            if d_comm == self.inf: continue
                            
                            # Total = Calibrate + NavToView + Take(1) + NavToComm + Send(1)
                            # We don't double count nav if calibration is on the way, but 
                            # summing them is a safe admissible-ish heuristic for HTN.
                            c = cost_cal + min_d_view + 1 + d_comm + 1
                            if c < min_task_cost:
                                min_task_cost = c

                if min_task_cost != self.inf:
                    cost += min_task_cost
                else:
                    cost += 50
            
            elif t_name.startswith('send_'):
                # Subtask handling: send_soil_data ?x ?from
                if len(args) >= 1:
                    r = args[0]
                    if r in rover_locs:
                        d = self.dist_to_comm[r][rover_locs[r]]
                        if d != self.inf:
                            cost += d + 1
            else:
                # Generic cost for other tasks
                cost += 1
                
        return cost