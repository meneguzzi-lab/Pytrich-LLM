import sys
import math
from collections import deque, defaultdict
from pytrich.heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic10(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_9=None):
        super().__init__(model)
        # Initialize data structures for domain analysis
        self.waypoints = set()
        self.rovers = set()
        self.objectives = set()
        self.cameras = set()
        self.stores = set()
        self.landers = set()
        self.modes = set()

        # Static Relations
        self.visible_graph = defaultdict(list)
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.store_of = {}
        self.on_board = defaultdict(list)  # rover -> [cameras]
        self.calibration_target = {}       # camera -> objective
        self.supports = defaultdict(set)   # camera -> {modes}
        self.visible_from = defaultdict(list) # objective -> [waypoints]
        self.at_lander = {} # lander -> waypoint (assuming static for standard rover, or check init)
        
        # Dynamic Predicate ID Mappings (for state checks)
        self.fact_ids = {}
        
        # Parse all ground facts to build static knowledge and ID maps
        self._parse_static_knowledge()

        # Precompute Distances (All-Pairs Shortest Path via BFS)
        self.distances = self._compute_all_distances()

        # Precompute nearest communication points
        self.comm_points = {wp for loc in self.at_lander.values() for wp in self.waypoints if self._get_dist(wp, loc) != float('inf') and self._is_visible(wp, loc)}
        # In rover domain, you usually communicate from a waypoint visible to the lander location.
        # The predicate is (visible ?w1 ?w2) and (at_lander ?l ?w2).
        # So comm points are neighbors of lander locations in the visibility graph.

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
        return name, []

    def _parse_static_knowledge(self):
        # We iterate over all possible facts in the model to find static relations and map dynamic ones
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            
            # Map fact signature to global ID for fast state lookup
            sig = (pred, tuple(args))
            self.fact_ids[sig] = fact.global_id

            # Collect Types and Static Predicates
            if pred == 'visible' and len(args) == 2:
                self.visible_graph[args[0]].append(args[1])
                self.waypoints.add(args[0])
                self.waypoints.add(args[1])
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_for_soil.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_for_rock.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_for_imaging.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'store_of':
                self.store_of[args[1]] = args[0] # rover -> store
                self.stores.add(args[0])
                self.rovers.add(args[1])
            elif pred == 'on_board':
                self.on_board[args[1]].append(args[0])
                self.cameras.add(args[0])
                self.rovers.add(args[1])
            elif pred == 'calibration_target':
                self.calibration_target[args[0]] = args[1]
                self.cameras.add(args[0])
                self.objectives.add(args[1])
            elif pred == 'supports':
                self.supports[args[0]].add(args[1])
                self.cameras.add(args[0])
                self.modes.add(args[1])
            elif pred == 'visible_from':
                self.visible_from[args[0]].append(args[1])
                self.objectives.add(args[0])
                self.waypoints.add(args[1])
            elif pred == 'at_lander':
                # at_lander is often static in these problems defined in init
                # If it's dynamic, we handle it in state check, but usually landers don't move.
                # We'll assume static for heuristic pre-computation but could check state if needed.
                self.at_lander[args[0]] = args[1]
                self.landers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == 'at':
                # Just collecting types
                if args[0].startswith('rover'): self.rovers.add(args[0])
                if args[0].startswith('waypoint'): self.waypoints.add(args[0])

    def _compute_all_distances(self):
        # BFS for all-pairs shortest paths on the visibility graph
        dists = {}
        for start_node in self.waypoints:
            queue = deque([(start_node, 0)])
            dists[(start_node, start_node)] = 0
            visited = {start_node}
            while queue:
                curr, d = queue.popleft()
                dists[(start_node, curr)] = d
                for neighbor in self.visible_graph.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
        return dists

    def _get_dist(self, w1, w2):
        if w1 == w2: return 0
        return self.distances.get((w1, w2), float('inf'))

    def _is_visible(self, w1, w2):
        # Check adjacency
        return w2 in self.visible_graph.get(w1, [])

    def _check_fact(self, state_bitwise, pred, *args):
        sig = (pred, tuple(args))
        gid = self.fact_ids.get(sig)
        if gid is not None:
            return (state_bitwise >> gid) & 1
        return 0

    def _get_rover_loc(self, rover, state):
        # Inefficient to scan all waypoints, but robust. 
        # Optimization: Could cache global_ids for (at rover w) for each w.
        for w in self.waypoints:
            if self._check_fact(state, 'at', rover, w):
                return w
        return None

    def _get_closest_comm(self, w_from):
        # Find nearest waypoint that is visible to a lander
        best_dist = float('inf')
        for l, l_loc in self.at_lander.items():
            # We need a waypoint w_comm such that visible(w_comm, l_loc)
            # We want to minimize dist(w_from, w_comm)
            # l_loc neighbors are candidates
            for comm_cand in self.visible_graph.get(l_loc, []):
                d = self._get_dist(w_from, comm_cand)
                if d < best_dist:
                    best_dist = d
            # Also check if w_from is already a comm location
            if self._is_visible(w_from, l_loc):
                return 0
        return best_dist

    def __call__(self, task_network, state, plan=None):
        # h(n) = sum of estimated costs of tasks in the network
        cost = 0
        
        # Cache rover locations for this state to avoid repeated lookups
        rover_locs = {}
        for r in self.rovers:
            loc = self._get_rover_loc(r, state)
            if loc: rover_locs[r] = loc

        for task in task_network:
            # Task is an Operator or AbstractTask object, but we need name/args
            # In Pytrich, task objects have a .name attribute which is the grounded string
            t_name, t_args = self._parse_fact_name(task.name)
            
            if t_name == 'do_navigate1':
                # do_navigate1 ?x ?to
                rover, to_wp = t_args
                r_loc = rover_locs.get(rover)
                if r_loc:
                    cost += self._get_dist(r_loc, to_wp)
            
            elif t_name == 'empty_store':
                # empty_store ?s ?x
                store, rover = t_args
                # If store full, add drop cost (1)
                if self._check_fact(state, 'full', store):
                    cost += 1
            
            elif t_name == 'get_soil_data':
                # get_soil_data ?from
                wp = t_args[0]
                if self._check_fact(state, 'communicated_soil_data', wp):
                    continue
                
                # Check if anyone has analysis
                has_analysis = False
                for r in self.equipped_for_soil:
                    if self._check_fact(state, 'have_soil_analysis', r, wp):
                        has_analysis = True
                        # Cost: navigate to comm + 1
                        r_loc = rover_locs.get(r)
                        if r_loc:
                            cost += self._get_closest_comm(r_loc) + 1
                        break
                
                if not has_analysis:
                    # Estimate best rover
                    best_rover_cost = float('inf')
                    for r in self.equipped_for_soil:
                        r_loc = rover_locs.get(r)
                        if r_loc:
                            # Nav to sample + Sample + Nav to comm + Comm
                            # Drop cost if store full?
                            store = self.store_of.get(r)
                            drop_cost = 1 if store and self._check_fact(state, 'full', store) else 0
                            
                            c = self._get_dist(r_loc, wp) + 1 + drop_cost + self._get_closest_comm(wp) + 1
                            if c < best_rover_cost:
                                best_rover_cost = c
                    if best_rover_cost != float('inf'):
                        cost += best_rover_cost

            elif t_name == 'get_rock_data':
                # get_rock_data ?from
                wp = t_args[0]
                if self._check_fact(state, 'communicated_rock_data', wp):
                    continue

                has_analysis = False
                for r in self.equipped_for_rock:
                    if self._check_fact(state, 'have_rock_analysis', r, wp):
                        has_analysis = True
                        r_loc = rover_locs.get(r)
                        if r_loc:
                            cost += self._get_closest_comm(r_loc) + 1
                        break
                
                if not has_analysis:
                    best_rover_cost = float('inf')
                    for r in self.equipped_for_rock:
                        r_loc = rover_locs.get(r)
                        if r_loc:
                            store = self.store_of.get(r)
                            drop_cost = 1 if store and self._check_fact(state, 'full', store) else 0
                            c = self._get_dist(r_loc, wp) + 1 + drop_cost + self._get_closest_comm(wp) + 1
                            if c < best_rover_cost:
                                best_rover_cost = c
                    if best_rover_cost != float('inf'):
                        cost += best_rover_cost

            elif t_name == 'get_image_data':
                # get_image_data ?o ?m
                obj, mode = t_args
                if self._check_fact(state, 'communicated_image_data', obj, mode):
                    continue
                
                has_image = False
                for r in self.equipped_for_imaging:
                    if self._check_fact(state, 'have_image', r, obj, mode):
                        has_image = True
                        r_loc = rover_locs.get(r)
                        if r_loc:
                            cost += self._get_closest_comm(r_loc) + 1
                        break
                
                if not has_image:
                    best_rover_cost = float('inf')
                    # Find rovers equipped for imaging AND supporting mode
                    capable_rovers = []
                    for r in self.equipped_for_imaging:
                        for cam in self.on_board[r]:
                            if mode in self.supports[cam]:
                                capable_rovers.append((r, cam))
                                break # One camera is enough
                    
                    for r, cam in capable_rovers:
                        r_loc = rover_locs.get(r)
                        if r_loc:
                            # Calibration
                            cal_cost = 0
                            if not self._check_fact(state, 'calibrated', cam, r):
                                cal_target = self.calibration_target.get(cam)
                                if cal_target:
                                    # Must navigate to somewhere where cal_target is visible
                                    # Usually calibration target is an objective. Pred: visible_from ?o ?w
                                    # But HDDL says: (calibration_target ?c ?o)
                                    # Action calibrate: (visible_from ?o ?p) (at ?x ?p)
                                    # So we need to go to a waypoint seeing the calibration target
                                    best_cal_dist = float('inf')
                                    for w_see in self.visible_from.get(cal_target, []):
                                        d = self._get_dist(r_loc, w_see)
                                        if d < best_cal_dist: best_cal_dist = d
                                    cal_cost = best_cal_dist + 1 # Nav + Calibrate
                                else:
                                    cal_cost = 1 # Fallback
                            
                            # Taking Image
                            # Must go to waypoint seeing objective
                            best_img_dist = float('inf')
                            best_w_img = None
                            # Cost depends on where we are AFTER calibration
                            # Heuristic approximation: just add calibration cost to trip
                            # More accurate: dist(r, cal_pos) + dist(cal_pos, img_pos)
                            # We'll use a simplified additive model: cal_cost + min_dist_to_img_pos
                            
                            for w_see in self.visible_from.get(obj, []):
                                d = self._get_dist(r_loc, w_see)
                                if d < best_img_dist:
                                    best_img_dist = d
                                    best_w_img = w_see
                            
                            # Send
                            send_dist = 0
                            if best_w_img:
                                send_dist = self._get_closest_comm(best_w_img)
                            
                            total = cal_cost + best_img_dist + 1 + send_dist + 1
                            if total < best_rover_cost:
                                best_rover_cost = total
                                
                    if best_rover_cost != float('inf'):
                        cost += best_rover_cost

            elif t_name == 'do_calibrate':
                # do_calibrate ?x ?c
                rover, cam = t_args
                if not self._check_fact(state, 'calibrated', cam, rover):
                    r_loc = rover_locs.get(rover)
                    if r_loc:
                        cal_target = self.calibration_target.get(cam)
                        best_dist = float('inf')
                        for w in self.visible_from.get(cal_target, []):
                            d = self._get_dist(r_loc, w)
                            if d < best_dist: best_dist = d
                        cost += best_dist + 1
            
            elif t_name == 'send_soil_data' or t_name == 'send_rock_data' or t_name == 'send_image_data':
                # send_* ?x ...
                # Assuming args[0] is rover
                if t_args:
                    rover = t_args[0]
                    r_loc = rover_locs.get(rover)
                    if r_loc:
                        cost += self._get_closest_comm(r_loc) + 1

            # Primitive tasks (operators)
            elif t_name in ['navigate', 'sample_soil', 'sample_rock', 'drop', 'calibrate', 'take_image', 
                            'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data']:
                cost += 1
            
            # Ignore 'visit', 'unvisit', 'nop' cost = 0
            
        return cost
