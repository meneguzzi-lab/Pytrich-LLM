import sys
import re
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic12(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_11=None):
        super().__init__(model)
        self.model = model
        self.inf = 99999

        # Domain maps
        self.rover_graph = defaultdict(lambda: defaultdict(list))
        self.rover_dists = defaultdict(lambda: defaultdict(lambda: self.inf))
        self.comm_points = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.rover_cameras = defaultdict(list)
        self.camera_modes = defaultdict(set)
        self.camera_targets = defaultdict(str)
        self.visible_from = defaultdict(set)
        self.rover_store = {}
        
        # Fact ID lookups for dynamic state checking
        self.f_at_ids = [] # (global_id, rover, waypoint)
        self.f_full_ids = {} # store_name -> global_id
        self.f_have_soil_ids = {} # (rover, waypoint) -> global_id
        self.f_have_rock_ids = {} # (rover, waypoint) -> global_id
        self.f_have_image_ids = {} # (rover, obj, mode) -> global_id
        self.f_calibrated_ids = {} # (camera, rover) -> global_id
        self.f_comm_soil_ids = {} # waypoint -> global_id
        self.f_comm_rock_ids = {} # waypoint -> global_id
        self.f_comm_image_ids = {} # (obj, mode) -> global_id

        # Pre-parse tasks/operators by global_id for speed in h()
        self.task_info = {}
        
        self._parse_static_info()
        self._compute_distances()
        self._preparse_tasks()

    def _parse_atom(self, name):
        name = name.strip()
        is_neg = False
        if name.startswith("-"):
            is_neg = True
            name = name[1:]
        elif name.startswith("+"):
            name = name[1:]

        # Handle +pred[arg1,arg2]
        if '[' in name and name.endswith(']'):
            pred = name[:name.find('[')]
            args = name[name.find('[')+1:-1].split(',')
            return pred, args, is_neg
        
        # Handle pred arg1 arg2
        parts = name.split()
        return parts[0], parts[1:], is_neg

    def _parse_static_info(self):
        # Temporary structures
        lander_locs = set()
        vis_graph = defaultdict(set)
        
        # Iterate all possible facts to build mappings
        for fact in self.model.facts:
            pred, args, is_neg = self._parse_atom(fact.name)
            gid = fact.global_id
            
            if is_neg: continue

            # Static facts (check if in initial state)
            is_true_initially = False
            if isinstance(self.model.initial_state, set):
                if gid in self.model.initial_state: is_true_initially = True
            elif isinstance(self.model.initial_state, int):
                if (self.model.initial_state >> gid) & 1: is_true_initially = True
            
            if is_true_initially:
                if pred == 'can_traverse':
                    # args: rover, from, to
                    self.rover_graph[args[0]][args[1]].append(args[2])
                elif pred == 'equipped_for_soil_analysis':
                    self.equipped_soil.add(args[0])
                elif pred == 'equipped_for_rock_analysis':
                    self.equipped_rock.add(args[0])
                elif pred == 'equipped_for_imaging':
                    self.equipped_image.add(args[0])
                elif pred == 'on_board':
                    self.rover_cameras[args[1]].append(args[0])
                elif pred == 'supports':
                    self.camera_modes[args[0]].add(args[1])
                elif pred == 'calibration_target':
                    self.camera_targets[args[0]] = args[1]
                elif pred == 'visible_from':
                    self.visible_from[args[0]].add(args[1])
                elif pred == 'at_lander':
                    lander_locs.add(args[1])
                elif pred == 'store_of':
                    self.rover_store[args[1]] = args[0]
                elif pred == 'visible':
                    vis_graph[args[0]].add(args[1])
                    vis_graph[args[1]].add(args[0])
            
            # Dynamic fact ID mapping
            if pred == 'at':
                self.f_at_ids.append((gid, args[0], args[1]))
            elif pred == 'full':
                self.f_full_ids[args[0]] = gid
            elif pred == 'have_soil_analysis':
                self.f_have_soil_ids[(args[0], args[1])] = gid
            elif pred == 'have_rock_analysis':
                self.f_have_rock_ids[(args[0], args[1])] = gid
            elif pred == 'have_image':
                self.f_have_image_ids[(args[0], args[1], args[2])] = gid
            elif pred == 'calibrated':
                self.f_calibrated_ids[(args[0], args[1])] = gid
            elif pred == 'communicated_soil_data':
                self.f_comm_soil_ids[args[0]] = gid
            elif pred == 'communicated_rock_data':
                self.f_comm_rock_ids[args[0]] = gid
            elif pred == 'communicated_image_data':
                self.f_comm_image_ids[(args[0], args[1])] = gid

        # Identify comm points
        for p1, neighbors in vis_graph.items():
            # If p1 is at lander, or p1 can see a lander loc
            if p1 in lander_locs:
                self.comm_points.add(p1)
            for n in neighbors:
                if n in lander_locs:
                    self.comm_points.add(p1)
                    break

    def _compute_distances(self):
        # BFS for each rover
        for rover, graph in self.rover_graph.items():
            nodes = list(graph.keys())
            for start_node in nodes:
                self.rover_dists[rover][start_node][start_node] = 0
                q = deque([(start_node, 0)])
                visited = {start_node}
                while q:
                    curr, dist = q.popleft()
                    self.rover_dists[rover][start_node][curr] = dist
                    for n in graph[curr]:
                        if n not in visited:
                            visited.add(n)
                            q.append((n, dist + 1))

    def _preparse_tasks(self):
        # Pre-parse abstract tasks
        for task in self.model.abstract_tasks:
            pred, args, _ = self._parse_atom(task.name)
            self.task_info[task.global_id] = (pred, args)
        # Pre-parse operators (primitives)
        for op in self.model.operators:
            pred, args, _ = self._parse_atom(op.name)
            self.task_info[op.global_id] = (pred, args)

    def h(self, state, task_network):
        # 1. Decode dynamic state
        rover_locs = {}
        for gid, r, w in self.f_at_ids:
            if (state >> gid) & 1:
                rover_locs[r] = w
        
        # Use a copy to track estimated location during this heuristic pass
        est_rover_locs = rover_locs.copy()
        total_cost = 0

        for task_obj in task_network:
            # task_obj can be Operator or AbstractTask
            if task_obj.global_id not in self.task_info:
                # Should typically not happen if pre-parsing worked
                continue
            
            t_name, t_args = self.task_info[task_obj.global_id]
            cost = 0

            # --- Navigation ---
            if t_name in ['do_navigate1', 'do_navigate2', 'navigate']:
                r = t_args[0]
                to_wp = t_args[-1]
                if r in est_rover_locs:
                    d = self.rover_dists[r][est_rover_locs[r]][to_wp]
                    if d < self.inf:
                        cost = d
                        est_rover_locs[r] = to_wp
                    else:
                        return self.inf

            # --- Soil Data ---
            elif t_name == 'get_soil_data':
                wp = t_args[0]
                # Check if done
                gid = self.f_comm_soil_ids.get(wp)
                if gid is not None and (state >> gid) & 1:
                    continue
                
                min_cost = self.inf
                for r in self.equipped_soil:
                    if r not in est_rover_locs: continue
                    curr_loc = est_rover_locs[r]
                    
                    # Cost to sample + drop if full
                    d_sample = self.rover_dists[r][curr_loc][wp]
                    if d_sample == self.inf: continue
                    
                    store = self.rover_store.get(r)
                    is_full = False
                    if store:
                        sgid = self.f_full_ids.get(store)
                        if sgid is not None and (state >> sgid) & 1:
                            is_full = True
                    
                    op_cost = 1 + (1 if is_full else 0) # Sample + Drop
                    
                    # Cost to comm
                    # Check if analysis held
                    hid = self.f_have_soil_ids.get((r, wp))
                    has_data = (hid is not None and (state >> hid) & 1)
                    
                    if has_data:
                        # Already have data, just go to comm
                        start_node = curr_loc
                        op_cost = 1 # Just comm
                    else:
                        # Go to sample then comm
                        start_node = wp
                        op_cost += d_sample # Add travel to sample
                        
                    d_comm = self.inf
                    for cp in self.comm_points:
                        d = self.rover_dists[r][start_node][cp]
                        if d < d_comm: d_comm = d
                    
                    if d_comm < self.inf:
                        total = op_cost + d_comm
                        if total < min_cost: min_cost = total
                
                if min_cost < self.inf: cost = min_cost
                else: return self.inf

            # --- Rock Data ---
            elif t_name == 'get_rock_data':
                wp = t_args[0]
                gid = self.f_comm_rock_ids.get(wp)
                if gid is not None and (state >> gid) & 1:
                    continue
                
                min_cost = self.inf
                for r in self.equipped_rock:
                    if r not in est_rover_locs: continue
                    curr_loc = est_rover_locs[r]
                    
                    d_sample = self.rover_dists[r][curr_loc][wp]
                    if d_sample == self.inf: continue
                    
                    store = self.rover_store.get(r)
                    is_full = False
                    if store:
                        sgid = self.f_full_ids.get(store)
                        if sgid is not None and (state >> sgid) & 1:
                            is_full = True
                    
                    op_cost = 1 + (1 if is_full else 0)
                    
                    hid = self.f_have_rock_ids.get((r, wp))
                    has_data = (hid is not None and (state >> hid) & 1)
                    
                    if has_data:
                        start_node = curr_loc
                        op_cost = 1
                    else:
                        start_node = wp
                        op_cost += d_sample
                        
                    d_comm = self.inf
                    for cp in self.comm_points:
                        d = self.rover_dists[r][start_node][cp]
                        if d < d_comm: d_comm = d
                    
                    if d_comm < self.inf:
                        total = op_cost + d_comm
                        if total < min_cost: min_cost = total
                
                if min_cost < self.inf: cost = min_cost
                else: return self.inf

            # --- Image Data ---
            elif t_name == 'get_image_data':
                obj = t_args[0]
                mode = t_args[1]
                
                gid = self.f_comm_image_ids.get((obj, mode))
                if gid is not None and (state >> gid) & 1: continue
                
                min_cost = self.inf
                for r in self.equipped_image:
                    if r not in est_rover_locs: continue
                    curr_loc = est_rover_locs[r]
                    
                    # Check valid camera
                    valid_cam = False
                    needs_calib = True
                    cal_target = None
                    
                    for cam in self.rover_cameras[r]:
                        if mode in self.camera_modes[cam]:
                            valid_cam = True
                            cid = self.f_calibrated_ids.get((cam, r))
                            if cid is not None and (state >> cid) & 1:
                                needs_calib = False
                            else:
                                cal_target = self.camera_targets[cam]
                            break
                    
                    if not valid_cam: continue
                    
                    hid = self.f_have_image_ids.get((r, obj, mode))
                    has_img = (hid is not None and (state >> hid) & 1)
                    
                    ops = 0
                    sim_loc = curr_loc
                    
                    if not has_img:
                        # Calibration
                        if needs_calib and cal_target:
                            best_cal_dist = self.inf
                            best_cal_wp = None
                            for wp in self.visible_from[cal_target]:
                                d = self.rover_dists[r][sim_loc][wp]
                                if d < best_cal_dist:
                                    best_cal_dist = d
                                    best_cal_wp = wp
                            if best_cal_dist == self.inf: continue
                            ops += best_cal_dist + 1 # nav + calibrate
                            sim_loc = best_cal_wp
                        
                        # Imaging
                        best_img_dist = self.inf
                        best_img_wp = None
                        for wp in self.visible_from[obj]:
                            d = self.rover_dists[r][sim_loc][wp]
                            if d < best_img_dist:
                                best_img_dist = d
                                best_img_wp = wp
                        if best_img_dist == self.inf: continue
                        ops += best_img_dist + 1 # nav + take_image
                        sim_loc = best_img_wp
                    
                    # Comm
                    best_comm_dist = self.inf
                    for cp in self.comm_points:
                        d = self.rover_dists[r][sim_loc][cp]
                        if d < best_comm_dist: best_comm_dist = d
                    
                    if best_comm_dist < self.inf:
                        total = ops + best_comm_dist + 1 # + communicate
                        if total < min_cost: min_cost = total
                
                if min_cost < self.inf: cost = min_cost
                else: return self.inf

            # --- Sending Data (Explicit) ---
            elif t_name.startswith('send_'):
                # send_soil_data ?x ?from
                # Primitive actions usually don't appear in abstract plan if decomposed, 
                # but if they do, minimal cost is nav to comm + 1
                r = t_args[0]
                if r in est_rover_locs:
                    d_comm = self.inf
                    for cp in self.comm_points:
                        d = self.rover_dists[r][est_rover_locs[r]][cp]
                        if d < d_comm: d_comm = d
                    if d_comm < self.inf:
                        cost = d_comm + 1
                    else:
                        return self.inf
            
            elif t_name in ['sample_soil', 'sample_rock', 'take_image', 'calibrate', 'drop']:
                cost = 1
            
            elif t_name.startswith('communicate_'):
                cost = 1
            
            elif t_name == 'empty_store':
                store = t_args[0]
                gid = self.f_full_ids.get(store)
                if gid is not None and (state >> gid) & 1:
                    cost = 1

            total_cost += cost

        return total_cost
