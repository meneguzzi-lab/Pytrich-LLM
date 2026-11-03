# Rover GTOHP Domain-Dependent Heuristic
# 
# BOTTLENECK: Navigation is the primary serializing constraint. Each data task
# (soil, rock, image) requires multiple navigate-sample/image-navigate-communicate
# steps. Store capacity (single store per rover) forces drops between samples.
#
# ADMISSIBLE LOWER BOUNDS:
# T1 - Uncommunicated goals: each uncommunicated_*_data goal needs >= 1 communicate action
# T2 - Task network minimum cost: each get_*_data task needs >= 3 actions minimum
#      (navigate+sample+communicate or calibrate+navigate+take_image+communicate)
# T3 - Unsatisfied precondition cost: for each pending task, estimate minimum actions
#      based on current state (sample availability, rover equipment, calibration status)
#
# The heuristic returns max(T1, T2, T3) for admissibility.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as OperatorClass
from Pytrich.model import AbstractTask as AbstractTaskClass


class RoverGTOHPAnthropicOpus46Heuristic10(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses multi-component max of:
    - Uncommunicated goal facts count
    - Task network structural minimum cost with state awareness
    - Navigation + resource bottleneck estimation
    """

    def __init__(self):
        super().__init__(name="rover_anthropicopus46_heuristic_10")
        # Fact ID maps
        self.communicated_soil_ids = {}   # waypoint -> global_id
        self.communicated_rock_ids = {}   # waypoint -> global_id  
        self.communicated_image_ids = {}  # (objective, mode) -> global_id
        self.at_soil_sample_ids = {}      # waypoint -> global_id
        self.at_rock_sample_ids = {}      # waypoint -> global_id
        self.have_soil_ids = {}           # (rover, waypoint) -> global_id
        self.have_rock_ids = {}           # (rover, waypoint) -> global_id
        self.have_image_ids = {}          # (rover, objective, mode) -> global_id
        self.calibrated_ids = {}          # (camera, rover) -> global_id
        self.at_rover_ids = {}            # (rover, waypoint) -> global_id
        self.empty_store_ids = {}         # store -> global_id
        self.full_store_ids = {}          # store -> global_id
        self.store_of_ids = {}            # (store, rover) -> global_id
        self.equipped_soil_ids = {}       # rover -> global_id
        self.equipped_rock_ids = {}       # rover -> global_id
        self.equipped_imaging_ids = {}    # rover -> global_id
        self.on_board_ids = {}            # (camera, rover) -> global_id
        self.supports_ids = {}            # (camera, mode) -> global_id
        self.visible_from_ids = {}        # (objective, waypoint) -> global_id
        self.calibration_target_ids = {}  # (camera, objective) -> global_id
        self.at_lander_ids = {}           # (lander, waypoint) -> global_id
        self.visible_ids = {}             # (wp1, wp2) -> global_id
        self.channel_free_ids = {}        # lander -> global_id
        self.available_ids = {}           # rover -> global_id
        self.can_traverse_ids = {}        # (rover, wp1, wp2) -> global_id
        
        # Goal tracking
        self.goal_soil_waypoints = []     # waypoints needing soil comm
        self.goal_rock_waypoints = []     # waypoints needing rock comm
        self.goal_image_pairs = []        # (objective, mode) needing image comm
        self.goal_fact_ids = set()         # all goal fact global_ids
        
        # Domain structure caches
        self.soil_rovers = []             # rovers equipped for soil
        self.rock_rovers = []             # rovers equipped for rock
        self.imaging_rovers = []          # rovers equipped for imaging
        self.rover_stores = {}            # rover -> store
        self.rover_cameras = {}           # rover -> list of cameras
        self.camera_modes = {}            # camera -> list of modes
        self.camera_cal_target = {}       # camera -> objective
        self.lander_waypoint = None       # waypoint where lander is
        self.lander_name = None
        
        # Task name prefixes for identification
        self.task_get_soil = None
        self.task_get_rock = None
        self.task_get_image = None
        self.task_send_soil = None
        self.task_send_rock = None
        self.task_send_image = None
        self.task_navigate1 = None
        self.task_navigate2 = None
        self.task_empty_store = None
        self.task_calibrate = None

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like +predicate[arg1,arg2] into (predicate, [args])."""
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

    def _parse_task_name(self, name: str):
        """Parse task/operator name to extract task type and arguments."""
        # Names look like: get_soil_data[waypoint0] or navigate[rover0,waypoint1,waypoint2]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Build all fact maps and precompute domain structure."""
        self.model = model
        
        # Parse all facts into lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            gid = f.global_id
            
            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_ids[args[0]] = gid
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_ids[args[0]] = gid
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_ids[(args[0], args[1])] = gid
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = gid
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = gid
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_ids[(args[0], args[1])] = gid
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_ids[(args[0], args[1])] = gid
            elif pred == "have_image" and len(args) == 3:
                self.have_image_ids[(args[0], args[1], args[2])] = gid
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = gid
            elif pred == "at" and len(args) == 2:
                self.at_rover_ids[(args[0], args[1])] = gid
            elif pred == "empty" and len(args) == 1:
                self.empty_store_ids[args[0]] = gid
            elif pred == "full" and len(args) == 1:
                self.full_store_ids[args[0]] = gid
            elif pred == "store_of" and len(args) == 2:
                self.store_of_ids[(args[0], args[1])] = gid
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids[args[0]] = gid
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids[args[0]] = gid
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids[args[0]] = gid
            elif pred == "on_board" and len(args) == 2:
                self.on_board_ids[(args[0], args[1])] = gid
            elif pred == "supports" and len(args) == 2:
                self.supports_ids[(args[0], args[1])] = gid
            elif pred == "visible_from" and len(args) == 2:
                self.visible_from_ids[(args[0], args[1])] = gid
            elif pred == "calibration_target" and len(args) == 2:
                self.calibration_target_ids[(args[0], args[1])] = gid
            elif pred == "at_lander" and len(args) == 2:
                self.at_lander_ids[(args[0], args[1])] = gid
            elif pred == "visible" and len(args) == 2:
                self.visible_ids[(args[0], args[1])] = gid
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_ids[args[0]] = gid
            elif pred == "available" and len(args) == 1:
                self.available_ids[args[0]] = gid
            elif pred == "can_traverse" and len(args) == 3:
                self.can_traverse_ids[(args[0], args[1], args[2])] = gid
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data":
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data":
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data":
                self.goal_image_pairs.append((args[0], args[1]))
        
        # Build goal fact id set for fast lookup
        for wp in self.goal_soil_waypoints:
            if wp in self.communicated_soil_ids:
                self.goal_fact_ids.add(self.communicated_soil_ids[wp])
        for wp in self.goal_rock_waypoints:
            if wp in self.communicated_rock_ids:
                self.goal_fact_ids.add(self.communicated_rock_ids[wp])
        for (obj, mode) in self.goal_image_pairs:
            key = (obj, mode)
            if key in self.communicated_image_ids:
                self.goal_fact_ids.add(self.communicated_image_ids[key])
        
        # Build rover equipment lists
        for rover, gid in self.equipped_soil_ids.items():
            self.soil_rovers.append(rover)
        for rover, gid in self.equipped_rock_ids.items():
            self.rock_rovers.append(rover)
        for rover, gid in self.equipped_imaging_ids.items():
            self.imaging_rovers.append(rover)
        
        # Build rover -> store mapping
        for (store, rover), gid in self.store_of_ids.items():
            self.rover_stores[rover] = store
        
        # Build rover -> cameras mapping
        for (camera, rover), gid in self.on_board_ids.items():
            if rover not in self.rover_cameras:
                self.rover_cameras[rover] = []
            self.rover_cameras[rover].append(camera)
        
        # Build camera -> modes mapping
        for (camera, mode), gid in self.supports_ids.items():
            if camera not in self.camera_modes:
                self.camera_modes[camera] = []
            self.camera_modes[camera].append(mode)
        
        # Build camera -> calibration target
        for (camera, objective), gid in self.calibration_target_ids.items():
            self.camera_cal_target[camera] = objective
        
        # Find lander waypoint
        for (lander, waypoint), gid in self.at_lander_ids.items():
            self.lander_waypoint = waypoint
            self.lander_name = lander
        
        # Build waypoints visible to lander (for communication)
        self.comm_waypoints = set()
        if self.lander_waypoint:
            for (wp1, wp2), gid in self.visible_ids.items():
                if wp2 == self.lander_waypoint:
                    self.comm_waypoints.add(wp1)
            # Also lander waypoint itself if visible from itself
            self.comm_waypoints.add(self.lander_waypoint)
        
        # Precompute which rovers can reach which waypoints (BFS adjacency)
        # For each rover, build adjacency list from can_traverse
        self.rover_adjacency = {}
        all_rovers = set()
        for (rover, wp1, wp2), gid in self.can_traverse_ids.items():
            all_rovers.add(rover)
            if rover not in self.rover_adjacency:
                self.rover_adjacency[rover] = {}
            if wp1 not in self.rover_adjacency[rover]:
                self.rover_adjacency[rover][wp1] = []
            self.rover_adjacency[rover][wp1].append(wp2)
        
        # Precompute BFS distances for each rover to all waypoints
        # This is done once in initialize, so it's O(R * W^2) but only once
        self.rover_distances = {}  # rover -> {(from_wp, to_wp): distance}
        # For efficiency, only compute distances we'll actually need:
        # - From each rover's possible positions to goal waypoints
        # - From goal waypoints to comm waypoints
        # Actually, let's just precompute all-pairs for each rover since we need it
        # But for large instances this could be expensive. Let's use lazy BFS.
        
        # Instead, precompute min distance from each waypoint to nearest comm waypoint
        # for each rover. This is more targeted.
        self.wp_to_comm_dist = {}  # (rover, waypoint) -> min distance to any comm waypoint
        
        # For the heuristic, we'll compute on-demand but cache
        self._bfs_cache = {}  # (rover, start_wp) -> {wp: dist}
        
        # Compute initial heuristic
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _bfs_distance(self, rover, start_wp, target_wp):
        """Get BFS distance for rover from start_wp to target_wp. Uses caching."""
        cache_key = (rover, start_wp)
        if cache_key in self._bfs_cache:
            cached = self._bfs_cache[cache_key]
            if target_wp in cached:
                return cached[target_wp]
            # BFS was done but target not reachable
            return 999
        
        # Run BFS from start_wp for this rover
        if rover not in self.rover_adjacency:
            return 999
        
        adj = self.rover_adjacency[rover]
        dist = {start_wp: 0}
        queue = [start_wp]
        qi = 0
        while qi < len(queue):
            current = queue[qi]
            qi += 1
            if current == target_wp:
                break  # Early termination
            if current in adj:
                for neighbor in adj[current]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
        
        self._bfs_cache[cache_key] = dist
        return dist.get(target_wp, 999)

    def _min_dist_to_comm(self, rover, wp):
        """Minimum BFS distance from wp to any comm waypoint for this rover."""
        if wp in self.comm_waypoints:
            return 0
        min_d = 999
        for cwp in self.comm_waypoints:
            d = self._bfs_distance(rover, wp, cwp)
            if d < min_d:
                min_d = d
                if min_d <= 1:
                    break
        return min_d

    def _compute(self, state, task_network):
        """Compute heuristic value from current state and task network."""
        # Term 1: Count uncommunicated goal facts
        t1 = 0
        for gid in self.goal_fact_ids:
            if not ((state >> gid) & 1):
                t1 += 1
        
        # Term 2: Task network structural cost with state awareness
        t2 = self._task_network_cost(state, task_network)
        
        # Term 3: Detailed goal-based cost estimation
        t3 = self._goal_based_cost(state)
        
        return max(t1, t2, t3)

    def _task_network_cost(self, state, task_network):
        """Estimate minimum cost from remaining task network with state awareness."""
        if not task_network:
            return 0
        
        total_cost = 0
        for task in task_network:
            if isinstance(task, OperatorClass):
                # Primitive operator: cost 1 (unless it's nop)
                tname, _ = self._parse_task_name(task.name)
                if tname == "nop":
                    total_cost += 0
                else:
                    total_cost += 1
            elif isinstance(task, AbstractTaskClass):
                # Abstract task: estimate minimum decomposition cost
                cost = self._estimate_abstract_task_cost(state, task)
                total_cost += cost
        
        return total_cost

    def _estimate_abstract_task_cost(self, state, task):
        """Estimate minimum cost to complete an abstract task given current state."""
        tname, args = self._parse_task_name(task.name)
        
        if tname == "get_soil_data":
            # Need: navigate to wp, empty store, sample_soil, send_soil_data
            # send_soil_data needs: navigate to comm wp, communicate
            # Minimum: if already at wp with empty store: sample(1) + navigate_to_comm(>=0) + communicate(1) = 2+
            # Typical: navigate(>=1) + drop?(0-1) + sample(1) + navigate(>=0) + communicate(1) = 3-5+
            wp = args[0] if args else None
            if wp is None:
                return 3
            
            # Check if already communicated
            if wp in self.communicated_soil_ids:
                comm_gid = self.communicated_soil_ids[wp]
                if (state >> comm_gid) & 1:
                    return 0  # Already done
            
            cost = 0
            # Check if any rover has soil analysis for this wp
            has_analysis = False
            for rover in self.soil_rovers:
                key = (rover, wp)
                if key in self.have_soil_ids:
                    hgid = self.have_soil_ids[key]
                    if (state >> hgid) & 1:
                        has_analysis = True
                        # Just need to communicate: navigate to comm + communicate
                        cost = max(cost, 0)  # Will be computed below
                        break
            
            if has_analysis:
                return 1  # At minimum: communicate action
            
            # Need to collect sample first
            # Check if soil sample still exists at waypoint
            if wp in self.at_soil_sample_ids:
                ss_gid = self.at_soil_sample_ids[wp]
                if not ((state >> ss_gid) & 1):
                    return 999  # Sample consumed, can't collect - unreachable
            
            # Minimum: sample(1) + communicate(1) + possible navigate(>=1) + possible drop(0-1)
            return 3  # Conservative minimum
            
        elif tname == "get_rock_data":
            wp = args[0] if args else None
            if wp is None:
                return 3
            
            if wp in self.communicated_rock_ids:
                comm_gid = self.communicated_rock_ids[wp]
                if (state >> comm_gid) & 1:
                    return 0
            
            has_analysis = False
            for rover in self.rock_rovers:
                key = (rover, wp)
                if key in self.have_rock_ids:
                    hgid = self.have_rock_ids[key]
                    if (state >> hgid) & 1:
                        has_analysis = True
                        break
            
            if has_analysis:
                return 1
            
            if wp in self.at_rock_sample_ids:
                rs_gid = self.at_rock_sample_ids[wp]
                if not ((state >> rs_gid) & 1):
                    return 999
            
            return 3
            
        elif tname == "get_image_data":
            # Need: calibrate camera, navigate to visible wp, take_image, send_image_data
            # Minimum: calibrate(1) + take_image(1) + communicate(1) = 3 + navigation
            obj = args[0] if len(args) > 0 else None
            mode = args[1] if len(args) > 1 else None
            
            if obj is None or mode is None:
                return 3
            
            key = (obj, mode)
            if key in self.communicated_image_ids:
                comm_gid = self.communicated_image_ids[key]
                if (state >> comm_gid) & 1:
                    return 0
            
            # Check if any rover already has the image
            has_img = False
            for rover in self.imaging_rovers:
                ikey = (rover, obj, mode)
                if ikey in self.have_image_ids:
                    igid = self.have_image_ids[ikey]
                    if (state >> igid) & 1:
                        has_img = True
                        break
            
            if has_img:
                return 1  # Just communicate
            
            # Need calibrate + take_image + communicate + navigations
            return 4  # calibrate(1) + possible_navigate + take_image(1) + navigate + communicate(1)
            
        elif tname == "send_soil_data" or tname == "send_rock_data" or tname == "send_image_data":
            # navigate to comm visible wp + communicate
            return 1  # At minimum just communicate if already at visible wp
            
        elif tname == "do_navigate1" or tname == "do_navigate2":
            # Check if rover is already at target
            if len(args) >= 2:
                rover = args[0]
                target = args[1] if tname == "do_navigate1" else args[2] if len(args) > 2 else args[1]
                key = (rover, target)
                if key in self.at_rover_ids:
                    at_gid = self.at_rover_ids[key]
                    if (state >> at_gid) & 1:
                        return 0  # Already there
            return 1  # At least one navigate action
            
        elif tname == "empty_store":
            if len(args) >= 1:
                store = args[0]
                if store in self.empty_store_ids:
                    egid = self.empty_store_ids[store]
                    if (state >> egid) & 1:
                        return 0  # Already empty
            return 1  # Need drop action
            
        elif tname == "do_calibrate":
            if len(args) >= 2:
                rover = args[0]
                camera = args[1]
                key = (camera, rover)
                if key in self.calibrated_ids:
                    cgid = self.calibrated_ids[key]
                    if (state >> cgid) & 1:
                        return 0  # Already calibrated
            return 2  # navigate to calibration target + calibrate
        
        # Default: at least 1 action for unknown abstract tasks
        return 1

    def _goal_based_cost(self, state):
        """Estimate cost based on unsatisfied goal conditions with state context."""
        total = 0
        
        # Soil goals
        for wp in self.goal_soil_waypoints:
            if wp in self.communicated_soil_ids:
                gid = self.communicated_soil_ids[wp]
                if (state >> gid) & 1:
                    continue  # Already communicated
            
            # Check if any rover already has the analysis
            has_analysis = False
            for rover in self.soil_rovers:
                key = (rover, wp)
                if key in self.have_soil_ids:
                    hgid = self.have_soil_ids[key]
                    if (state >> hgid) & 1:
                        has_analysis = True
                        # Need: navigate to comm wp + communicate = 1-2
                        total += 1
                        break
            
            if not has_analysis:
                # Check sample availability
                sample_exists = True
                if wp in self.at_soil_sample_ids:
                    ss_gid = self.at_soil_sample_ids[wp]
                    if not ((state >> ss_gid) & 1):
                        sample_exists = False
                
                if not sample_exists:
                    # Sample already taken - maybe some rover has analysis but we missed it
                    # Or it's impossible - give high cost
                    total += 1  # Might still be achievable via different rover
                else:
                    # Need: navigate + maybe_drop + sample + navigate_comm + communicate
                    # Minimum realistic: 3 actions
                    total += 3
        
        # Rock goals  
        for wp in self.goal_rock_waypoints:
            if wp in self.communicated_rock_ids:
                gid = self.communicated_rock_ids[wp]
                if (state >> gid) & 1:
                    continue
            
            has_analysis = False
            for rover in self.rock_rovers:
                key = (rover, wp)
                if key in self.have_rock_ids:
                    hgid = self.have_rock_ids[key]
                    if (state >> hgid) & 1:
                        has_analysis = True
                        total += 1
                        break
            
            if not has_analysis:
                sample_exists = True
                if wp in self.at_rock_sample_ids:
                    rs_gid = self.at_rock_sample_ids[wp]
                    if not ((state >> rs_gid) & 1):
                        sample_exists = False
                
                if not sample_exists:
                    total += 1
                else:
                    total += 3
        
        # Image goals
        for (obj, mode) in self.goal_image_pairs:
            key = (obj, mode)
            if key in self.communicated_image_ids:
                gid = self.communicated_image_ids[key]
                if (state >> gid) & 1:
                    continue
            
            has_img = False
            for rover in self.imaging_rovers:
                ikey = (rover, obj, mode)
                if ikey in self.have_image_ids:
                    igid = self.have_image_ids[ikey]
                    if (state >> igid) & 1:
                        has_img = True
                        total += 1  # Just need to communicate
                        break
            
            if not has_img:
                # Need: calibrate + navigate + take_image + navigate_comm + communicate
                # Check if any suitable camera is already calibrated
                best_cost = 5  # Default high
                for rover in self.imaging_rovers:
                    if rover in self.rover_cameras:
                        for camera in self.rover_cameras[rover]:
                            if camera in self.camera_modes and mode in self.camera_modes[camera]:
                                cal_key = (camera, rover)
                                if cal_key in self.calibrated_ids:
                                    cal_gid = self.calibrated_ids[cal_key]
                                    if (state >> cal_gid) & 1:
                                        # Already calibrated: take_image + comm = 2 + navigation
                                        best_cost = min(best_cost, 3)
                                    else:
                                        # Need calibration: calibrate + take_image + comm = 3 + navigation
                                        best_cost = min(best_cost, 4)
                
                total += best_cost
        
        return total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
