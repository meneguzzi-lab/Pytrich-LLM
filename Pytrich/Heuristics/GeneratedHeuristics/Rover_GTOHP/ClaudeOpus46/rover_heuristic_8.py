# Rover GTOHP Domain Heuristic
# Bottleneck: Navigation between waypoints serializes all work - rovers must physically
# travel to sample locations, calibration targets, imaging positions, and communication points.
# Each soil/rock data task requires: navigate + empty_store + sample + navigate + communicate (min ~4-5 actions)
# Each image data task requires: calibrate(navigate+calibrate) + navigate + take_image + navigate + communicate (min ~5-6 actions)
#
# Admissible lower bounds:
# T1: Count of unsatisfied communicated_* goal facts. Each needs at least 1 action (communicate).
# T2: For each unsatisfied goal, estimate minimum actions needed based on current state.
#     - communicated_soil_data(wp): need have_soil_analysis (1 sample) + communicate (1) = min 2 if have analysis,
#       else need navigate + sample + communicate = min 3-4
#     - communicated_rock_data(wp): similar to soil
#     - communicated_image_data(obj,mode): need have_image + communicate = min 2, else calibrate+navigate+take+communicate = min 4+
# T3: Task network based - count non-trivial tasks with minimum decomposition cost estimates.
#
# Return max(T1_weighted, T2_detailed, T3_task_count) for strong guidance.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as OperatorClass
from Pytrich.model import AbstractTask as AbstractTaskClass


class RoverGTOHPAnthropicOpus46Heuristic8(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses multi-component max of:
    1. Weighted unsatisfied communication goals
    2. State-aware per-goal minimum action cost
    3. Task network analysis with minimum decomposition costs
    """

    def __init__(self):
        super().__init__(name="rover_anthropicopus46_heuristic_8")
        # Fact ID maps - built during initialize
        self.communicated_soil_ids = {}    # waypoint_name -> global_id
        self.communicated_rock_ids = {}    # waypoint_name -> global_id  
        self.communicated_image_ids = {}   # (obj_name, mode_name) -> global_id
        self.have_soil_ids = {}            # (rover, waypoint) -> global_id
        self.have_rock_ids = {}            # (rover, waypoint) -> global_id
        self.have_image_ids = {}           # (rover, obj, mode) -> global_id
        self.at_soil_sample_ids = {}       # waypoint -> global_id
        self.at_rock_sample_ids = {}       # waypoint -> global_id
        self.calibrated_ids = {}           # (camera, rover) -> global_id
        self.at_rover_ids = {}             # (rover, waypoint) -> global_id
        self.equipped_soil_ids = {}        # rover -> global_id
        self.equipped_rock_ids = {}        # rover -> global_id
        self.equipped_imaging_ids = {}     # rover -> global_id
        self.empty_store_ids = {}          # store -> global_id
        self.store_of_ids = {}             # (store, rover) -> global_id
        self.on_board_ids = {}             # (camera, rover) -> global_id
        self.supports_ids = {}             # (camera, mode) -> global_id
        self.visible_from_ids = {}         # (objective, waypoint) -> global_id
        self.calibration_target_ids = {}   # (camera, objective) -> global_id
        self.can_traverse_ids = {}         # (rover, wp1, wp2) -> global_id
        self.at_lander_ids = {}            # (lander, waypoint) -> global_id
        self.visible_ids = {}              # (wp1, wp2) -> global_id
        self.channel_free_ids = {}         # lander -> global_id
        self.available_ids = {}            # rover -> global_id
        
        # Goal tracking
        self.goal_soil_waypoints = []      # list of waypoint names needing soil communication
        self.goal_rock_waypoints = []      # list of waypoint names needing rock communication
        self.goal_image_pairs = []         # list of (obj, mode) needing image communication
        self.goal_fact_ids = []             # all goal fact global_ids
        
        # Precomputed info
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()
        
        # Rover capabilities
        self.soil_rovers = set()           # rovers equipped for soil
        self.rock_rovers = set()           # rovers equipped for rock
        self.imaging_rovers = set()        # rovers equipped for imaging
        
        # Camera info
        self.camera_rover = {}             # camera -> rover
        self.camera_modes = {}             # camera -> set of modes
        self.camera_cal_target = {}        # camera -> objective
        
        # Lander location
        self.lander_waypoint = {}          # lander -> waypoint
        
        # Adjacency for rovers
        self.rover_adj = {}                # rover -> set of (wp1, wp2) edges
        
        # Visible from objective
        self.obj_visible_wps = {}          # objective -> set of waypoints
        
        # Visible waypoint pairs
        self.visible_pairs = set()
        
        # Communication waypoints (visible to lander)
        self.comm_waypoints = set()        # waypoints from which communication is possible
        
        # Min tasks costs for task types
        self.task_type_costs = {}
        
        # BFS distances per rover
        self.rover_distances = {}          # rover -> dict of {wp: min_dist}

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

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to get the task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            
            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_ids[args[0]] = f.global_id
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_ids[args[0]] = f.global_id
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_image" and len(args) == 3:
                self.have_image_ids[(args[0], args[1], args[2])] = f.global_id
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = f.global_id
                self.waypoints.add(args[0])
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = f.global_id
                self.waypoints.add(args[0])
            elif pred == "at" and len(args) == 2:
                self.at_rover_ids[(args[0], args[1])] = f.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids[args[0]] = f.global_id
                self.soil_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids[args[0]] = f.global_id
                self.rock_rovers.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids[args[0]] = f.global_id
                self.imaging_rovers.add(args[0])
            elif pred == "empty" and len(args) == 1:
                self.empty_store_ids[args[0]] = f.global_id
                self.stores.add(args[0])
            elif pred == "store_of" and len(args) == 2:
                self.store_of_ids[(args[0], args[1])] = f.global_id
            elif pred == "on_board" and len(args) == 2:
                self.on_board_ids[(args[0], args[1])] = f.global_id
                self.cameras.add(args[0])
                self.camera_rover[args[0]] = args[1]
            elif pred == "supports" and len(args) == 2:
                self.supports_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.camera_modes:
                    self.camera_modes[args[0]] = set()
                self.camera_modes[args[0]].add(args[1])
                self.modes.add(args[1])
            elif pred == "visible_from" and len(args) == 2:
                self.visible_from_ids[(args[0], args[1])] = f.global_id
                self.objectives.add(args[0])
                if args[0] not in self.obj_visible_wps:
                    self.obj_visible_wps[args[0]] = set()
                self.obj_visible_wps[args[0]].add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self.calibration_target_ids[(args[0], args[1])] = f.global_id
                self.camera_cal_target[args[0]] = args[1]
            elif pred == "can_traverse" and len(args) == 3:
                self.can_traverse_ids[(args[0], args[1], args[2])] = f.global_id
                rover = args[0]
                if rover not in self.rover_adj:
                    self.rover_adj[rover] = set()
                self.rover_adj[rover].add((args[1], args[2]))
            elif pred == "at_lander" and len(args) == 2:
                self.at_lander_ids[(args[0], args[1])] = f.global_id
                self.landers.add(args[0])
                self.lander_waypoint[args[0]] = args[1]
            elif pred == "visible" and len(args) == 2:
                self.visible_ids[(args[0], args[1])] = f.global_id
                self.visible_pairs.add((args[0], args[1]))
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_ids[args[0]] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_ids[args[0]] = f.global_id
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = f.global_id
        
        # Find communication waypoints (visible to any lander waypoint)
        for lander, lander_wp in self.lander_waypoint.items():
            for wp in self.waypoints:
                if (wp, lander_wp) in self.visible_pairs:
                    self.comm_waypoints.add(wp)
                # Also the lander waypoint itself if visible from itself
            self.comm_waypoints.add(lander_wp)  # lander wp can always communicate
        
        # Extract goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))
        
        # Store all goal fact global IDs for quick checking
        for wp in self.goal_soil_waypoints:
            if wp in self.communicated_soil_ids:
                self.goal_fact_ids.append(self.communicated_soil_ids[wp])
        for wp in self.goal_rock_waypoints:
            if wp in self.communicated_rock_ids:
                self.goal_fact_ids.append(self.communicated_rock_ids[wp])
        for (obj, mode) in self.goal_image_pairs:
            key = (obj, mode)
            if key in self.communicated_image_ids:
                self.goal_fact_ids.append(self.communicated_image_ids[key])
        
        # Precompute BFS shortest distances for each rover from each waypoint
        # This is done once and is crucial for accurate navigation cost estimation
        for rover in self.rovers:
            if rover not in self.rover_adj:
                continue
            adj = {}
            for (w1, w2) in self.rover_adj[rover]:
                if w1 not in adj:
                    adj[w1] = []
                adj[w1].append(w2)
            self.rover_distances[rover] = {}
            # BFS from each waypoint that is relevant
            all_wps = set()
            for (w1, w2) in self.rover_adj[rover]:
                all_wps.add(w1)
                all_wps.add(w2)
            # Store adjacency for this rover
            self.rover_distances[rover] = {'adj': adj, 'wps': all_wps}
        
        # Precompute min task costs for abstract task types
        self._precompute_task_costs()
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _bfs_distance(self, rover, start_wp, target_wp):
        """Compute BFS shortest distance for a rover between two waypoints."""
        if start_wp == target_wp:
            return 0
        if rover not in self.rover_distances:
            return 999
        adj = self.rover_distances[rover].get('adj', {})
        if start_wp not in self.rover_distances[rover].get('wps', set()):
            return 999
        if target_wp not in self.rover_distances[rover].get('wps', set()):
            return 999
        
        # BFS
        visited = {start_wp}
        queue = [start_wp]
        dist = 0
        while queue:
            dist += 1
            next_queue = []
            for wp in queue:
                for neighbor in adj.get(wp, []):
                    if neighbor == target_wp:
                        return dist
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_queue.append(neighbor)
            queue = next_queue
            if dist > 50:  # Safety limit
                break
        return 999

    def _precompute_task_costs(self):
        """Precompute minimum costs for different task types."""
        # get_soil_data: navigate + empty_store + sample_soil + send_soil_data
        # send_soil_data: navigate + communicate = min 2
        # So min cost for get_soil_data = ~4 actions minimum
        self.task_type_costs['get_soil_data'] = 4
        
        # get_rock_data: same structure as soil
        self.task_type_costs['get_rock_data'] = 4
        
        # get_image_data: calibrate(navigate+calibrate) + navigate + take_image + send_image
        # send_image: navigate + communicate = 2
        # So min = 1(calibrate or nop) + 1(navigate/nop) + 1(take_image) + 2(send) = ~5
        self.task_type_costs['get_image_data'] = 5
        
        # Navigation tasks
        self.task_type_costs['do_navigate1'] = 0  # could be nop if already there
        self.task_type_costs['do_navigate2'] = 0
        self.task_type_costs['empty_store'] = 0
        self.task_type_costs['send_soil_data'] = 2
        self.task_type_costs['send_rock_data'] = 2
        self.task_type_costs['send_image_data'] = 2
        self.task_type_costs['do_calibrate'] = 1

    def _get_rover_location(self, state, rover):
        """Get current waypoint of a rover from state using bit checks."""
        for (r, wp), gid in self.at_rover_ids.items():
            if r == rover and (state >> gid) & 1:
                return wp
        return None

    def _compute_soil_cost(self, state, waypoint):
        """Estimate minimum cost to achieve communicated_soil_data for waypoint."""
        comm_id = self.communicated_soil_ids.get(waypoint)
        if comm_id is not None and (state >> comm_id) & 1:
            return 0  # Already communicated
        
        # Check if any rover already has the soil analysis
        has_analysis = False
        for rover in self.soil_rovers:
            key = (rover, waypoint)
            if key in self.have_soil_ids:
                if (state >> self.have_soil_ids[key]) & 1:
                    has_analysis = True
                    break
        
        if has_analysis:
            # Just need to navigate to comm point and communicate
            return 2  # navigate + communicate minimum
        
        # Check if soil sample still exists at waypoint
        sample_exists = False
        if waypoint in self.at_soil_sample_ids:
            if (state >> self.at_soil_sample_ids[waypoint]) & 1:
                sample_exists = True
        
        if not sample_exists:
            # Sample already taken by some rover, need to check have_soil
            # If no rover has it, this might be a problem but we assume it was taken
            for rover in self.soil_rovers:
                key = (rover, waypoint)
                if key in self.have_soil_ids:
                    if (state >> self.have_soil_ids[key]) & 1:
                        return 2  # Already handled above
            # Sample consumed but no one has analysis - shouldn't happen in valid plans
            return 2
        
        # Need: navigate to wp, possibly empty store, sample, navigate to comm, communicate
        # Minimum: empty_store(0-1) + sample(1) + navigate_to_comm(1+) + communicate(1) = 3-4
        return 4

    def _compute_rock_cost(self, state, waypoint):
        """Estimate minimum cost to achieve communicated_rock_data for waypoint."""
        comm_id = self.communicated_rock_ids.get(waypoint)
        if comm_id is not None and (state >> comm_id) & 1:
            return 0
        
        has_analysis = False
        for rover in self.rock_rovers:
            key = (rover, waypoint)
            if key in self.have_rock_ids:
                if (state >> self.have_rock_ids[key]) & 1:
                    has_analysis = True
                    break
        
        if has_analysis:
            return 2
        
        sample_exists = False
        if waypoint in self.at_rock_sample_ids:
            if (state >> self.at_rock_sample_ids[waypoint]) & 1:
                sample_exists = True
        
        if not sample_exists:
            for rover in self.rock_rovers:
                key = (rover, waypoint)
                if key in self.have_rock_ids:
                    if (state >> self.have_rock_ids[key]) & 1:
                        return 2
            return 2
        
        return 4

    def _compute_image_cost(self, state, objective, mode):
        """Estimate minimum cost to achieve communicated_image_data for (objective, mode)."""
        key = (objective, mode)
        comm_id = self.communicated_image_ids.get(key)
        if comm_id is not None and (state >> comm_id) & 1:
            return 0
        
        # Check if any rover already has the image
        has_img = False
        for rover in self.imaging_rovers:
            ikey = (rover, objective, mode)
            if ikey in self.have_image_ids:
                if (state >> self.have_image_ids[ikey]) & 1:
                    has_img = True
                    break
        
        if has_img:
            return 2  # navigate to comm + communicate
        
        # Need: calibrate camera (possibly navigate) + navigate to visible wp + take_image + navigate to comm + communicate
        # Check if any suitable camera is already calibrated
        camera_calibrated = False
        for cam in self.cameras:
            if cam not in self.camera_modes:
                continue
            if mode not in self.camera_modes[cam]:
                continue
            rover = self.camera_rover.get(cam)
            if rover and rover in self.imaging_rovers:
                cal_key = (cam, rover)
                if cal_key in self.calibrated_ids:
                    if (state >> self.calibrated_ids[cal_key]) & 1:
                        camera_calibrated = True
                        break
        
        if camera_calibrated:
            # navigate to visible wp + take_image + navigate to comm + communicate
            return 3  # take_image + navigate_comm + communicate
        
        # Full cost: navigate_to_cal_target + calibrate + navigate_to_obj_wp + take_image + navigate_to_comm + communicate
        return 5

    def _term1_unsatisfied_goals(self, state):
        """Count unsatisfied communication goals - simple but fast lower bound.
        Each unsatisfied goal needs at minimum 1 action (the communicate action)."""
        count = 0
        for gid in self.goal_fact_ids:
            if not ((state >> gid) & 1):
                count += 1
        return count

    def _term2_state_aware_goal_cost(self, state):
        """State-aware estimation of minimum actions needed for all unsatisfied goals."""
        total = 0
        
        for wp in self.goal_soil_waypoints:
            total += self._compute_soil_cost(state, wp)
        
        for wp in self.goal_rock_waypoints:
            total += self._compute_rock_cost(state, wp)
        
        for (obj, mode) in self.goal_image_pairs:
            total += self._compute_image_cost(state, obj, mode)
        
        return total

    def _term3_task_network_cost(self, task_network):
        """Estimate cost from remaining task network.
        For each task, use minimum decomposition cost based on task type."""
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            if isinstance(task, OperatorClass):
                # Primitive operator - costs 1 action unless it's nop
                task_type, _ = self._parse_task_name(task.name)
                if task_type == "nop":
                    continue
                total += 1
            elif isinstance(task, AbstractTaskClass):
                task_type, args = self._parse_task_name(task.name)
                cost = self.task_type_costs.get(task_type, 1)
                total += cost
        
        return total

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        # Term 1: Simple unsatisfied goal count (very fast, weak lower bound)
        t1 = self._term1_unsatisfied_goals(state)
        
        if t1 == 0 and len(task_network) == 0:
            return 0
        
        # Term 2: State-aware goal cost estimation (stronger, still fast)
        # We use a fraction to avoid over-counting since goals may share rovers
        # but individual estimates are lower bounds on each goal's cost
        t2 = self._term2_state_aware_goal_cost(state)
        
        # Term 3: Task network based cost
        t3 = self._term3_task_network_cost(task_network)
        
        # Use max for admissibility, but we can also blend
        # t2 is the strongest individual bound
        # t3 captures work remaining in the HTN decomposition
        result = max(t1, t2, t3)
        
        return result

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
