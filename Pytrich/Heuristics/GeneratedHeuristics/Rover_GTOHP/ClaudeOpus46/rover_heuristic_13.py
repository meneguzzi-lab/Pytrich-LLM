# Rover_GTOHP Domain Heuristic
# Bottleneck: Navigation is the primary serializer - rovers must travel to sample locations,
# then to communication points visible to the lander. Each data task requires multiple nav steps.
#
# Admissible lower bounds:
# T1 - Uncommunicated goals: each uncommunicated_X_data goal needs at minimum: navigate + sample/image + navigate + communicate = 4 actions for soil/rock, ~5-6 for images
# T2 - Task network estimation: count remaining abstract tasks with state-aware cost estimation
# T3 - Store contention: multiple soil/rock tasks on same rover require drop actions

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic13(Heuristic):
    """Domain-dependent heuristic for Rover_GTOHP.
    
    Uses multi-component max of:
    1. Weighted uncommunicated goal count (state-aware)
    2. Task network cost estimation with per-task-type weights
    3. Navigation/resource lower bound
    """

    def __init__(self, name="rover_anthropicopus46_heuristic_13"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}   # waypoint -> global_id
        self.communicated_rock_ids = {}   # waypoint -> global_id
        self.communicated_image_ids = {}  # (objective, mode) -> global_id
        self.at_soil_sample_ids = {}      # waypoint -> global_id
        self.at_rock_sample_ids = {}      # waypoint -> global_id
        self.have_soil_ids = {}           # (rover, waypoint) -> global_id
        self.have_rock_ids = {}           # (rover, waypoint) -> global_id
        self.have_image_ids = {}          # (rover, objective, mode) -> global_id
        self.at_ids = {}                  # (rover, waypoint) -> global_id
        self.empty_ids = {}              # store -> global_id
        self.full_ids = {}               # store -> global_id
        self.calibrated_ids = {}         # (camera, rover) -> global_id
        self.at_lander_ids = {}          # (lander, waypoint) -> global_id
        self.visible_ids = set()          # set of (wp1, wp2)
        self.store_of_ids = {}           # (store, rover) -> global_id
        self.equipped_soil_ids = {}      # rover -> global_id
        self.equipped_rock_ids = {}      # rover -> global_id
        self.equipped_imaging_ids = {}   # rover -> global_id
        self.can_traverse_set = {}       # rover -> set of (wp1, wp2)
        self.visible_from_ids = {}       # (objective, waypoint) -> global_id
        self.on_board_ids = {}           # (camera, rover) -> global_id
        self.supports_ids = {}           # (camera, mode) -> global_id
        self.calibration_target_ids = {} # (camera, objective) -> global_id
        self.channel_free_ids = {}       # lander -> global_id
        
        # Goal structures
        self.goal_soil_waypoints = []    # list of waypoints needing soil data communicated
        self.goal_rock_waypoints = []    # list of waypoints needing rock data communicated
        self.goal_image_pairs = []       # list of (objective, mode) needing image data communicated
        
        # Task type name patterns for identification
        self.task_type_costs = {}  # task_name_prefix -> estimated min cost
        
        # Lander waypoint for distance estimation
        self.lander_waypoint = None
        
        # Rover info
        self.rovers = set()
        self.waypoints = set()
        
        # BFS distance cache per rover (computed lazily or in init)
        self.rover_distances = {}  # rover -> {wp -> {wp -> dist}}
        
        # Visible-to-lander waypoints
        self.lander_visible_wps = set()

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
        """Parse a grounded task/operator name to get the base task and arguments."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _compute_rover_bfs(self, rover):
        """Compute BFS shortest distances for a rover's traversal graph."""
        if rover not in self.can_traverse_set:
            return {}
        edges = self.can_traverse_set[rover]
        # Build adjacency list
        adj = {}
        for (w1, w2) in edges:
            if w1 not in adj:
                adj[w1] = []
            adj[w1].append(w2)
        
        # BFS from each waypoint the rover could reach
        all_wps = set()
        for (w1, w2) in edges:
            all_wps.add(w1)
            all_wps.add(w2)
        
        distances = {}
        for start in all_wps:
            dist = {start: 0}
            queue = [start]
            qi = 0
            while qi < len(queue):
                current = queue[qi]
                qi += 1
                if current in adj:
                    for neighbor in adj[current]:
                        if neighbor not in dist:
                            dist[neighbor] = dist[current] + 1
                            queue.append(neighbor)
            distances[start] = dist
        return distances

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            
            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_ids[args[0]] = fact.global_id
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_ids[args[0]] = fact.global_id
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_ids[(args[0], args[1])] = fact.global_id
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = fact.global_id
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = fact.global_id
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_image" and len(args) == 3:
                self.have_image_ids[(args[0], args[1], args[2])] = fact.global_id
            elif pred == "at" and len(args) == 2:
                self.at_ids[(args[0], args[1])] = fact.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = fact.global_id
            elif pred == "full" and len(args) == 1:
                self.full_ids[args[0]] = fact.global_id
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = fact.global_id
            elif pred == "at_lander" and len(args) == 2:
                self.at_lander_ids[(args[0], args[1])] = fact.global_id
                self.lander_waypoint = args[1]
            elif pred == "visible" and len(args) == 2:
                self.visible_ids.add((args[0], args[1]))
            elif pred == "store_of" and len(args) == 2:
                self.store_of_ids[(args[0], args[1])] = fact.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids[args[0]] = fact.global_id
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids[args[0]] = fact.global_id
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids[args[0]] = fact.global_id
            elif pred == "can_traverse" and len(args) == 3:
                rover = args[0]
                if rover not in self.can_traverse_set:
                    self.can_traverse_set[rover] = set()
                self.can_traverse_set[rover].add((args[1], args[2]))
            elif pred == "visible_from" and len(args) == 2:
                self.visible_from_ids[(args[0], args[1])] = fact.global_id
            elif pred == "on_board" and len(args) == 2:
                self.on_board_ids[(args[0], args[1])] = fact.global_id
            elif pred == "supports" and len(args) == 2:
                self.supports_ids[(args[0], args[1])] = fact.global_id
            elif pred == "calibration_target" and len(args) == 2:
                self.calibration_target_ids[(args[0], args[1])] = fact.global_id
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_ids[args[0]] = fact.global_id
        
        # Identify lander-visible waypoints
        if self.lander_waypoint:
            for (w1, w2) in self.visible_ids:
                if w2 == self.lander_waypoint:
                    self.lander_visible_wps.add(w1)
                if w1 == self.lander_waypoint:
                    self.lander_visible_wps.add(w2)
            # The lander waypoint itself is always "visible" for communicate actions
            self.lander_visible_wps.add(self.lander_waypoint)
        
        # Parse goals to identify what needs to be communicated
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))
        
        # Compute BFS distances for each rover (limit to rovers that appear in goals' capable set)
        # Only compute for rovers with traversal edges
        for rover in self.rovers:
            if rover in self.can_traverse_set:
                self.rover_distances[rover] = self._compute_rover_bfs(rover)
        
        # Precompute which rovers can do soil, rock, imaging
        self.soil_rovers = set(self.equipped_soil_ids.keys())
        self.rock_rovers = set(self.equipped_rock_ids.keys())
        self.imaging_rovers = set(self.equipped_imaging_ids.keys())
        
        # Precompute store for each rover
        self.rover_store = {}
        for (store, rover), fid in self.store_of_ids.items():
            self.rover_store[rover] = store
        
        # Build abstract task name mapping for quick lookup
        self.abstract_task_types = {}
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.abstract_task_types[at.global_id] = (ttype, targs)
        
        # Also map operators
        self.operator_types = {}
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            self.operator_types[op.global_id] = (ttype, targs)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _min_nav_cost_to_wp(self, rover, target_wp, state):
        """Estimate minimum navigation cost for rover to reach target_wp from current position."""
        if rover not in self.rover_distances:
            return 1  # Can't compute, return conservative estimate
        
        # Find rover's current position from state
        current_wp = None
        for (r, wp), fid in self.at_ids.items():
            if r == rover:
                if (state >> fid) & 1:
                    current_wp = wp
                    break
        
        if current_wp is None:
            return 1
        if current_wp == target_wp:
            return 0
        
        dist_map = self.rover_distances.get(rover, {})
        if current_wp in dist_map:
            d = dist_map[current_wp].get(target_wp, None)
            if d is not None:
                return d
        return 2  # Conservative fallback

    def _min_dist_between(self, rover, wp1, wp2):
        """Get precomputed distance between two waypoints for a rover."""
        if rover not in self.rover_distances:
            return 1
        dist_map = self.rover_distances.get(rover, {})
        if wp1 in dist_map:
            d = dist_map[wp1].get(wp2, None)
            if d is not None:
                return d
        return 2

    def _compute_goal_term(self, state):
        """Term 1: Weighted count of uncommunicated goals with state-aware cost estimation.
        
        Each uncommunicated soil/rock goal needs at minimum:
        - 1 action to sample (if sample still exists at waypoint)
        - 1 action to communicate
        - Navigation to sample point + navigation to comm point
        
        Each uncommunicated image goal needs at minimum:
        - 1 calibrate + 1 take_image + 1 communicate
        - Navigation to calibration point + image point + comm point
        """
        h = 0
        
        # Soil goals
        for wp in self.goal_soil_waypoints:
            if wp in self.communicated_soil_ids:
                fid = self.communicated_soil_ids[wp]
                if (state >> fid) & 1:
                    continue  # Already communicated
            
            # Check if any rover already has the soil analysis
            has_analysis = False
            for (r, w), fid in self.have_soil_ids.items():
                if w == wp and (state >> fid) & 1:
                    has_analysis = True
                    # Still need to communicate: navigate to comm point + communicate
                    h += 2  # At minimum: navigate + communicate
                    break
            
            if not has_analysis:
                # Check if soil sample still exists
                sample_exists = False
                if wp in self.at_soil_sample_ids:
                    sfid = self.at_soil_sample_ids[wp]
                    if (state >> sfid) & 1:
                        sample_exists = True
                
                if sample_exists:
                    # Need: navigate to wp, empty store, sample, navigate to comm, communicate
                    # Minimum: sample(1) + communicate(1) + at least navigate(1) to comm point = 3
                    # Plus possible store emptying
                    h += 4
                else:
                    # Sample already taken by someone but not communicated yet
                    # Either someone has it (checked above) or it's consumed but analysis not done
                    # This shouldn't happen in well-formed problems, but be safe
                    h += 3
        
        # Rock goals
        for wp in self.goal_rock_waypoints:
            if wp in self.communicated_rock_ids:
                fid = self.communicated_rock_ids[wp]
                if (state >> fid) & 1:
                    continue
            
            has_analysis = False
            for (r, w), fid in self.have_rock_ids.items():
                if w == wp and (state >> fid) & 1:
                    has_analysis = True
                    h += 2
                    break
            
            if not has_analysis:
                sample_exists = False
                if wp in self.at_rock_sample_ids:
                    sfid = self.at_rock_sample_ids[wp]
                    if (state >> sfid) & 1:
                        sample_exists = True
                
                if sample_exists:
                    h += 4
                else:
                    h += 3
        
        # Image goals
        for (obj, mode) in self.goal_image_pairs:
            if (obj, mode) in self.communicated_image_ids:
                fid = self.communicated_image_ids[(obj, mode)]
                if (state >> fid) & 1:
                    continue
            
            has_img = False
            for (r, o, m), fid in self.have_image_ids.items():
                if o == obj and m == mode and (state >> fid) & 1:
                    has_img = True
                    h += 2  # navigate to comm + communicate
                    break
            
            if not has_img:
                # Need: calibrate(1) + navigate to obj visible point + take_image(1) + navigate to comm + communicate(1)
                # Check if any suitable camera is already calibrated
                needs_calibration = True
                for (cam, rov), fid in self.calibrated_ids.items():
                    if (state >> fid) & 1:
                        # Check if this camera supports the mode and rover has imaging
                        if (cam, mode) in self.supports_ids and rov in self.imaging_rovers:
                            needs_calibration = False
                            break
                
                if needs_calibration:
                    h += 5  # calibrate + nav + take_image + nav + communicate
                else:
                    h += 4  # nav + take_image + nav + communicate
        
        return h

    def _compute_task_network_term(self, task_network):
        """Term 2: Estimate minimum actions from remaining task network.
        
        Each task type has a minimum decomposition cost:
        - get_soil_data: ~4 actions minimum (navigate + empty_store/nop + sample + send_soil)
        - get_rock_data: ~4 actions minimum
        - get_image_data: ~5 actions minimum (calibrate + navigate + take_image + send_image)
        - send_X_data: ~2 actions minimum (navigate + communicate)
        - do_navigate1/2: ~1 action minimum
        - empty_store: ~0-1 actions
        - do_calibrate: ~2 actions minimum (navigate + calibrate)
        - primitive actions: 1 each (except nop = 0)
        """
        h = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an abstract task
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                if gid in self.abstract_task_types:
                    ttype, targs = self.abstract_task_types[gid]
                    if ttype == "get_soil_data":
                        h += 4
                    elif ttype == "get_rock_data":
                        h += 4
                    elif ttype == "get_image_data":
                        h += 5
                    elif ttype == "send_soil_data":
                        h += 2
                    elif ttype == "send_rock_data":
                        h += 2
                    elif ttype == "send_image_data":
                        h += 2
                    elif ttype == "do_navigate1":
                        h += 1
                    elif ttype == "do_navigate2":
                        h += 1
                    elif ttype == "do_calibrate":
                        h += 2
                    elif ttype == "empty_store":
                        h += 0  # might be nop
                    else:
                        h += 1
                else:
                    h += 1
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive operator
                if gid in self.operator_types:
                    ttype, _ = self.operator_types[gid]
                    if ttype == "nop":
                        h += 0
                    else:
                        h += 1
                else:
                    h += 1
        
        return h

    def _compute_resource_term(self, state, task_network):
        """Term 3: Resource contention lower bound.
        
        Count how many soil+rock sampling tasks remain in the task network.
        Each pair sharing a rover's store requires at least one drop action.
        Also consider navigation distances.
        """
        # Count remaining sampling tasks by type
        soil_tasks = 0
        rock_tasks = 0
        image_tasks = 0
        
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                if gid in self.abstract_task_types:
                    ttype, targs = self.abstract_task_types[gid]
                    if ttype == "get_soil_data":
                        soil_tasks += 1
                    elif ttype == "get_rock_data":
                        rock_tasks += 1
                    elif ttype == "get_image_data":
                        image_tasks += 1
        
        # Each soil/rock task after the first on a rover needs a drop action
        # In the worst case (single rover), we need (soil_tasks + rock_tasks - 1) extra drops
        # But with multiple rovers, it's distributed
        total_sample_tasks = soil_tasks + rock_tasks
        num_capable_rovers = max(1, min(len(self.soil_rovers | self.rock_rovers), total_sample_tasks))
        
        # Lower bound on extra drop actions needed
        if total_sample_tasks > num_capable_rovers:
            extra_drops = total_sample_tasks - num_capable_rovers
        else:
            extra_drops = 0
        
        # Each image task needs calibration, but cameras might stay calibrated
        # Lower bound: at least 1 action per uncalibrated camera use
        
        # Communication bottleneck: all data must go through lander
        # channel_free constraint means communications are serialized
        # Each communication is at least 1 action
        total_comms_needed = 0
        for wp in self.goal_soil_waypoints:
            if wp in self.communicated_soil_ids:
                fid = self.communicated_soil_ids[wp]
                if not ((state >> fid) & 1):
                    total_comms_needed += 1
        for wp in self.goal_rock_waypoints:
            if wp in self.communicated_rock_ids:
                fid = self.communicated_rock_ids[wp]
                if not ((state >> fid) & 1):
                    total_comms_needed += 1
        for (obj, mode) in self.goal_image_pairs:
            if (obj, mode) in self.communicated_image_ids:
                fid = self.communicated_image_ids[(obj, mode)]
                if not ((state >> fid) & 1):
                    total_comms_needed += 1
        
        return extra_drops + total_comms_needed

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible terms."""
        if self.model.goal_reached(state, task_network):
            return 0
        
        t1 = self._compute_goal_term(state)
        t2 = self._compute_task_network_term(task_network)
        t3 = self._compute_resource_term(state, task_network)
        
        return max(t1, t2, t3)
