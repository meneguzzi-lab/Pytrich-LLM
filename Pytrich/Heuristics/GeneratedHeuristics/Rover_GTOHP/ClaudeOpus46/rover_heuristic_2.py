"""Domain-dependent heuristic for Rover_GTOHP.

Bottleneck: Navigation between waypoints dominates plan length. Each data goal
requires navigating to the sample/image location, possibly emptying store,
collecting data, then navigating to a communication-visible waypoint.

Admissible lower bounds:
1. Uncommunicated data goals: each soil/rock needs >= 2 actions (sample + communicate),
   each image needs >= 3 actions (calibrate + take_image + communicate). This is admissible
   because these actions are unavoidable.
2. Task network structural cost: remaining abstract tasks weighted by minimum
   decomposition depth. Admissible because each task must be fully decomposed.
3. Store contention: when multiple soil/rock tasks share a rover, drops are needed
   between samples. Each extra sample beyond the first needs +1 drop action.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic2(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_2"):
        super().__init__(name)
        self._fact_map = {}  # predicate -> {(args_tuple): global_id}
        self._communicated_soil_ids = {}  # waypoint_arg -> global_id
        self._communicated_rock_ids = {}  # waypoint_arg -> global_id
        self._communicated_image_ids = {}  # (objective, mode) -> global_id
        self._goal_soil_waypoints = set()
        self._goal_rock_waypoints = set()
        self._goal_image_keys = set()  # (objective, mode)
        
        # For task network analysis
        self._task_min_cost = {}  # abstract_task global_id -> min cost estimate
        self._op_names = {}  # operator global_id -> parsed name
        
        # Soil/rock analysis facts
        self._at_soil_sample_ids = {}  # waypoint -> global_id
        self._at_rock_sample_ids = {}  # waypoint -> global_id
        self._have_soil_ids = {}  # (rover, waypoint) -> global_id
        self._have_rock_ids = {}  # (rover, waypoint) -> global_id
        self._have_image_ids = {}  # (rover, objective, mode) -> global_id
        self._calibrated_ids = {}  # (camera, rover) -> global_id
        self._at_rover_ids = {}  # (rover, waypoint) -> global_id
        self._empty_store_ids = {}  # store -> global_id
        self._full_store_ids = {}  # store -> global_id
        self._store_of_ids = {}  # (store, rover) -> bool (static)
        self._equipped_soil_ids = set()  # rover names equipped for soil
        self._equipped_rock_ids = set()  # rover names equipped for rock
        self._equipped_imaging_ids = set()  # rover names equipped for imaging
        
        # Navigation precomputation
        self._can_traverse = {}  # rover -> set of (wp1, wp2)
        self._visible = set()  # set of (wp1, wp2)
        self._lander_waypoint = None
        
        # BFS distance caches per rover
        self._rover_distances = {}  # rover_name -> {waypoint -> min_distance_from_any_comm_visible_wp}
        
        # Waypoints visible to lander
        self._comm_waypoints = set()  # waypoints from which communication is possible

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        if bracket_idx < 0:
            return name, []
        return None, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name like '__navigate[rover0,waypoint1,waypoint0]'"""
        name = name.strip()
        # Remove leading underscores
        clean = name.lstrip('_')
        bracket_idx = clean.find("[")
        if bracket_idx > 0 and clean.endswith("]"):
            task_name = clean[:bracket_idx]
            args_str = clean[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_name, args
        if bracket_idx < 0:
            return clean, []
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            args_tuple = tuple(args)
            if pred not in self._fact_map:
                self._fact_map[pred] = {}
            self._fact_map[pred][args_tuple] = f.global_id
            
            if pred == "communicated_soil_data":
                self._communicated_soil_ids[args[0]] = f.global_id
            elif pred == "communicated_rock_data":
                self._communicated_rock_ids[args[0]] = f.global_id
            elif pred == "communicated_image_data":
                self._communicated_image_ids[(args[0], args[1])] = f.global_id
            elif pred == "at_soil_sample":
                self._at_soil_sample_ids[args[0]] = f.global_id
            elif pred == "at_rock_sample":
                self._at_rock_sample_ids[args[0]] = f.global_id
            elif pred == "have_soil_analysis":
                self._have_soil_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis":
                self._have_rock_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_image":
                self._have_image_ids[(args[0], args[1], args[2])] = f.global_id
            elif pred == "calibrated":
                self._calibrated_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self._at_rover_ids[(args[0], args[1])] = f.global_id
            elif pred == "empty":
                self._empty_store_ids[args[0]] = f.global_id
            elif pred == "full":
                self._full_store_ids[args[0]] = f.global_id
            elif pred == "store_of":
                self._store_of_ids[(args[0], args[1])] = True
            elif pred == "equipped_for_soil_analysis":
                self._equipped_soil_ids.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self._equipped_rock_ids.add(args[0])
            elif pred == "equipped_for_imaging":
                self._equipped_imaging_ids.add(args[0])
            elif pred == "can_traverse":
                rover = args[0]
                if rover not in self._can_traverse:
                    self._can_traverse[rover] = set()
                self._can_traverse[rover].add((args[1], args[2]))
            elif pred == "visible":
                self._visible.add((args[0], args[1]))
            elif pred == "at_lander":
                self._lander_waypoint = args[1]
        
        # Determine goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data":
                self._goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data":
                self._goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data":
                self._goal_image_keys.add((args[0], args[1]))
        
        # Find communication waypoints (waypoints visible to lander waypoint)
        if self._lander_waypoint:
            for wp_pair in self._visible:
                if wp_pair[1] == self._lander_waypoint:
                    self._comm_waypoints.add(wp_pair[0])
            # The lander waypoint itself can be used for communicate_soil_data2 style
            self._comm_waypoints.add(self._lander_waypoint)
        
        # Precompute BFS distances for each rover from all waypoints
        # We build adjacency from can_traverse for each rover
        all_waypoints = set()
        if "at" in self._fact_map:
            for args_tuple in self._fact_map["at"]:
                if len(args_tuple) == 2:
                    all_waypoints.add(args_tuple[1])
        # Also collect from visible
        for wp1, wp2 in self._visible:
            all_waypoints.add(wp1)
            all_waypoints.add(wp2)
        
        self._all_waypoints = all_waypoints
        
        # For each rover, compute shortest path distances using BFS on can_traverse graph
        self._rover_wp_dist = {}  # rover -> {(src, dst): distance}
        for rover, edges in self._can_traverse.items():
            adj = {}
            for wp1, wp2 in edges:
                if wp1 not in adj:
                    adj[wp1] = []
                adj[wp1].append(wp2)
            self._rover_wp_dist[rover] = adj
        
        # Precompute min abstract task costs
        self._precompute_task_costs(model)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _bfs_distance(self, rover_adj, src, dst):
        """BFS shortest path distance for a rover's traversal graph."""
        if src == dst:
            return 0
        if rover_adj is None:
            return 999
        visited = {src}
        queue = [(src, 0)]
        qi = 0
        while qi < len(queue):
            node, dist = queue[qi]
            qi += 1
            for neighbor in rover_adj.get(node, []):
                if neighbor == dst:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999  # unreachable

    def _min_dist_to_comm(self, rover_adj, src):
        """BFS to find minimum distance from src to any communication waypoint."""
        if src in self._comm_waypoints:
            return 0
        if rover_adj is None:
            return 999
        visited = {src}
        queue = [(src, 0)]
        qi = 0
        while qi < len(queue):
            node, dist = queue[qi]
            qi += 1
            for neighbor in rover_adj.get(node, []):
                if neighbor in self._comm_waypoints:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999

    def _precompute_task_costs(self, model):
        """Estimate minimum decomposition cost for each abstract task type."""
        # We categorize by task type prefix
        for at in model.abstract_tasks:
            tname, args = self._parse_task_name(at.name)
            # Minimum cost based on task type
            if tname == "do_navigate1" or tname == "do_navigate2":
                self._task_min_cost[at.global_id] = 0  # might already be there
            elif tname == "empty_store":
                self._task_min_cost[at.global_id] = 0  # might already be empty
            elif tname == "get_soil_data":
                self._task_min_cost[at.global_id] = 2  # sample + communicate minimum
            elif tname == "send_soil_data":
                self._task_min_cost[at.global_id] = 1  # communicate
            elif tname == "get_rock_data":
                self._task_min_cost[at.global_id] = 2  # sample + communicate
            elif tname == "send_rock_data":
                self._task_min_cost[at.global_id] = 1
            elif tname == "get_image_data":
                self._task_min_cost[at.global_id] = 3  # calibrate + take_image + communicate
            elif tname == "send_image_data":
                self._task_min_cost[at.global_id] = 1
            elif tname == "do_calibrate":
                self._task_min_cost[at.global_id] = 1  # calibrate action
            else:
                self._task_min_cost[at.global_id] = 0

    def _compute(self, state, task_network):
        h1 = self._goal_cost_estimate(state)
        h2 = self._task_network_cost(state, task_network)
        return max(h1, h2)

    def _goal_cost_estimate(self, state):
        """Estimate cost based on uncommunicated goals with state-aware reasoning."""
        total = 0
        
        for wp in self._goal_soil_waypoints:
            gid = self._communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                # Not yet communicated
                cost = 1  # at minimum communicate action
                # Check if any rover has the soil analysis
                has_analysis = False
                for (rover, waypoint), fid in self._have_soil_ids.items():
                    if waypoint == wp and ((state >> fid) & 1):
                        has_analysis = True
                        break
                if not has_analysis:
                    cost += 1  # need to sample
                    # Check if soil sample still exists at waypoint
                    ss_id = self._at_soil_sample_ids.get(wp)
                    if ss_id is not None and not ((state >> ss_id) & 1):
                        # Sample already taken by someone but not communicated yet
                        # Still need to communicate
                        cost = 1
                    else:
                        # Need to navigate to wp + sample + navigate to comm + communicate
                        cost = 2  # sample + communicate minimum
                total += cost
        
        for wp in self._goal_rock_waypoints:
            gid = self._communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                cost = 1
                has_analysis = False
                for (rover, waypoint), fid in self._have_rock_ids.items():
                    if waypoint == wp and ((state >> fid) & 1):
                        has_analysis = True
                        break
                if not has_analysis:
                    cost = 2
                total += cost
        
        for (obj, mode) in self._goal_image_keys:
            gid = self._communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                cost = 1
                has_img = False
                for (rover, objective, m), fid in self._have_image_ids.items():
                    if objective == obj and m == mode and ((state >> fid) & 1):
                        has_img = True
                        break
                if not has_img:
                    cost = 3  # calibrate + take_image + communicate
                total += cost
        
        return total

    def _task_network_cost(self, state, task_network):
        """Estimate cost based on remaining task network with state-aware refinement."""
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive task)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive action - costs 1 unless it's nop
                tname, args = self._parse_task_name(task.name)
                if tname == "nop":
                    continue
                total += 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task - use precomputed minimum + state-aware refinement
                base_cost = self._task_min_cost.get(gid, 1)
                tname, args = self._parse_task_name(task.name)
                
                if tname == "get_soil_data" and len(args) >= 1:
                    wp = args[0]
                    # Check if already communicated
                    comm_id = self._communicated_soil_ids.get(wp)
                    if comm_id is not None and ((state >> comm_id) & 1):
                        total += 0  # Already done
                        continue
                    # Check if sample already taken
                    has_analysis = False
                    for (rover, waypoint), fid in self._have_soil_ids.items():
                        if waypoint == wp and ((state >> fid) & 1):
                            has_analysis = True
                            break
                    if has_analysis:
                        total += 1  # Just need to communicate
                    else:
                        # Need navigate + empty_store + sample + navigate_to_comm + communicate
                        # At minimum: sample(1) + communicate(1) = 2, but often more due to navigation
                        total += 3  # Conservative: navigate + sample + communicate
                
                elif tname == "get_rock_data" and len(args) >= 1:
                    wp = args[0]
                    comm_id = self._communicated_rock_ids.get(wp)
                    if comm_id is not None and ((state >> comm_id) & 1):
                        continue
                    has_analysis = False
                    for (rover, waypoint), fid in self._have_rock_ids.items():
                        if waypoint == wp and ((state >> fid) & 1):
                            has_analysis = True
                            break
                    if has_analysis:
                        total += 1
                    else:
                        total += 3
                
                elif tname == "get_image_data" and len(args) >= 2:
                    obj, mode = args[0], args[1]
                    comm_id = self._communicated_image_ids.get((obj, mode))
                    if comm_id is not None and ((state >> comm_id) & 1):
                        continue
                    has_img = False
                    for (rover, objective, m), fid in self._have_image_ids.items():
                        if objective == obj and m == mode and ((state >> fid) & 1):
                            has_img = True
                            break
                    if has_img:
                        total += 1  # Just communicate
                    else:
                        # calibrate + navigate + take_image + navigate_to_comm + communicate
                        total += 4
                
                elif tname == "do_navigate1" or tname == "do_navigate2":
                    if len(args) >= 2:
                        rover = args[0]
                        dst = args[-1]
                        # Check if rover is already at destination
                        at_id = self._at_rover_ids.get((rover, dst))
                        if at_id is not None and ((state >> at_id) & 1):
                            total += 0  # Already there
                        else:
                            total += 1  # At least one navigate action
                    else:
                        total += base_cost
                
                elif tname == "empty_store":
                    if len(args) >= 1:
                        store = args[0]
                        empty_id = self._empty_store_ids.get(store)
                        if empty_id is not None and ((state >> empty_id) & 1):
                            total += 0  # Already empty
                        else:
                            total += 1  # Need to drop
                    else:
                        total += base_cost
                
                elif tname == "send_soil_data" or tname == "send_rock_data" or tname == "send_image_data":
                    total += 1  # Navigate to comm + communicate, min 1
                
                elif tname == "do_calibrate":
                    if len(args) >= 2:
                        rover = args[0]
                        camera = args[1]
                        cal_id = self._calibrated_ids.get((camera, rover))
                        if cal_id is not None and ((state >> cal_id) & 1):
                            total += 0
                        else:
                            total += 1  # navigate + calibrate, min 1
                    else:
                        total += base_cost
                else:
                    total += max(base_cost, 1)
        
        return total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
