"""Rover GTOHP Domain-Dependent Heuristic

Bottleneck: Navigation distances between waypoints combined with single-store
capacity per rover serialize soil/rock data collection. Communication requires
visibility to lander waypoint, adding another travel constraint.

Admissible lower bounds:
1. Uncommunicated goals: each uncommunicated_soil/rock/image_data goal needs
   at minimum 1 communicate action (admissible: can't communicate without acting).
2. Missing prerequisites: for each uncommunicated goal, if the rover doesn't
   have the analysis/image yet, we need at least sample+communicate or
   calibrate+take_image+communicate actions.
3. Task network estimate: count remaining non-trivial abstract tasks weighted
   by minimum decomposition depth.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic9(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_9"):
        super().__init__(name)
        self._fact_map = {}  # predicate -> list of (args, global_id)
        self._communicated_soil_ids = {}  # waypoint_arg -> global_id
        self._communicated_rock_ids = {}  # waypoint_arg -> global_id
        self._communicated_image_ids = {}  # (objective, mode) -> global_id
        self._have_soil_ids = {}  # (rover, waypoint) -> global_id
        self._have_rock_ids = {}  # (rover, waypoint) -> global_id
        self._have_image_ids = {}  # (rover, objective, mode) -> global_id
        self._calibrated_ids = {}  # (camera, rover) -> global_id
        self._at_ids = {}  # (rover, waypoint) -> global_id
        self._at_soil_sample_ids = {}  # waypoint -> global_id
        self._at_rock_sample_ids = {}  # waypoint -> global_id
        self._empty_store_ids = {}  # store -> global_id
        self._full_store_ids = {}  # store -> global_id
        self._goal_soil_waypoints = set()
        self._goal_rock_waypoints = set()
        self._goal_image_keys = set()  # (objective, mode)
        self._goal_fact_ids = []
        # Abstract task name -> min cost estimate
        self._abstract_task_min_cost = {}
        # Operator/AbstractTask identification by name prefix
        self._nop_op_ids = set()
        self._visit_op_ids = set()
        self._unvisit_op_ids = set()
        # Store rover capabilities
        self._soil_rovers = set()
        self._rock_rovers = set()
        self._imaging_rovers = set()
        self._store_of = {}  # store -> rover
        self._rover_store = {}  # rover -> store
        # Lander info
        self._lander_waypoint = None
        self._visible_pairs = set()  # (wp1, wp2)
        # Can traverse adjacency for BFS
        self._can_traverse = {}  # rover -> set of (wp1, wp2)
        self._rover_adjacency = {}  # rover -> {wp: set of neighbors}
        # Precomputed min distances to lander-visible waypoints per rover
        self._min_dist_to_comm = {}  # rover -> {wp: min_dist_to_communication_point}

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
        return name, []

    def _parse_task_name(self, name: str):
        """Parse operator/task name to extract the base name and arguments."""
        # Names like: navigate[rover0,waypoint1,waypoint0]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            base = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return base, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self._communicated_soil_ids[args[0]] = f.global_id
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self._communicated_rock_ids[args[0]] = f.global_id
            elif pred == "communicated_image_data" and len(args) >= 2:
                self._communicated_image_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis" and len(args) >= 2:
                self._have_soil_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis" and len(args) >= 2:
                self._have_rock_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_image" and len(args) >= 3:
                self._have_image_ids[(args[0], args[1], args[2])] = f.global_id
            elif pred == "calibrated" and len(args) >= 2:
                self._calibrated_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) >= 2:
                self._at_ids[(args[0], args[1])] = f.global_id
            elif pred == "at_soil_sample" and len(args) >= 1:
                self._at_soil_sample_ids[args[0]] = f.global_id
            elif pred == "at_rock_sample" and len(args) >= 1:
                self._at_rock_sample_ids[args[0]] = f.global_id
            elif pred == "empty" and len(args) >= 1:
                self._empty_store_ids[args[0]] = f.global_id
            elif pred == "full" and len(args) >= 1:
                self._full_store_ids[args[0]] = f.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) >= 1:
                self._soil_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) >= 1:
                self._rock_rovers.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) >= 1:
                self._imaging_rovers.add(args[0])
            elif pred == "store_of" and len(args) >= 2:
                self._store_of[args[0]] = args[1]  # store -> rover
                self._rover_store[args[1]] = args[0]  # rover -> store
            elif pred == "at_lander" and len(args) >= 2:
                self._lander_waypoint = args[1]
            elif pred == "visible" and len(args) >= 2:
                self._visible_pairs.add((args[0], args[1]))
            elif pred == "can_traverse" and len(args) >= 3:
                rover = args[0]
                wp1 = args[1]
                wp2 = args[2]
                if rover not in self._rover_adjacency:
                    self._rover_adjacency[rover] = {}
                if wp1 not in self._rover_adjacency[rover]:
                    self._rover_adjacency[rover][wp1] = set()
                self._rover_adjacency[rover][wp1].add(wp2)

        # Parse goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self._goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self._goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data" and len(args) >= 2:
                self._goal_image_keys.add((args[0], args[1]))

        # Collect all goal fact global_ids for fast checking
        for wp in self._goal_soil_waypoints:
            if wp in self._communicated_soil_ids:
                self._goal_fact_ids.append(self._communicated_soil_ids[wp])
        for wp in self._goal_rock_waypoints:
            if wp in self._communicated_rock_ids:
                self._goal_fact_ids.append(self._communicated_rock_ids[wp])
        for key in self._goal_image_keys:
            if key in self._communicated_image_ids:
                self._goal_fact_ids.append(self._communicated_image_ids[key])

        # Find waypoints visible to lander for communication
        self._comm_waypoints = set()
        if self._lander_waypoint:
            for (wp1, wp2) in self._visible_pairs:
                if wp2 == self._lander_waypoint:
                    self._comm_waypoints.add(wp1)
                if wp1 == self._lander_waypoint:
                    self._comm_waypoints.add(wp2)
            # The lander waypoint itself is also a comm point if visible to itself
            # Actually, communication requires being at wp1 where visible(wp1, lander_wp)
            # So comm_waypoints are all wp where visible(wp, lander_waypoint)

        # Precompute BFS distances from each waypoint to nearest comm waypoint per rover
        # This is done per rover since traversal capabilities differ
        all_rovers = set()
        for (r, w) in self._at_ids:
            all_rovers.add(r)

        # For efficiency, compute only for rovers that have capabilities
        relevant_rovers = self._soil_rovers | self._rock_rovers | self._imaging_rovers
        
        # Collect all waypoints
        all_waypoints = set()
        for wp in self._at_soil_sample_ids:
            all_waypoints.add(wp)
        for wp in self._at_rock_sample_ids:
            all_waypoints.add(wp)
        for (r, wp) in self._at_ids:
            all_waypoints.add(wp)
        for wp in self._comm_waypoints:
            all_waypoints.add(wp)

        # BFS from comm waypoints (reverse direction) for each rover
        for rover in relevant_rovers:
            if rover not in self._rover_adjacency:
                continue
            adj = self._rover_adjacency[rover]
            # Build reverse adjacency
            rev_adj = {}
            for wp1 in adj:
                for wp2 in adj[wp1]:
                    if wp2 not in rev_adj:
                        rev_adj[wp2] = set()
                    rev_adj[wp2].add(wp1)

            # BFS from all comm waypoints simultaneously
            dist = {}
            from collections import deque
            queue = deque()
            for cw in self._comm_waypoints:
                if cw in adj or cw in rev_adj or any(cw in adj.get(w, set()) for w in adj):
                    dist[cw] = 0
                    queue.append(cw)
                # Also check if cw is a valid waypoint for this rover
                # A rover can be at cw even if no traversal from cw
                dist[cw] = 0
                queue.append(cw)

            while queue:
                wp = queue.popleft()
                d = dist[wp]
                # Reverse: from wp, who can reach wp? Those in rev_adj[wp]
                if wp in rev_adj:
                    for prev_wp in rev_adj[wp]:
                        if prev_wp not in dist:
                            dist[prev_wp] = d + 1
                            queue.append(prev_wp)
            self._min_dist_to_comm[rover] = dist

        # Precompute abstract task minimum costs
        # For each abstract task type, estimate minimum actions needed
        for at in model.abstract_tasks:
            base, args = self._parse_task_name(at.name)
            if base in ("get_soil_data", "get_rock_data"):
                # Minimum: navigate(maybe 0) + empty_store(maybe 0) + sample + navigate_to_comm + communicate = ~3
                self._abstract_task_min_cost[at.global_id] = 3
            elif base == "get_image_data":
                # Minimum: calibrate(navigate+calibrate) + navigate + take_image + navigate_to_comm + communicate = ~4
                self._abstract_task_min_cost[at.global_id] = 4
            elif base in ("send_soil_data", "send_rock_data", "send_image_data"):
                self._abstract_task_min_cost[at.global_id] = 1
            elif base in ("do_navigate1", "do_navigate2"):
                self._abstract_task_min_cost[at.global_id] = 0  # might be nop
            elif base == "empty_store":
                self._abstract_task_min_cost[at.global_id] = 0  # might already be empty
            elif base == "do_calibrate":
                self._abstract_task_min_cost[at.global_id] = 1
            else:
                self._abstract_task_min_cost[at.global_id] = 0

        # Identify nop/visit/unvisit operators for filtering
        for op in model.operators:
            base, args = self._parse_task_name(op.name)
            if base == "nop":
                self._nop_op_ids.add(op.global_id)
            elif base == "visit":
                self._visit_op_ids.add(op.global_id)
            elif base == "unvisit":
                self._unvisit_op_ids.add(op.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible terms."""
        h1 = self._goal_distance_estimate(state)
        h2 = self._task_network_estimate(state, task_network)
        h3 = self._prerequisite_estimate(state)
        return max(h1, h2, h3)

    def _goal_distance_estimate(self, state):
        """Count uncommunicated goal facts.
        Each uncommunicated goal requires at least 1 action (communicate).
        This is admissible."""
        count = 0
        for gid in self._goal_fact_ids:
            if not ((state >> gid) & 1):
                count += 1
        return count

    def _prerequisite_estimate(self, state):
        """For each uncommunicated goal, estimate minimum actions needed
        based on current state prerequisites.
        
        Admissible because:
        - If not communicated AND no analysis/image: need sample/image + communicate >= 2
        - If not communicated AND have analysis but need travel to comm: >= 1 + travel
        - If communicated: 0
        """
        total = 0

        for wp in self._goal_soil_waypoints:
            if wp in self._communicated_soil_ids:
                gid = self._communicated_soil_ids[wp]
                if (state >> gid) & 1:
                    continue  # Already communicated
            else:
                continue

            # Check if any rover has soil analysis for this waypoint
            has_analysis = False
            for rover in self._soil_rovers:
                key = (rover, wp)
                if key in self._have_soil_ids:
                    fid = self._have_soil_ids[key]
                    if (state >> fid) & 1:
                        has_analysis = True
                        break

            if has_analysis:
                # Need at least: navigate to comm + communicate = 2
                total += 2
            else:
                # Check if soil sample still exists
                sample_exists = True
                if wp in self._at_soil_sample_ids:
                    sid = self._at_soil_sample_ids[wp]
                    if not ((state >> sid) & 1):
                        sample_exists = False
                
                if sample_exists:
                    # Need: navigate to wp + maybe empty store + sample + navigate to comm + communicate = ~4
                    total += 4
                else:
                    # Sample already taken by some rover, need to find which
                    # At minimum need communicate = 2 (navigate + comm)
                    total += 2

        for wp in self._goal_rock_waypoints:
            if wp in self._communicated_rock_ids:
                gid = self._communicated_rock_ids[wp]
                if (state >> gid) & 1:
                    continue
            else:
                continue

            has_analysis = False
            for rover in self._rock_rovers:
                key = (rover, wp)
                if key in self._have_rock_ids:
                    fid = self._have_rock_ids[key]
                    if (state >> fid) & 1:
                        has_analysis = True
                        break

            if has_analysis:
                total += 2
            else:
                sample_exists = True
                if wp in self._at_rock_sample_ids:
                    sid = self._at_rock_sample_ids[wp]
                    if not ((state >> sid) & 1):
                        sample_exists = False
                
                if sample_exists:
                    total += 4
                else:
                    total += 2

        for (obj, mode) in self._goal_image_keys:
            key = (obj, mode)
            if key in self._communicated_image_ids:
                gid = self._communicated_image_ids[key]
                if (state >> gid) & 1:
                    continue
            else:
                continue

            has_image = False
            for rover in self._imaging_rovers:
                ikey = (rover, obj, mode)
                if ikey in self._have_image_ids:
                    fid = self._have_image_ids[ikey]
                    if (state >> fid) & 1:
                        has_image = True
                        break

            if has_image:
                total += 2  # navigate to comm + communicate
            else:
                # Need: calibrate(1-2) + navigate to visible + take_image + navigate to comm + communicate
                total += 5

        return total

    def _task_network_estimate(self, state, task_network):
        """Estimate based on remaining tasks in the task network.
        
        For each task, use precomputed minimum cost.
        Admissible because we use lower bounds on decomposition costs.
        """
        if not task_network:
            return 0

        total = 0
        for task in task_network:
            gid = task.global_id
            # Check if it's an operator
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                # It's an operator - costs 1 action unless nop/visit/unvisit
                if gid in self._nop_op_ids or gid in self._visit_op_ids or gid in self._unvisit_op_ids:
                    pass  # 0 meaningful cost
                else:
                    total += 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # It's an abstract task
                if gid in self._abstract_task_min_cost:
                    total += self._abstract_task_min_cost[gid]
                else:
                    total += 1  # default

        return total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
