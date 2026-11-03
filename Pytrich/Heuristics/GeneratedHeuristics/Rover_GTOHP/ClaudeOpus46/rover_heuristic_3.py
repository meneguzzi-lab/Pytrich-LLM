"""Domain-dependent heuristic for Rover_GTOHP.

Bottleneck: Navigation + sequential store usage. Each data goal requires:
  - Navigate to sample/image location
  - Perform sample/calibrate+image action
  - Navigate to communication-visible waypoint
  - Communicate
These steps are sequential per rover, creating serialization.

Admissible lower bounds:
  T1: Number of uncommunicated goal facts. Each requires >= 1 action (communicate).
      Actually each requires >= 2 actions minimum (sample/image + communicate).
  T2: Task network depth - count of non-trivial abstract tasks remaining,
      weighted by minimum actions needed per task type.
  T3: Resource contention - rovers with multiple pending tasks must serialize.

All terms are admissible lower bounds on remaining cost.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict


class RoverGTOHPAnthropicOpus46Heuristic3(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_3"):
        super().__init__(name)
        self.goal_fact_ids = []
        self.goal_fact_min_costs = {}
        # Maps from task name prefix to minimum action cost
        self.task_min_costs = {}
        # Fact ID lookups
        self.communicated_soil_facts = {}  # waypoint_arg -> fact_global_id
        self.communicated_rock_facts = {}  # waypoint_arg -> fact_global_id
        self.communicated_image_facts = {}  # (obj, mode) -> fact_global_id
        # Task name classification
        self.soil_task_prefix = "get_soil_data"
        self.rock_task_prefix = "get_rock_data"
        self.image_task_prefix = "get_image_data"
        self.send_soil_prefix = "send_soil_data"
        self.send_rock_prefix = "send_rock_data"
        self.send_image_prefix = "send_image_data"
        self.navigate_prefix = "do_navigate"
        self.calibrate_prefix = "do_calibrate"
        self.empty_store_prefix = "empty_store"
        # Rover equipment info
        self.rover_has_soil = set()
        self.rover_has_rock = set()
        self.rover_has_imaging = set()
        # Store associations
        self.rover_stores = {}  # rover -> store
        # At facts for rovers
        self.rover_at_facts = {}  # rover -> {waypoint -> fact_id}
        # Lander location
        self.lander_waypoint = None
        # Visible waypoints from lander
        self.visible_to_lander = set()
        # Can traverse graph per rover
        self.rover_graph = defaultdict(lambda: defaultdict(set))
        # BFS distances per rover (precomputed)
        self.rover_distances = {}
        # All waypoints
        self.all_waypoints = set()
        self.all_rovers = set()
        # Fact IDs for at predicates
        self.at_fact_ids = {}  # (rover, waypoint) -> global_id
        # soil/rock sample locations
        self.soil_sample_facts = {}  # waypoint -> global_id
        self.rock_sample_facts = {}  # waypoint -> global_id
        # have_soil/rock/image facts
        self.have_soil_facts = {}  # (rover, waypoint) -> global_id
        self.have_rock_facts = {}  # (rover, waypoint) -> global_id
        self.have_image_facts = {}  # (rover, obj, mode) -> global_id
        # Equipment facts
        self.equipped_soil_facts = {}  # rover -> global_id
        self.equipped_rock_facts = {}  # rover -> global_id
        self.equipped_imaging_facts = {}  # rover -> global_id
        # Store facts
        self.empty_store_facts = {}  # store -> global_id
        self.store_of_facts = {}  # (store, rover) -> global_id
        # Calibrated facts
        self.calibrated_facts = {}  # (camera, rover) -> global_id
        # Channel free
        self.channel_free_fact = None
        # Available facts
        self.available_facts = {}  # rover -> global_id

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
        """Parse grounded task/operator name to get task type and args."""
        name = name.strip()
        # Task names look like: task_name[arg1,arg2,...]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _bfs_distances(self, rover, adjacency):
        """Compute BFS shortest path distances for a rover."""
        if rover not in adjacency or not adjacency[rover]:
            return {}
        distances = {}
        all_nodes = set()
        for wp in adjacency[rover]:
            all_nodes.add(wp)
            for neighbor in adjacency[rover][wp]:
                all_nodes.add(neighbor)
        
        for start in all_nodes:
            dist = {start: 0}
            queue = [start]
            qi = 0
            while qi < len(queue):
                current = queue[qi]
                qi += 1
                if current in adjacency[rover]:
                    for neighbor in adjacency[rover][current]:
                        if neighbor not in dist:
                            dist[neighbor] = dist[current] + 1
                            queue.append(neighbor)
            distances[start] = dist
        return distances

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.communicated_soil_facts[args[0]] = f.global_id
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.communicated_rock_facts[args[0]] = f.global_id
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.communicated_image_facts[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) >= 2:
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                self.all_rovers.add(args[0])
                self.all_waypoints.add(args[1])
            elif pred == "can_traverse" and len(args) >= 3:
                self.rover_graph[args[0]][args[1]].add(args[2])
            elif pred == "at_soil_sample" and len(args) >= 1:
                self.soil_sample_facts[args[0]] = f.global_id
            elif pred == "at_rock_sample" and len(args) >= 1:
                self.rock_sample_facts[args[0]] = f.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) >= 1:
                self.equipped_soil_facts[args[0]] = f.global_id
                self.rover_has_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) >= 1:
                self.equipped_rock_facts[args[0]] = f.global_id
                self.rover_has_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) >= 1:
                self.equipped_imaging_facts[args[0]] = f.global_id
                self.rover_has_imaging.add(args[0])
            elif pred == "store_of" and len(args) >= 2:
                self.store_of_facts[(args[0], args[1])] = f.global_id
                self.rover_stores[args[1]] = args[0]
            elif pred == "empty" and len(args) >= 1:
                self.empty_store_facts[args[0]] = f.global_id
            elif pred == "have_soil_analysis" and len(args) >= 2:
                self.have_soil_facts[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis" and len(args) >= 2:
                self.have_rock_facts[(args[0], args[1])] = f.global_id
            elif pred == "have_image" and len(args) >= 3:
                self.have_image_facts[(args[0], args[1], args[2])] = f.global_id
            elif pred == "at_lander" and len(args) >= 2:
                self.lander_waypoint = args[1]
            elif pred == "visible" and len(args) >= 2:
                pass  # We'll use this for lander visibility
            elif pred == "channel_free" and len(args) >= 1:
                self.channel_free_fact = f.global_id
            elif pred == "available" and len(args) >= 1:
                self.available_facts[args[0]] = f.global_id
            elif pred == "calibrated" and len(args) >= 2:
                self.calibrated_facts[(args[0], args[1])] = f.global_id

        # Find waypoints visible to lander
        if self.lander_waypoint:
            for f in model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred == "visible" and len(args) >= 2:
                    if args[1] == self.lander_waypoint:
                        self.visible_to_lander.add(args[0])
                    if args[0] == self.lander_waypoint:
                        self.visible_to_lander.add(args[1])  # visible is symmetric check

        # Extract goal fact IDs
        goal_state = model.goals
        for f in model.facts:
            if (goal_state >> f.global_id) & 1:
                self.goal_fact_ids.append(f.global_id)
        
        # Precompute BFS distances for each rover (limit to rovers that are actually used)
        # Only compute for rovers that have relevant equipment
        relevant_rovers = self.rover_has_soil | self.rover_has_rock | self.rover_has_imaging
        for rover in relevant_rovers:
            if rover in self.rover_graph:
                self.rover_distances[rover] = self._bfs_distances(rover, self.rover_graph)

        # Compute minimum cost per goal type
        # For soil data: navigate(>=1) + empty_store(0-1) + sample(1) + navigate(>=0) + communicate(1) = min 3
        # For rock data: same as soil = min 3  
        # For image data: navigate(>=1) + calibrate(1) + navigate(>=0) + take_image(1) + navigate(>=0) + communicate(1) = min 4
        self.soil_min_cost = 3  # sample + navigate_to_comm + communicate (at minimum)
        self.rock_min_cost = 3
        self.image_min_cost = 4  # calibrate + take_image + navigate_to_comm + communicate

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_rover_position(self, state, rover):
        """Get the current waypoint of a rover from state, using bit checks."""
        for (r, wp), fid in self.at_fact_ids.items():
            if r == rover and (state >> fid) & 1:
                return wp
        return None

    def _min_nav_distance(self, rover, from_wp, to_wp):
        """Get minimum navigation distance for rover between waypoints."""
        if from_wp == to_wp:
            return 0
        if rover in self.rover_distances:
            dists = self.rover_distances[rover]
            if from_wp in dists and to_wp in dists[from_wp]:
                return dists[from_wp][to_wp]
        return 999  # unreachable

    def _min_dist_to_comm(self, rover, from_wp):
        """Minimum distance from a waypoint to any waypoint visible to lander."""
        if from_wp in self.visible_to_lander:
            return 0
        best = 999
        if rover in self.rover_distances and from_wp in self.rover_distances[rover]:
            dists = self.rover_distances[rover][from_wp]
            for vwp in self.visible_to_lander:
                if vwp in dists:
                    best = min(best, dists[vwp])
        return best

    def _compute_goal_cost(self, state):
        """Count uncommunicated goals with weighted costs."""
        cost = 0
        for gid in self.goal_fact_ids:
            if not ((state >> gid) & 1):
                # This goal is not yet achieved
                fact = self.model.facts[gid]
                pred, args = self._parse_fact_name(fact.name)
                if pred == "communicated_soil_data":
                    cost += self.soil_min_cost
                elif pred == "communicated_rock_data":
                    cost += self.rock_min_cost
                elif pred == "communicated_image_data":
                    cost += self.image_min_cost
                else:
                    cost += 1
        return cost

    def _compute_task_network_cost(self, task_network):
        """Estimate cost from remaining task network."""
        if not task_network:
            return 0
        cost = 0
        for task in task_network:
            tname = task.name
            # Check if it's an abstract task by checking if it has decompositions
            is_abstract = hasattr(task, 'decompositions')
            if is_abstract:
                ttype, targs = self._parse_task_name(tname)
                if ttype == "get_soil_data":
                    # navigate + empty_store + sample + send_soil_data
                    # send_soil_data = navigate + communicate
                    cost += 4  # minimum: navigate(1) + sample(1) + navigate(1) + communicate(1)
                elif ttype == "get_rock_data":
                    cost += 4
                elif ttype == "get_image_data":
                    # calibrate_nav + calibrate + navigate + take_image + send_image
                    cost += 5
                elif ttype == "send_soil_data" or ttype == "send_rock_data":
                    cost += 2  # navigate + communicate
                elif ttype == "send_image_data":
                    cost += 2  # navigate + communicate
                elif ttype == "do_navigate1" or ttype == "do_navigate2":
                    cost += 1  # at least 1 navigate action (could be 0 if already there, but we estimate)
                elif ttype == "do_calibrate":
                    cost += 2  # navigate + calibrate
                elif ttype == "empty_store":
                    cost += 0  # could be nop
                else:
                    cost += 1
            else:
                # Primitive operator
                if "nop" not in tname:
                    cost += 1
        return cost

    def _compute_state_aware_task_cost(self, state, task_network):
        """More refined task cost using state information."""
        if not task_network:
            return 0
        cost = 0
        for task in task_network:
            is_abstract = hasattr(task, 'decompositions')
            if not is_abstract:
                # Primitive: cost 1 unless nop/visit/unvisit
                if "nop" not in task.name and "visit" not in task.name:
                    cost += 1
                continue
            
            ttype, targs = self._parse_task_name(task.name)
            
            if ttype == "get_soil_data" and len(targs) >= 1:
                waypoint = targs[0]
                # Check if soil data already communicated
                if waypoint in self.communicated_soil_facts:
                    cid = self.communicated_soil_facts[waypoint]
                    if (state >> cid) & 1:
                        cost += 0  # already done
                        continue
                
                # Find best rover: one with soil equipment
                best_rover_cost = 999
                for rover in self.rover_has_soil:
                    pos = self._get_rover_position(state, rover)
                    if pos is None:
                        continue
                    nav_to_sample = self._min_nav_distance(rover, pos, waypoint)
                    dist_to_comm = self._min_dist_to_comm(rover, waypoint)
                    # sample(1) + possibly drop(1) + navigate_to_comm + communicate(1)
                    rover_cost = nav_to_sample + 1 + dist_to_comm + 1
                    # Check if store is empty
                    if rover in self.rover_stores:
                        store = self.rover_stores[rover]
                        if store in self.empty_store_facts:
                            sid = self.empty_store_facts[store]
                            if not ((state >> sid) & 1):
                                rover_cost += 1  # need to drop
                    best_rover_cost = min(best_rover_cost, rover_cost)
                cost += min(best_rover_cost, 8)  # cap to avoid overestimation from unreachable

            elif ttype == "get_rock_data" and len(targs) >= 1:
                waypoint = targs[0]
                if waypoint in self.communicated_rock_facts:
                    cid = self.communicated_rock_facts[waypoint]
                    if (state >> cid) & 1:
                        cost += 0
                        continue
                
                best_rover_cost = 999
                for rover in self.rover_has_rock:
                    pos = self._get_rover_position(state, rover)
                    if pos is None:
                        continue
                    nav_to_sample = self._min_nav_distance(rover, pos, waypoint)
                    dist_to_comm = self._min_dist_to_comm(rover, waypoint)
                    rover_cost = nav_to_sample + 1 + dist_to_comm + 1
                    if rover in self.rover_stores:
                        store = self.rover_stores[rover]
                        if store in self.empty_store_facts:
                            sid = self.empty_store_facts[store]
                            if not ((state >> sid) & 1):
                                rover_cost += 1
                    best_rover_cost = min(best_rover_cost, rover_cost)
                cost += min(best_rover_cost, 8)

            elif ttype == "get_image_data" and len(targs) >= 2:
                obj, mode = targs[0], targs[1]
                if (obj, mode) in self.communicated_image_facts:
                    cid = self.communicated_image_facts[(obj, mode)]
                    if (state >> cid) & 1:
                        cost += 0
                        continue
                # Image tasks: calibrate + navigate + take_image + navigate_comm + communicate
                # At minimum: 3 actions (calibrate + take_image + communicate) + navigation
                cost += 4  # reasonable lower bound

            elif ttype == "send_soil_data" or ttype == "send_rock_data":
                cost += 1  # at least communicate action
            elif ttype == "send_image_data":
                cost += 1
            elif ttype == "do_navigate1" or ttype == "do_navigate2":
                # Could be 0 if already there
                cost += 0  # conservative - might already be there
            elif ttype == "do_calibrate":
                cost += 1  # calibrate action at minimum
            elif ttype == "empty_store":
                cost += 0  # could be nop
            else:
                cost += 1
        return cost

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple terms."""
        if self.model.goal_reached(state, task_network):
            return 0
        
        # Term 1: Uncommunicated goal facts with type-based weights
        t1 = self._compute_goal_cost(state)
        
        # Term 2: State-aware task network estimation
        t2 = self._compute_state_aware_task_cost(state, task_network)
        
        # Term 3: Simple task count as fallback
        t3 = 0
        if task_network:
            for task in task_network:
                is_abstract = hasattr(task, 'decompositions')
                if is_abstract:
                    t3 += 1
                elif "nop" not in task.name and "visit" not in task.name and "unvisit" not in task.name:
                    t3 += 1
        
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
