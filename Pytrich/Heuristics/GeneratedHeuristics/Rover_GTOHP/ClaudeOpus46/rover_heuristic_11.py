"""Rover GTOHP Domain-Dependent Heuristic

Bottleneck: Navigation serializes all work. Each data goal requires
navigate-to-sample/image-location, perform action, navigate-to-comm-location,
communicate. Store capacity (1 per rover) further serializes soil/rock collection.

Admissible lower bounds:
1. For each uncommunicated goal fact, at least 1 action (communicate) is needed.
   If the prerequisite data (have_soil/rock/image) is also missing, at least 2 more
   actions are needed (navigate + sample/image). This gives per-goal lower bounds.
2. Task network remaining: count non-nop primitive tasks and abstract tasks with
   minimum decomposition depth.
3. Resource contention: if multiple goals need the same rover's store, they must
   be serialized (drop + re-sample), adding extra actions.

We return max(goal_based_h, task_network_h) for admissibility.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic11(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_11"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}  # waypoint -> global_id
        self.communicated_rock_ids = {}  # waypoint -> global_id
        self.communicated_image_ids = {}  # (objective, mode) -> global_id
        self.have_soil_ids = {}  # (rover, waypoint) -> global_id
        self.have_rock_ids = {}  # (rover, waypoint) -> global_id
        self.have_image_ids = {}  # (rover, objective, mode) -> global_id
        self.at_soil_sample_ids = {}  # waypoint -> global_id
        self.at_rock_sample_ids = {}  # waypoint -> global_id
        self.calibrated_ids = {}  # (camera, rover) -> global_id
        self.at_ids = {}  # (rover, waypoint) -> global_id
        self.empty_ids = {}  # store -> global_id
        self.full_ids = {}  # store -> global_id
        self.store_of_map = {}  # store -> rover
        self.equipped_soil = set()  # rovers equipped for soil
        self.equipped_rock = set()  # rovers equipped for rock
        self.equipped_imaging = set()  # rovers equipped for imaging
        self.on_board_map = {}  # camera -> rover
        self.supports_map = {}  # camera -> set of modes
        self.calibration_target_map = {}  # camera -> objective
        
        # Goal tracking
        self.goal_soil_waypoints = []  # waypoints needing soil communication
        self.goal_rock_waypoints = []  # waypoints needing rock communication
        self.goal_image_pairs = []  # (objective, mode) needing image communication
        
        # Abstract task and operator name caches
        self.nop_op_ids = set()
        self.navigate_op_ids = set()
        self.visit_op_ids = set()
        self.unvisit_op_ids = set()
        
        # Precomputed minimum task costs
        self.abstract_task_min_cost = {}

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
        """Parse operator/abstract task name to extract task type."""
        # Names like: navigate[rover0,waypoint1,waypoint0]
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            task_type = name[:bracket_idx]
            if name.endswith("]"):
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return task_type, args
            return task_type, []
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
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = f.global_id
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_ids[(args[0], args[1])] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = f.global_id
            elif pred == "full" and len(args) == 1:
                self.full_ids[args[0]] = f.global_id
            elif pred == "store_of" and len(args) == 2:
                self.store_of_map[args[0]] = args[1]  # store -> rover
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging.add(args[0])
            elif pred == "on_board" and len(args) == 2:
                self.on_board_map[args[0]] = args[1]  # camera -> rover
            elif pred == "supports" and len(args) == 2:
                if args[0] not in self.supports_map:
                    self.supports_map[args[0]] = set()
                self.supports_map[args[0]].add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self.calibration_target_map[args[0]] = args[1]  # camera -> objective

        # Parse goal facts to identify what needs to be achieved
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))

        # Build rover -> store mapping for contention analysis
        self.rover_store = {}  # rover -> store
        for store, rover in self.store_of_map.items():
            self.rover_store[rover] = store

        # Map which rovers can handle each soil/rock waypoint
        self.soil_capable_rovers = {}  # waypoint -> list of rovers
        self.rock_capable_rovers = {}  # waypoint -> list of rovers
        for wp in self.goal_soil_waypoints:
            self.soil_capable_rovers[wp] = list(self.equipped_soil)
        for wp in self.goal_rock_waypoints:
            self.rock_capable_rovers[wp] = list(self.equipped_rock)

        # Map which rover+camera combos can handle each image goal
        self.image_capable = {}  # (obj, mode) -> list of (rover, camera)
        for obj, mode in self.goal_image_pairs:
            capable = []
            for camera, rover in self.on_board_map.items():
                if rover in self.equipped_imaging:
                    if camera in self.supports_map and mode in self.supports_map[camera]:
                        capable.append((rover, camera))
            self.image_capable[(obj, mode)] = capable

        # Classify operators for task network analysis
        for op in model.operators:
            task_type, _ = self._parse_task_name(op.name)
            if task_type == "nop":
                self.nop_op_ids.add(op.global_id)
            elif task_type == "navigate":
                self.navigate_op_ids.add(op.global_id)
            elif task_type == "visit":
                self.visit_op_ids.add(op.global_id)
            elif task_type == "unvisit":
                self.unvisit_op_ids.add(op.global_id)

        # Precompute minimum cost for each abstract task type
        # Use BFS over decomposition tree to find min primitive actions
        self._precompute_abstract_task_costs()

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _precompute_abstract_task_costs(self):
        """Compute minimum number of non-trivial primitive actions for each abstract task."""
        from Pytrich.model import Model
        # We'll do iterative fixed-point computation
        # Start with operators: nop=0, others=1
        cost = {}
        for op in self.model.operators:
            if op.global_id in self.nop_op_ids:
                cost[op.global_id] = 0
            elif op.global_id in self.visit_op_ids or op.global_id in self.unvisit_op_ids:
                cost[op.global_id] = 0  # visit/unvisit are bookkeeping
            else:
                cost[op.global_id] = 1

        # Initialize abstract tasks with infinity
        for at in self.model.abstract_tasks:
            cost[at.global_id] = 999999

        # Iterate until convergence
        changed = True
        iterations = 0
        max_iterations = 50
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for dec in self.model.decompositions:
                # Cost of decomposition = sum of costs of subtasks
                dec_cost = 0
                valid = True
                for subtask in dec.task_network:
                    if subtask.global_id in cost:
                        dec_cost += cost[subtask.global_id]
                    else:
                        valid = False
                        break
                if valid:
                    ct = dec.compound_task
                    if dec_cost < cost.get(ct.global_id, 999999):
                        cost[ct.global_id] = dec_cost
                        changed = True

        self.abstract_task_min_cost = cost

    def _compute(self, state, task_network):
        """Compute heuristic value from current state and task network."""
        h_goal = self._goal_based_h(state)
        h_tn = self._task_network_h(state, task_network)
        h_contention = self._store_contention_h(state)
        return max(h_goal, h_tn, h_contention)

    def _goal_based_h(self, state):
        """Goal-based admissible heuristic.
        
        For each uncommunicated goal:
        - If communicated: 0
        - If have_data but not communicated: >= 1 (communicate)
        - If not have_data and sample exists: >= 3 (navigate + sample + communicate)
        - If not have_data and sample consumed: someone already has it, >= 1 (communicate)
        
        For images:
        - If communicated: 0
        - If have_image but not communicated: >= 1
        - If not have_image and not calibrated: >= 4 (nav + calibrate + nav + take_image + nav + communicate)
        - If not have_image and calibrated: >= 3 (nav + take_image + communicate)
        """
        h = 0

        # Soil goals
        for wp in self.goal_soil_waypoints:
            comm_id = self.communicated_soil_ids.get(wp)
            if comm_id is not None and (state >> comm_id) & 1:
                continue  # Already communicated
            
            # Check if any rover has the soil analysis
            any_has_soil = False
            for rover in self.equipped_soil:
                have_id = self.have_soil_ids.get((rover, wp))
                if have_id is not None and (state >> have_id) & 1:
                    any_has_soil = True
                    break
            
            if any_has_soil:
                # Need to navigate to comm point + communicate
                h += 2  # at least navigate + communicate
            else:
                # Check if soil sample still exists at waypoint
                sample_id = self.at_soil_sample_ids.get(wp)
                if sample_id is not None and (state >> sample_id) & 1:
                    # Need: navigate to wp + maybe empty store + sample + navigate to comm + communicate
                    h += 4  # navigate + sample + navigate + communicate minimum
                else:
                    # Sample already taken by someone, they need to communicate
                    # Or it's impossible (shouldn't happen in valid problems)
                    h += 2  # navigate + communicate

        # Rock goals
        for wp in self.goal_rock_waypoints:
            comm_id = self.communicated_rock_ids.get(wp)
            if comm_id is not None and (state >> comm_id) & 1:
                continue
            
            any_has_rock = False
            for rover in self.equipped_rock:
                have_id = self.have_rock_ids.get((rover, wp))
                if have_id is not None and (state >> have_id) & 1:
                    any_has_rock = True
                    break
            
            if any_has_rock:
                h += 2
            else:
                sample_id = self.at_rock_sample_ids.get(wp)
                if sample_id is not None and (state >> sample_id) & 1:
                    h += 4
                else:
                    h += 2

        # Image goals
        for obj, mode in self.goal_image_pairs:
            comm_id = self.communicated_image_ids.get((obj, mode))
            if comm_id is not None and (state >> comm_id) & 1:
                continue
            
            any_has_image = False
            best_image_cost = 5  # worst case: calibrate + nav + take_image + nav + communicate
            
            for rover, camera in self.image_capable.get((obj, mode), []):
                have_id = self.have_image_ids.get((rover, obj, mode))
                if have_id is not None and (state >> have_id) & 1:
                    any_has_image = True
                    best_image_cost = 2  # nav + communicate
                    break
                
                # Check if camera is calibrated
                cal_id = self.calibrated_ids.get((camera, rover))
                if cal_id is not None and (state >> cal_id) & 1:
                    best_image_cost = min(best_image_cost, 3)  # nav + take_image + communicate
                else:
                    best_image_cost = min(best_image_cost, 4)  # nav + calibrate + take_image + communicate
            
            if any_has_image:
                h += 2
            else:
                h += best_image_cost

        return h

    def _task_network_h(self, state, task_network):
        """Task network based heuristic using precomputed minimum costs."""
        if not task_network:
            return 0
        
        h = 0
        for task in task_network:
            gid = task.global_id
            min_cost = self.abstract_task_min_cost.get(gid, 0)
            if min_cost >= 999999:
                # Check if it's an abstract task with applicable decompositions
                # Use state-aware cost estimation
                if hasattr(task, 'decompositions'):
                    best_cost = 999999
                    for dec in task.decompositions:
                        if dec.applicable(state):
                            dec_cost = 0
                            for subtask in dec.task_network:
                                dec_cost += self.abstract_task_min_cost.get(subtask.global_id, 1)
                            best_cost = min(best_cost, dec_cost)
                    if best_cost < 999999:
                        h += best_cost
                    else:
                        # No applicable decomposition found, use structural estimate
                        h += 1
                else:
                    h += min_cost if min_cost < 999999 else 1
            else:
                h += min_cost
        
        return h

    def _store_contention_h(self, state):
        """Estimate extra cost from store contention.
        
        If a rover needs to collect multiple samples (soil and rock),
        it must drop between collections, adding at least 1 action per extra sample.
        """
        # Count how many uncollected soil+rock goals exist per rover capability group
        # This is a rough lower bound on drops needed
        
        # For each rover, count pending collections
        rover_pending = {}  # rover -> count of pending collections
        
        # Count uncollected soil goals
        uncollected_soil = []
        for wp in self.goal_soil_waypoints:
            comm_id = self.communicated_soil_ids.get(wp)
            if comm_id is not None and (state >> comm_id) & 1:
                continue
            any_has = False
            for rover in self.equipped_soil:
                have_id = self.have_soil_ids.get((rover, wp))
                if have_id is not None and (state >> have_id) & 1:
                    any_has = True
                    break
            if not any_has:
                sample_id = self.at_soil_sample_ids.get(wp)
                if sample_id is not None and (state >> sample_id) & 1:
                    uncollected_soil.append(wp)
        
        # Count uncollected rock goals
        uncollected_rock = []
        for wp in self.goal_rock_waypoints:
            comm_id = self.communicated_rock_ids.get(wp)
            if comm_id is not None and (state >> comm_id) & 1:
                continue
            any_has = False
            for rover in self.equipped_rock:
                have_id = self.have_rock_ids.get((rover, wp))
                if have_id is not None and (state >> have_id) & 1:
                    any_has = True
                    break
            if not any_has:
                sample_id = self.at_rock_sample_ids.get(wp)
                if sample_id is not None and (state >> sample_id) & 1:
                    uncollected_rock.append(wp)
        
        total_collections = len(uncollected_soil) + len(uncollected_rock)
        
        # Each collection after the first per rover requires a drop (1 action)
        # With N rovers capable, we need at least ceil(total/N) - 1 drops per rover
        # But as a simple lower bound: if total_collections > num_capable_rovers,
        # we need at least (total_collections - num_capable_rovers) drops
        num_capable = len(self.equipped_soil | self.equipped_rock)
        if num_capable > 0 and total_collections > num_capable:
            extra_drops = total_collections - num_capable
            return extra_drops
        
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
