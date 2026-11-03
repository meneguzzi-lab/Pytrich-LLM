"""
Rover GTOHP Domain-Dependent Heuristic

Bottleneck Analysis:
  The primary bottleneck in the Rover domain is navigation: rovers must traverse
  waypoints to reach sample locations, imaging positions, and communication positions.
  Each high-level task (get_soil_data, get_rock_data, get_image_data) requires
  navigation + sampling/imaging + communication, all serialized. The store capacity
  (one sample at a time per rover) further serializes soil/rock collection.

Admissible Lower Bound Terms:
  T1 - Uncommunicated goals count: Each uncommunicated goal (soil/rock/image) requires
       at minimum one communicate action, so count of unsatisfied goal predicates is
       a lower bound on remaining actions.
  T2 - Unachieved intermediate state: For each uncommunicated soil/rock data, if the
       rover doesn't have the analysis yet, we need at least sample + communicate = 2.
       If the sample isn't taken and rover isn't at the waypoint, navigate + sample +
       navigate_to_comm + communicate = 4 minimum. For images: calibrate + navigate +
       take_image + navigate_to_comm + communicate = 5 minimum.
  T3 - Task network structural estimate: Count remaining compound tasks weighted by
       their minimum decomposition depth, using state-aware checks to avoid counting
       already-satisfied subtasks.

The heuristic returns max(T1, T2, T3) for strong guidance while remaining efficient.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic0(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_0"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}    # waypoint -> fact global_id
        self.communicated_rock_ids = {}    # waypoint -> fact global_id
        self.communicated_image_ids = {}   # (objective, mode) -> fact global_id
        self.have_soil_ids = {}            # (rover, waypoint) -> fact global_id
        self.have_rock_ids = {}            # (rover, waypoint) -> fact global_id
        self.have_image_ids = {}           # (rover, objective, mode) -> fact global_id
        self.at_rover_ids = {}             # (rover, waypoint) -> fact global_id
        self.calibrated_ids = {}           # (camera, rover) -> fact global_id
        self.at_soil_sample_ids = {}       # waypoint -> fact global_id
        self.at_rock_sample_ids = {}       # waypoint -> fact global_id
        self.empty_store_ids = {}          # store -> fact global_id
        self.full_store_ids = {}           # store -> fact global_id
        self.store_of_ids = {}             # (store, rover) -> True
        self.equipped_soil_ids = set()     # set of rovers
        self.equipped_rock_ids = set()     # set of rovers
        self.equipped_imaging_ids = set()  # set of rovers
        
        # Goal tracking
        self.goal_soil_waypoints = []      # list of waypoints needing soil comm
        self.goal_rock_waypoints = []      # list of waypoints needing rock comm
        self.goal_image_pairs = []         # list of (objective, mode) needing image comm
        
        # Task name patterns for task network analysis
        self.task_type_cache = {}          # task global_id -> estimated min cost

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
        """Parse a task/operator name to extract the task type and arguments."""
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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact ID maps by parsing all facts
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
            elif pred == "at" and len(args) == 2:
                self.at_rover_ids[(args[0], args[1])] = f.global_id
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = f.global_id
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = f.global_id
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_store_ids[args[0]] = f.global_id
            elif pred == "full" and len(args) == 1:
                self.full_store_ids[args[0]] = f.global_id
            elif pred == "store_of" and len(args) == 2:
                self.store_of_ids[(args[0], args[1])] = True
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids.add(args[0])
        
        # Parse goal state to identify what needs to be communicated
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))
        
        # Precompute rover-to-store mapping for quick lookup
        self.rover_stores = {}  # rover -> list of stores
        for (store, rover) in self.store_of_ids:
            if rover not in self.rover_stores:
                self.rover_stores[rover] = []
            self.rover_stores[rover].append(store)
        
        # Precompute which rovers can do soil, rock, imaging
        self.soil_rovers = list(self.equipped_soil_ids)
        self.rock_rovers = list(self.equipped_rock_ids)
        self.imaging_rovers = list(self.equipped_imaging_ids)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the current state using bitwise ops."""
        return (state >> global_id) & 1

    def _compute(self, state, task_network):
        t1 = self._goal_distance_estimate(state)
        t2 = self._task_network_estimate(state, task_network)
        return max(t1, t2)

    def _goal_distance_estimate(self, state):
        """T1+T2 combined: State-aware estimate of remaining work for each goal.
        
        For each uncommunicated goal:
        - If communicated: 0
        - If have_analysis but not communicated: 1 (just communicate)
        - If sample exists but not analyzed: 2+ (sample + communicate, maybe navigate)
        - If sample doesn't exist (already taken by someone): check if any rover has it
        For images: similar staged estimation.
        """
        total = 0
        
        # Soil goals
        for wp in self.goal_soil_waypoints:
            if wp not in self.communicated_soil_ids:
                total += 1
                continue
            comm_id = self.communicated_soil_ids[wp]
            if self._check_fact(state, comm_id):
                continue  # Already done
            
            # Check if any rover has the soil analysis
            any_has = False
            for rover in self.soil_rovers:
                key = (rover, wp)
                if key in self.have_soil_ids:
                    if self._check_fact(state, self.have_soil_ids[key]):
                        any_has = True
                        break
            
            if any_has:
                # Just need to navigate to comm position + communicate
                total += 2  # navigate + communicate (minimum)
            else:
                # Need to: navigate to wp, maybe empty store, sample, navigate to comm, communicate
                # Check if soil sample still exists at waypoint
                if wp in self.at_soil_sample_ids and self._check_fact(state, self.at_soil_sample_ids[wp]):
                    total += 4  # navigate + empty_store/nop + sample + communicate (min)
                else:
                    # Sample already taken but not by a soil-equipped rover? 
                    # This shouldn't normally happen in well-formed problems
                    total += 4
        
        # Rock goals
        for wp in self.goal_rock_waypoints:
            if wp not in self.communicated_rock_ids:
                total += 1
                continue
            comm_id = self.communicated_rock_ids[wp]
            if self._check_fact(state, comm_id):
                continue
            
            any_has = False
            for rover in self.rock_rovers:
                key = (rover, wp)
                if key in self.have_rock_ids:
                    if self._check_fact(state, self.have_rock_ids[key]):
                        any_has = True
                        break
            
            if any_has:
                total += 2
            else:
                if wp in self.at_rock_sample_ids and self._check_fact(state, self.at_rock_sample_ids[wp]):
                    total += 4
                else:
                    total += 4
        
        # Image goals
        for (obj, mode) in self.goal_image_pairs:
            key = (obj, mode)
            if key not in self.communicated_image_ids:
                total += 1
                continue
            comm_id = self.communicated_image_ids[key]
            if self._check_fact(state, comm_id):
                continue
            
            # Check if any rover has the image
            any_has = False
            for rover in self.imaging_rovers:
                ikey = (rover, obj, mode)
                if ikey in self.have_image_ids:
                    if self._check_fact(state, self.have_image_ids[ikey]):
                        any_has = True
                        break
            
            if any_has:
                total += 2  # navigate to comm + communicate
            else:
                # Need: calibrate + navigate to visible + take_image + navigate to comm + communicate
                total += 5
        
        return total

    def _task_network_estimate(self, state, task_network):
        """T3: Estimate cost from remaining task network.
        
        Count compound tasks with state-aware minimum cost estimation.
        Primitive tasks (operators) count as 1 each (except nop = 0).
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive task)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                # It's a primitive action
                task_type, args = self._parse_task_name(task.name)
                if task_type == "nop":
                    continue
                total += 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # It's an abstract task - estimate based on type
                task_type, args = self._parse_task_name(task.name)
                cost = self._estimate_abstract_task_cost(state, task_type, args)
                total += cost
        
        return total

    def _estimate_abstract_task_cost(self, state, task_type, args):
        """Estimate minimum actions needed for an abstract task given current state."""
        
        if task_type == "do_navigate1":
            # Navigate rover to waypoint
            if len(args) >= 2:
                rover, to_wp = args[0], args[1]
                key = (rover, to_wp)
                if key in self.at_rover_ids:
                    if self._check_fact(state, self.at_rover_ids[key]):
                        return 0  # Already there
                return 1  # At least one navigate action
            return 1
        
        elif task_type == "do_navigate2":
            if len(args) >= 3:
                rover, from_wp, to_wp = args[0], args[1], args[2]
                if from_wp == to_wp:
                    return 0
                key = (rover, to_wp)
                if key in self.at_rover_ids:
                    if self._check_fact(state, self.at_rover_ids[key]):
                        return 0
            return 1
        
        elif task_type == "empty_store":
            if len(args) >= 2:
                store = args[0]
                if store in self.empty_store_ids:
                    if self._check_fact(state, self.empty_store_ids[store]):
                        return 0  # Already empty
                return 1  # Need to drop
            return 1
        
        elif task_type == "get_soil_data":
            if len(args) >= 1:
                wp = args[0]
                # Check if already communicated
                if wp in self.communicated_soil_ids:
                    if self._check_fact(state, self.communicated_soil_ids[wp]):
                        return 0
                # navigate + maybe_empty + sample + navigate_comm + communicate = ~5
                # Check if any rover already has the analysis
                for rover in self.soil_rovers:
                    key = (rover, wp)
                    if key in self.have_soil_ids:
                        if self._check_fact(state, self.have_soil_ids[key]):
                            return 2  # Just need send_soil_data (navigate + comm)
                return 4  # Full pipeline minimum
            return 4
        
        elif task_type == "get_rock_data":
            if len(args) >= 1:
                wp = args[0]
                if wp in self.communicated_rock_ids:
                    if self._check_fact(state, self.communicated_rock_ids[wp]):
                        return 0
                for rover in self.rock_rovers:
                    key = (rover, wp)
                    if key in self.have_rock_ids:
                        if self._check_fact(state, self.have_rock_ids[key]):
                            return 2
                return 4
            return 4
        
        elif task_type == "get_image_data":
            if len(args) >= 2:
                obj, mode = args[0], args[1]
                key = (obj, mode)
                if key in self.communicated_image_ids:
                    if self._check_fact(state, self.communicated_image_ids[key]):
                        return 0
                for rover in self.imaging_rovers:
                    ikey = (rover, obj, mode)
                    if ikey in self.have_image_ids:
                        if self._check_fact(state, self.have_image_ids[ikey]):
                            return 2  # Just send
                return 5  # calibrate + navigate + take_image + navigate + communicate
            return 5
        
        elif task_type == "send_soil_data":
            if len(args) >= 2:
                rover, wp = args[0], args[1]
                if wp in self.communicated_soil_ids:
                    if self._check_fact(state, self.communicated_soil_ids[wp]):
                        return 0
            return 2  # navigate + communicate
        
        elif task_type == "send_rock_data":
            if len(args) >= 2:
                rover, wp = args[0], args[1]
                if wp in self.communicated_rock_ids:
                    if self._check_fact(state, self.communicated_rock_ids[wp]):
                        return 0
            return 2
        
        elif task_type == "send_image_data":
            if len(args) >= 3:
                rover, obj, mode = args[0], args[1], args[2]
                key = (obj, mode)
                if key in self.communicated_image_ids:
                    if self._check_fact(state, self.communicated_image_ids[key]):
                        return 0
            return 2
        
        elif task_type == "do_calibrate":
            if len(args) >= 2:
                rover, camera = args[0], args[1]
                key = (camera, rover)
                if key in self.calibrated_ids:
                    if self._check_fact(state, self.calibrated_ids[key]):
                        return 0
            return 2  # navigate + calibrate
        
        # Unknown task type - conservative estimate
        return 1
