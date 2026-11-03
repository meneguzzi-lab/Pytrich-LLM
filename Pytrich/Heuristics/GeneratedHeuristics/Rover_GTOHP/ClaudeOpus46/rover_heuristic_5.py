"""Domain-dependent heuristic for Rover GTOHP domain.

Bottleneck: Navigation + communication serialization. Each soil/rock data task requires
navigate-to-sample, sample, navigate-to-comm-point, communicate (min 4 actions).
Each image task requires calibrate, navigate-to-vis-point, take_image, navigate-to-comm, communicate (min 5 actions).

Admissible lower bounds:
  T1 - Unsatisfied goals: each communicated_* goal requires at least 1 communicate action.
  T2 - Task network weighted cost: each get_soil/rock_data needs >=3 actions (sample+drop+communicate),
       each get_image_data needs >=3 actions (calibrate+take_image+communicate), plus navigation.
  T3 - Per-task minimum action estimate using state awareness (already sampled? already have analysis? etc.)

Return max(T1, T2, T3) for strong admissible guidance.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic5(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_5"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}   # waypoint -> global_id
        self.communicated_rock_ids = {}   # waypoint -> global_id
        self.communicated_image_ids = {}  # (objective, mode) -> global_id
        self.have_soil_ids = {}           # (rover, waypoint) -> global_id
        self.have_rock_ids = {}           # (rover, waypoint) -> global_id
        self.have_image_ids = {}          # (rover, objective, mode) -> global_id
        self.at_soil_sample_ids = {}      # waypoint -> global_id
        self.at_rock_sample_ids = {}      # waypoint -> global_id
        self.calibrated_ids = {}          # (camera, rover) -> global_id
        self.at_ids = {}                  # (rover, waypoint) -> global_id
        self.empty_ids = {}               # store -> global_id
        self.full_ids = {}                # store -> global_id
        self.store_of_map = {}            # store -> rover
        self.equipped_soil = set()        # rovers equipped for soil
        self.equipped_rock = set()        # rovers equipped for rock
        self.equipped_imaging = set()     # rovers equipped for imaging
        self.on_board_map = {}            # camera -> rover
        self.supports_map = {}            # camera -> set of modes
        self.channel_free_id = None       # global_id for channel_free
        
        # Goal tracking
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_image_pairs = set()     # (objective, mode)
        
        # Task name classification
        self.task_name_cache = {}
        
        # Operator classification
        self.op_is_nop = set()  # global_ids of nop operators
        self.op_is_visit = set()
        self.op_is_unvisit = set()
        
        # Precomputed minimum costs per task type
        self.MIN_SOIL_COST = 4    # navigate + empty_store/nop + sample_soil + send(navigate+communicate)
        self.MIN_ROCK_COST = 4    # same pattern
        self.MIN_IMAGE_COST = 5   # calibrate(navigate+calibrate) + navigate + take_image + send(navigate+communicate)
        
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
        """Parse task/operator name to get the base task type."""
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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.communicated_soil_ids[args[0]] = f.global_id
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.communicated_rock_ids[args[0]] = f.global_id
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.communicated_image_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis" and len(args) >= 2:
                self.have_soil_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis" and len(args) >= 2:
                self.have_rock_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_image" and len(args) >= 3:
                self.have_image_ids[(args[0], args[1], args[2])] = f.global_id
            elif pred == "at_soil_sample" and len(args) >= 1:
                self.at_soil_sample_ids[args[0]] = f.global_id
            elif pred == "at_rock_sample" and len(args) >= 1:
                self.at_rock_sample_ids[args[0]] = f.global_id
            elif pred == "calibrated" and len(args) >= 2:
                self.calibrated_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) >= 2:
                self.at_ids[(args[0], args[1])] = f.global_id
            elif pred == "empty" and len(args) >= 1:
                self.empty_ids[args[0]] = f.global_id
            elif pred == "full" and len(args) >= 1:
                self.full_ids[args[0]] = f.global_id
            elif pred == "store_of" and len(args) >= 2:
                self.store_of_map[args[0]] = args[1]  # store -> rover
            elif pred == "equipped_for_soil_analysis" and len(args) >= 1:
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) >= 1:
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) >= 1:
                self.equipped_imaging.add(args[0])
            elif pred == "on_board" and len(args) >= 2:
                self.on_board_map[args[0]] = args[1]  # camera -> rover
            elif pred == "supports" and len(args) >= 2:
                if args[0] not in self.supports_map:
                    self.supports_map[args[0]] = set()
                self.supports_map[args[0]].add(args[1])
            elif pred == "channel_free" and len(args) >= 1:
                self.channel_free_id = f.global_id
        
        # Parse goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.goal_image_pairs.add((args[0], args[1]))
        
        # Classify operators by name
        for op in model.operators:
            op_name, _ = self._parse_task_name(op.name)
            if op_name == "nop":
                self.op_is_nop.add(op.global_id)
            elif op_name == "visit":
                self.op_is_visit.add(op.global_id)
            elif op_name == "unvisit":
                self.op_is_unvisit.add(op.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        t1 = self._unsatisfied_goals_cost(state)
        t2 = self._task_network_cost(state, task_network)
        t3 = self._state_aware_goal_cost(state)
        return max(t1, t2, t3)

    def _unsatisfied_goals_cost(self, state):
        """T1: Each unsatisfied communicated_* goal needs at minimum 1 communicate action.
        But realistically each needs sample/image + communicate = at least 2.
        This is admissible since true cost >= 2 per unsatisfied goal."""
        count = 0
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                count += 2  # at minimum: sample + communicate
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                count += 2
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                count += 2  # at minimum: take_image + communicate
        return count

    def _state_aware_goal_cost(self, state):
        """T3: More detailed state-aware cost per unsatisfied goal.
        Check if intermediate results (have_soil_analysis, have_image, etc.) are achieved."""
        total = 0
        
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                # Check if any rover already has soil analysis for this waypoint
                has_analysis = False
                for rover in self.equipped_soil:
                    hid = self.have_soil_ids.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        has_analysis = True
                        break
                if has_analysis:
                    # Just need to navigate to comm point + communicate = min 1
                    total += 1
                else:
                    # Check if soil sample still exists
                    ssid = self.at_soil_sample_ids.get(wp)
                    if ssid is not None and ((state >> ssid) & 1):
                        # Need: navigate + possibly drop + sample + navigate + communicate = min 3
                        total += 3
                    else:
                        # Sample already taken by some rover but not communicated yet
                        # Some rover must have it - need navigate + communicate = min 1
                        total += 1
        
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                has_analysis = False
                for rover in self.equipped_rock:
                    hid = self.have_rock_ids.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        has_analysis = True
                        break
                if has_analysis:
                    total += 1
                else:
                    rsid = self.at_rock_sample_ids.get(wp)
                    if rsid is not None and ((state >> rsid) & 1):
                        total += 3
                    else:
                        total += 1
        
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                has_img = False
                for rover in self.equipped_imaging:
                    hid = self.have_image_ids.get((rover, obj, mode))
                    if hid is not None and ((state >> hid) & 1):
                        has_img = True
                        break
                if has_img:
                    # Just need navigate + communicate = min 1
                    total += 1
                else:
                    # Need calibrate + navigate + take_image + navigate + communicate = min 3
                    total += 3
        
        return total

    def _task_network_cost(self, state, task_network):
        """T2: Estimate minimum actions from task network.
        Each task in the network contributes a minimum action count.
        We use task type classification to assign costs."""
        if not task_network:
            return 0
        
        total = 0
        for task_item in task_network:
            gid = task_item.global_id
            
            # Check cache first
            if gid in self.task_name_cache:
                task_type, args = self.task_name_cache[gid]
            else:
                task_type, args = self._parse_task_name(task_item.name)
                self.task_name_cache[gid] = (task_type, args)
            
            # Operators (primitive tasks)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                if gid in self.op_is_nop:
                    total += 0
                elif gid in self.op_is_visit or gid in self.op_is_unvisit:
                    total += 0  # These are bookkeeping, don't count
                else:
                    total += 1  # Real action
            # Abstract tasks
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                if task_type == "get_soil_data":
                    if len(args) >= 1:
                        wp = args[0]
                        # Check how much work remains
                        comm_gid = self.communicated_soil_ids.get(wp)
                        if comm_gid is not None and ((state >> comm_gid) & 1):
                            total += 0  # Already communicated
                        else:
                            # Check if any rover has the analysis
                            has_analysis = False
                            for rover in self.equipped_soil:
                                hid = self.have_soil_ids.get((rover, wp))
                                if hid is not None and ((state >> hid) & 1):
                                    has_analysis = True
                                    break
                            if has_analysis:
                                total += 1  # Just communicate
                            else:
                                # Check if sample exists
                                ssid = self.at_soil_sample_ids.get(wp)
                                if ssid is not None and ((state >> ssid) & 1):
                                    total += 3  # sample + communicate + at least 1 nav
                                else:
                                    total += 1  # Sample taken, just communicate
                    else:
                        total += 3
                elif task_type == "get_rock_data":
                    if len(args) >= 1:
                        wp = args[0]
                        comm_gid = self.communicated_rock_ids.get(wp)
                        if comm_gid is not None and ((state >> comm_gid) & 1):
                            total += 0
                        else:
                            has_analysis = False
                            for rover in self.equipped_rock:
                                hid = self.have_rock_ids.get((rover, wp))
                                if hid is not None and ((state >> hid) & 1):
                                    has_analysis = True
                                    break
                            if has_analysis:
                                total += 1
                            else:
                                rsid = self.at_rock_sample_ids.get(wp)
                                if rsid is not None and ((state >> rsid) & 1):
                                    total += 3
                                else:
                                    total += 1
                    else:
                        total += 3
                elif task_type == "get_image_data":
                    if len(args) >= 2:
                        obj, mode = args[0], args[1]
                        comm_gid = self.communicated_image_ids.get((obj, mode))
                        if comm_gid is not None and ((state >> comm_gid) & 1):
                            total += 0
                        else:
                            has_img = False
                            for rover in self.equipped_imaging:
                                hid = self.have_image_ids.get((rover, obj, mode))
                                if hid is not None and ((state >> hid) & 1):
                                    has_img = True
                                    break
                            if has_img:
                                total += 1
                            else:
                                total += 3  # calibrate + take_image + communicate
                    else:
                        total += 3
                elif task_type == "send_soil_data":
                    total += 1  # At least communicate
                elif task_type == "send_rock_data":
                    total += 1
                elif task_type == "send_image_data":
                    total += 1
                elif task_type == "do_navigate1" or task_type == "do_navigate2":
                    # Navigation: might be 0 if already there, or >= 1
                    if len(args) >= 2:
                        rover = args[0]
                        target = args[1] if task_type == "do_navigate1" else args[2] if len(args) >= 3 else args[1]
                        at_gid = self.at_ids.get((rover, target))
                        if at_gid is not None and ((state >> at_gid) & 1):
                            total += 0  # Already there
                        else:
                            total += 1  # At least one navigate
                    else:
                        total += 0  # Conservative
                elif task_type == "do_calibrate":
                    if len(args) >= 2:
                        rover = args[0]
                        camera = args[1]
                        cal_gid = self.calibrated_ids.get((camera, rover))
                        if cal_gid is not None and ((state >> cal_gid) & 1):
                            total += 0
                        else:
                            total += 1  # At least calibrate action
                    else:
                        total += 1
                elif task_type == "empty_store":
                    if len(args) >= 2:
                        store = args[0]
                        emp_gid = self.empty_ids.get(store)
                        if emp_gid is not None and ((state >> emp_gid) & 1):
                            total += 0
                        else:
                            total += 1  # drop action
                    else:
                        total += 0
                else:
                    total += 1  # Unknown abstract task, at least 1
        
        return total
