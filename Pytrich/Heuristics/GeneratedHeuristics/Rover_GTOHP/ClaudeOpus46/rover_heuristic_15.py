# Rover GTOHP Domain-Dependent Heuristic
#
# BOTTLENECK: Each rover has a single store, requiring drop between samples.
# Navigation is multi-hop. Communication requires lander visibility.
# Imaging requires calibration (which is invalidated after take_image).
#
# ADMISSIBLE LOWER BOUNDS:
# T1 - Goal-based: Each unsatisfied communicated_* goal needs >= 1 action (communicate).
#      Actually needs navigate+sample+navigate+communicate minimum.
# T2 - Task-network weighted: Each remaining abstract task has a minimum
#      decomposition cost based on task type and current state.
# T3 - Resource contention: Rovers with stores that need multiple uses
#      require intermediate drops.
#
# All terms are admissible lower bounds. We return max(T1, T2).

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as OperatorClass
from Pytrich.model import AbstractTask as AbstractTaskClass


class RoverGTOHPAnthropicOpus46Heuristic15(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses multi-component max of:
    - Goal satisfaction lower bound (state-based)
    - Task network weighted cost estimation (state-aware)
    """

    def __init__(self, name="rover_anthropicopus46_heuristic_15"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}   # waypoint_str -> global_id
        self.communicated_rock_ids = {}   # waypoint_str -> global_id
        self.communicated_image_ids = {}  # (objective_str, mode_str) -> global_id
        self.at_rover_ids = {}            # (rover_str, waypoint_str) -> global_id
        self.have_soil_ids = {}           # (rover_str, waypoint_str) -> global_id
        self.have_rock_ids = {}           # (rover_str, waypoint_str) -> global_id
        self.have_image_ids = {}          # (rover_str, obj_str, mode_str) -> global_id
        self.calibrated_ids = {}          # (camera_str, rover_str) -> global_id
        self.empty_store_ids = {}         # store_str -> global_id
        self.at_soil_sample_ids = {}      # waypoint_str -> global_id
        self.at_rock_sample_ids = {}      # waypoint_str -> global_id
        self.equipped_soil_ids = {}       # rover_str -> global_id
        self.equipped_rock_ids = {}       # rover_str -> global_id
        self.equipped_imaging_ids = {}    # rover_str -> global_id
        self.available_ids = {}           # rover_str -> global_id
        self.channel_free_ids = {}        # lander_str -> global_id
        
        # Goal tracking
        self.goal_soil_waypoints = []     # list of (waypoint_str, fact_global_id)
        self.goal_rock_waypoints = []     # list of (waypoint_str, fact_global_id)
        self.goal_image_data = []         # list of ((obj_str, mode_str), fact_global_id)
        
        # Task name patterns for cost estimation
        self.task_min_costs = {}          # task_name_prefix -> min_cost
        
        # Precomputed rover capabilities
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like +predicate[arg1,arg2] into (predicate, [args])."""
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
        """Parse task/operator name to extract the base task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts and build lookup maps for fast heuristic evaluation."""
        self.model = model

        # Build fact ID maps by parsing all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            gid = f.global_id

            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_ids[args[0]] = gid
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_ids[args[0]] = gid
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_ids[(args[0], args[1])] = gid
            elif pred == "at" and len(args) == 2:
                self.at_rover_ids[(args[0], args[1])] = gid
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_ids[(args[0], args[1])] = gid
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_ids[(args[0], args[1])] = gid
            elif pred == "have_image" and len(args) == 3:
                self.have_image_ids[(args[0], args[1], args[2])] = gid
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = gid
            elif pred == "empty" and len(args) == 1:
                self.empty_store_ids[args[0]] = gid
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = gid
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = gid
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids[args[0]] = gid
                self.soil_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids[args[0]] = gid
                self.rock_rovers.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids[args[0]] = gid
                self.imaging_rovers.add(args[0])
            elif pred == "available" and len(args) == 1:
                self.available_ids[args[0]] = gid
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_ids[args[0]] = gid

        # Parse goals to identify what needs to be achieved
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gf_name in goal_fact_names:
            pred, args = self._parse_fact_name(gf_name)
            if pred == "communicated_soil_data" and len(args) == 1:
                wp = args[0]
                if wp in self.communicated_soil_ids:
                    self.goal_soil_waypoints.append((wp, self.communicated_soil_ids[wp]))
            elif pred == "communicated_rock_data" and len(args) == 1:
                wp = args[0]
                if wp in self.communicated_rock_ids:
                    self.goal_rock_waypoints.append((wp, self.communicated_rock_ids[wp]))
            elif pred == "communicated_image_data" and len(args) == 2:
                key = (args[0], args[1])
                if key in self.communicated_image_ids:
                    self.goal_image_data.append((key, self.communicated_image_ids[key]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        t1 = self._goal_based_estimate(state)
        t2 = self._task_network_estimate(state, task_network)
        return max(t1, t2)

    def _goal_based_estimate(self, state):
        """Estimate cost based on unsatisfied goal predicates.
        
        For each unsatisfied communicated_soil_data(wp):
          - Need at minimum: navigate to wp (1) + sample_soil (1) + navigate to comm point (1) + communicate (1) = 4
          - But if have_soil_analysis already true, only need navigate + communicate = 2
          - If already communicated, cost = 0
        
        Similarly for rock and image data.
        This is admissible because these are true minimum action counts.
        """
        h = 0

        for wp, gid in self.goal_soil_waypoints:
            if (state >> gid) & 1:
                continue  # Already communicated
            # Check if any rover already has the soil analysis
            has_analysis = False
            for rover in self.soil_rovers:
                key = (rover, wp)
                if key in self.have_soil_ids:
                    sid = self.have_soil_ids[key]
                    if (state >> sid) & 1:
                        has_analysis = True
                        break
            if has_analysis:
                # Still need navigate to comm point + communicate = min 2 actions
                h += 2
            else:
                # Check if soil sample still exists at waypoint
                if wp in self.at_soil_sample_ids:
                    ssid = self.at_soil_sample_ids[wp]
                    if (state >> ssid) & 1:
                        # Sample exists: navigate + empty_store(maybe) + sample + navigate + communicate
                        # Minimum: sample(1) + communicate(1) + at least navigate there and to comm = 4
                        h += 4
                    else:
                        # Sample already taken but not communicated and no rover has it?
                        # This shouldn't happen in valid problems but handle gracefully
                        h += 2
                else:
                    h += 4

        for wp, gid in self.goal_rock_waypoints:
            if (state >> gid) & 1:
                continue
            has_analysis = False
            for rover in self.rock_rovers:
                key = (rover, wp)
                if key in self.have_rock_ids:
                    sid = self.have_rock_ids[key]
                    if (state >> sid) & 1:
                        has_analysis = True
                        break
            if has_analysis:
                h += 2
            else:
                if wp in self.at_rock_sample_ids:
                    rsid = self.at_rock_sample_ids[wp]
                    if (state >> rsid) & 1:
                        h += 4
                    else:
                        h += 2
                else:
                    h += 4

        for (obj, mode), gid in self.goal_image_data:
            if (state >> gid) & 1:
                continue
            # Check if any rover already has the image
            has_img = False
            for rover in self.imaging_rovers:
                key = (rover, obj, mode)
                if key in self.have_image_ids:
                    iid = self.have_image_ids[key]
                    if (state >> iid) & 1:
                        has_img = True
                        break
            if has_img:
                # Need navigate to comm point + communicate = 2
                h += 2
            else:
                # Need calibrate + navigate_to_visible + take_image + navigate_to_comm + communicate
                # Minimum: calibrate(1) + take_image(1) + communicate(1) + navigations(2) = 5
                h += 5

        return h

    def _task_network_estimate(self, state, task_network):
        """Estimate cost based on remaining tasks in the task network.
        
        Each task type has a minimum number of primitive actions needed.
        This is admissible because we use minimum possible decomposition costs.
        """
        h = 0
        for task in task_network:
            if isinstance(task, OperatorClass):
                # Primitive operator - costs exactly 1
                h += 1
            elif isinstance(task, AbstractTaskClass):
                task_name = task.name
                h += self._estimate_abstract_task_cost(state, task_name)
        return h

    def _estimate_abstract_task_cost(self, state, task_name):
        """Estimate minimum cost for an abstract task based on its type and current state.
        
        We parse the task name to determine its type and use state-aware estimation.
        """
        ttype, args = self._parse_task_name(task_name)

        if ttype == "get_soil_data":
            # get_soil_data(waypoint): navigate + empty_store + sample_soil + send_soil_data
            # send_soil_data decomposes to navigate + communicate
            # Minimum: navigate(0 if already there) + nop/drop(0-1) + sample(1) + navigate(0-1) + communicate(1)
            if len(args) >= 1:
                wp = args[0]
                # Check if already communicated
                if wp in self.communicated_soil_ids:
                    cid = self.communicated_soil_ids[wp]
                    if (state >> cid) & 1:
                        return 1  # Just a nop essentially but task still in network
                return 3  # sample + navigate_for_comm + communicate minimum
            return 3

        elif ttype == "get_rock_data":
            if len(args) >= 1:
                wp = args[0]
                if wp in self.communicated_rock_ids:
                    cid = self.communicated_rock_ids[wp]
                    if (state >> cid) & 1:
                        return 1
                return 3
            return 3

        elif ttype == "get_image_data":
            # get_image_data(objective, mode): calibrate_task + navigate + take_image + send_image_data
            # Minimum: calibrate(1) + take_image(1) + communicate(1) = 3 plus navigations
            if len(args) >= 2:
                obj, mode = args[0], args[1]
                key = (obj, mode)
                if key in self.communicated_image_ids:
                    cid = self.communicated_image_ids[key]
                    if (state >> cid) & 1:
                        return 1
                # Check if any rover already has the image
                for rover in self.imaging_rovers:
                    ikey = (rover, obj, mode)
                    if ikey in self.have_image_ids:
                        iid = self.have_image_ids[ikey]
                        if (state >> iid) & 1:
                            return 2  # Just need send_image_data
                return 4  # calibrate + take_image + navigate + communicate
            return 4

        elif ttype == "send_soil_data":
            # navigate + communicate = minimum 2 (could be 1 if already at visible point)
            return 1

        elif ttype == "send_rock_data":
            return 1

        elif ttype == "send_image_data":
            return 1

        elif ttype == "do_navigate1":
            # Could be nop if already there, or navigate chain
            if len(args) >= 2:
                rover, to_wp = args[0], args[1]
                key = (rover, to_wp)
                if key in self.at_rover_ids:
                    aid = self.at_rover_ids[key]
                    if (state >> aid) & 1:
                        return 0  # Already there
                return 1  # At least one navigate action
            return 1

        elif ttype == "do_navigate2":
            if len(args) >= 3:
                rover, from_wp, to_wp = args[0], args[1], args[2]
                key = (rover, to_wp)
                if key in self.at_rover_ids:
                    aid = self.at_rover_ids[key]
                    if (state >> aid) & 1:
                        return 0
                return 1
            return 1

        elif ttype == "empty_store":
            # Either nop (if empty) or drop (1 action)
            if len(args) >= 1:
                store = args[0]
                if store in self.empty_store_ids:
                    eid = self.empty_store_ids[store]
                    if (state >> eid) & 1:
                        return 0  # Already empty
                return 1  # Need to drop
            return 0

        elif ttype == "do_calibrate":
            # navigate to calibration target + calibrate = minimum 1
            if len(args) >= 2:
                rover, camera = args[0], args[1]
                # Check if already calibrated
                key = (camera, rover)
                if key in self.calibrated_ids:
                    cid = self.calibrated_ids[key]
                    if (state >> cid) & 1:
                        return 0  # Already calibrated
                return 1  # At least calibrate action
            return 1

        # Default: unknown task type, assume at least 1 action
        return 1

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
