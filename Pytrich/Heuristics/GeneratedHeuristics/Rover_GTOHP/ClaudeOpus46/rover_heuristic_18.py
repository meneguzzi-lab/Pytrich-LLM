"""Rover GTOHP Domain-Dependent Heuristic

Bottleneck: Navigation between waypoints serializes all work. Each data goal
(soil/rock/image) requires navigate-sample-navigate-communicate, and the single
store per rover forces sequential sample/drop cycles.

Admissible lower bounds:
1. Communication bound: each uncommunicated goal needs >= 1 action (communicate)
2. Sample+communicate bound: each unsampled soil/rock needs >= 2 actions (sample + communicate)
3. Image pipeline bound: each uncaptured image needs >= 3 actions (calibrate + take_image + communicate),
   plus navigation steps
4. Task network structural bound: minimum actions from remaining abstract tasks
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic18(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_18"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}    # waypoint -> global_id
        self.communicated_rock_ids = {}    # waypoint -> global_id
        self.communicated_image_ids = {}   # (objective, mode) -> global_id
        self.have_soil_ids = {}            # (rover, waypoint) -> global_id
        self.have_rock_ids = {}            # (rover, waypoint) -> global_id
        self.have_image_ids = {}           # (rover, objective, mode) -> global_id
        self.calibrated_ids = {}           # (camera, rover) -> global_id
        self.at_soil_sample_ids = {}       # waypoint -> global_id
        self.at_rock_sample_ids = {}       # waypoint -> global_id
        self.at_rover_ids = {}             # (rover, waypoint) -> global_id
        self.empty_store_ids = {}          # store -> global_id
        self.full_store_ids = {}           # store -> global_id
        self.store_of_ids = {}             # (store, rover) -> global_id
        self.equipped_soil_ids = {}        # rover -> global_id
        self.equipped_rock_ids = {}        # rover -> global_id
        self.equipped_imaging_ids = {}     # rover -> global_id
        self.on_board_ids = {}             # (camera, rover) -> global_id
        self.supports_ids = {}             # (camera, mode) -> global_id
        self.visible_from_ids = {}         # (objective, waypoint) -> global_id
        self.calibration_target_ids = {}   # (camera, objective) -> global_id
        self.channel_free_ids = {}         # lander -> global_id
        self.available_ids = {}            # rover -> global_id
        
        # Goal structures
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_images = set()  # (objective, mode)
        
        # Rover capabilities
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        
        # Camera info
        self.rover_cameras = {}  # rover -> list of cameras
        self.camera_modes = {}   # camera -> set of modes
        self.camera_cal_target = {}  # camera -> objective
        
        # Abstract task type mapping
        self.task_type_map = {}  # global_id -> (type_name, estimated_min_cost)
        
        # Operator name map
        self.op_navigate_ids = set()
        self.op_sample_soil_ids = set()
        self.op_sample_rock_ids = set()
        self.op_communicate_soil_ids = set()
        self.op_communicate_rock_ids = set()
        self.op_communicate_image_ids = set()
        self.op_calibrate_ids = set()
        self.op_take_image_ids = set()
        self.op_drop_ids = set()
        self.op_nop_ids = set()

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

    def _parse_component_name(self, name: str):
        """Parse operator/task names like 'navigate[rover0,waypoint1,waypoint2]'"""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            pred = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return pred, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
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
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_ids[(args[0], args[1])] = gid
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_ids[(args[0], args[1])] = gid
            elif pred == "have_image" and len(args) == 3:
                self.have_image_ids[(args[0], args[1], args[2])] = gid
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = gid
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = gid
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = gid
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
                self.camera_cal_target[args[0]] = args[1]
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_ids[args[0]] = gid
            elif pred == "available" and len(args) == 1:
                self.available_ids[args[0]] = gid
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.goal_images.add((args[0], args[1]))
        
        # Identify rover capabilities from facts
        for rover_name in self.equipped_soil_ids:
            self.soil_rovers.add(rover_name)
        for rover_name in self.equipped_rock_ids:
            self.rock_rovers.add(rover_name)
        for rover_name in self.equipped_imaging_ids:
            self.imaging_rovers.add(rover_name)
        
        # Build camera info
        for (camera, rover), gid in self.on_board_ids.items():
            if rover not in self.rover_cameras:
                self.rover_cameras[rover] = []
            self.rover_cameras[rover].append(camera)
        for (camera, mode), gid in self.supports_ids.items():
            if camera not in self.camera_modes:
                self.camera_modes[camera] = set()
            self.camera_modes[camera].add(mode)
        
        # Classify abstract tasks by type and estimate minimum cost
        for at in model.abstract_tasks:
            task_name, task_args = self._parse_component_name(at.name)
            self.task_type_map[at.global_id] = task_name
        
        # Classify operators
        for op in model.operators:
            op_name, op_args = self._parse_component_name(op.name)
            if op_name == "navigate":
                self.op_navigate_ids.add(op.global_id)
            elif op_name == "sample_soil":
                self.op_sample_soil_ids.add(op.global_id)
            elif op_name == "sample_rock":
                self.op_sample_rock_ids.add(op.global_id)
            elif op_name in ("communicate_soil_data1", "communicate_soil_data2"):
                self.op_communicate_soil_ids.add(op.global_id)
            elif op_name in ("communicate_rock_data1", "communicate_rock_data2"):
                self.op_communicate_rock_ids.add(op.global_id)
            elif op_name == "communicate_image_data":
                self.op_communicate_image_ids.add(op.global_id)
            elif op_name == "calibrate":
                self.op_calibrate_ids.add(op.global_id)
            elif op_name == "take_image":
                self.op_take_image_ids.add(op.global_id)
            elif op_name == "drop":
                self.op_drop_ids.add(op.global_id)
            elif op_name == "nop":
                self.op_nop_ids.add(op.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_bit(self, state, global_id):
        return (state >> global_id) & 1

    def _compute_goal_based_h(self, state):
        """Compute heuristic based on what goal predicates are still unsatisfied.
        
        For each unsatisfied communicated_soil_data(wp):
          - If no rover has have_soil_analysis for wp: need sample(1) + navigate_to_wp(>=1) + navigate_to_comm(>=1) + communicate(1) = 4 min
          - If some rover has soil analysis: need navigate_to_comm(>=1) + communicate(1) = 2 min
          - Also if store not empty, need drop(1) extra
        
        Similar for rock and image data.
        """
        h = 0
        
        # Soil data goals
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and self._check_bit(state, gid):
                continue  # Already communicated
            
            # Check if any rover already has the soil analysis
            has_analysis = False
            for rover in self.soil_rovers:
                key = (rover, wp)
                if key in self.have_soil_ids:
                    if self._check_bit(state, self.have_soil_ids[key]):
                        has_analysis = True
                        break
            
            if has_analysis:
                # Need: navigate to comm point + communicate = at least 2
                h += 2
            else:
                # Check if soil sample still exists at waypoint
                sample_exists = True
                if wp in self.at_soil_sample_ids:
                    sample_exists = self._check_bit(state, self.at_soil_sample_ids[wp])
                
                if sample_exists:
                    # Need: navigate to wp (>=1) + possibly drop (0-1) + sample(1) + navigate to comm(>=1) + communicate(1) = 4+
                    h += 4
                else:
                    # Sample already taken by some rover but not communicated
                    # Some rover must have it, need navigate + communicate = 2
                    h += 2
        
        # Rock data goals
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and self._check_bit(state, gid):
                continue
            
            has_analysis = False
            for rover in self.rock_rovers:
                key = (rover, wp)
                if key in self.have_rock_ids:
                    if self._check_bit(state, self.have_rock_ids[key]):
                        has_analysis = True
                        break
            
            if has_analysis:
                h += 2
            else:
                sample_exists = True
                if wp in self.at_rock_sample_ids:
                    sample_exists = self._check_bit(state, self.at_rock_sample_ids[wp])
                
                if sample_exists:
                    h += 4
                else:
                    h += 2
        
        # Image data goals
        for (obj, mode) in self.goal_images:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and self._check_bit(state, gid):
                continue
            
            # Check if any rover has the image
            has_img = False
            for rover in self.imaging_rovers:
                key = (rover, obj, mode)
                if key in self.have_image_ids:
                    if self._check_bit(state, self.have_image_ids[key]):
                        has_img = True
                        break
            
            if has_img:
                # Need navigate to comm + communicate = 2
                h += 2
            else:
                # Need calibrate(1) + navigate to visible wp(>=1) + take_image(1) + navigate to comm(>=1) + communicate(1) = 5
                # Check if any suitable camera is already calibrated
                any_calibrated = False
                for rover in self.imaging_rovers:
                    if rover in self.rover_cameras:
                        for cam in self.rover_cameras[rover]:
                            if cam in self.camera_modes and mode in self.camera_modes[cam]:
                                cal_key = (cam, rover)
                                if cal_key in self.calibrated_ids:
                                    if self._check_bit(state, self.calibrated_ids[cal_key]):
                                        any_calibrated = True
                                        break
                    if any_calibrated:
                        break
                
                if any_calibrated:
                    # calibrate already done: navigate(1) + take_image(1) + navigate(1) + communicate(1) = 4
                    h += 4
                else:
                    # Full pipeline: navigate_cal(1) + calibrate(1) + navigate_img(1) + take_image(1) + navigate_comm(1) + communicate(1) = 6
                    h += 5
        
        return h

    def _compute_task_network_h(self, task_network):
        """Estimate minimum actions from remaining task network.
        
        Each abstract task contributes a minimum cost based on its type.
        Each operator (primitive task) contributes 1 (except nop which contributes 0).
        """
        h = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator (primitive)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                if gid not in self.op_nop_ids:
                    h += 1
            # Check if it's an abstract task
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                task_type = self.task_type_map.get(gid, "")
                if task_type == "get_soil_data":
                    # Minimum: navigate(nop possible) + empty_store(nop possible) + sample_soil + send_soil_data
                    # send_soil_data needs at least: navigate(nop possible) + communicate = 1
                    # So minimum is 2 (sample + communicate)
                    h += 2
                elif task_type == "get_rock_data":
                    h += 2
                elif task_type == "get_image_data":
                    # calibrate(subtask) + navigate + take_image + send_image_data
                    # Minimum: calibrate(1) + take_image(1) + communicate(1) = 3
                    h += 3
                elif task_type == "send_soil_data":
                    h += 1  # At least communicate
                elif task_type == "send_rock_data":
                    h += 1
                elif task_type == "send_image_data":
                    h += 1
                elif task_type == "do_navigate1":
                    # Could be nop if already there, or at least 1 navigate
                    h += 0  # Conservative: might already be there
                elif task_type == "do_navigate2":
                    h += 0
                elif task_type == "empty_store":
                    h += 0  # Could be nop if already empty
                elif task_type == "do_calibrate":
                    h += 1  # At least calibrate action
                else:
                    h += 1  # Unknown abstract task, at least 1 action
        
        return h

    def _compute_store_penalty(self, state):
        """Add penalty for store management overhead.
        
        If a rover needs to do multiple samples and its store is full,
        it needs extra drop actions.
        """
        penalty = 0
        # Count how many sampling tasks remain per rover type
        # For now, check if any rover's store is full - that means we need a drop
        for store_name in self.full_store_ids:
            fid = self.full_store_ids[store_name]
            if self._check_bit(state, fid):
                penalty += 1  # Will need at least one drop
        return penalty

    def _compute(self, state, task_network):
        """Main heuristic computation using max of multiple bounds."""
        if len(task_network) == 0:
            # Check if goals are satisfied
            if (self.model.goals & state) == self.model.goals:
                return 0
            return float('inf')
        
        # Term 1: Goal-based lower bound (how much work remains to satisfy goals)
        h_goal = self._compute_goal_based_h(state)
        
        # Term 2: Task network structural bound
        h_tn = self._compute_task_network_h(task_network)
        
        # Term 3: Store management penalty
        h_store = self._compute_store_penalty(state)
        
        # Use max for admissibility, add store penalty as it's orthogonal
        h = max(h_goal, h_tn) + h_store
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
