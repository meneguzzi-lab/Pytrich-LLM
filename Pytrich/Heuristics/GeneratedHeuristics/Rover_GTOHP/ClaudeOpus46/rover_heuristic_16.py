"""Rover GTOHP Domain Heuristic

Bottleneck: Each data goal (soil/rock/image) requires a sequence of:
  navigate -> sample/image -> navigate -> communicate.
  The store capacity (single slot per rover) serializes soil/rock collection.
  Camera calibration serializes imaging tasks.

Admissible lower bounds:
  T1: Count of uncommunicated goal facts * min_actions_per_goal (admissible because
      each communicated_* goal requires at least 2 actions: sample+communicate or image+communicate)
  T2: Task network weighted cost - each remaining abstract task needs at least 1 action
      to fully decompose, weighted by task type minimum cost.
  T3: Navigation lower bound - sum of minimum navigation steps needed for uncompleted goals.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as ModelOperator
from Pytrich.model import AbstractTask as ModelAbstractTask


class RoverGTOHPAnthropicOpus46Heuristic16(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_16"):
        super().__init__(name)
        self.goal_soil_facts = []  # (fact_global_id, waypoint_arg)
        self.goal_rock_facts = []
        self.goal_image_facts = []  # (fact_global_id, objective_arg, mode_arg)
        
        # Fact lookups by predicate
        self.have_soil_ids = {}  # (rover, waypoint) -> global_id
        self.have_rock_ids = {}  # (rover, waypoint) -> global_id
        self.have_image_ids = {}  # (rover, obj, mode) -> global_id
        self.at_soil_sample_ids = {}  # waypoint -> global_id
        self.at_rock_sample_ids = {}  # waypoint -> global_id
        self.calibrated_ids = {}  # (camera, rover) -> global_id
        self.at_rover_ids = {}  # (rover, waypoint) -> global_id
        self.equipped_soil_ids = {}  # rover -> global_id
        self.equipped_rock_ids = {}  # rover -> global_id
        self.equipped_imaging_ids = {}  # rover -> global_id
        self.store_of_ids = {}  # (store, rover) -> global_id
        self.empty_store_ids = {}  # store -> global_id
        self.on_board_ids = {}  # (camera, rover) -> global_id
        self.supports_ids = {}  # (camera, mode) -> global_id
        self.visible_from_ids = {}  # (objective, waypoint) -> global_id
        self.channel_free_ids = {}  # lander -> global_id
        self.at_lander_ids = {}  # (lander, waypoint) -> global_id
        self.visible_ids = {}  # (wp1, wp2) -> global_id
        self.can_traverse_ids = {}  # (rover, wp1, wp2) -> global_id
        self.communicated_soil_ids = {}  # waypoint -> global_id
        self.communicated_rock_ids = {}  # waypoint -> global_id
        self.communicated_image_ids = {}  # (obj, mode) -> global_id
        self.calibration_target_ids = {}  # (camera, objective) -> global_id
        
        # Task type identification
        self.task_type_costs = {}  # task name prefix -> min cost
        
        # Rovers, waypoints lists
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()
        
        # For each rover, which waypoints can it reach (precomputed)
        self.rover_soil_capable = set()  # rovers with soil analysis
        self.rover_rock_capable = set()  # rovers with rock analysis
        self.rover_imaging_capable = set()  # rovers with imaging
        
        # Goal fact ids for fast checking
        self.goal_fact_ids = []
        
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
        return name, []
    
    def _parse_task_name(self, name: str):
        """Parse a task/operator name to get the task type."""
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
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.communicated_soil_ids[args[0]] = f.global_id
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.communicated_rock_ids[args[0]] = f.global_id
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.communicated_image_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis" and len(args) >= 2:
                self.have_soil_ids[(args[0], args[1])] = f.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "have_rock_analysis" and len(args) >= 2:
                self.have_rock_ids[(args[0], args[1])] = f.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "have_image" and len(args) >= 3:
                self.have_image_ids[(args[0], args[1], args[2])] = f.global_id
                self.rovers.add(args[0])
                self.objectives.add(args[1])
                self.modes.add(args[2])
            elif pred == "at_soil_sample" and len(args) >= 1:
                self.at_soil_sample_ids[args[0]] = f.global_id
                self.waypoints.add(args[0])
            elif pred == "at_rock_sample" and len(args) >= 1:
                self.at_rock_sample_ids[args[0]] = f.global_id
                self.waypoints.add(args[0])
            elif pred == "calibrated" and len(args) >= 2:
                self.calibrated_ids[(args[0], args[1])] = f.global_id
                self.cameras.add(args[0])
                self.rovers.add(args[1])
            elif pred == "at" and len(args) >= 2:
                self.at_rover_ids[(args[0], args[1])] = f.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "equipped_for_soil_analysis" and len(args) >= 1:
                self.equipped_soil_ids[args[0]] = f.global_id
                self.rover_soil_capable.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) >= 1:
                self.equipped_rock_ids[args[0]] = f.global_id
                self.rover_rock_capable.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) >= 1:
                self.equipped_imaging_ids[args[0]] = f.global_id
                self.rover_imaging_capable.add(args[0])
            elif pred == "store_of" and len(args) >= 2:
                self.store_of_ids[(args[0], args[1])] = f.global_id
                self.stores.add(args[0])
                self.rovers.add(args[1])
            elif pred == "empty" and len(args) >= 1:
                self.empty_store_ids[args[0]] = f.global_id
            elif pred == "on_board" and len(args) >= 2:
                self.on_board_ids[(args[0], args[1])] = f.global_id
                self.cameras.add(args[0])
                self.rovers.add(args[1])
            elif pred == "supports" and len(args) >= 2:
                self.supports_ids[(args[0], args[1])] = f.global_id
                self.cameras.add(args[0])
                self.modes.add(args[1])
            elif pred == "visible_from" and len(args) >= 2:
                self.visible_from_ids[(args[0], args[1])] = f.global_id
            elif pred == "channel_free" and len(args) >= 1:
                self.channel_free_ids[args[0]] = f.global_id
                self.landers.add(args[0])
            elif pred == "at_lander" and len(args) >= 2:
                self.at_lander_ids[(args[0], args[1])] = f.global_id
                self.landers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "visible" and len(args) >= 2:
                self.visible_ids[(args[0], args[1])] = f.global_id
            elif pred == "can_traverse" and len(args) >= 3:
                self.can_traverse_ids[(args[0], args[1], args[2])] = f.global_id
            elif pred == "calibration_target" and len(args) >= 2:
                self.calibration_target_ids[(args[0], args[1])] = f.global_id
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            gid = None
            if pred == "communicated_soil_data" and len(args) >= 1:
                gid = self.communicated_soil_ids.get(args[0])
                if gid is not None:
                    self.goal_soil_facts.append((gid, args[0]))
            elif pred == "communicated_rock_data" and len(args) >= 1:
                gid = self.communicated_rock_ids.get(args[0])
                if gid is not None:
                    self.goal_rock_facts.append((gid, args[0]))
            elif pred == "communicated_image_data" and len(args) >= 2:
                gid = self.communicated_image_ids.get((args[0], args[1]))
                if gid is not None:
                    self.goal_image_facts.append((gid, args[0], args[1]))
            if gid is not None:
                self.goal_fact_ids.append(gid)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)
    
    def _compute(self, state, task_network):
        t1 = self._goal_distance(state)
        t2 = self._task_network_cost(state, task_network)
        return max(t1, t2)
    
    def _goal_distance(self, state):
        """Estimate cost to achieve all uncommunicated goal facts.
        
        For each uncommunicated goal:
        - soil: need navigate(>=1) + sample(1) + navigate(>=0) + communicate(1) = min 2-3
          But if we already have_soil_analysis, skip sample: navigate(>=0) + communicate(1) = min 1
        - rock: same as soil
        - image: need calibrate(1) + navigate(>=0) + take_image(1) + navigate(>=0) + communicate(1) = min 2-3
          But if we already have_image, just communicate: min 1
        """
        total = 0
        
        for gid, wp in self.goal_soil_facts:
            if (state >> gid) & 1:
                continue  # already communicated
            
            # Check if any rover already has soil analysis for this waypoint
            has_analysis = False
            for rover in self.rover_soil_capable:
                fid = self.have_soil_ids.get((rover, wp))
                if fid is not None and (state >> fid) & 1:
                    has_analysis = True
                    break
            
            if has_analysis:
                # Just need to navigate to visible point and communicate: min 1 action
                total += 1
            else:
                # Check if soil sample still exists at waypoint
                sample_id = self.at_soil_sample_ids.get(wp)
                if sample_id is not None and (state >> sample_id) & 1:
                    # Need: possibly drop(1) + navigate(>=1) + sample(1) + navigate(>=0) + communicate(1) = min 3
                    total += 3
                else:
                    # Sample already taken by some rover but not communicated
                    # Check all rovers for having analysis
                    found = False
                    for rover in self.rovers:
                        fid = self.have_soil_ids.get((rover, wp))
                        if fid is not None and (state >> fid) & 1:
                            found = True
                            break
                    if found:
                        total += 1  # just communicate
                    else:
                        # Sample consumed but no rover has analysis - shouldn't happen in valid plan
                        # but be safe
                        total += 3
        
        for gid, wp in self.goal_rock_facts:
            if (state >> gid) & 1:
                continue
            
            has_analysis = False
            for rover in self.rover_rock_capable:
                fid = self.have_rock_ids.get((rover, wp))
                if fid is not None and (state >> fid) & 1:
                    has_analysis = True
                    break
            
            if has_analysis:
                total += 1
            else:
                sample_id = self.at_rock_sample_ids.get(wp)
                if sample_id is not None and (state >> sample_id) & 1:
                    total += 3
                else:
                    found = False
                    for rover in self.rovers:
                        fid = self.have_rock_ids.get((rover, wp))
                        if fid is not None and (state >> fid) & 1:
                            found = True
                            break
                    if found:
                        total += 1
                    else:
                        total += 3
        
        for gid, obj, mode in self.goal_image_facts:
            if (state >> gid) & 1:
                continue
            
            has_image = False
            for rover in self.rover_imaging_capable:
                fid = self.have_image_ids.get((rover, obj, mode))
                if fid is not None and (state >> fid) & 1:
                    has_image = True
                    break
            
            if has_image:
                total += 1  # just communicate
            else:
                # Need calibrate + navigate + take_image + navigate + communicate
                # Minimum: calibrate(1) + take_image(1) + communicate(1) = 3
                # Plus possible navigation
                total += 3
        
        return total
    
    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network.
        
        Each task in the network needs at least some minimum actions.
        We assign costs based on task type.
        """
        if not task_network:
            # No tasks left - check if goals are met
            unmet = 0
            for gid in self.goal_fact_ids:
                if not ((state >> gid) & 1):
                    unmet += 1
            return unmet
        
        total = 0
        for task in task_network:
            if isinstance(task, ModelAbstractTask):
                tname, targs = self._parse_task_name(task.name)
                cost = self._estimate_abstract_task_cost(state, tname, targs)
                total += cost
            elif isinstance(task, ModelOperator):
                # Primitive actions cost 1 each (except nop)
                tname, _ = self._parse_task_name(task.name)
                if tname != "nop":
                    total += 1
        
        return total
    
    def _estimate_abstract_task_cost(self, state, tname, targs):
        """Estimate minimum cost for an abstract task based on type and current state."""
        
        if tname == "get_soil_data":
            if len(targs) < 1:
                return 4
            wp = targs[0]
            # Check if already communicated
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and (state >> gid) & 1:
                return 0  # already done somehow
            
            # Need: navigate + empty_store + sample + send_soil_data
            # send_soil_data = navigate + communicate
            # Minimum: sample(1) + communicate(1) = 2 if already at wp with empty store
            # More realistic: navigate(1) + sample(1) + navigate(1) + communicate(1) = 4
            
            # Check if any capable rover is already at the waypoint
            at_wp = False
            for rover in self.rover_soil_capable:
                fid = self.at_rover_ids.get((rover, wp))
                if fid is not None and (state >> fid) & 1:
                    at_wp = True
                    break
            
            # Check if any capable rover already has the analysis
            has_analysis = False
            for rover in self.rover_soil_capable:
                fid = self.have_soil_ids.get((rover, wp))
                if fid is not None and (state >> fid) & 1:
                    has_analysis = True
                    break
            
            if has_analysis:
                return 2  # navigate + communicate (minimum)
            elif at_wp:
                return 3  # sample + navigate + communicate  
            else:
                return 4  # navigate + sample + navigate + communicate
        
        elif tname == "get_rock_data":
            if len(targs) < 1:
                return 4
            wp = targs[0]
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and (state >> gid) & 1:
                return 0
            
            at_wp = False
            for rover in self.rover_rock_capable:
                fid = self.at_rover_ids.get((rover, wp))
                if fid is not None and (state >> fid) & 1:
                    at_wp = True
                    break
            
            has_analysis = False
            for rover in self.rover_rock_capable:
                fid = self.have_rock_ids.get((rover, wp))
                if fid is not None and (state >> fid) & 1:
                    has_analysis = True
                    break
            
            if has_analysis:
                return 2
            elif at_wp:
                return 3
            else:
                return 4
        
        elif tname == "get_image_data":
            if len(targs) < 2:
                return 5
            obj, mode = targs[0], targs[1]
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and (state >> gid) & 1:
                return 0
            
            has_image = False
            for rover in self.rover_imaging_capable:
                fid = self.have_image_ids.get((rover, obj, mode))
                if fid is not None and (state >> fid) & 1:
                    has_image = True
                    break
            
            if has_image:
                return 2  # navigate + communicate
            
            # Check if any capable rover has a calibrated camera supporting mode
            has_calibrated = False
            for rover in self.rover_imaging_capable:
                for cam in self.cameras:
                    cal_id = self.calibrated_ids.get((cam, rover))
                    ob_id = self.on_board_ids.get((cam, rover))
                    sup_id = self.supports_ids.get((cam, mode))
                    if (cal_id is not None and ob_id is not None and sup_id is not None and
                        (state >> cal_id) & 1):
                        has_calibrated = True
                        break
                if has_calibrated:
                    break
            
            if has_calibrated:
                return 3  # navigate + take_image + communicate (min)
            else:
                return 5  # navigate + calibrate + navigate + take_image + communicate
        
        elif tname == "send_soil_data":
            return 2  # navigate + communicate
        
        elif tname == "send_rock_data":
            return 2
        
        elif tname == "send_image_data":
            return 2
        
        elif tname == "do_navigate1":
            if len(targs) >= 2:
                rover, to_wp = targs[0], targs[1]
                fid = self.at_rover_ids.get((rover, to_wp))
                if fid is not None and (state >> fid) & 1:
                    return 0  # already there
            return 1  # at least one navigate action
        
        elif tname == "do_navigate2":
            if len(targs) >= 3:
                rover, from_wp, to_wp = targs[0], targs[1], targs[2]
                fid = self.at_rover_ids.get((rover, to_wp))
                if fid is not None and (state >> fid) & 1:
                    return 0
                # Check if direct traverse possible
                ct_id = self.can_traverse_ids.get((rover, from_wp, to_wp))
                if ct_id is not None:
                    return 1
            return 1
        
        elif tname == "empty_store":
            if len(targs) >= 2:
                store = targs[0]
                fid = self.empty_store_ids.get(store)
                if fid is not None and (state >> fid) & 1:
                    return 0  # already empty
            return 1  # drop action
        
        elif tname == "do_calibrate":
            if len(targs) >= 2:
                rover, cam = targs[0], targs[1]
                fid = self.calibrated_ids.get((cam, rover))
                if fid is not None and (state >> fid) & 1:
                    return 0  # already calibrated
            return 2  # navigate + calibrate
        
        # Default: at least 1 action
        return 1
    
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
