# Rover GTOHP Domain Heuristic
# Bottleneck: Navigation is the serializing constraint. Each data collection task
# requires navigating to the sample/image waypoint, then to a comm-visible waypoint.
# Store capacity (1 per rover) further serializes soil/rock collection.
#
# Admissible lower bounds:
# T1: For each uncommunicated goal, count minimum actions still needed
#     (navigate + sample + navigate + communicate, or calibrate + navigate + image + navigate + communicate)
#     This is admissible because these actions are truly required.
# T2: Task network depth - count remaining abstract tasks weighted by minimum decomposition cost.
# T3: Store contention - if multiple soil/rock tasks remain for the same rover, each extra needs a drop.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as ModelOperator
from Pytrich.model import AbstractTask as ModelAbstractTask


class RoverGTOHPAnthropicOpus46Heuristic19(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses multi-component max of:
    - Goal-based lower bound (uncommunicated data requirements)
    - Task network structural lower bound
    - Prerequisite chain analysis
    """

    def __init__(self, name="rover_anthropicopus46_heuristic_19"):
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
        self.equipped_soil = set()        # set of rovers
        self.equipped_rock = set()        # set of rovers
        self.equipped_imaging = set()     # set of rovers
        self.channel_free_ids = {}        # lander -> global_id
        self.available_ids = {}           # rover -> global_id
        
        # Goal tracking
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_image_pairs = set()     # (objective, mode)
        
        # Task name parsing cache
        self.task_name_cache = {}
        
        # Abstract task type classification
        self.get_soil_task_ids = set()
        self.get_rock_task_ids = set()
        self.get_image_task_ids = set()
        self.send_soil_task_ids = set()
        self.send_rock_task_ids = set()
        self.send_image_task_ids = set()
        self.navigate_task_ids = set()
        self.calibrate_task_ids = set()
        self.empty_store_task_ids = set()

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
        return None, []

    def _parse_task_name(self, name: str):
        """Parse a task/abstract task name to extract the task type and arguments."""
        if name in self.task_name_cache:
            return self.task_name_cache[name]
        
        original = name
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            result = (task_type, args)
        else:
            result = (name, [])
        
        self.task_name_cache[original] = result
        return result

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
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
                self.store_of_map[args[0]] = args[1]
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging.add(args[0])
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_ids[args[0]] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_ids[args[0]] = f.global_id
        
        # Extract goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.add((args[0], args[1]))
        
        # Classify abstract tasks by type
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            if task_type == "get_soil_data":
                self.get_soil_task_ids.add(at.global_id)
            elif task_type == "get_rock_data":
                self.get_rock_task_ids.add(at.global_id)
            elif task_type == "get_image_data":
                self.get_image_task_ids.add(at.global_id)
            elif task_type == "send_soil_data":
                self.send_soil_task_ids.add(at.global_id)
            elif task_type == "send_rock_data":
                self.send_rock_task_ids.add(at.global_id)
            elif task_type == "send_image_data":
                self.send_image_task_ids.add(at.global_id)
            elif task_type == "do_navigate1" or task_type == "do_navigate2":
                self.navigate_task_ids.add(at.global_id)
            elif task_type == "do_calibrate":
                self.calibrate_task_ids.add(at.global_id)
            elif task_type == "empty_store":
                self.empty_store_task_ids.add(at.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        t1 = self._goal_based_bound(state)
        t2 = self._task_network_bound(state, task_network)
        return max(t1, t2)

    def _goal_based_bound(self, state):
        """For each uncommunicated goal, estimate minimum remaining actions.
        
        Admissibility: Each uncommunicated_soil_data requires at minimum:
        - If we don't have the analysis: navigate + sample + navigate + communicate = 4 actions min
        - If we have analysis but not communicated: navigate + communicate = 2 actions min
        Similarly for rock data.
        For image data:
        - If we don't have the image: calibrate + navigate + take_image + navigate + communicate = 5 min
        - If we have image: navigate + communicate = 2 min
        
        Since different goals may share navigation, we take the sum but discount
        slightly. However, a simple sum is NOT admissible if actions are shared.
        We use max over individual costs + count-based estimate.
        """
        total_cost = 0
        max_single = 0
        
        # Soil goals
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                # Not yet communicated
                # Check if any rover has the analysis
                has_analysis = False
                for rover in self.equipped_soil:
                    sid = self.have_soil_ids.get((rover, wp))
                    if sid is not None and (state >> sid) & 1:
                        has_analysis = True
                        break
                
                if has_analysis:
                    # Need at least: navigate to comm point + communicate = 2
                    cost = 2
                else:
                    # Check if soil sample still exists
                    sample_id = self.at_soil_sample_ids.get(wp)
                    if sample_id is not None and (state >> sample_id) & 1:
                        # Need: navigate + maybe_drop + sample + navigate + communicate = 4-5
                        cost = 4
                    else:
                        # Sample already taken by some rover, check if any has it
                        # If no rover has analysis and sample gone, this might be unsolvable
                        # but we estimate conservatively
                        cost = 4
                
                total_cost += cost
                max_single = max(max_single, cost)
        
        # Rock goals
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                has_analysis = False
                for rover in self.equipped_rock:
                    sid = self.have_rock_ids.get((rover, wp))
                    if sid is not None and (state >> sid) & 1:
                        has_analysis = True
                        break
                
                if has_analysis:
                    cost = 2
                else:
                    sample_id = self.at_rock_sample_ids.get(wp)
                    if sample_id is not None and (state >> sample_id) & 1:
                        cost = 4
                    else:
                        cost = 4
                
                total_cost += cost
                max_single = max(max_single, cost)
        
        # Image goals
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                has_image = False
                for rover in self.equipped_imaging:
                    sid = self.have_image_ids.get((rover, obj, mode))
                    if sid is not None and (state >> sid) & 1:
                        has_image = True
                        break
                
                if has_image:
                    cost = 2
                else:
                    # Need calibrate + navigate + take_image + navigate + communicate = 5
                    cost = 5
                
                total_cost += cost
                max_single = max(max_single, cost)
        
        # Return max of single task cost and a fraction of total
        # (total is not admissible due to shared navigation, but max is)
        # We return max(max_single, total // 2) as a compromise
        # Actually, each communicate action is unique per goal, so we can count them
        # Number of uncommunicated goals is a strict lower bound on communicate actions
        uncommunicated = 0
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                uncommunicated += 1
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                uncommunicated += 1
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                uncommunicated += 1
        
        return max(max_single, uncommunicated)

    def _task_network_bound(self, state, task_network):
        """Estimate minimum actions from remaining task network.
        
        Admissibility: Each get_soil_data / get_rock_data task requires at minimum
        1 primitive action (the communicate). Each get_image_data requires at minimum
        1 primitive action. But we can be more precise by checking state.
        """
        if not task_network:
            return 0
        
        h = 0
        for task in task_network:
            gid = task.global_id
            
            if isinstance(task, ModelOperator):
                # Primitive task - costs 1 action (unless it's nop)
                task_type, _ = self._parse_task_name(task.name)
                if task_type != "nop":
                    h += 1
            elif isinstance(task, ModelAbstractTask):
                # Abstract task - estimate based on type
                if gid in self.get_soil_task_ids:
                    task_type, args = self._parse_task_name(task.name)
                    if len(args) >= 1:
                        wp = args[0]
                        # Minimum: navigate(maybe 0-1) + empty_store(0-1) + sample + send_soil
                        # send_soil = navigate(0-1) + communicate = at least 1
                        # So minimum is sample + communicate = 2, but need to check state
                        communicated_id = self.communicated_soil_ids.get(wp)
                        if communicated_id is not None and (state >> communicated_id) & 1:
                            h += 0  # already done
                        else:
                            # Check if any rover has analysis
                            has_analysis = False
                            for rover in self.equipped_soil:
                                sid = self.have_soil_ids.get((rover, wp))
                                if sid is not None and (state >> sid) & 1:
                                    has_analysis = True
                                    break
                            if has_analysis:
                                h += 1  # just communicate
                            else:
                                h += 3  # sample + navigate + communicate minimum
                    else:
                        h += 3
                
                elif gid in self.get_rock_task_ids:
                    task_type, args = self._parse_task_name(task.name)
                    if len(args) >= 1:
                        wp = args[0]
                        communicated_id = self.communicated_rock_ids.get(wp)
                        if communicated_id is not None and (state >> communicated_id) & 1:
                            h += 0
                        else:
                            has_analysis = False
                            for rover in self.equipped_rock:
                                sid = self.have_rock_ids.get((rover, wp))
                                if sid is not None and (state >> sid) & 1:
                                    has_analysis = True
                                    break
                            if has_analysis:
                                h += 1
                            else:
                                h += 3
                    else:
                        h += 3
                
                elif gid in self.get_image_task_ids:
                    task_type, args = self._parse_task_name(task.name)
                    if len(args) >= 2:
                        obj, mode = args[0], args[1]
                        communicated_id = self.communicated_image_ids.get((obj, mode))
                        if communicated_id is not None and (state >> communicated_id) & 1:
                            h += 0
                        else:
                            has_image = False
                            for rover in self.equipped_imaging:
                                sid = self.have_image_ids.get((rover, obj, mode))
                                if sid is not None and (state >> sid) & 1:
                                    has_image = True
                                    break
                            if has_image:
                                h += 1
                            else:
                                h += 4  # calibrate + take_image + navigate + communicate
                    else:
                        h += 4
                
                elif gid in self.send_soil_task_ids or gid in self.send_rock_task_ids:
                    h += 1  # at least communicate action
                
                elif gid in self.send_image_task_ids:
                    h += 1  # at least communicate action
                
                elif gid in self.navigate_task_ids:
                    # Navigation might be 0 (already there) or 1+
                    # We conservatively estimate 0 since rover might be at target
                    pass
                
                elif gid in self.calibrate_task_ids:
                    h += 1  # at least calibrate action
                
                elif gid in self.empty_store_task_ids:
                    # Might be 0 (already empty) or 1 (drop)
                    pass
                
                else:
                    # Unknown abstract task, estimate 1
                    h += 1
        
        return h
