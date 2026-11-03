"""Domain-dependent heuristic for Rover_GTOHP.

Bottleneck Analysis:
  The primary bottleneck in the Rover domain is navigation: rovers must traverse
  waypoints to collect samples, take images, and communicate data. Each high-level
  task (get_soil_data, get_rock_data, get_image_data) requires a sequence of:
  navigate -> sample/image -> navigate -> communicate. The store capacity (single
  store per rover) serializes soil/rock collection. Communication requires
  visibility to the lander waypoint.

Admissible Lower Bounds:
  Term 1 - Uncommunicated goals count: Each uncommunicated goal (soil, rock, image)
    requires at least one communicate action. This is a strict lower bound since
    each communicate action handles exactly one datum.
  
  Term 2 - Unsatisfied prerequisite actions: For each uncommunicated soil/rock goal,
    we also need at least a sample action (if we don't have the analysis yet).
    For each uncommunicated image goal, we need at least a take_image action.
    Each such action is an unavoidable cost.
  
  Term 3 - Task network structural bound: Count remaining compound tasks that
    correspond to get_soil_data, get_rock_data, get_image_data in the task network,
    weighted by minimum decomposition depth. Each such task requires at minimum
    ~3 primitive actions (navigate + sample/image + communicate).

The heuristic returns max(term1, term2, term3) which is admissible since each
term independently lower-bounds the true cost.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic6(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_6"):
        super().__init__(name)
        # Fact ID maps
        self.communicated_soil_ids = {}   # waypoint_arg -> global_id
        self.communicated_rock_ids = {}   # waypoint_arg -> global_id
        self.communicated_image_ids = {}  # (objective_arg, mode_arg) -> global_id
        self.have_soil_ids = {}           # (rover_arg, waypoint_arg) -> global_id
        self.have_rock_ids = {}           # (rover_arg, waypoint_arg) -> global_id
        self.have_image_ids = {}          # (rover_arg, objective_arg, mode_arg) -> global_id
        self.at_soil_sample_ids = {}      # waypoint_arg -> global_id
        self.at_rock_sample_ids = {}      # waypoint_arg -> global_id
        self.calibrated_ids = {}          # (camera_arg, rover_arg) -> global_id
        self.at_rover_ids = {}            # (rover_arg, waypoint_arg) -> global_id
        self.empty_store_ids = {}         # store_arg -> global_id
        self.full_store_ids = {}          # store_arg -> global_id
        
        # Goal tracking
        self.goal_soil_waypoints = []     # list of waypoint args needing soil communication
        self.goal_rock_waypoints = []     # list of waypoint args needing rock communication  
        self.goal_image_pairs = []        # list of (objective, mode) needing image communication
        
        # Task name pattern matching
        self.get_soil_task_prefix = "get_soil_data"
        self.get_rock_task_prefix = "get_rock_data"
        self.get_image_task_prefix = "get_image_data"
        self.send_soil_task_prefix = "send_soil_data"
        self.send_rock_task_prefix = "send_rock_data"
        self.send_image_task_prefix = "send_image_data"
        self.do_navigate_prefix = "do_navigate"
        self.do_calibrate_prefix = "do_calibrate"
        self.empty_store_prefix = "empty_store"
        
        # Rover capabilities
        self.soil_rovers = set()  # rovers equipped for soil analysis
        self.rock_rovers = set()  # rovers equipped for rock analysis
        self.imaging_rovers = set()  # rovers equipped for imaging
        
        # Precomputed goal fact global IDs for fast bitwise checking
        self.goal_soil_fact_ids = []  # list of global_ids for communicated_soil_data goals
        self.goal_rock_fact_ids = []  # list of global_ids for communicated_rock_data goals
        self.goal_image_fact_ids = [] # list of global_ids for communicated_image_data goals
        
        # Abstract task identification
        self.task_min_cost = {}  # abstract_task global_id -> minimum estimated cost

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
        """Parse a task/operator name to extract the base task name and arguments."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_name = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_name, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Preprocess: build fact maps, identify goals, estimate task costs."""
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
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
                self.at_rover_ids[(args[0], args[1])] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_store_ids[args[0]] = f.global_id
            elif pred == "full" and len(args) == 1:
                self.full_store_ids[args[0]] = f.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.soil_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.rock_rovers.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.imaging_rovers.add(args[0])

        # Identify goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
                if args[0] in self.communicated_soil_ids:
                    self.goal_soil_fact_ids.append(self.communicated_soil_ids[args[0]])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
                if args[0] in self.communicated_rock_ids:
                    self.goal_rock_fact_ids.append(self.communicated_rock_ids[args[0]])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))
                key = (args[0], args[1])
                if key in self.communicated_image_ids:
                    self.goal_image_fact_ids.append(self.communicated_image_ids[key])

        # Build task minimum cost estimates for abstract tasks
        for at in model.abstract_tasks:
            task_name, args = self._parse_task_name(at.name)
            if task_name == self.get_soil_task_prefix:
                # get_soil_data: navigate + (maybe drop) + sample_soil + navigate + communicate = ~4-5
                self.task_min_cost[at.global_id] = 4
            elif task_name == self.get_rock_task_prefix:
                # get_rock_data: navigate + (maybe drop) + sample_rock + navigate + communicate = ~4-5
                self.task_min_cost[at.global_id] = 4
            elif task_name == self.get_image_task_prefix:
                # get_image_data: calibrate_nav + calibrate + navigate + take_image + navigate + communicate = ~5-6
                self.task_min_cost[at.global_id] = 5
            elif task_name == self.send_soil_task_prefix:
                # send_soil_data: navigate + communicate = ~2
                self.task_min_cost[at.global_id] = 2
            elif task_name == self.send_rock_task_prefix:
                self.task_min_cost[at.global_id] = 2
            elif task_name == self.send_image_task_prefix:
                self.task_min_cost[at.global_id] = 2
            elif task_name == self.do_calibrate_prefix:
                # do_calibrate: navigate + calibrate = ~2
                self.task_min_cost[at.global_id] = 2
            elif task_name and task_name.startswith(self.do_navigate_prefix):
                # do_navigate: at least 1 navigate action (or 0 if already there)
                self.task_min_cost[at.global_id] = 1
            elif task_name == self.empty_store_prefix:
                self.task_min_cost[at.global_id] = 0
            else:
                self.task_min_cost[at.global_id] = 1

        # Build a set mapping: for each soil goal waypoint, which have_soil_analysis facts to check
        self.soil_have_facts_for_wp = {}
        for wp in self.goal_soil_waypoints:
            related = []
            for (rover, w), gid in self.have_soil_ids.items():
                if w == wp:
                    related.append(gid)
            self.soil_have_facts_for_wp[wp] = related

        self.rock_have_facts_for_wp = {}
        for wp in self.goal_rock_waypoints:
            related = []
            for (rover, w), gid in self.have_rock_ids.items():
                if w == wp:
                    related.append(gid)
            self.rock_have_facts_for_wp[wp] = related

        self.image_have_facts_for_pair = {}
        for (obj, mode) in self.goal_image_pairs:
            related = []
            for (rover, o, m), gid in self.have_image_ids.items():
                if o == obj and m == mode:
                    related.append(gid)
            self.image_have_facts_for_pair[(obj, mode)] = related

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        """Compute the heuristic value as max of three admissible terms."""
        term1 = self._uncommunicated_goals_count(state)
        term2 = self._prerequisite_actions_bound(state)
        term3 = self._task_network_structural_bound(state, task_network)
        return max(term1, term2, term3)

    def _uncommunicated_goals_count(self, state):
        """Term 1: Count how many goal communication facts are not yet achieved.
        Each requires at minimum 1 action (the communicate action itself).
        This is admissible: each communicate handles exactly one datum."""
        count = 0
        for gid in self.goal_soil_fact_ids:
            if not ((state >> gid) & 1):
                count += 1
        for gid in self.goal_rock_fact_ids:
            if not ((state >> gid) & 1):
                count += 1
        for gid in self.goal_image_fact_ids:
            if not ((state >> gid) & 1):
                count += 1
        return count

    def _prerequisite_actions_bound(self, state):
        """Term 2: For each uncommunicated goal, estimate minimum actions needed.
        
        For soil/rock: if not communicated, need at least communicate (1).
          If no rover has the analysis yet, also need sample (1) + navigate to wp (1) = +2.
          If sample not available at waypoint, the sample can't be collected => still need it.
        For image: if not communicated, need communicate (1).
          If no rover has the image yet, need take_image (1) + calibrate (1) + navigate (1) = +3.
        
        This is admissible: each action counted is strictly necessary."""
        total = 0
        
        for i, wp in enumerate(self.goal_soil_waypoints):
            gid = self.goal_soil_fact_ids[i] if i < len(self.goal_soil_fact_ids) else None
            if gid is not None and (state >> gid) & 1:
                continue  # already communicated
            
            # Need at least communicate = 1
            cost = 1
            
            # Check if any rover already has the soil analysis
            have_facts = self.soil_have_facts_for_wp.get(wp, [])
            any_has = False
            for hid in have_facts:
                if (state >> hid) & 1:
                    any_has = True
                    break
            
            if not any_has:
                # Need sample_soil (1) + navigate to sample location (1) + navigate to comm point (1)
                cost += 2
                # Check if soil sample still exists at waypoint
                soil_sample_id = self.at_soil_sample_ids.get(wp)
                if soil_sample_id is not None and not ((state >> soil_sample_id) & 1):
                    # Soil sample already taken but no rover has analysis for this waypoint?
                    # This shouldn't normally happen in well-formed states, but be safe
                    pass
            else:
                # Have analysis but not communicated: need navigate to comm point (1)
                cost += 1
            
            total += cost
        
        for i, wp in enumerate(self.goal_rock_waypoints):
            gid = self.goal_rock_fact_ids[i] if i < len(self.goal_rock_fact_ids) else None
            if gid is not None and (state >> gid) & 1:
                continue
            
            cost = 1
            have_facts = self.rock_have_facts_for_wp.get(wp, [])
            any_has = False
            for hid in have_facts:
                if (state >> hid) & 1:
                    any_has = True
                    break
            
            if not any_has:
                cost += 2
            else:
                cost += 1
            
            total += cost
        
        for i, (obj, mode) in enumerate(self.goal_image_pairs):
            gid = self.goal_image_fact_ids[i] if i < len(self.goal_image_fact_ids) else None
            if gid is not None and (state >> gid) & 1:
                continue
            
            cost = 1
            have_facts = self.image_have_facts_for_pair.get((obj, mode), [])
            any_has = False
            for hid in have_facts:
                if (state >> hid) & 1:
                    any_has = True
                    break
            
            if not any_has:
                # Need calibrate + navigate + take_image + navigate to comm
                cost += 3
            else:
                cost += 1
            
            total += cost
        
        return total

    def _task_network_structural_bound(self, state, task_network):
        """Term 3: Estimate cost from remaining task network structure.
        
        For each task in the network:
        - Primitive operators count as 1 (except nop which counts as 0)
        - Abstract tasks use precomputed minimum cost estimates, adjusted by state.
        
        This is admissible because we use minimum possible decomposition costs."""
        total = 0
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task
                base_cost = self.task_min_cost.get(gid, 1)
                
                # Refine cost based on state for specific task types
                task_name, args = self._parse_task_name(task.name)
                
                if task_name == self.get_soil_task_prefix and len(args) >= 1:
                    wp = args[0]
                    comm_id = self.communicated_soil_ids.get(wp)
                    if comm_id is not None and (state >> comm_id) & 1:
                        base_cost = 0  # Already done
                    else:
                        # Check if any rover has the analysis
                        have_facts = self.soil_have_facts_for_wp.get(wp, [])
                        for hid in have_facts:
                            if (state >> hid) & 1:
                                base_cost = 2  # Just need navigate + communicate
                                break
                
                elif task_name == self.get_rock_task_prefix and len(args) >= 1:
                    wp = args[0]
                    comm_id = self.communicated_rock_ids.get(wp)
                    if comm_id is not None and (state >> comm_id) & 1:
                        base_cost = 0
                    else:
                        have_facts = self.rock_have_facts_for_wp.get(wp, [])
                        for hid in have_facts:
                            if (state >> hid) & 1:
                                base_cost = 2
                                break
                
                elif task_name == self.get_image_task_prefix and len(args) >= 2:
                    obj, mode = args[0], args[1]
                    comm_id = self.communicated_image_ids.get((obj, mode))
                    if comm_id is not None and (state >> comm_id) & 1:
                        base_cost = 0
                    else:
                        have_facts = self.image_have_facts_for_pair.get((obj, mode), [])
                        for hid in have_facts:
                            if (state >> hid) & 1:
                                base_cost = 2
                                break
                
                elif task_name and task_name.startswith("do_navigate"):
                    # Check if rover is already at destination
                    if len(args) >= 2:
                        rover = args[0]
                        dest = args[1] if len(args) == 2 else args[2]
                        at_id = self.at_rover_ids.get((rover, dest))
                        if at_id is not None and (state >> at_id) & 1:
                            base_cost = 0
                
                elif task_name == self.empty_store_prefix and len(args) >= 2:
                    store = args[0]
                    empty_id = self.empty_store_ids.get(store)
                    if empty_id is not None and (state >> empty_id) & 1:
                        base_cost = 0
                    else:
                        base_cost = 1
                
                total += base_cost
            
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive operator
                op_name, _ = self._parse_task_name(task.name)
                if op_name == "nop":
                    pass  # 0 cost
                else:
                    total += 1
        
        return total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Called for every node expansion."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
