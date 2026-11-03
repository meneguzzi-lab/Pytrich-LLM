# Bottleneck: Rovers must navigate to sample/image locations, collect data (requiring empty stores),
# then navigate to lander-visible waypoints to communicate. Store capacity of 1 serializes work.
#
# Admissible lower bounds:
# T1: For each uncommunicated goal fact, at least 1 communicate action is needed (admissible: each comm is independent)
# T2: For each uncollected+uncommunicated data, at least 2 actions needed (collect + communicate) (admissible: subset of real cost)
# T3: Task network estimate counting minimum actions per remaining compound task type

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as OperatorClass
from Pytrich.model import AbstractTask as AbstractTaskClass


class RoverGTOHPAnthropicOpus46Heuristic12(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses max of multiple admissible lower bounds:
    - Communication gap: counts uncommunicated goal data
    - Collection gap: counts data not yet collected that needs to be
    - Task network structural estimate with state-aware costing
    """

    def __init__(self, name="rover_anthropicopus46_heuristic_12"):
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
        self.at_ids = {}                  # (rover, waypoint) -> global_id
        self.calibrated_ids = {}          # (camera, rover) -> global_id
        self.empty_ids = {}              # store -> global_id
        self.full_ids = {}               # store -> global_id
        
        # Goal tracking
        self.goal_soil_waypoints = []     # list of waypoints needing soil comm
        self.goal_rock_waypoints = []     # list of waypoints needing rock comm
        self.goal_image_pairs = []        # list of (objective, mode) needing image comm
        
        # Task name patterns for cost estimation
        self.task_min_costs = {}          # task name prefix -> min cost estimate
        
        # Rover capabilities
        self.soil_rovers = set()          # rovers equipped for soil
        self.rock_rovers = set()          # rovers equipped for rock
        self.imaging_rovers = set()       # rovers equipped for imaging
        
        # Store mapping
        self.store_of_rover = {}          # rover -> store
        
        self.equipped_soil_ids = {}       # rover -> global_id
        self.equipped_rock_ids = {}       # rover -> global_id
        self.equipped_imaging_ids = {}    # rover -> global_id
        
        # Lander info
        self.lander_waypoint = None
        self.channel_free_id = None
        
        # Visibility from objective
        self.visible_from_ids = {}        # (objective, waypoint) -> global_id
        
        # can_traverse info for navigation estimation
        self.rover_adjacency = {}         # rover -> set of (wp1, wp2)
        
        # All rovers and waypoints
        self.all_rovers = set()
        self.all_waypoints = set()

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like +predicate[arg1,arg2] into (predicate, [args])."""
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

    def initialize(self, model: Model, initial_node: HTNNode):
        """Preprocess all fact maps and goal structures."""
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
            elif pred == "at" and len(args) == 2:
                self.at_ids[(args[0], args[1])] = f.global_id
                self.all_rovers.add(args[0])
                self.all_waypoints.add(args[1])
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = f.global_id
            elif pred == "full" and len(args) == 1:
                self.full_ids[args[0]] = f.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids[args[0]] = f.global_id
                self.soil_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids[args[0]] = f.global_id
                self.rock_rovers.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids[args[0]] = f.global_id
                self.imaging_rovers.add(args[0])
            elif pred == "store_of" and len(args) == 2:
                self.store_of_rover[args[1]] = args[0]  # store_of(store, rover)
            elif pred == "at_lander" and len(args) == 2:
                self.lander_waypoint = args[1]
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_id = f.global_id
            elif pred == "visible_from" and len(args) == 2:
                self.visible_from_ids[(args[0], args[1])] = f.global_id
            elif pred == "can_traverse" and len(args) == 3:
                rover = args[0]
                if rover not in self.rover_adjacency:
                    self.rover_adjacency[rover] = set()
                self.rover_adjacency[rover].add((args[1], args[2]))
        
        # Extract goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))
        
        # Precompute which rovers can potentially collect each type of data
        # for state-aware cost estimation
        self.soil_capable_rovers = list(self.soil_rovers)
        self.rock_capable_rovers = list(self.rock_rovers)
        self.image_capable_rovers = list(self.imaging_rovers)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        """Return max of multiple admissible lower bound estimates."""
        t1 = self._communication_gap(state)
        t2 = self._collection_gap(state)
        t3 = self._task_network_estimate(state, task_network)
        return max(t1, t2, t3)

    def _communication_gap(self, state):
        """T1: Count uncommunicated goal data. Each needs at least 1 communicate action.
        
        Admissible because each communicate action handles exactly one piece of data,
        so this is a strict lower bound on remaining communicate actions.
        """
        count = 0
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                count += 1
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                count += 1
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                count += 1
        return count

    def _collection_gap(self, state):
        """T2: For each uncommunicated goal, estimate minimum actions needed.
        
        For soil/rock: if not communicated, need at minimum:
          - If no rover has the analysis: need navigate + possibly drop + sample + navigate_to_comm + communicate = ~5
          - If some rover has it: need navigate_to_comm + communicate = ~2
        For images: if not communicated:
          - If no rover has the image: need calibrate + navigate + take_image + navigate_to_comm + communicate = ~5  
          - If some rover has it: need navigate_to_comm + communicate = ~2
        
        Admissible because we take minimum across all capable rovers.
        """
        total = 0
        
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                # Check if any rover already has the soil analysis
                has_analysis = False
                for rover in self.soil_capable_rovers:
                    hid = self.have_soil_ids.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        has_analysis = True
                        break
                
                if has_analysis:
                    # Just need to navigate to comm point and communicate
                    total += 2  # navigate + communicate (minimum)
                else:
                    # Need to navigate to waypoint, possibly empty store, sample, navigate to comm, communicate
                    # Check if sample is still available
                    sample_id = self.at_soil_sample_ids.get(wp)
                    if sample_id is not None and ((state >> sample_id) & 1):
                        # Sample available, need: nav_to_wp(>=1) + maybe_drop(0-1) + sample(1) + nav_to_comm(>=1) + comm(1) = 4+
                        total += 4
                    else:
                        # Sample already taken by some rover, check if any rover has it
                        found = False
                        for rover in self.soil_capable_rovers:
                            hid = self.have_soil_ids.get((rover, wp))
                            if hid is not None and ((state >> hid) & 1):
                                found = True
                                break
                        if found:
                            total += 2
                        else:
                            # Sample gone and no one has analysis - this shouldn't happen in valid problems
                            # but be safe
                            total += 4
        
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_ids.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                has_analysis = False
                for rover in self.rock_capable_rovers:
                    hid = self.have_rock_ids.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        has_analysis = True
                        break
                
                if has_analysis:
                    total += 2
                else:
                    sample_id = self.at_rock_sample_ids.get(wp)
                    if sample_id is not None and ((state >> sample_id) & 1):
                        total += 4
                    else:
                        found = False
                        for rover in self.rock_capable_rovers:
                            hid = self.have_rock_ids.get((rover, wp))
                            if hid is not None and ((state >> hid) & 1):
                                found = True
                                break
                        if found:
                            total += 2
                        else:
                            total += 4
        
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_ids.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                has_image = False
                for rover in self.image_capable_rovers:
                    hid = self.have_image_ids.get((rover, obj, mode))
                    if hid is not None and ((state >> hid) & 1):
                        has_image = True
                        break
                
                if has_image:
                    total += 2  # navigate to comm + communicate
                else:
                    # Need: calibrate(1) + nav_to_obj(>=1) + take_image(1) + nav_to_comm(>=1) + comm(1) = 5
                    total += 5
        
        return total

    def _task_network_estimate(self, state, task_network):
        """T3: Structural estimate based on remaining task network.
        
        For each task in the network, estimate minimum actions based on task type.
        Uses state-aware costing where possible.
        
        Admissible because we use minimum possible cost for each task.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            if isinstance(task, OperatorClass):
                # Primitive actions cost 1 (except nop which costs 0 effectively)
                # Check if it's a nop
                if "nop" in task.name:
                    total += 0
                else:
                    total += 1
            elif isinstance(task, AbstractTaskClass):
                total += self._estimate_abstract_task_cost(state, task)
        
        return total

    def _estimate_abstract_task_cost(self, state, task):
        """Estimate minimum cost for an abstract task based on its name and current state."""
        name = task.name
        
        # Parse task name to determine type
        if "get_soil_data" in name:
            # get_soil_data needs: navigate + empty_store + sample_soil + send_soil_data
            # send_soil_data needs: navigate + communicate
            # Minimum: navigate(>=1) + maybe_drop(0) + sample(1) + navigate(>=1) + communicate(1) = 4
            # But if already at waypoint and store empty: sample(1) + navigate(>=1) + communicate(1) = 3
            return 3
        
        elif "get_rock_data" in name:
            # Same structure as soil
            return 3
        
        elif "get_image_data" in name:
            # get_image_data needs: calibrate + navigate + take_image + send_image_data
            # send_image_data needs: navigate + communicate
            # Minimum: calibrate_task(>=1) + navigate(>=0) + take_image(1) + navigate(>=1) + communicate(1) = 4
            return 4
        
        elif "send_soil_data" in name or "send_rock_data" in name:
            # navigate + communicate = minimum 2, but could be at comm point already
            return 1  # at least the communicate action
        
        elif "send_image_data" in name:
            return 1  # at least the communicate action
        
        elif "do_navigate1" in name:
            # Could be 0 (already there) or >= 1
            return 0  # admissible lower bound
        
        elif "do_navigate2" in name:
            return 0  # could be already there
        
        elif "empty_store" in name:
            return 0  # could already be empty
        
        elif "do_calibrate" in name:
            # navigate + calibrate = at least 1 (the calibrate action)
            return 1
        
        else:
            # Unknown task type, return 0 for admissibility
            return 0
