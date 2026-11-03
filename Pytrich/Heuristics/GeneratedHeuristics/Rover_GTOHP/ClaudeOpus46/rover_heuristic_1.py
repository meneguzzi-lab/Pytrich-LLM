# Rover GTOHP Domain Heuristic
# Bottleneck: Navigation between waypoints serializes all work. Each goal requires
# navigate-to-sample + collect + navigate-to-comm + communicate. Store capacity (1 per rover)
# forces sequential collect-drop cycles.
#
# Admissible lower bounds:
# T1: Count of uncommunicated goal facts - each needs >= 1 communicate action
# T2: Count of uncollected+uncommunicated data - each needs >= 2 actions (collect + communicate)
# T3: Task-network based estimate - remaining abstract tasks weighted by minimum decomposition cost
#     using state-aware checks (e.g., if rover already at target, navigation cost is 0)

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as ModelOperator
from Pytrich.model import AbstractTask as ModelAbstractTask


class RoverGTOHPAnthropicOpus46Heuristic1(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses a multi-component max of:
    1. Goal-distance: count uncommunicated goals weighted by minimum actions needed
    2. State-aware task network cost: estimates remaining work per task using current state
    3. Resource bottleneck: accounts for store emptying and calibration needs
    """

    def __init__(self):
        super().__init__(name="rover_anthropicopus46_heuristic_1")
        
        # Fact ID maps built during initialization
        self.communicated_soil_facts = {}   # waypoint -> global_id
        self.communicated_rock_facts = {}   # waypoint -> global_id
        self.communicated_image_facts = {}  # (objective, mode) -> global_id
        
        self.have_soil_facts = {}    # (rover, waypoint) -> global_id
        self.have_rock_facts = {}    # (rover, waypoint) -> global_id
        self.have_image_facts = {}   # (rover, objective, mode) -> global_id
        
        self.at_facts = {}           # (rover, waypoint) -> global_id
        self.at_soil_sample_facts = {}  # waypoint -> global_id
        self.at_rock_sample_facts = {}  # waypoint -> global_id
        self.empty_facts = {}        # store -> global_id
        self.full_facts = {}         # store -> global_id
        self.calibrated_facts = {}   # (camera, rover) -> global_id
        self.store_of_facts = {}     # store -> rover
        self.equipped_soil = set()   # rovers equipped for soil
        self.equipped_rock = set()   # rovers equipped for rock
        self.equipped_imaging = set()  # rovers equipped for imaging
        self.channel_free_facts = {} # lander -> global_id
        self.available_facts = {}    # rover -> global_id
        self.on_board_facts = {}     # (camera, rover) -> global_id
        self.visible_from_facts = {} # (objective, waypoint) -> global_id
        self.at_lander_facts = {}    # (lander, waypoint) -> global_id
        self.visible_facts = {}      # (wp1, wp2) -> global_id
        
        # Goal facts
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_image_pairs = set()  # (objective, mode)
        
        # Task name to type mapping
        self.task_type_cache = {}  # task global_id -> (type_str, estimated_min_cost)
        
        # Abstract task name patterns
        self.abstract_task_info = {}  # global_id -> (task_type, params)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like '+predicate[arg1,arg2]' into (predicate, [args])."""
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
        """Parse task/operator name to extract type and parameters."""
        # Names look like: get_soil_data[waypoint0] or navigate[rover0,waypoint1,waypoint2]
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            gid = fact.global_id
            
            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_facts[args[0]] = gid
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_facts[args[0]] = gid
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_facts[(args[0], args[1])] = gid
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_facts[(args[0], args[1])] = gid
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_facts[(args[0], args[1])] = gid
            elif pred == "have_image" and len(args) == 3:
                self.have_image_facts[(args[0], args[1], args[2])] = gid
            elif pred == "at" and len(args) == 2:
                self.at_facts[(args[0], args[1])] = gid
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_facts[args[0]] = gid
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_facts[args[0]] = gid
            elif pred == "empty" and len(args) == 1:
                self.empty_facts[args[0]] = gid
            elif pred == "full" and len(args) == 1:
                self.full_facts[args[0]] = gid
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_facts[(args[0], args[1])] = gid
            elif pred == "store_of" and len(args) == 2:
                self.store_of_facts[args[0]] = args[1]
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging.add(args[0])
            elif pred == "channel_free" and len(args) == 1:
                self.channel_free_facts[args[0]] = gid
            elif pred == "available" and len(args) == 1:
                self.available_facts[args[0]] = gid
            elif pred == "on_board" and len(args) == 2:
                self.on_board_facts[(args[0], args[1])] = gid
            elif pred == "visible_from" and len(args) == 2:
                self.visible_from_facts[(args[0], args[1])] = gid
            elif pred == "at_lander" and len(args) == 2:
                self.at_lander_facts[(args[0], args[1])] = gid
            elif pred == "visible" and len(args) == 2:
                self.visible_facts[(args[0], args[1])] = gid
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gfn in goal_fact_names:
            pred, args = self._parse_fact_name(gfn)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.add((args[0], args[1]))
        
        self.total_goals = len(self.goal_soil_waypoints) + len(self.goal_rock_waypoints) + len(self.goal_image_pairs)
        
        # Precompute abstract task info
        for at in model.abstract_tasks:
            task_type, params = self._parse_task_name(at.name)
            self.abstract_task_info[at.global_id] = (task_type, params)
        
        # Also cache operator info
        self.operator_info = {}
        for op in model.operators:
            task_type, params = self._parse_task_name(op.name)
            self.operator_info[op.global_id] = (task_type, params)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_fact(self, state, global_id):
        """Check if a fact is true in the given state using bitwise operations."""
        return (state >> global_id) & 1

    def _count_uncommunicated_goals(self, state):
        """Count goal facts not yet achieved. Each needs at least 1 action (communicate)."""
        count = 0
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_facts.get(wp)
            if gid is not None and not self._check_fact(state, gid):
                count += 1
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_facts.get(wp)
            if gid is not None and not self._check_fact(state, gid):
                count += 1
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_facts.get((obj, mode))
            if gid is not None and not self._check_fact(state, gid):
                count += 1
        return count

    def _data_collection_estimate(self, state):
        """Estimate actions needed for uncollected + uncommunicated data.
        
        For each uncommunicated goal:
        - If data not yet collected: need navigate + empty_store(maybe) + sample + navigate_to_comm + communicate = ~4-5
        - If data collected but not communicated: need navigate_to_comm + communicate = ~2
        - If communicated: 0
        
        This is admissible because each step is a necessary minimum.
        """
        total = 0
        
        for wp in self.goal_soil_waypoints:
            comm_gid = self.communicated_soil_facts.get(wp)
            if comm_gid is not None and self._check_fact(state, comm_gid):
                continue  # Already communicated
            
            # Check if any rover has collected this soil data
            has_data = False
            for rover in self.equipped_soil:
                have_gid = self.have_soil_facts.get((rover, wp))
                if have_gid is not None and self._check_fact(state, have_gid):
                    has_data = True
                    break
            
            if has_data:
                # Need at least: navigate to comm point + communicate = 2 actions minimum
                total += 2
            else:
                # Check if soil sample still exists at waypoint
                sample_gid = self.at_soil_sample_facts.get(wp)
                if sample_gid is not None and self._check_fact(state, sample_gid):
                    # Need: navigate to wp + maybe empty store + sample + navigate to comm + communicate
                    # Minimum: navigate(1) + sample(1) + navigate(1) + communicate(1) = 4
                    # But navigate might be 0 if already there
                    total += 4
                else:
                    # Sample already taken by some rover but we haven't found have_data
                    # This means some rover took it - they need to communicate
                    total += 2
        
        for wp in self.goal_rock_waypoints:
            comm_gid = self.communicated_rock_facts.get(wp)
            if comm_gid is not None and self._check_fact(state, comm_gid):
                continue
            
            has_data = False
            for rover in self.equipped_rock:
                have_gid = self.have_rock_facts.get((rover, wp))
                if have_gid is not None and self._check_fact(state, have_gid):
                    has_data = True
                    break
            
            if has_data:
                total += 2
            else:
                sample_gid = self.at_rock_sample_facts.get(wp)
                if sample_gid is not None and self._check_fact(state, sample_gid):
                    total += 4
                else:
                    total += 2
        
        for (obj, mode) in self.goal_image_pairs:
            comm_gid = self.communicated_image_facts.get((obj, mode))
            if comm_gid is not None and self._check_fact(state, comm_gid):
                continue
            
            has_data = False
            for rover in self.equipped_imaging:
                have_gid = self.have_image_facts.get((rover, obj, mode))
                if have_gid is not None and self._check_fact(state, have_gid):
                    has_data = True
                    break
            
            if has_data:
                # Need navigate to comm + communicate = 2
                total += 2
            else:
                # Need: calibrate(maybe navigate + calibrate) + navigate to visible_from + take_image + navigate to comm + communicate
                # Minimum: calibrate(1) + navigate(1) + take_image(1) + navigate(1) + communicate(1) = 5
                # Check if any rover has calibrated camera
                has_calibrated = False
                for rover in self.equipped_imaging:
                    for (cam, rov), gid in self.calibrated_facts.items():
                        if rov == rover and self._check_fact(state, gid):
                            has_calibrated = True
                            break
                    if has_calibrated:
                        break
                
                if has_calibrated:
                    # navigate + take_image + navigate + communicate = 4
                    total += 4
                else:
                    # navigate_to_calib + calibrate + navigate_to_obj + take_image + navigate_to_comm + communicate = 6
                    total += 5
        
        return total

    def _task_network_estimate(self, state, task_network):
        """Estimate minimum actions from remaining task network.
        
        Each abstract task has a minimum decomposition cost based on its type
        and current state. Operators cost 1 (except nop which costs 0).
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's an operator
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                info = self.operator_info.get(gid)
                if info is not None:
                    task_type = info[0]
                    if task_type == "nop":
                        total += 0
                    elif task_type == "visit" or task_type == "unvisit":
                        total += 0  # These are bookkeeping, basically free
                    else:
                        total += 1
                else:
                    total += 1
            # Check if it's an abstract task
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                info = self.abstract_task_info.get(gid)
                if info is not None:
                    task_type, params = info
                    total += self._estimate_abstract_task_cost(state, task_type, params)
                else:
                    total += 1
        
        return total

    def _estimate_abstract_task_cost(self, state, task_type, params):
        """Estimate minimum cost for an abstract task based on type and state."""
        
        if task_type == "do_navigate1":
            # navigate to a waypoint - costs 0 if already there, else >= 1
            if len(params) >= 2:
                rover, target_wp = params[0], params[1]
                at_gid = self.at_facts.get((rover, target_wp))
                if at_gid is not None and self._check_fact(state, at_gid):
                    return 0  # Already there
            return 1  # At least one navigate action
        
        elif task_type == "do_navigate2":
            # Similar but from a specific waypoint
            if len(params) >= 3:
                rover, from_wp, to_wp = params[0], params[1], params[2]
                at_gid = self.at_facts.get((rover, to_wp))
                if at_gid is not None and self._check_fact(state, at_gid):
                    return 0
            return 1
        
        elif task_type == "empty_store":
            # Costs 0 if already empty, else 1 (drop)
            if len(params) >= 1:
                store = params[0]
                empty_gid = self.empty_facts.get(store)
                if empty_gid is not None and self._check_fact(state, empty_gid):
                    return 0
            return 1
        
        elif task_type == "get_soil_data":
            # navigate + empty_store + sample + send_soil_data
            # send_soil_data = navigate + communicate
            # Minimum: sample(1) + communicate(1) = 2 if at location and store empty
            if len(params) >= 1:
                wp = params[0]
                # Check if already communicated
                comm_gid = self.communicated_soil_facts.get(wp)
                if comm_gid is not None and self._check_fact(state, comm_gid):
                    return 0  # Though this shouldn't happen in well-formed plans
                
                # Check if any equipped rover is at the waypoint with empty store
                min_cost = 6  # Default: navigate + empty + sample + navigate + communicate + overhead
                for rover in self.equipped_soil:
                    cost = 0
                    at_gid = self.at_facts.get((rover, wp))
                    if at_gid is None or not self._check_fact(state, at_gid):
                        cost += 1  # Need to navigate
                    
                    # Check store
                    for store, rov in self.store_of_facts.items():
                        if rov == rover:
                            empty_gid = self.empty_facts.get(store)
                            if empty_gid is not None and not self._check_fact(state, empty_gid):
                                cost += 1  # Need to drop
                            break
                    
                    cost += 1  # sample action
                    cost += 1  # communicate action
                    cost += 1  # navigate to comm point (minimum)
                    
                    min_cost = min(min_cost, cost)
                
                return max(min_cost, 2)  # At least sample + communicate
            return 4
        
        elif task_type == "get_rock_data":
            # Same structure as soil
            if len(params) >= 1:
                wp = params[0]
                comm_gid = self.communicated_rock_facts.get(wp)
                if comm_gid is not None and self._check_fact(state, comm_gid):
                    return 0
                
                min_cost = 6
                for rover in self.equipped_rock:
                    cost = 0
                    at_gid = self.at_facts.get((rover, wp))
                    if at_gid is None or not self._check_fact(state, at_gid):
                        cost += 1
                    
                    for store, rov in self.store_of_facts.items():
                        if rov == rover:
                            empty_gid = self.empty_facts.get(store)
                            if empty_gid is not None and not self._check_fact(state, empty_gid):
                                cost += 1
                            break
                    
                    cost += 1  # sample
                    cost += 1  # communicate
                    cost += 1  # navigate to comm
                    min_cost = min(min_cost, cost)
                
                return max(min_cost, 2)
            return 4
        
        elif task_type == "get_image_data":
            # calibrate + navigate + take_image + send_image_data
            if len(params) >= 2:
                obj, mode = params[0], params[1]
                comm_gid = self.communicated_image_facts.get((obj, mode))
                if comm_gid is not None and self._check_fact(state, comm_gid):
                    return 0
                
                # Check if any rover already has this image
                for rover in self.equipped_imaging:
                    have_gid = self.have_image_facts.get((rover, obj, mode))
                    if have_gid is not None and self._check_fact(state, have_gid):
                        return 2  # Just need navigate + communicate
                
                return 4  # calibrate + navigate + take_image + communicate minimum
            return 4
        
        elif task_type == "send_soil_data" or task_type == "send_rock_data":
            # navigate to visible waypoint + communicate
            return 2
        
        elif task_type == "send_image_data":
            return 2
        
        elif task_type == "do_calibrate":
            # navigate to calibration target visible point + calibrate
            if len(params) >= 2:
                rover, camera = params[0], params[1]
                cal_gid = self.calibrated_facts.get((camera, rover))
                if cal_gid is not None and self._check_fact(state, cal_gid):
                    return 0  # Already calibrated
            return 2  # navigate + calibrate
        
        return 1  # Default minimum for unknown tasks

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible components."""
        
        # Term 1: Simple count of uncommunicated goals
        t1 = self._count_uncommunicated_goals(state)
        
        # Term 2: Data collection estimate (state-aware)
        t2 = self._data_collection_estimate(state)
        
        # Term 3: Task network estimate
        t3 = self._task_network_estimate(state, task_network)
        
        # Return max of the three terms (admissible since each is a lower bound)
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
