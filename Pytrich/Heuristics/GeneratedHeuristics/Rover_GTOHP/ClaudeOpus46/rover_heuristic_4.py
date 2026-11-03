# Rover GTOHP Domain-Dependent Heuristic
#
# BOTTLENECK: Navigation between waypoints serializes all work. Each data goal
# requires navigate-to-sample, collect, navigate-to-comm-point, communicate.
# Store capacity (1 per rover) further serializes soil/rock collection.
#
# ADMISSIBLE LOWER BOUNDS:
# T1 - Goal fact gap: Count uncommunicated goal facts. Each needs >= 1 action.
# T2 - Data collection gap: For uncommunicated data where the rover doesn't
#      have the analysis/image yet, we need >= 2 actions (collect + communicate).
#      For soil/rock where sample still exists at waypoint, need >= 3 (navigate + sample + communicate).
# T3 - Task network lower bound: Count minimum primitive actions in remaining
#      task network by estimating each abstract task's minimum decomposition depth.
#
# Return max(T1, T2, T3) for admissible multi-component heuristic.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic4(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP domain.
    
    Uses multi-component max of admissible lower bounds focusing on:
    - Uncommunicated goal facts
    - Data collection requirements (sample existence, analysis state)
    - Task network structural estimates
    """
    
    def __init__(self, name="rover_anthropicopus46_heuristic_4"):
        super().__init__(name)
        # Fact ID maps built during initialization
        self.communicated_soil_ids = {}  # waypoint -> global_id
        self.communicated_rock_ids = {}  # waypoint -> global_id
        self.communicated_image_ids = {}  # (objective, mode) -> global_id
        
        self.have_soil_ids = {}  # (rover, waypoint) -> global_id
        self.have_rock_ids = {}  # (rover, waypoint) -> global_id
        self.have_image_ids = {}  # (rover, objective, mode) -> global_id
        
        self.at_soil_sample_ids = {}  # waypoint -> global_id
        self.at_rock_sample_ids = {}  # waypoint -> global_id
        
        self.calibrated_ids = {}  # (camera, rover) -> global_id
        self.at_rover_ids = {}  # (rover, waypoint) -> global_id
        self.empty_store_ids = {}  # store -> global_id
        self.full_store_ids = {}  # store -> global_id
        self.store_of_ids = {}  # (store, rover) -> global_id
        
        self.equipped_soil_ids = {}  # rover -> global_id
        self.equipped_rock_ids = {}  # rover -> global_id
        self.equipped_imaging_ids = {}  # rover -> global_id
        
        # Goal information
        self.goal_soil_waypoints = []  # list of waypoints needing soil comm
        self.goal_rock_waypoints = []  # list of waypoints needing rock comm
        self.goal_image_pairs = []  # list of (objective, mode) needing image comm
        
        self.goal_fact_ids = []  # all goal fact global_ids
        
        # Rovers and their capabilities
        self.soil_rovers = []  # rovers equipped for soil
        self.rock_rovers = []  # rovers equipped for rock
        self.imaging_rovers = []  # rovers equipped for imaging
        
        # All rover names
        self.all_rovers = []
        
        # Abstract task minimum cost estimates
        self.min_task_cost = {}  # task name prefix -> min cost
        
        # Operator/AbstractTask identification
        self.is_operator = set()  # global_ids that are operators
        self.is_abstract = set()  # global_ids that are abstract tasks
        self.abstract_min_cost = {}  # abstract task global_id -> min primitive cost
        
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
        return name, []
    
    def _parse_task_name(self, name: str):
        """Parse task/operator name to get the task type."""
        # Names like: get_soil_data[waypoint0], navigate[rover0,waypoint1,waypoint2]
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            return name[:bracket_idx]
        return name
    
    def _compute_abstract_min_costs(self):
        """Compute minimum number of primitive actions for each abstract task.
        
        Uses iterative fixed-point computation.
        """
        # Initialize: operators cost 1, abstract tasks cost infinity
        costs = {}
        for op in self.model.operators:
            op_type = self._parse_task_name(op.name)
            if op_type == 'nop':
                costs[op.global_id] = 0
            else:
                costs[op.global_id] = 1
        
        for at in self.model.abstract_tasks:
            costs[at.global_id] = 999999
        
        # Fixed-point iteration
        changed = True
        iterations = 0
        max_iterations = 50  # Safety limit
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for at in self.model.abstract_tasks:
                best_cost = 999999
                for decomp in at.decompositions:
                    decomp_cost = 0
                    for subtask in decomp.task_network:
                        subtask_cost = costs.get(subtask.global_id, 999999)
                        decomp_cost += subtask_cost
                        if decomp_cost >= best_cost:
                            break
                    best_cost = min(best_cost, decomp_cost)
                
                if best_cost < costs[at.global_id]:
                    costs[at.global_id] = best_cost
                    changed = True
        
        self.abstract_min_cost = costs
    
    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts and build lookup maps for fast heuristic computation."""
        self.model = model
        
        # Build fact ID maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_ids[args[0]] = fact.global_id
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_ids[args[0]] = fact.global_id
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_image" and len(args) == 3:
                self.have_image_ids[(args[0], args[1], args[2])] = fact.global_id
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_ids[args[0]] = fact.global_id
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_ids[args[0]] = fact.global_id
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_ids[(args[0], args[1])] = fact.global_id
            elif pred == "at" and len(args) == 2:
                self.at_rover_ids[(args[0], args[1])] = fact.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_store_ids[args[0]] = fact.global_id
            elif pred == "full" and len(args) == 1:
                self.full_store_ids[args[0]] = fact.global_id
            elif pred == "store_of" and len(args) == 2:
                self.store_of_ids[(args[0], args[1])] = fact.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil_ids[args[0]] = fact.global_id
                if args[0] not in self.soil_rovers:
                    self.soil_rovers.append(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock_ids[args[0]] = fact.global_id
                if args[0] not in self.rock_rovers:
                    self.rock_rovers.append(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging_ids[args[0]] = fact.global_id
                if args[0] not in self.imaging_rovers:
                    self.imaging_rovers.append(args[0])
        
        # Collect all rover names
        rover_set = set()
        for key in self.at_rover_ids:
            rover_set.add(key[0])
        self.all_rovers = list(rover_set)
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))
        
        # Collect all goal fact IDs for fast checking
        for wp in self.goal_soil_waypoints:
            if wp in self.communicated_soil_ids:
                self.goal_fact_ids.append(self.communicated_soil_ids[wp])
        for wp in self.goal_rock_waypoints:
            if wp in self.communicated_rock_ids:
                self.goal_fact_ids.append(self.communicated_rock_ids[wp])
        for obj, mode in self.goal_image_pairs:
            key = (obj, mode)
            if key in self.communicated_image_ids:
                self.goal_fact_ids.append(self.communicated_image_ids[key])
        
        # Compute minimum decomposition costs for abstract tasks
        self._compute_abstract_min_costs()
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)
    
    def _check_fact(self, state, global_id):
        """Check if a fact is true in the given state using bitwise ops."""
        return (state >> global_id) & 1
    
    def _count_uncommunicated_goals(self, state):
        """T1: Count how many goal communication facts are not yet achieved.
        Each requires at least 1 action (the communicate action itself)."""
        count = 0
        for fid in self.goal_fact_ids:
            if not self._check_fact(state, fid):
                count += 1
        return count
    
    def _data_collection_gap(self, state):
        """T2: Estimate minimum actions needed considering data collection state.
        
        For each uncommunicated goal:
        - If data already collected by some rover: need >= 1 (communicate)
        - If data not collected but sample exists: need >= 2 (sample + communicate)
          For soil/rock, if store is full: need >= 3 (drop + sample + communicate)
        - If data not collected and sample doesn't exist: infeasible or already done
        
        For images:
        - If image not taken: need calibrate + navigate + take_image + navigate + communicate >= 3
        - If image taken: need >= 1 (communicate)
        """
        total_cost = 0
        
        # Soil goals
        for wp in self.goal_soil_waypoints:
            comm_id = self.communicated_soil_ids.get(wp)
            if comm_id is None or self._check_fact(state, comm_id):
                continue  # Already communicated
            
            # Check if any rover has the soil analysis
            has_analysis = False
            for rover in self.soil_rovers:
                key = (rover, wp)
                if key in self.have_soil_ids:
                    if self._check_fact(state, self.have_soil_ids[key]):
                        has_analysis = True
                        break
            
            if has_analysis:
                # Just need to communicate: navigate to visible point + communicate
                total_cost += 2  # navigate + communicate (at minimum)
            else:
                # Need to collect + communicate
                sample_id = self.at_soil_sample_ids.get(wp)
                if sample_id is not None and self._check_fact(state, sample_id):
                    # Sample exists, need: navigate + (maybe drop) + sample + navigate + communicate
                    total_cost += 4  # navigate + sample + navigate + communicate
                else:
                    # Sample already taken by someone but not communicated
                    # or infeasible - estimate conservatively
                    total_cost += 2
        
        # Rock goals
        for wp in self.goal_rock_waypoints:
            comm_id = self.communicated_rock_ids.get(wp)
            if comm_id is None or self._check_fact(state, comm_id):
                continue
            
            has_analysis = False
            for rover in self.rock_rovers:
                key = (rover, wp)
                if key in self.have_rock_ids:
                    if self._check_fact(state, self.have_rock_ids[key]):
                        has_analysis = True
                        break
            
            if has_analysis:
                total_cost += 2
            else:
                sample_id = self.at_rock_sample_ids.get(wp)
                if sample_id is not None and self._check_fact(state, sample_id):
                    total_cost += 4
                else:
                    total_cost += 2
        
        # Image goals
        for obj, mode in self.goal_image_pairs:
            key = (obj, mode)
            comm_id = self.communicated_image_ids.get(key)
            if comm_id is None or self._check_fact(state, comm_id):
                continue
            
            has_image = False
            for rover in self.imaging_rovers:
                img_key = (rover, obj, mode)
                if img_key in self.have_image_ids:
                    if self._check_fact(state, self.have_image_ids[img_key]):
                        has_image = True
                        break
            
            if has_image:
                total_cost += 2  # navigate + communicate
            else:
                # Need calibrate + navigate + take_image + navigate + communicate
                total_cost += 5
        
        return total_cost
    
    def _task_network_estimate(self, task_network):
        """T3: Estimate minimum primitive actions from remaining task network.
        
        Uses precomputed minimum decomposition costs.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            cost = self.abstract_min_cost.get(task.global_id, 1)
            if cost >= 999999:
                cost = 1  # Fallback for unreachable tasks
            total += cost
        
        return total
    
    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible lower bounds."""
        # T1: Simple count of uncommunicated goals
        t1 = self._count_uncommunicated_goals(state)
        
        # T2: Data collection gap (more informed)
        t2 = self._data_collection_gap(state)
        
        # T3: Task network structural estimate
        t3 = self._task_network_estimate(task_network)
        
        return max(t1, t2, t3)
    
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
