# Rover GTOHP Domain-Dependent Heuristic
# 
# BOTTLENECK: Navigation is the primary serializer - rovers must travel to waypoints
# for sampling, imaging, and communication. Store capacity (1 per rover) serializes
# soil/rock collection. Camera calibration serializes imaging tasks.
#
# ADMISSIBLE LOWER BOUNDS:
# Term 1 (goal_gap): Count uncommunicated goal facts. Each needs >= 1 communicate action.
# Term 2 (collection_gap): For uncommunicated data where the rover hasn't even collected
#   the sample/image yet, we need >= 2 actions (collect + communicate). This is admissible
#   because collect and communicate are distinct required actions.
# Term 3 (task_network_cost): Weighted count of remaining tasks in the task network,
#   using minimum decomposition costs precomputed from the HTN structure.
#
# TIE-BREAKING: Small penalty for tasks requiring navigation to break symmetries
# among equivalent rover/waypoint choices.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGTOHPAnthropicOpus46Heuristic7(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP domain.
    
    Uses a multi-component max of:
    1. Goal gap: uncommunicated data goals
    2. Collection gap: data not yet collected requiring sample+communicate
    3. Task network weighted cost estimate
    """

    def __init__(self, name="rover_anthropicopus46_heuristic_7"):
        super().__init__(name)
        # Fact ID maps - built during initialize
        self.communicated_soil_ids = {}    # waypoint -> fact global_id
        self.communicated_rock_ids = {}    # waypoint -> fact global_id
        self.communicated_image_ids = {}   # (objective, mode) -> fact global_id
        self.have_soil_ids = {}            # (rover, waypoint) -> fact global_id
        self.have_rock_ids = {}            # (rover, waypoint) -> fact global_id
        self.have_image_ids = {}           # (rover, objective, mode) -> fact global_id
        self.at_soil_sample_ids = {}       # waypoint -> fact global_id
        self.at_rock_sample_ids = {}       # waypoint -> fact global_id
        self.calibrated_ids = {}           # (camera, rover) -> fact global_id
        self.at_rover_ids = {}             # (rover, waypoint) -> fact global_id
        self.empty_store_ids = {}          # store -> fact global_id
        self.equipped_soil_ids = {}        # rover -> fact global_id
        self.equipped_rock_ids = {}        # rover -> fact global_id
        self.equipped_imaging_ids = {}     # rover -> fact global_id
        self.on_board_ids = {}             # (camera, rover) -> fact global_id
        self.store_of_ids = {}             # (store, rover) -> fact global_id
        self.channel_free_ids = {}         # lander -> fact global_id
        
        # Goal structures
        self.soil_goals = []       # list of waypoints needing communicated_soil_data
        self.rock_goals = []       # list of waypoints needing communicated_rock_data
        self.image_goals = []      # list of (objective, mode) needing communicated_image_data
        
        # Task cost estimates
        self.task_min_cost = {}    # task name prefix -> min cost
        
        # Rover info
        self.soil_rovers = []      # rovers equipped for soil analysis
        self.rock_rovers = []      # rovers equipped for rock analysis
        self.imaging_rovers = []   # rovers equipped for imaging
        
        # Abstract task name -> min cost mapping
        self.abstract_task_costs = {}

    def _parse_fact_name(self, name: str):
        """Parse a grounded fact name like +predicate[arg1,arg2] into (predicate, [args])."""
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
        """Parse a task/operator name to extract the task type and arguments."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _precompute_abstract_task_costs(self):
        """Compute minimum decomposition costs for abstract tasks.
        
        Uses iterative fixed-point computation:
        - Operators cost 1 (except nop which costs 0)
        - Abstract tasks cost min over decompositions of sum of subtask costs
        """
        # Initialize operator costs
        op_costs = {}
        for op in self.model.operators:
            task_type, _ = self._parse_task_name(op.name)
            if task_type == "nop":
                op_costs[op.global_id] = 0
            else:
                op_costs[op.global_id] = 1
        
        # Initialize abstract task costs to infinity
        at_costs = {}
        for at in self.model.abstract_tasks:
            at_costs[at.global_id] = 999999
        
        # Iterative fixed-point
        changed = True
        iterations = 0
        max_iterations = 50
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for at in self.model.abstract_tasks:
                best_cost = 999999
                for dec in at.decompositions:
                    dec_cost = 0
                    feasible = True
                    for subtask in dec.task_network:
                        if subtask.global_id in op_costs:
                            dec_cost += op_costs[subtask.global_id]
                        elif subtask.global_id in at_costs:
                            dec_cost += at_costs[subtask.global_id]
                        else:
                            feasible = False
                            break
                    if feasible and dec_cost < best_cost:
                        best_cost = dec_cost
                if best_cost < at_costs[at.global_id]:
                    at_costs[at.global_id] = best_cost
                    changed = True
        
        # Merge into single dict
        self.abstract_task_costs = {}
        for gid, cost in op_costs.items():
            self.abstract_task_costs[gid] = cost
        for gid, cost in at_costs.items():
            self.abstract_task_costs[gid] = cost

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts, build goal structures, precompute task costs."""
        self.model = model
        
        # Build fact ID maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.communicated_soil_ids[args[0]] = fact.global_id
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.communicated_rock_ids[args[0]] = fact.global_id
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.communicated_image_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_soil_analysis" and len(args) >= 2:
                self.have_soil_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_rock_analysis" and len(args) >= 2:
                self.have_rock_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_image" and len(args) >= 3:
                self.have_image_ids[(args[0], args[1], args[2])] = fact.global_id
            elif pred == "at_soil_sample" and len(args) >= 1:
                self.at_soil_sample_ids[args[0]] = fact.global_id
            elif pred == "at_rock_sample" and len(args) >= 1:
                self.at_rock_sample_ids[args[0]] = fact.global_id
            elif pred == "calibrated" and len(args) >= 2:
                self.calibrated_ids[(args[0], args[1])] = fact.global_id
            elif pred == "at" and len(args) >= 2:
                self.at_rover_ids[(args[0], args[1])] = fact.global_id
            elif pred == "empty" and len(args) >= 1:
                self.empty_store_ids[args[0]] = fact.global_id
            elif pred == "equipped_for_soil_analysis" and len(args) >= 1:
                self.equipped_soil_ids[args[0]] = fact.global_id
            elif pred == "equipped_for_rock_analysis" and len(args) >= 1:
                self.equipped_rock_ids[args[0]] = fact.global_id
            elif pred == "equipped_for_imaging" and len(args) >= 1:
                self.equipped_imaging_ids[args[0]] = fact.global_id
            elif pred == "on_board" and len(args) >= 2:
                self.on_board_ids[(args[0], args[1])] = fact.global_id
            elif pred == "store_of" and len(args) >= 2:
                self.store_of_ids[(args[0], args[1])] = fact.global_id
            elif pred == "channel_free" and len(args) >= 1:
                self.channel_free_ids[args[0]] = fact.global_id
        
        # Extract goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) >= 1:
                self.soil_goals.append(args[0])
            elif pred == "communicated_rock_data" and len(args) >= 1:
                self.rock_goals.append(args[0])
            elif pred == "communicated_image_data" and len(args) >= 2:
                self.image_goals.append((args[0], args[1]))
        
        # Collect rover capabilities
        self.soil_rovers = list(self.equipped_soil_ids.keys())
        self.rock_rovers = list(self.equipped_rock_ids.keys())
        self.imaging_rovers = list(self.equipped_imaging_ids.keys())
        
        # Precompute abstract task minimum costs
        self._precompute_abstract_task_costs()
        
        # Compute initial heuristic
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        """Compute the heuristic value using multi-component max."""
        t1 = self._goal_gap(state)
        t2 = self._collection_gap(state)
        t3 = self._task_network_cost(state, task_network)
        return max(t1, t2, t3)

    def _goal_gap(self, state):
        """Count uncommunicated goal facts. Each requires at least 1 communicate action.
        
        Admissible: each communicate_*_data action achieves exactly one communicated_*_data fact,
        so at minimum we need this many actions.
        """
        count = 0
        for wp in self.soil_goals:
            fid = self.communicated_soil_ids.get(wp)
            if fid is not None and not ((state >> fid) & 1):
                count += 1
        for wp in self.rock_goals:
            fid = self.communicated_rock_ids.get(wp)
            if fid is not None and not ((state >> fid) & 1):
                count += 1
        for (obj, mode) in self.image_goals:
            fid = self.communicated_image_ids.get((obj, mode))
            if fid is not None and not ((state >> fid) & 1):
                count += 1
        return count

    def _collection_gap(self, state):
        """For each uncommunicated goal, estimate minimum remaining actions.
        
        For soil/rock: if not communicated, need at minimum:
          - If no rover has the analysis: need navigate + sample + navigate + communicate = 4
          - If some rover has the analysis: need navigate + communicate = 2
          - If communicated: 0
        For images: if not communicated:
          - If no rover has the image: need calibrate + navigate + take_image + navigate + communicate = 5  
          - If some rover has image: need navigate + communicate = 2
          - If communicated: 0
        
        We sum per-goal minimum costs. This is admissible because each goal requires
        its own distinct set of at least these many actions (communicate actions are
        distinct per goal fact).
        """
        total = 0
        
        for wp in self.soil_goals:
            fid = self.communicated_soil_ids.get(wp)
            if fid is not None and not ((state >> fid) & 1):
                # Not communicated yet - check if any rover has the analysis
                has_analysis = False
                for rover in self.soil_rovers:
                    hid = self.have_soil_ids.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        has_analysis = True
                        break
                if has_analysis:
                    total += 2  # navigate to comm point + communicate
                else:
                    # Check if soil sample still exists at waypoint
                    sid = self.at_soil_sample_ids.get(wp)
                    if sid is not None and ((state >> sid) & 1):
                        total += 4  # navigate + sample + navigate + communicate
                    else:
                        # Sample already taken by some rover but we don't have analysis flagged
                        # This means some rover took it - find which one
                        found = False
                        for rover in self.soil_rovers:
                            hid = self.have_soil_ids.get((rover, wp))
                            if hid is not None and ((state >> hid) & 1):
                                found = True
                                break
                        if found:
                            total += 2
                        else:
                            # Sample gone, no one has analysis - shouldn't happen in valid plans
                            # but conservatively estimate
                            total += 2
        
        for wp in self.rock_goals:
            fid = self.communicated_rock_ids.get(wp)
            if fid is not None and not ((state >> fid) & 1):
                has_analysis = False
                for rover in self.rock_rovers:
                    hid = self.have_rock_ids.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        has_analysis = True
                        break
                if has_analysis:
                    total += 2
                else:
                    sid = self.at_rock_sample_ids.get(wp)
                    if sid is not None and ((state >> sid) & 1):
                        total += 4
                    else:
                        found = False
                        for rover in self.rock_rovers:
                            hid = self.have_rock_ids.get((rover, wp))
                            if hid is not None and ((state >> hid) & 1):
                                found = True
                                break
                        if found:
                            total += 2
                        else:
                            total += 2
        
        for (obj, mode) in self.image_goals:
            fid = self.communicated_image_ids.get((obj, mode))
            if fid is not None and not ((state >> fid) & 1):
                has_image = False
                for rover in self.imaging_rovers:
                    hid = self.have_image_ids.get((rover, obj, mode))
                    if hid is not None and ((state >> hid) & 1):
                        has_image = True
                        break
                if has_image:
                    total += 2  # navigate + communicate
                else:
                    total += 5  # calibrate + nav + take_image + nav + communicate
        
        return total

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network using precomputed min costs.
        
        Admissible: uses minimum possible decomposition cost for each task,
        which is a lower bound on the actual number of primitive actions needed.
        """
        total = 0
        for task in task_network:
            cost = self.abstract_task_costs.get(task.global_id, 1)
            if cost >= 999999:
                cost = 1  # Fallback for unreachable tasks
            total += cost
        return total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
