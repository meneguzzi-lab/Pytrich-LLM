# Rover GTOHP Domain Heuristic
# Bottleneck: Navigation between waypoints serializes all work. Each data goal requires
# navigate-to-sample + sample + navigate-to-comm + communicate (soil/rock) or
# navigate-to-calibrate + calibrate + navigate-to-objective + take_image + navigate-to-comm + communicate (image).
#
# Admissible Lower Bounds:
# T1: Count of uncommunicated goal facts. Each needs >= 1 communicate action.
# T2: Count of uncollected data (no have_X_analysis / have_image). Each needs >= 2 actions (sample+communicate or take_image+communicate).
# T3: Task network structural estimate - count remaining meaningful tasks weighted by minimum decomposition cost.
# Return max(T1, T2, T3) which is admissible since each term individually underestimates.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as ModelOperator
from Pytrich.model import AbstractTask as ModelAbstractTask


class RoverGTOHPAnthropicOpus46Heuristic14(Heuristic):
    """Domain-dependent heuristic for Rover GTOHP.
    
    Uses multi-component max of:
    - Uncommunicated goals count (each needs at least 1 action)
    - Uncollected data estimate (each uncollected item needs sample/image + communicate)
    - Task network structural cost estimate
    """

    def __init__(self, name="rover_anthropicopus46_heuristic_14"):
        super().__init__(name)
        # Fact indices by predicate
        self.communicated_soil_facts = {}  # waypoint -> global_id
        self.communicated_rock_facts = {}  # waypoint -> global_id
        self.communicated_image_facts = {}  # (objective, mode) -> global_id
        self.have_soil_facts = {}  # (rover, waypoint) -> global_id
        self.have_rock_facts = {}  # (rover, waypoint) -> global_id
        self.have_image_facts = {}  # (rover, objective, mode) -> global_id
        self.calibrated_facts = {}  # (camera, rover) -> global_id
        self.at_facts = {}  # (rover, waypoint) -> global_id
        self.empty_facts = {}  # store -> global_id
        self.full_facts = {}  # store -> global_id
        self.at_soil_sample_facts = {}  # waypoint -> global_id
        self.at_rock_sample_facts = {}  # waypoint -> global_id
        
        # Goal-related
        self.goal_soil_waypoints = []  # waypoints needing communicated_soil_data
        self.goal_rock_waypoints = []  # waypoints needing communicated_rock_data
        self.goal_image_pairs = []  # (objective, mode) needing communicated_image_data
        
        # Task name -> minimum cost estimate
        self.task_min_cost = {}
        
        # Operator/task name prefixes for identification
        self.nop_ops = set()
        self.visit_ops = set()
        self.unvisit_ops = set()

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
        # No brackets - predicate with no args
        return name, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to extract the base task type."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            base = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return base, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts and build lookup maps for fast heuristic computation."""
        self.model = model

        # Build fact index maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.communicated_soil_facts[args[0]] = f.global_id
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.communicated_rock_facts[args[0]] = f.global_id
            elif pred == "communicated_image_data" and len(args) == 2:
                self.communicated_image_facts[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis" and len(args) == 2:
                self.have_soil_facts[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_facts[(args[0], args[1])] = f.global_id
            elif pred == "have_image" and len(args) == 3:
                self.have_image_facts[(args[0], args[1], args[2])] = f.global_id
            elif pred == "calibrated" and len(args) == 2:
                self.calibrated_facts[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_facts[(args[0], args[1])] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_facts[args[0]] = f.global_id
            elif pred == "full" and len(args) == 1:
                self.full_facts[args[0]] = f.global_id
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_facts[args[0]] = f.global_id
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_facts[args[0]] = f.global_id

        # Parse goal facts to identify what needs to be communicated
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.append((args[0], args[1]))

        # Collect all rover names from have_soil_facts keys
        self.rover_names = set()
        for key in self.have_soil_facts:
            self.rover_names.add(key[0])
        for key in self.have_rock_facts:
            self.rover_names.add(key[0])
        for key in self.have_image_facts:
            self.rover_names.add(key[0])

        # Identify nop operators for task network cost estimation
        for op in model.operators:
            base, _ = self._parse_task_name(op.name)
            if base == "nop":
                self.nop_ops.add(op.global_id)
            elif base == "visit":
                self.visit_ops.add(op.global_id)
            elif base == "unvisit":
                self.unvisit_ops.add(op.global_id)

        # Precompute minimum cost estimates for abstract task types
        # get_soil_data: navigate + empty_store + sample_soil + navigate + communicate = ~5 min
        # get_rock_data: navigate + empty_store + sample_rock + navigate + communicate = ~5 min
        # get_image_data: calibrate_nav + calibrate + navigate + take_image + navigate + communicate = ~6 min
        # send_soil/rock_data: navigate + communicate = ~2 min
        # send_image_data: navigate + communicate = ~2 min
        # do_navigate1: 0 (already there) or 1+ (navigate)
        # do_navigate2: 0 or 1+
        # empty_store: 0 or 1 (drop)
        # do_calibrate: navigate + calibrate = ~2 min
        self.abstract_task_costs = {
            "get_soil_data": 4,
            "get_rock_data": 4,
            "get_image_data": 5,
            "send_soil_data": 1,
            "send_rock_data": 1,
            "send_image_data": 1,
            "do_navigate1": 1,
            "do_navigate2": 1,
            "empty_store": 0,
            "do_calibrate": 1,
        }
        
        self.operator_costs = {
            "nop": 0,
            "visit": 0,
            "unvisit": 0,
            "navigate": 1,
            "sample_soil": 1,
            "sample_rock": 1,
            "drop": 1,
            "calibrate": 1,
            "take_image": 1,
            "communicate_soil_data1": 1,
            "communicate_soil_data2": 1,
            "communicate_rock_data1": 1,
            "communicate_rock_data2": 1,
            "communicate_image_data": 1,
        }

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        """Compute max of multiple admissible lower bound terms."""
        t1 = self._uncommunicated_goals_count(state)
        t2 = self._uncollected_data_estimate(state)
        t3 = self._task_network_estimate(state, task_network)
        return max(t1, t2, t3)

    def _uncommunicated_goals_count(self, state):
        """T1: Each uncommunicated goal needs at least 1 communicate action.
        This is admissible because communicate is a required action for each goal."""
        count = 0
        for wp in self.goal_soil_waypoints:
            fid = self.communicated_soil_facts.get(wp)
            if fid is not None and not ((state >> fid) & 1):
                count += 1
        for wp in self.goal_rock_waypoints:
            fid = self.communicated_rock_facts.get(wp)
            if fid is not None and not ((state >> fid) & 1):
                count += 1
        for (obj, mode) in self.goal_image_pairs:
            fid = self.communicated_image_facts.get((obj, mode))
            if fid is not None and not ((state >> fid) & 1):
                count += 1
        return count

    def _uncollected_data_estimate(self, state):
        """T2: For each uncommunicated goal, estimate minimum actions needed.
        
        If data not yet communicated:
          - If data not yet collected (no have_X_analysis for any rover): need sample + communicate = 2
          - If data collected but not communicated: need communicate = 1
        For images:
          - If no rover has the image: need calibrate + take_image + communicate = 3
          - If some rover has image but not communicated: need communicate = 1
        
        This is admissible as it underestimates the true cost (ignoring navigation)."""
        total = 0

        for wp in self.goal_soil_waypoints:
            comm_fid = self.communicated_soil_facts.get(wp)
            if comm_fid is not None and not ((state >> comm_fid) & 1):
                # Check if any rover has the soil analysis
                has_data = False
                for rover in self.rover_names:
                    hfid = self.have_soil_facts.get((rover, wp))
                    if hfid is not None and ((state >> hfid) & 1):
                        has_data = True
                        break
                if has_data:
                    total += 1  # just communicate
                else:
                    # Check if soil sample still exists at waypoint
                    soil_fid = self.at_soil_sample_facts.get(wp)
                    if soil_fid is not None and ((state >> soil_fid) & 1):
                        total += 2  # sample + communicate
                    else:
                        # Sample already taken by some rover but analysis not recorded?
                        # This shouldn't happen normally, but be safe
                        total += 2

        for wp in self.goal_rock_waypoints:
            comm_fid = self.communicated_rock_facts.get(wp)
            if comm_fid is not None and not ((state >> comm_fid) & 1):
                has_data = False
                for rover in self.rover_names:
                    hfid = self.have_rock_facts.get((rover, wp))
                    if hfid is not None and ((state >> hfid) & 1):
                        has_data = True
                        break
                if has_data:
                    total += 1
                else:
                    total += 2

        for (obj, mode) in self.goal_image_pairs:
            comm_fid = self.communicated_image_facts.get((obj, mode))
            if comm_fid is not None and not ((state >> comm_fid) & 1):
                has_img = False
                for rover in self.rover_names:
                    hfid = self.have_image_facts.get((rover, obj, mode))
                    if hfid is not None and ((state >> hfid) & 1):
                        has_img = True
                        break
                if has_img:
                    total += 1  # just communicate
                else:
                    total += 3  # calibrate + take_image + communicate

        return total

    def _task_network_estimate(self, state, task_network):
        """T3: Estimate cost from remaining task network.
        
        For each task in the network, assign a minimum cost based on task type
        and current state. This is admissible because we use minimum possible
        decomposition costs."""
        total = 0
        for task in task_network:
            if isinstance(task, ModelAbstractTask):
                base, args = self._parse_task_name(task.name)
                min_cost = self.abstract_task_costs.get(base, 1)
                
                # State-aware refinement for specific task types
                if base == "get_soil_data" and len(args) >= 1:
                    wp = args[0]
                    # Check if already communicated
                    comm_fid = self.communicated_soil_facts.get(wp)
                    if comm_fid is not None and ((state >> comm_fid) & 1):
                        min_cost = 0  # Already done (shouldn't appear but be safe)
                    else:
                        # Check if any rover already has the analysis
                        has_data = False
                        for rover in self.rover_names:
                            hfid = self.have_soil_facts.get((rover, wp))
                            if hfid is not None and ((state >> hfid) & 1):
                                has_data = True
                                break
                        if has_data:
                            min_cost = 1  # Just need to communicate
                        else:
                            min_cost = 3  # sample + navigate + communicate minimum
                
                elif base == "get_rock_data" and len(args) >= 1:
                    wp = args[0]
                    comm_fid = self.communicated_rock_facts.get(wp)
                    if comm_fid is not None and ((state >> comm_fid) & 1):
                        min_cost = 0
                    else:
                        has_data = False
                        for rover in self.rover_names:
                            hfid = self.have_rock_facts.get((rover, wp))
                            if hfid is not None and ((state >> hfid) & 1):
                                has_data = True
                                break
                        if has_data:
                            min_cost = 1
                        else:
                            min_cost = 3
                
                elif base == "get_image_data" and len(args) >= 2:
                    obj, mode = args[0], args[1]
                    comm_fid = self.communicated_image_facts.get((obj, mode))
                    if comm_fid is not None and ((state >> comm_fid) & 1):
                        min_cost = 0
                    else:
                        has_img = False
                        for rover in self.rover_names:
                            hfid = self.have_image_facts.get((rover, obj, mode))
                            if hfid is not None and ((state >> hfid) & 1):
                                has_img = True
                                break
                        if has_img:
                            min_cost = 1  # communicate
                        else:
                            min_cost = 4  # calibrate + take_image + navigate + communicate
                
                elif base == "send_soil_data" and len(args) >= 2:
                    # rover, waypoint
                    min_cost = 1  # at minimum: communicate
                
                elif base == "send_rock_data" and len(args) >= 2:
                    min_cost = 1
                
                elif base == "send_image_data" and len(args) >= 3:
                    min_cost = 1
                
                elif base == "do_navigate1" and len(args) >= 2:
                    rover, to_wp = args[0], args[1]
                    at_fid = self.at_facts.get((rover, to_wp))
                    if at_fid is not None and ((state >> at_fid) & 1):
                        min_cost = 0  # already there
                    else:
                        min_cost = 1  # at least one navigate
                
                elif base == "do_navigate2":
                    min_cost = 0  # could be already there (nop)
                
                elif base == "empty_store" and len(args) >= 1:
                    store = args[0]
                    empty_fid = self.empty_facts.get(store)
                    if empty_fid is not None and ((state >> empty_fid) & 1):
                        min_cost = 0
                    else:
                        min_cost = 1  # drop
                
                elif base == "do_calibrate" and len(args) >= 2:
                    rover, camera = args[0], args[1]
                    cal_fid = self.calibrated_facts.get((camera, rover))
                    if cal_fid is not None and ((state >> cal_fid) & 1):
                        min_cost = 0  # already calibrated (though method doesn't check this)
                    else:
                        min_cost = 1  # calibrate action
                
                total += min_cost
            
            elif isinstance(task, ModelOperator):
                base, _ = self._parse_task_name(task.name)
                cost = self.operator_costs.get(base, 1)
                total += cost
        
        return total
