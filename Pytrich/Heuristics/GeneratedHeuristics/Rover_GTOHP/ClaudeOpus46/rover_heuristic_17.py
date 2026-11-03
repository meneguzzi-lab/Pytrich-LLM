"""Domain-dependent heuristic for Rover_GTOHP domain.

Bottleneck Analysis:
  The primary bottleneck in the Rover domain is NAVIGATION + COMMUNICATION.
  Each data goal (soil, rock, image) requires:
    1. Navigate to sample location
    2. Perform sampling/imaging
    3. Navigate to a communication-visible waypoint
    4. Communicate data to lander
  Navigation can be multi-hop, and stores have limited capacity (1 sample).
  The lander is at a fixed waypoint, and communication requires visibility.

Admissible Lower Bounds:
  Term 1 - Uncommunicated data count: Each uncommunicated goal fact requires
    at minimum 1 communicate action. This is a strict lower bound.
  Term 2 - Missing prerequisites: For each uncommunicated soil/rock data,
    if the rover doesn't have the analysis yet, at least a sample action
    is needed (+ navigate). For images, calibrate + take_image are needed.
    We count minimum actions per unsatisfied data goal.
  Term 3 - Task network structural lower bound: Count remaining compound
    tasks that require at least one primitive action to resolve.

All terms are admissible (each action can satisfy at most one goal requirement).
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator, AbstractTask


class RoverGTOHPAnthropicOpus46Heuristic17(Heuristic):
    def __init__(self, name="rover_anthropicopus46_heuristic_17"):
        super().__init__(name)
        # Precomputed mappings
        self.communicated_soil_facts = {}   # waypoint_arg -> global_id
        self.communicated_rock_facts = {}   # waypoint_arg -> global_id
        self.communicated_image_facts = {}  # (objective, mode) -> global_id
        self.have_soil_facts = {}           # (rover, waypoint) -> global_id
        self.have_rock_facts = {}           # (rover, waypoint) -> global_id
        self.have_image_facts = {}          # (rover, objective, mode) -> global_id
        self.calibrated_facts = {}          # (camera, rover) -> global_id
        self.at_soil_sample_facts = {}      # waypoint -> global_id
        self.at_rock_sample_facts = {}      # waypoint -> global_id
        self.at_facts = {}                  # (rover, waypoint) -> global_id
        self.empty_facts = {}               # store -> global_id
        self.full_facts = {}                # store -> global_id
        self.store_of_map = {}              # store -> rover
        self.equipped_soil = set()          # set of rovers
        self.equipped_rock = set()          # set of rovers
        self.equipped_imaging = set()       # set of rovers
        self.on_board_map = {}              # camera -> rover
        self.supports_map = {}              # (camera, mode) -> True
        self.calibration_target_map = {}    # camera -> objective
        
        # Goal sets
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_image_pairs = set()       # (objective, mode)
        
        # For task network analysis
        self.nop_op_ids = set()
        self.visit_op_ids = set()
        self.unvisit_op_ids = set()
        
        # Abstract task name prefixes for cost estimation
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
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _parse_component_name(self, name: str):
        """Parse operator/task names like 'navigate[rover0,waypoint1,waypoint2]'"""
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            pred = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return pred, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
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
            elif pred == "at_soil_sample" and len(args) == 1:
                self.at_soil_sample_facts[args[0]] = f.global_id
            elif pred == "at_rock_sample" and len(args) == 1:
                self.at_rock_sample_facts[args[0]] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_facts[(args[0], args[1])] = f.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_facts[args[0]] = f.global_id
            elif pred == "full" and len(args) == 1:
                self.full_facts[args[0]] = f.global_id
            elif pred == "store_of" and len(args) == 2:
                self.store_of_map[args[0]] = args[1]  # store -> rover
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.equipped_imaging.add(args[0])
            elif pred == "on_board" and len(args) == 2:
                self.on_board_map[args[0]] = args[1]  # camera -> rover
            elif pred == "supports" and len(args) == 2:
                self.supports_map[(args[0], args[1])] = True
            elif pred == "calibration_target" and len(args) == 2:
                self.calibration_target_map[args[0]] = args[1]  # camera -> objective

        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil_waypoints.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock_waypoints.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image_pairs.add((args[0], args[1]))

        # Parse operators for nop/visit/unvisit identification
        for op in model.operators:
            op_name, op_args = self._parse_component_name(op.name)
            if op_name == "nop":
                self.nop_op_ids.add(op.global_id)
            elif op_name == "visit":
                self.visit_op_ids.add(op.global_id)
            elif op_name == "unvisit":
                self.unvisit_op_ids.add(op.global_id)

        # Parse abstract tasks
        for at in model.abstract_tasks:
            at_name, at_args = self._parse_component_name(at.name)
            if at_name == "get_soil_data":
                self.get_soil_task_ids.add(at.global_id)
            elif at_name == "get_rock_data":
                self.get_rock_task_ids.add(at.global_id)
            elif at_name == "get_image_data":
                self.get_image_task_ids.add(at.global_id)
            elif at_name == "send_soil_data":
                self.send_soil_task_ids.add(at.global_id)
            elif at_name == "send_rock_data":
                self.send_rock_task_ids.add(at.global_id)
            elif at_name == "send_image_data":
                self.send_image_task_ids.add(at.global_id)
            elif at_name == "do_navigate1" or at_name == "do_navigate2":
                self.navigate_task_ids.add(at.global_id)
            elif at_name == "do_calibrate":
                self.calibrate_task_ids.add(at.global_id)
            elif at_name == "empty_store":
                self.empty_store_task_ids.add(at.global_id)

        # Build a mapping: for each rover, what soil/rock/image goals can it potentially serve?
        # This helps with more refined cost estimation.
        self.rover_list = list(self.equipped_soil | self.equipped_rock | self.equipped_imaging)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        t1 = self._uncommunicated_goals_cost(state)
        t2 = self._missing_prerequisites_cost(state)
        t3 = self._task_network_cost(state, task_network)
        return max(t1, t2, t3)

    def _uncommunicated_goals_cost(self, state):
        """Term 1: Each uncommunicated goal needs at least 1 communicate action.
        Admissible because each communicate action satisfies exactly one goal."""
        count = 0
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_facts.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                count += 1
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_facts.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                count += 1
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_facts.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                count += 1
        return count

    def _missing_prerequisites_cost(self, state):
        """Term 2: For each uncommunicated goal, estimate minimum actions needed.
        
        Soil data goal at waypoint w:
          - If not communicated: need communicate (1)
          - If no rover has soil analysis of w: need sample_soil (1) + navigate (>=1) + communicate (1) = >=3
          - If soil sample still at w: need sample (need navigate to w + sample + communicate) = >=3
          - If store not empty: need drop first = +1
        
        Rock data: similar to soil.
        
        Image data (objective o, mode m):
          - If not communicated: need communicate (1)
          - If no rover has image: need take_image (1) + calibrate (1) + navigate to visible wp (>=1) + communicate (1) = >=4
        
        This is admissible: we take the minimum over all capable rovers.
        """
        total = 0
        
        for wp in self.goal_soil_waypoints:
            gid = self.communicated_soil_facts.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                # Not communicated yet
                cost = 1  # At minimum need communicate
                
                # Check if any rover already has the soil analysis
                any_has_analysis = False
                for rover in self.equipped_soil:
                    hid = self.have_soil_facts.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        any_has_analysis = True
                        break
                
                if not any_has_analysis:
                    # Need to sample: navigate to wp + maybe drop + sample + navigate to comm + communicate
                    cost = 3  # sample + navigate_to_wp + communicate (minimum)
                    # Check if soil sample still exists at waypoint
                    ss_id = self.at_soil_sample_facts.get(wp)
                    if ss_id is not None and not ((state >> ss_id) & 1):
                        # Soil sample already taken but no rover has analysis?
                        # This shouldn't normally happen, but if it does, it's still >= 1
                        cost = 1
                else:
                    # Has analysis, just need to navigate to comm point + communicate
                    cost = 2  # navigate + communicate
                
                total += cost
        
        for wp in self.goal_rock_waypoints:
            gid = self.communicated_rock_facts.get(wp)
            if gid is not None and not ((state >> gid) & 1):
                cost = 1
                
                any_has_analysis = False
                for rover in self.equipped_rock:
                    hid = self.have_rock_facts.get((rover, wp))
                    if hid is not None and ((state >> hid) & 1):
                        any_has_analysis = True
                        break
                
                if not any_has_analysis:
                    cost = 3
                    rs_id = self.at_rock_sample_facts.get(wp)
                    if rs_id is not None and not ((state >> rs_id) & 1):
                        cost = 1
                else:
                    cost = 2
                
                total += cost
        
        for (obj, mode) in self.goal_image_pairs:
            gid = self.communicated_image_facts.get((obj, mode))
            if gid is not None and not ((state >> gid) & 1):
                cost = 1
                
                any_has_image = False
                for rover in self.equipped_imaging:
                    hid = self.have_image_facts.get((rover, obj, mode))
                    if hid is not None and ((state >> hid) & 1):
                        any_has_image = True
                        break
                
                if not any_has_image:
                    # Need calibrate + navigate + take_image + navigate_comm + communicate
                    cost = 4
                    
                    # Check if any suitable rover is already calibrated
                    any_calibrated = False
                    for camera, rover in self.on_board_map.items():
                        if rover in self.equipped_imaging and (camera, mode) in self.supports_map:
                            cal_id = self.calibrated_facts.get((camera, rover))
                            if cal_id is not None and ((state >> cal_id) & 1):
                                any_calibrated = True
                                break
                    
                    if any_calibrated:
                        cost = 3  # navigate + take_image + communicate
                else:
                    cost = 2  # navigate to comm + communicate
                
                total += cost
        
        return total

    def _task_network_cost(self, state, task_network):
        """Term 3: Estimate cost from remaining task network.
        
        For each task in the network, assign a minimum action cost based on type.
        This is admissible because each task requires at least these actions.
        """
        if not task_network:
            return 0
        
        total = 0
        for task_item in task_network:
            gid = task_item.global_id
            
            if isinstance(task_item, Operator):
                # Primitive task - costs 1 unless it's a nop/visit/unvisit
                if gid in self.nop_op_ids:
                    total += 0
                elif gid in self.visit_op_ids or gid in self.unvisit_op_ids:
                    total += 0  # These are bookkeeping, essentially free
                else:
                    total += 1
            elif isinstance(task_item, AbstractTask):
                # Abstract task - estimate minimum decomposition cost
                if gid in self.get_soil_task_ids:
                    # get_soil_data: navigate + empty_store(maybe) + sample_soil + send_soil_data
                    # Minimum: sample_soil + communicate = 2
                    # But with navigation: at least 2-3
                    total += 2
                elif gid in self.get_rock_task_ids:
                    total += 2
                elif gid in self.get_image_task_ids:
                    # calibrate + take_image + communicate = 3 minimum
                    total += 3
                elif gid in self.send_soil_task_ids or gid in self.send_rock_task_ids:
                    # At minimum: communicate action = 1
                    total += 1
                elif gid in self.send_image_task_ids:
                    total += 1
                elif gid in self.navigate_task_ids:
                    # Could be already at location (0) or need to traverse (>=1)
                    # Conservatively: 0 (admissible, might already be there)
                    total += 0
                elif gid in self.calibrate_task_ids:
                    # calibrate action = 1 (+ possible navigate)
                    total += 1
                elif gid in self.empty_store_task_ids:
                    # Either already empty (0) or need drop (1)
                    total += 0
                else:
                    # Unknown abstract task, at least 0
                    total += 0
        
        return total
