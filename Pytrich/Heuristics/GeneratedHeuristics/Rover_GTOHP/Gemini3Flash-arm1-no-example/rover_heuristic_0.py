from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic0(Heuristic):
    """
    Heuristic for the Rover_GTOHP domain.
    
    Bottleneck Analysis:
    The main bottleneck is the sequential nature of data acquisition and communication. 
    Each rover must navigate to specific waypoints to collect data (soil, rock, or images), 
    and then navigate to waypoints with line-of-sight visibility to a lander to transmit that data.
    
    Admissible Lower Bounds:
    1. Work Term: Each abstract task in the task network requires a minimum number of primitive actions 
       to be resolved (e.g., get_soil_data requires at least a sample and a communicate action). 
       Summing these minimum costs across the current task network is admissible.
    2. Navigation Term: For any task in the network, a rover must navigate to a target waypoint and 
       likely a communication waypoint. The maximum of these minimum travel distances (number of 
       navigate actions) across the tasks in the network is an admissible lower bound on navigation cost.
    """

    def __init__(self, rover_gemini3flash_heuristic_0="rover_gemini3flash_heuristic_0"):
        super().__init__(name=rover_gemini3flash_heuristic_0)
        self.at_fact_id = {} # (rover, waypoint) -> id
        self.comm_soil_fact_id = {} # waypoint -> id
        self.comm_rock_fact_id = {} # waypoint -> id
        self.comm_image_fact_id = {} # (objective, mode) -> id
        
        self.rovers = []
        self.waypoints = []
        self.objectives = []
        self.modes = []
        self.landers = []
        
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_imaging = set()
        self.rover_cameras = collections.defaultdict(list)
        self.cam_modes = collections.defaultdict(set)
        self.obj_vis_waypoints = collections.defaultdict(list)
        self.comm_waypoints = set()
        
        self.dist_matrix = {} # rover -> w1 -> w2 -> cost
        self.dist_to_comm = {} # rover -> waypoint -> min_cost

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse facts and identify objects
        state = initial_node.state
        static_visible = set()
        static_can_traverse = collections.defaultdict(set)
        lander_waypoints = set()
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "at":
                self.at_fact_id[(args[0], args[1])] = f.global_id
                if args[0] not in self.rovers: self.rovers.append(args[0])
                if args[1] not in self.waypoints: self.waypoints.append(args[1])
            elif pred == "at_lander":
                lander_waypoints.add(args[1])
            elif pred == "visible":
                static_visible.add((args[0], args[1]))
            elif pred == "can_traverse":
                static_can_traverse[args[0]].add((args[1], args[2]))
            elif pred == "visible_from":
                self.obj_vis_waypoints[args[0]].append(args[1])
            elif pred == "equipped_for_soil_analysis":
                self.rover_equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.rover_equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.rover_equipped_imaging.add(args[0])
            elif pred == "on_board":
                self.rover_cameras[args[1]].append(args[0])
            elif pred == "supports":
                self.cam_modes[args[0]].add(args[1])
            elif pred == "communicated_soil_data":
                self.comm_soil_fact_id[args[0]] = f.global_id
            elif pred == "communicated_rock_data":
                self.comm_rock_fact_id[args[0]] = f.global_id
            elif pred == "communicated_image_data":
                self.comm_image_fact_id[(args[0], args[1])] = f.global_id

        # Comm waypoints: w1 where visible(w1, w_lander)
        for w1 in self.waypoints:
            for w_lander in lander_waypoints:
                if (w1, w_lander) in static_visible:
                    self.comm_waypoints.add(w1)
                    break

        # BFS for all-pairs shortest paths per rover
        for r in self.rovers:
            self.dist_matrix[r] = {w: {w2: 999 for w2 in self.waypoints} for w in self.waypoints}
            adj = collections.defaultdict(list)
            for p1, p2 in static_can_traverse[r]:
                if (p1, p2) in static_visible:
                    adj[p1].append(p2)
            
            for start_w in self.waypoints:
                self.dist_matrix[r][start_w][start_w] = 0
                queue = collections.deque([(start_w, 0)])
                visited = {start_w}
                while queue:
                    curr, d = queue.popleft()
                    self.dist_matrix[r][start_w][curr] = d
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))
            
            # Min dist to communication point
            self.dist_to_comm[r] = {}
            for w in self.waypoints:
                m = 999
                for cw in self.comm_waypoints:
                    d = self.dist_matrix[r][w][cw]
                    if d < m: m = d
                self.dist_to_comm[r][w] = m

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network:
            return 0
            
        rover_positions = {}
        for r in self.rovers:
            for w in self.waypoints:
                if (state >> self.at_fact_id.get((r, w), 0)) & 1:
                    rover_positions[r] = w
                    break
        
        total_work = 0
        max_nav = 0
        
        for task in task_network:
            task_type, args = self._parse_task_name(task.name)
            work = 0
            nav = 0
            
            if task_type == 'get_soil_data':
                w_target = args[0]
                if not ((state >> self.comm_soil_fact_id.get(w_target, 0)) & 1):
                    work = 2
                    nav = self._min_nav_for_collection(rover_positions, w_target, self.rover_equipped_soil)
            elif task_type == 'get_rock_data':
                w_target = args[0]
                if not ((state >> self.comm_rock_fact_id.get(w_target, 0)) & 1):
                    work = 2
                    nav = self._min_nav_for_collection(rover_positions, w_target, self.rover_equipped_rock)
            elif task_type == 'get_image_data':
                obj, mode = args[0], args[1]
                if not ((state >> self.comm_image_fact_id.get((obj, mode), 0)) & 1):
                    work = 3
                    # Imaging requires a specific rover with specific camera
                    imaging_rovers = [r for r in self.rover_equipped_imaging if any(mode in self.cam_modes[c] for c in self.rover_cameras[r])]
                    nav = self._min_nav_for_imaging(rover_positions, obj, imaging_rovers)
            elif task_type == 'send_soil_data' or task_type == 'send_rock_data' or task_type == 'send_image_data':
                work = 1
                r_assigned = args[0]
                nav = self.dist_to_comm[r_assigned].get(rover_positions.get(r_assigned, ''), 0)
            elif task_type == 'do_navigate1' or task_type == 'do_navigate2':
                r_assigned, w_target = args[0], args[-1]
                nav = self.dist_matrix[r_assigned][rover_positions.get(r_assigned, w_target)][w_target]
            elif task_type in ['sample_soil', 'sample_rock', 'take_image', 'calibrate', 'drop']:
                work = 1
            
            total_work += work
            if nav > max_nav: max_nav = nav
            
        return int(total_work + max_nav)

    def _min_nav_for_collection(self, rover_pos, target_w, qualified_rovers):
        best = 999
        for r in qualified_rovers:
            curr = rover_pos.get(r, target_w)
            d = self.dist_matrix[r][curr][target_w] + self.dist_to_comm[r][target_w]
            if d < best: best = d
        return best if best < 999 else 0

    def _min_nav_for_imaging(self, rover_pos, obj, qualified_rovers):
        best = 999
        for r in qualified_rovers:
            curr = rover_pos.get(r, '')
            if not curr: continue
            # Distance to some waypoint that can see the objective
            for w_vis in self.obj_vis_waypoints[obj]:
                d = self.dist_matrix[r][curr][w_vis] + self.dist_to_comm[r][w_vis]
                if d < best: best = d
        return best if best < 999 else 0

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

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        parts = name.split()
        return parts[0], parts[1:]
