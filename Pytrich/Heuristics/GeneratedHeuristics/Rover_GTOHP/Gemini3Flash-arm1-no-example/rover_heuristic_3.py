from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic3(Heuristic):
    """
    Rover Domain Bottleneck Analysis:
    The primary bottleneck in the Rover domain is navigation and movement serialization. 
    Every high-level goal (communicating soil, rock, or image data) requires the rover to 
    be at specific waypoints for collection and then potentially move to another waypoint 
    visible to a lander for transmission. Movement via 'navigate' is the most frequent 
    and distance-constrained action.

    Admissibility Sketches:
    1. Goal-Action Lower Bound: Each unachieved goal fact (communicated_soil_data, etc.) 
       requires at least one sampling action and one communication action. Soil/Rock 
       require 2 actions, and Images require 3 (Calibrate + Take + Communicate). 
       This is a lower bound because it ignores navigation and store-emptying costs.
    2. Navigation Lower Bound: The number of 'navigate' steps required to reach the target 
       waypoint of the current task from the rover's current location is an admissible 
       lower bound on the cost to finish that specific navigation sub-sequence.
    """

    def __init__(self, rover_gemini3flash_heuristic_3="rover_gemini3flash_heuristic_3"):
        super().__init__(name=rover_gemini3flash_heuristic_3)
        self.goal_bits = []
        self.goal_types = {} # bit -> type ('soil', 'rock', 'image')
        self.at_ids = {}     # rover -> waypoint -> bit
        self.waypoint_dist = {} # rover -> waypoint1 -> waypoint2 -> int
        self.rover_list = []
        self.waypoint_list = []

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Facts
        self.rover_list = []
        self.waypoint_list = []
        traversals = collections.defaultdict(lambda: collections.defaultdict(list))

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                rover, wp = args[0], args[1]
                self.at_ids[(rover, wp)] = f.global_id
                if rover not in self.rover_list: self.rover_list.append(rover)
                if wp not in self.waypoint_list: self.waypoint_list.append(wp)
            elif pred == "can_traverse":
                rover, w1, w2 = args[0], args[1], args[2]
                traversals[rover][w1].append(w2)

        # 2. Precompute Distances per Rover (BFS)
        for r in self.rover_list:
            self.waypoint_dist[r] = {}
            for start_node in self.waypoint_list:
                self.waypoint_dist[r][start_node] = {start_node: 0}
                queue = collections.deque([start_node])
                while queue:
                    curr = queue.popleft()
                    d = self.waypoint_dist[r][start_node][curr]
                    for neighbor in traversals[r][curr]:
                        if neighbor not in self.waypoint_dist[r][start_node]:
                            self.waypoint_dist[r][start_node][neighbor] = d + 1
                            queue.append(neighbor)

        # 3. Identify Goal Bits
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            # Find the Fact object for this name to get global_id
            for f in model.facts:
                if f.name == g_name:
                    self.goal_bits.append(f.global_id)
                    if "soil" in pred: self.goal_types[f.global_id] = 'soil'
                    elif "rock" in pred: self.goal_types[f.global_id] = 'rock'
                    elif "image" in pred: self.goal_types[f.global_id] = 'image'
                    break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Term 1: Missing Goals Cost
        # Each soil/rock goal needs sample + communicate (2)
        # Each image goal needs calibrate + take + communicate (3)
        h_goals = 0
        for bit in self.goal_bits:
            if not ((state >> bit) & 1):
                g_type = self.goal_types.get(bit, 'soil')
                h_goals += 3 if g_type == 'image' else 2

        # Term 2: Task Network Work
        h_tasks = 0
        # Track current location of rovers to estimate travel
        rover_locs = {}
        for r in self.rover_list:
            for w in self.waypoint_list:
                if (state >> self.at_ids.get((r, w), 0)) & 1:
                    rover_locs[r] = w
                    break

        h_travel = 0
        for task in task_network:
            # Ignore nop and small housekeeping
            if task.name == "nop" or task.name == "visit" or task.name == "unvisit":
                continue
            
            # Basic decomposition estimate (very rough lower bound)
            h_tasks += 1 
            
            # Navigation check
            # Task names are like "do_navigate1[rover0,waypoint3]" or "navigate[rover0,waypoint1,waypoint2]"
            name = task.name
            if "navigate" in name:
                # Attempt to parse rover and target waypoint from grounded task name
                try:
                    parts = name.split("[")[-1][:-1].split(",")
                    if len(parts) >= 2:
                        r, target_wp = parts[0], parts[-1]
                        curr_wp = rover_locs.get(r)
                        if curr_wp and target_wp in self.waypoint_dist.get(r, {}).get(curr_wp, {}):
                            h_travel = max(h_travel, self.waypoint_dist[r][curr_wp][target_wp])
                except:
                    pass

        # Tie-breaking: encourage fewer tasks in network
        tie_breaker = len(task_network) * 0.01

        return max(h_goals, h_tasks) + h_travel + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
