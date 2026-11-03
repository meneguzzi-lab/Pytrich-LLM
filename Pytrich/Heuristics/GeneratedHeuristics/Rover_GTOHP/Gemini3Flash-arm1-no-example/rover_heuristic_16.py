from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic16(Heuristic):
    """
    Heuristic for the Rover_GTOHP domain.
    
    1. Bottleneck: The primary bottleneck is rover navigation between waypoints. 
       Moving from a starting position to sampling locations, and then to locations visible 
       from a lander for data transmission, accounts for most of the plan cost.

    2. Admissible Lower Bounds:
       - Term 1 (Navigation Lower Bound): The shortest path distance (number of 'navigate' steps) 
         from a rover's current waypoint to the target waypoints of its active navigation tasks. 
         Admissibility: A rover cannot reach a destination in fewer steps than the shortest path 
         in the connectivity graph.
       - Term 2 (Unsatisfied Goals): Each remaining 'communicated_..._data' goal requires at least 
         two physical actions: one to acquire the data (sample_soil/rock or take_image) and one 
         to transmit it (communicate_..._data). Navigation is additional. 
         Admissibility: These actions are non-overlapping and mandatory for each distinct data goal.
    """

    def __init__(self, rover_gemini3flash_heuristic_16="rover_gemini3flash_heuristic_16"):
        super().__init__(rover_gemini3flash_heuristic_16)
        self.waypoint_dist = {}
        self.goal_data_facts = []
        self.at_fact_ids = {} # (rover, waypoint) -> global_id
        self.lander_visible_waypoints = collections.defaultdict(set) # lander -> {waypoints}

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
        super().initialize(model, 0)
        self.model = model
        
        # 1. Parse connectivity for shortest paths
        adj = collections.defaultdict(list)
        waypoints = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "can_traverse":
                # can_traverse ?rover ?p1 ?p2
                adj[args[1]].append(args[2])
                adj[args[2]].append(args[1])
                waypoints.add(args[1])
                waypoints.add(args[2])
            elif pred == "at":
                self.at_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "visible":
                # Precompute lander visibility later if needed, but BFS is more robust
                pass

        # 2. BFS for all-pairs shortest paths
        self.waypoint_dist = {w: {w2: 999 for w2 in waypoints} for w in waypoints}
        for start_node in waypoints:
            self.waypoint_dist[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.waypoint_dist[start_node][v] = d + 1
                        queue.append((v, d + 1))

        # 3. Goal identification
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred and "communicated" in pred:
                # Find the global fact object
                for f in model.facts:
                    if f.name == gname:
                        self.goal_data_facts.append(f)
                        break

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        self.initial_h = h_initial
        return h_initial

    def _get_current_waypoint(self, rover, state):
        for (r, w), fid in self.at_fact_ids.items():
            if r == rover and (state >> fid) & 1:
                return w
        return None

    def _compute(self, state, task_network):
        # Term 1: Goal Progress (Unsatisfied data communication goals)
        unsatisfied_goals = 0
        for gf in self.goal_data_facts:
            if not ((state >> gf.global_id) & 1):
                unsatisfied_goals += 1
        
        # Each unsatisfied goal needs at least 2 actions (sample + communicate)
        goal_h = unsatisfied_goals * 2

        # Term 2: Navigation in Task Network
        nav_h = 0
        # We estimate navigation costs for active navigation tasks
        # and baseline costs for decomposition levels
        task_cost = 0
        for task in task_network:
            task_cost += 1 # Every task costs at least 1
            tname = task.name.lower()
            parts = tname.split()
            # Format: "do_navigate1 rover0 waypoint1"
            if "do_navigate" in tname and len(parts) >= 3:
                rover = parts[1]
                target_wp = parts[-1]
                curr_wp = self._get_current_waypoint(rover, state)
                if curr_wp and target_wp in self.waypoint_dist[curr_wp]:
                    nav_h += self.waypoint_dist[curr_wp][target_wp]
            elif any(x in tname for x in ["get_soil", "get_rock", "get_image"]):
                # These tasks involve navigation, sampling, and communication
                task_cost += 2

        return max(goal_h, nav_h, task_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
