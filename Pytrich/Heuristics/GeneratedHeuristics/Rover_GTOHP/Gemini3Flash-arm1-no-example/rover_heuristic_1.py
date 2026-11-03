from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic1(Heuristic):
    """
    Rover Domain Bottleneck: Navigation and Sequential Action Requirements.
    
    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is the requirement for rovers to navigate between waypoints to gather 
    and communicate data. While movement and communication are conceptually parallel (in this HDDL version, lander 
    channels are not consumed), each piece of data requires a sequence of specific actions: 
    Navigate -> Sample/Take Image -> (Navigate) -> Communicate. Navigation is the most costly repetitive action.

    Admissible Lower Bounds:
    1. Goal Progress: Every uncommunicated goal requires at least 1 action (communicate) if the data is already held,
       and at least 2 actions (sample + communicate) if it is not. This is a lower bound because sampling and 
       communicating are distinct, necessary actions.
    2. Navigation Distance: For every task involving a specific waypoint, the rover must be at that waypoint. 
       The shortest path distance (number of navigate actions) from the rover's current location to the target
       is a strictly admissible lower bound on the number of 'navigate' actions required.
    """

    def __init__(self, rover_gemini3flash_heuristic_1="rover_heuristic"):
        super().__init__(rover_gemini3flash_heuristic_1)
        self.dist = {}
        self.waypoint_names = []
        self.rover_names = []
        self.at_ids = {} # (rover_name, waypoint_name) -> global_id
        
        # Goal related maps
        self.comm_soil_goal_ids = {} # waypoint -> global_id
        self.comm_rock_goal_ids = {} # waypoint -> global_id
        self.comm_image_goal_ids = {} # (obj, mode) -> global_id
        
        # Analysis status maps
        self.have_soil_ids = {} # (rover, waypoint) -> global_id
        self.have_rock_ids = {} # (rover, waypoint) -> global_id
        self.have_image_ids = {} # (rover, obj, mode) -> global_id

        self.task_parsers = {}

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        adj = collections.defaultdict(list)
        
        # 1. Parse waypoints, rovers, and predicates
        waypoints = set()
        rovers = set()
        
        for fact in model.facts:
            pred, args, _ = self._parse_fact_name(fact.name)
            if pred == "at":
                rovers.add(args[0])
                waypoints.add(args[1])
                self.at_ids[(args[0], args[1])] = fact.global_id
            elif pred == "can_traverse":
                adj[args[1]].append(args[2])
                adj[args[2]].append(args[1])
            elif pred == "communicated_soil_data":
                self.comm_soil_goal_ids[args[0]] = fact.global_id
            elif pred == "communicated_rock_data":
                self.comm_rock_goal_ids[args[0]] = fact.global_id
            elif pred == "communicated_image_data":
                self.comm_image_goal_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_soil_analysis":
                self.have_soil_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_rock_analysis":
                self.have_rock_ids[(args[0], args[1])] = fact.global_id
            elif pred == "have_image":
                self.have_image_ids[(args[0], args[1], args[2])] = fact.global_id

        self.waypoint_names = sorted(list(waypoints))
        self.rover_names = sorted(list(rovers))

        # 2. Precompute all-pairs shortest paths using BFS
        for start in self.waypoint_names:
            self.dist[start] = {w: 1000 for w in self.waypoint_names}
            self.dist[start][start] = 0
            queue = collections.deque([(start, 0)])
            visited = {start}
            while queue:
                curr, d = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Determine current locations of rovers
        rover_at = {}
        for r in self.rover_names:
            for w in self.waypoint_names:
                if (state >> self.at_ids.get((r, w), 0)) & 1:
                    rover_at[r] = w
                    break

        h_comm = 0
        # Count goals remaining - strictly serializable action 'communicate'
        for w, fid in self.comm_soil_goal_ids.items():
            if not ((state >> fid) & 1): h_comm += 1
        for w, fid in self.comm_rock_goal_ids.items():
            if not ((state >> fid) & 1): h_comm += 1
        for (o, m), fid in self.comm_image_goal_ids.items():
            if not ((state >> fid) & 1): h_comm += 1

        h_nav = 0
        # Evaluate navigation required by the task network
        for task in task_network:
            # task.name is like "get_soil_data[waypoint0]"
            name_str = task.name
            bracket_idx = name_str.find("[")
            if bracket_idx == -1: continue
            
            predicate = name_str[:bracket_idx]
            args = name_str[bracket_idx+1:-1].split(",")

            if predicate == "do_navigate1":
                # args = [rover, to_waypoint]
                r, to_w = args[0], args[1]
                at_w = rover_at.get(r)
                if at_w and to_w in self.dist[at_w]:
                    h_nav = max(h_nav, self.dist[at_w][to_w])
            
            elif predicate in ["get_soil_data", "get_rock_data"]:
                # args = [waypoint]
                target_w = args[0]
                # Minimum distance for any rover to get here
                min_d = 1000
                for r_at in rover_at.values():
                    min_d = min(min_d, self.dist[r_at].get(target_w, 1000))
                if min_d < 1000:
                    h_nav = max(h_nav, min_d)

        # Final h is a max of bottlenecks. CommCount is very reliable in this domain.
        return max(h_comm, h_nav)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
