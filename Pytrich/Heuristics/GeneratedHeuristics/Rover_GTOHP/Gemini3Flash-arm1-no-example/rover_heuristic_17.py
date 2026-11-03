from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic17(Heuristic):
    """
    Rover Domain Heuristic Analysis:
    1. Bottleneck: Navigation (travel) between waypoints is the dominant cost, as rovers must traverse 
       multiple waypoints to reach sample sites and communication points (landers).
    2. Admissible Lower Bound 1 (Task Base Cost): Each unfulfilled communication goal requires 
       at least two actions (sampling/taking image + communicating). 
    3. Admissible Lower Bound 2 (Traversal Distance): The minimum number of 'navigate' actions 
       required to move from a rover's current location to its target waypoint in the task network.
    """

    def __init__(self, rover_gemini3flash_heuristic_17="rover_gemini3flash_heuristic_17"):
        super().__init__(rover_gemini3flash_heuristic_17)
        self.dist_matrix = {}
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_image = set()
        self.at_rover = {} # rover -> fact_id
        self.at_soil_sample = {} # waypoint -> fact_id
        self.at_rock_sample = {} # waypoint -> fact_id
        self.visible_from = {} # objective -> list of waypoints
        self.at_lander = {} # lander -> waypoint
        self.visible_waypoints = collections.defaultdict(set)
        self.store_full = {} # store -> fact_id
        self.rover_stores = collections.defaultdict(list)

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at":
                self.at_rover[(args[0], args[1])] = fact.global_id
            elif pred == "at_soil_sample":
                self.at_soil_sample[args[0]] = fact.global_id
            elif pred == "at_rock_sample":
                self.at_rock_sample[args[0]] = fact.global_id
            elif pred == "visible_from":
                if args[0] not in self.visible_from: self.visible_from[args[0]] = []
                self.visible_from[args[0]].append(args[1])
            elif pred == "at_lander":
                self.at_lander[args[0]] = args[1]
            elif pred == "visible":
                self.visible_waypoints[args[0]].add(args[1])
                self.visible_waypoints[args[1]].add(args[0])
            elif pred == "can_traverse":
                # can_traverse ?r ?p1 ?p2
                if args[0] not in self.dist_matrix:
                    self.dist_matrix[args[0]] = collections.defaultdict(lambda: float('inf'))
                self.dist_matrix[args[0]][(args[1], args[2])] = 1
                self.dist_matrix[args[0]][(args[1], args[1])] = 0
            elif pred == "full":
                self.store_full[args[0]] = fact.global_id
            elif pred == "store_of":
                self.rover_stores[args[1]].append(args[0])

        # 2. Precompute All-Pairs Shortest Paths for each Rover
        for rover in self.dist_matrix:
            waypoints = set()
            for p1, p2 in self.dist_matrix[rover].keys():
                waypoints.add(p1)
                waypoints.add(p2)
            
            # Floyd-Warshall like logic for reachability
            for k in waypoints:
                for i in waypoints:
                    for j in waypoints:
                        self.dist_matrix[rover][(i, j)] = min(self.dist_matrix[rover][(i, j)], 
                                                              self.dist_matrix[rover][(i, k)] + self.dist_matrix[rover][(k, j)])

        # 3. Parse Goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "communicated_soil_data":
                self.goal_soil.add(args[0])
            elif pred == "communicated_rock_data":
                self.goal_rock.add(args[0])
            elif pred == "communicated_image_data":
                self.goal_image.add((args[0], args[1]))

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        if not task_network:
            return 0

        # Term 1: Unfinished Tasks cost (Admissible: 2 per goal type, 1 per communication stage)
        # Note: We prioritize counting based on the current state
        task_cost = 0
        
        # Term 2: Travel Cost (Admissible: Min distance to target site)
        travel_cost = 0

        # Find rovers' current positions
        rover_locs = {}
        for (r, p), fid in self.at_rover.items():
            if (state >> fid) & 1:
                rover_locs[r] = p

        # Check unfulfilled goals in state
        for task in task_network:
            # Simplification: analyze top-level tasks if they are operators or abstract tasks
            # If it's navigation, we can look at its 'to' parameter
            if hasattr(task, 'name'):
                name = task.name.lower()
                # Approximate work by counting remaining high level goals not yet met in state
                # However, Pytrich passes grounded tasks. We extract args from task.name for heuristic.
                # Format often: 'task_name rover_name arg1 arg2...'
                parts = name.split(" ")
                
                if "navigate" in parts[0]:
                    # navigation usually looks like: (do_navigate1 rover waypoint)
                    if len(parts) >= 3:
                        r, target_p = parts[1], parts[2]
                        if r in rover_locs:
                            dist = self.dist_matrix.get(r, {}).get((rover_locs[r], target_p), 0)
                            travel_cost = max(travel_cost, dist)
                
                # Add constant cost per unrefined data gathering task
                if "get_soil" in parts[0] or "get_rock" in parts[0] or "get_image" in parts[0]:
                    task_cost += 2
                elif "send_" in parts[0] or "communicate_" in parts[0]:
                    task_cost += 1

        return int(max(task_cost, travel_cost))

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
