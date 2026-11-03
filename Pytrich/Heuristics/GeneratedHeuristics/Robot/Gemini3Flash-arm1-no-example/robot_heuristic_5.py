from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic5(Heuristic):
    """
    Heuristic for the Robot domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the single robot arm (armempty/holding). 
    The robot can only carry one package at a time, meaning all deliveries must be serial.
    
    Admissibility Sketches:
    1. Delivery Action Bound: Each package not at its goal requires exactly one 'pickup' and one 'putdown' action. 
       If the robot is already holding a package, that package only needs one 'putdown'.
    2. Movement Bound: To deliver a package from A to B, the robot must perform at least dist(A, B) 'move' actions.
       If the robot is at R and the package is at A, it must also move dist(R, A) to reach it.
    3. Door Bound: If a door is closed and lies on the only path the robot takes, an 'open' action is mandatory.
    """

    def __init__(self, robot_gemini3flash_heuristic_5="robot_gemini3flash_heuristic_5"):
        super().__init__(name=robot_gemini3flash_heuristic_5)
        self.goal_locations = {}  # pkg -> room
        self.in_facts = {}       # pkg -> room -> global_id
        self.holding_facts = {}  # pkg -> global_id
        self.rloc_facts = {}     # room -> global_id
        self.door_facts = []     # list of (r1, r2, door_id)
        self.closed_facts = {}   # door -> global_id
        self.rooms = []
        self.adj = {}
        self.dist_matrix = {}

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
        # 1. Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "goal_in":
                self.goal_locations[args[0]] = args[1]
            elif pred == "in":
                if args[0] not in self.in_facts: self.in_facts[args[0]] = {}
                self.in_facts[args[0]][args[1]] = f.global_id
            elif pred == "holding":
                self.holding_facts[args[0]] = f.global_id
            elif pred == "rloc":
                self.rloc_facts[args[1]] = f.global_id
                if args[0] not in self.rooms: self.rooms.append(args[0])
            elif pred == "door":
                self.door_facts.append((args[0], args[1], args[2]))
                if args[0] not in self.adj: self.adj[args[0]] = []
                self.adj[args[0]].append(args[1])
            elif pred == "closed":
                self.closed_facts[args[0]] = f.global_id

        # Ensure all rooms are in rooms list
        room_set = set()
        for r1, r2, d in self.door_facts:
            room_set.add(r1)
            room_set.add(r2)
        self.rooms = list(room_set)

        # 2. Precompute distances using BFS
        for start_node in self.rooms:
            self.dist_matrix[start_node] = {r: float('inf') for r in self.rooms}
            self.dist_matrix[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for v in self.adj.get(u, []):
                    if self.dist_matrix[start_node][v] == float('inf'):
                        self.dist_matrix[start_node][v] = self.dist_matrix[start_node][u] + 1
                        queue.append(v)

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        # Get current robot location
        current_rloc = None
        for room, fid in self.rloc_facts.items():
            if (state >> fid) & 1:
                current_rloc = room
                break
        
        if current_rloc is None: return 0

        total_h = 0
        packages_to_deliver = []
        held_package = None

        # Check what the robot is holding
        for pkg, fid in self.holding_facts.items():
            if (state >> fid) & 1:
                held_package = pkg
                break

        # Identify packages not at goal
        for pkg, goal_room in self.goal_locations.items():
            if held_package == pkg:
                # Needs: 1 putdown + dist(current_rloc, goal_room)
                total_h += 1 + self.dist_matrix[current_rloc].get(goal_room, 0)
            else:
                # Find current room of package
                pkg_room = None
                pkg_in_goal = False
                if pkg in self.in_facts:
                    for room, fid in self.in_facts[pkg].items():
                        if (state >> fid) & 1:
                            pkg_room = room
                            if room == goal_room:
                                pkg_in_goal = True
                            break
                
                if not pkg_in_goal and pkg_room:
                    # Needs: pickup + putdown + dist(pkg_room, goal_room)
                    # We also add move cost to reach the package, but only for the 'first' one.
                    # Since we can only carry one, we can sum these basic costs.
                    total_h += 2 + self.dist_matrix[pkg_room].get(goal_room, 0)
                    packages_to_deliver.append(pkg_room)

        # Add distance to the nearest package if not holding anything
        if not held_package and packages_to_deliver:
            min_dist_to_pkg = min(self.dist_matrix[current_rloc].get(pr, 0) for pr in packages_to_deliver)
            total_h += min_dist_to_pkg
        
        # Add task network penalty: count non-abstract tasks or progress
        # In Robot domain, length of task network is a decent indicator of depth.
        total_h += len(task_network)

        return int(total_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
