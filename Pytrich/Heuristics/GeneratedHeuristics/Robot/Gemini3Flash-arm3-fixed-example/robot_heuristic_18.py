from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque, defaultdict
import time

class RobotGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Bottleneck Analysis:
    The primary bottleneck in this domain is the single-package capacity of the robot.
    Every package must be transported sequentially (Move to package, Pickup, Move to goal, Putdown).
    The robot's movement is constrained by the room-door topology, and doors may need to be opened.

    Admissibility Sketches:
    1. Action Cost (Admissible): Every package not currently in its goal room requires at least 
       one 'putdown' action. If the robot is not currently holding it, it also requires one 'pickup'. 
    2. Travel Cost (Admissible): The robot must move from its current location to a package's 
       starting room, and then from that room to the goal room. BFS distance in the room graph 
       provides a safe lower bound for the number of 'move' actions.
    3. Composite Max (Admissible): The heuristic returns the sum of these sequential costs 
       as it represents the total work remaining for a single robot capacity.
    """

    def __init__(self, robot_gemini3flash_heuristic_18=None, name="RobotGemini3FlashHeuristic18"):
        super().__init__(name=name)
        self.robot_gemini3flash_heuristic_18 = robot_gemini3flash_heuristic_18
        self.preprocessing_time = 0
        
        # Preprocessed Domain Knowledge
        self.rooms = []
        self.packages = []
        self.goal_rooms = {}        # pkg -> target room
        self.dist_matrix = {}       # room -> room -> int (shortest path distance)
        
        # Fact ID mappings for fast bitwise checks
        self.in_fact_ids = defaultdict(list)    # pkg -> [(room, fact_id)]
        self.holding_fact_ids = {}              # pkg -> fact_id
        self.rloc_fact_ids = {}                 # room -> fact_id
        self.armempty_id = -1
        self.adj = defaultdict(list)           # room -> [neighbor_rooms]

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # 1. Map Facts and Objects
        temp_rooms = set()
        temp_pkgs = set()
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                room = args[0]
                self.rloc_fact_ids[room] = fact.global_id
                temp_rooms.add(room)
            elif pred == "in":
                pkg, room = args[0], args[1]
                self.in_fact_ids[pkg].append((room, fact.global_id))
                temp_pkgs.add(pkg)
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "door":
                r1, r2, d = args
                self.adj[r1].append(r2)
            elif pred == "goal_in":
                # Use static goal_in predicates from initial state to define targets
                if (initial_node.state >> fact.global_id) & 1:
                    pkg, target_room = args[0], args[1]
                    self.goal_rooms[pkg] = target_room

        self.rooms = list(temp_rooms)
        self.packages = list(temp_pkgs)

        # 2. Precompute All-Pairs Shortest Paths (BFS)
        for start_room in self.rooms:
            self.dist_matrix[start_room] = {r: 1000 for r in self.rooms}
            self.dist_matrix[start_room][start_room] = 0
            queue = deque([start_room])
            while queue:
                curr = queue.popleft()
                curr_dist = self.dist_matrix[start_room][curr]
                for neighbor in self.adj.get(curr, []):
                    if self.dist_matrix[start_room][neighbor] == 1000:
                        self.dist_matrix[start_room][neighbor] = curr_dist + 1
                        queue.append(neighbor)

        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _compute(self, state: int, task_network) -> int:
        # Identify robot location
        robot_room = None
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                robot_room = room
                break

        if robot_room is None:
            return 0

        h_actions = 0
        h_move = 0
        unsettled_starts = []
        held_pkg = None

        # Check if holding something
        if self.armempty_id != -1 and not ((state >> self.armempty_id) & 1):
            for pkg, fid in self.holding_fact_ids.items():
                if (state >> fid) & 1:
                    held_pkg = pkg
                    break

        for pkg in self.packages:
            target = self.goal_rooms.get(pkg)
            if not target:
                continue

            if pkg == held_pkg:
                # Already picked up. Cost = Putdown + Move to goal
                h_actions += 1
                h_move += self.dist_matrix[robot_room].get(target, 0)
            else:
                # Find current location
                curr_room = None
                for r, fid in self.in_fact_ids[pkg]:
                    if (state >> fid) & 1:
                        curr_room = r
                        break
                
                if curr_room and curr_room != target:
                    # Needs Pickup + Putdown + Move from current to goal
                    h_actions += 2
                    h_move += self.dist_matrix[curr_room].get(target, 0)
                    unsettled_starts.append(curr_room)

        # If not holding, robot must travel to the first package
        if held_pkg is None and unsettled_starts:
            h_move += min(self.dist_matrix[robot_room].get(loc, 1000) for loc in unsettled_starts)

        # Admissible lower bound: Sum of unavoidable actions and movements
        # Sub-unit tie-breaker to prefer states with fewer tasks or more open doors (implicit)
        h = h_actions + h_move
        if task_network:
            h += 0.001 * len(task_network)
            
        return int(h) if h < 1000000 else 1000

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini3FlashHeuristic18(pkgs={len(self.packages)})"
