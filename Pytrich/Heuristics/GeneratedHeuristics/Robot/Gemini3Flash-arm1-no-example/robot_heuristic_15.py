from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic15(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Robot domain is the single-carry constraint (armempty/holding).
    The robot can only transport one package at a time, forcing a sequential execution of 
    pickup, travel, and putdown for every package that is not at its destination.

    Admissibility Sketches:
    1. Package Movement Cost (Admissible): Every package 'p' currently at room 'A' that must reach 
       room 'B' (where A != B) requires at least one 'pickup', one 'putdown', and 'dist(A, B)' 'move' actions.
       Since the robot carries only one package, these costs are strictly additive for each package.
       Sum(dist(start, goal) + 2) is a valid lower bound.
    2. Robot Travel to Package (Admissible): If the robot is not holding a package and some packages 
       still need to be moved, the robot must travel to the nearest package. This distance is a 
       necessary lower bound on the next move actions.
    """

    def __init__(self, robot_gemini3flash_heuristic_15="robot_gemini3flash_heuristic_15"):
        super().__init__(name=robot_gemini3flash_heuristic_15)
        self.package_ids = []
        self.room_ids = []
        self.goal_locations = {} # package_id -> room_id
        self.distances = {} # (room_id, room_id) -> int
        
        self.in_pred_id = {}
        self.holding_pred_id = {}
        self.rloc_pred_id = {}
        self.closed_pred_id = {}
        self.armempty_id = -1

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
                return prefix, predicate, args
        return None, None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        rooms = []
        packages = []
        adj = {}
        
        # 1. Parse all facts to map the world
        for fact in model.facts:
            prefix, pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "rloc":
                self.rloc_pred_id[args[0]] = fact.global_id
                if args[0] not in rooms: rooms.append(args[0])
            elif pred == "in":
                if args[0] not in packages: packages.append(args[0])
                if args[1] not in rooms: rooms.append(args[1])
                self.in_pred_id[(args[0], args[1])] = fact.global_id
            elif pred == "holding":
                if args[0] not in packages: packages.append(args[0])
                self.holding_pred_id[args[0]] = fact.global_id
            elif pred == "goal_in":
                # This is a static/setup fact in this domain
                # Check initial state to see which packages have which goal rooms
                if (initial_node.state >> fact.global_id) & 1:
                    self.goal_locations[args[0]] = args[1]
            elif pred == "door":
                r1, r2, d = args
                if r1 not in adj: adj[r1] = set()
                adj[r1].add(r2)
            elif pred == "closed":
                self.closed_pred_id[args[0]] = fact.global_id
            elif pred == "armempty":
                self.armempty_id = fact.global_id

        self.package_ids = packages
        self.room_ids = rooms

        # 2. Precompute All-Pairs Shortest Paths for Rooms (BFS)
        for start_node in rooms:
            self.distances[start_node] = {r: 1000 for r in rooms}
            self.distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}
            
            while queue:
                curr, d = queue.popleft()
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        # Determine Robot location
        robot_loc = None
        for r, fid in self.rloc_pred_id.items():
            if (state >> fid) & 1:
                robot_loc = r
                break
        
        if robot_loc is None: return 0

        # Determine what is being held
        held_pkg = None
        for p, fid in self.holding_pred_id.items():
            if (state >> fid) & 1:
                held_pkg = p
                break

        h = 0
        packages_to_move = []

        # Calculate cost for all packages
        for p in self.package_ids:
            goal_r = self.goal_locations.get(p)
            if not goal_r: continue

            if p == held_pkg:
                # Cost to drop off held package
                h += self.distances[robot_loc][goal_r] + 1
            else:
                # Find where it is
                current_r = None
                for r in self.room_ids:
                    fid = self.in_pred_id.get((p, r))
                    if fid is not None and (state >> fid) & 1:
                        current_r = r
                        break
                
                if current_r and current_r != goal_r:
                    # Needs pickup (1), move (dist), and putdown (1)
                    h += self.distances[current_r][goal_r] + 2
                    packages_to_move.append(current_r)
        
        # If not holding, must move to the first package
        if held_pkg is None and packages_to_move:
            min_dist_to_pkg = min(self.distances[robot_loc][pr] for pr in packages_to_move)
            h += min_dist_to_pkg

        # Small tie-breaker for closed doors to provide gradient when distances are equal
        closed_doors_count = 0
        for d_fid in self.closed_pred_id.values():
            if (state >> d_fid) & 1:
                closed_doors_count += 1
        
        return h + (0.01 * closed_doors_count)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
