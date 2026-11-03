from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic18(Heuristic):
    """
    Heuristic for the Robot HDDL domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the single robotic arm. The robot must transport packages 
    one by one from their starting locations to their target locations (defined by goal_in). 
    Each transportation requires a sequence of: move (to pickup), pickup, move (to destination), and putdown. 
    Doors between rooms might be closed, requiring additional 'open' actions.

    Admissible Lower Bounds:
    1. Transport Actions: Each package not at its goal requires at least 1 'putdown'. If not currently held, 
       it also requires a 'pickup'. Since the robot is serial, we can sum these (2 actions per unheld package, 
       1 per held package).
    2. Move Actions: Each package transport requires moving between the pickup and goal rooms. The minimum 
       number of 'move' actions is the shortest path distance. We sum these distances for all unfinished packages.
    3. Door Actions: Every closed door on a required shortest path must be opened at least once.
    4. Robot Travel: The robot must travel to the first package's location (if not holding one) or 
       to the goal of the current package.
    """

    def __init__(self, robot_gemini3flash_heuristic_18="robot_gemini3flash_heuristic_18"):
        super().__init__(robot_gemini3flash_heuristic_18)
        self.goal_in_map = {}  # pkg -> room_name
        self.package_names = set()
        self.room_names = set()
        
        self.holding_ids = {}  # pkg -> fact_id
        self.rloc_ids = {}     # room -> fact_id
        self.closed_ids = {}   # door -> fact_id
        
        self.pkg_in_mask = {}  # pkg -> bitmask of all in(pkg, room) facts
        self.fact_id_to_room = {}  # fact_id -> room_name (for in and rloc)
        self.rloc_mask = 0
        self.armempty_id = -1
        
        self.room_distances = {} # (r1, r2) -> int
        self.room_path_masks = {} # (r1, r2) -> bitmask of closed(d) fact_ids

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
        # 1. Parse all facts and categorize them
        doors_found = []
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "goal_in":
                if (model.initial_state >> f.global_id) & 1:
                    self.goal_in_map[args[0]] = args[1]
                self.package_names.add(args[0])
                self.room_names.add(args[1])
            elif pred == "in":
                pkg, room = args[0], args[1]
                self.package_names.add(pkg)
                self.room_names.add(room)
                if pkg not in self.pkg_in_mask: self.pkg_in_mask[pkg] = 0
                self.pkg_in_mask[pkg] |= (1 << f.global_id)
                self.fact_id_to_room[f.global_id] = room
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
                self.package_names.add(args[0])
            elif pred == "rloc":
                self.rloc_ids[args[0]] = f.global_id
                self.rloc_mask |= (1 << f.global_id)
                self.fact_id_to_room[f.global_id] = args[0]
                self.room_names.add(args[0])
            elif pred == "closed":
                self.closed_ids[args[0]] = f.global_id
            elif pred == "door":
                if (model.initial_state >> f.global_id) & 1:
                    doors_found.append((args[0], args[1], args[2]))
            elif pred == "armempty":
                self.armempty_id = f.global_id

        # 2. Build Adjacency and Precompute Distances/Paths (BFS)
        adj = {r: [] for r in self.room_names}
        for r1, r2, d_name in doors_found:
            c_id = self.closed_ids.get(d_name)
            adj[r1].append((r2, c_id))

        for start_node in self.room_names:
            q = deque([(start_node, 0, 0)])
            visited = {start_node}
            self.room_distances[(start_node, start_node)] = 0
            self.room_path_masks[(start_node, start_node)] = 0
            while q:
                curr, dist, mask = q.popleft()
                for neighbor, door_f_id in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_mask = mask
                        if door_f_id is not None: new_mask |= (1 << door_f_id)
                        self.room_distances[(start_node, neighbor)] = dist + 1
                        self.room_path_masks[(start_node, neighbor)] = new_mask
                        q.append((neighbor, dist + 1, new_mask))

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        # 1. Robot Location
        robot_bits = state & self.rloc_mask
        if not robot_bits: return 0
        robot_room_id = (robot_bits & -robot_bits).bit_length() - 1
        robot_room = self.fact_id_to_room.get(robot_room_id)

        # 2. Package Status
        h = 0
        unfinished_pickups = []
        all_path_masks = 0
        held_pkg_goal = None

        for pkg in self.package_names:
            goal_room = self.goal_in_map.get(pkg)
            if not goal_room: continue

            # Is package being held?
            if (state >> self.holding_ids.get(pkg, 0)) & 1:
                h += 1 # putdown
                dist = self.room_distances.get((robot_room, goal_room), 1)
                h += dist
                all_path_masks |= self.room_path_masks.get((robot_room, goal_room), 0)
                held_pkg_goal = goal_room
                continue

            # Is package in a room?
            in_bits = state & self.pkg_in_mask.get(pkg, 0)
            if in_bits:
                loc_id = (in_bits & -in_bits).bit_length() - 1
                current_loc = self.fact_id_to_room.get(loc_id)
                if current_loc != goal_room:
                    h += 2 # pickup + putdown
                    dist_move = self.room_distances.get((current_loc, goal_room), 1)
                    h += dist_move
                    all_path_masks |= self.room_path_masks.get((current_loc, goal_room), 0)
                    unfinished_pickups.append(current_loc)

        if not unfinished_pickups and held_pkg_goal is None:
            return 0

        # 3. Travel to closest next target
        if held_pkg_goal is None and unfinished_pickups:
            min_travel = min(self.room_distances.get((robot_room, loc), 1) for loc in unfinished_pickups)
            h += min_travel
            # Also count doors on the path to the closest pickup
            for loc in unfinished_pickups:
                if self.room_distances.get((robot_room, loc)) == min_travel:
                    all_path_masks |= self.room_path_masks.get((robot_room, loc), 0)
                    break

        # 4. Doors
        h += bin(state & all_path_masks).count('1')

        # 5. Task Network Guidance (Tie-breaking)
        h += 0.001 * len(task_network)

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
