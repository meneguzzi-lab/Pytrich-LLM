from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic4(Heuristic):
    """
    Heuristic for the Robot domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the single robot and its single arm. All packages must be delivered sequentially. 
    Each package not at its destination requires a pickup, a putdown, and a series of move actions. 
    The robot must also travel to the location of the next package to be picked up.

    Admissibility Sketches:
    1. Delivery Cost: For every package 'p' that is not at its goal room 'G_p':
       - If the robot is holding 'p', it needs at least dist(robot_loc, G_p) moves + 1 putdown.
       - If the robot is NOT holding 'p', 'p' needs at least dist(p_loc, G_p) moves + 1 pickup + 1 putdown.
       - Since the robot moves one package at a time, the sum of these actions across all undelivered 
         packages is a lower bound on the total plan length.
    2. Robot Travel: If the arm is empty, the robot must move to at least one package location.
       - h += min(dist(robot_loc, p_loc)) for all packages needing delivery.
    """

    def __init__(self, robot_gemini3flash_heuristic_4="robot_gemini3flash_heuristic_4"):
        super().__init__(name=robot_gemini3flash_heuristic_4)
        self.rooms = []
        self.packages = []
        self.dist_matrix = {}
        self.pkg_targets = {} # pkg_idx -> room_idx
        self.armempty_id = -1
        self.rloc_ids = [] # index is room_idx, value is global_id
        self.holding_ids = [] # index is pkg_idx, value is global_id
        self.in_ids = {} # (pkg_idx, room_idx) -> global_id
        self.room_name_to_idx = {}
        self.pkg_name_to_idx = {}

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
        # 1. Identify objects and static connectivity
        room_names = set()
        package_names = set()
        adj = collections.defaultdict(list)
        
        # Process all facts to find objects and goal locations
        state_facts = model.state_explicit_repr(initial_node.state)
        for f_name in state_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "rloc" or pred == "in":
                room_names.add(args[-1])
            if pred == "in":
                package_names.add(args[0])
            if pred == "door":
                room_names.add(args[0])
                room_names.add(args[1])
                adj[args[0]].append(args[1])
                adj[args[1]].append(args[0])
            if pred == "goal_in":
                package_names.add(args[0])
                room_names.add(args[1])

        self.rooms = sorted(list(room_names))
        self.packages = sorted(list(package_names))
        self.room_name_to_idx = {name: i for i, name in enumerate(self.rooms)}
        self.pkg_name_to_idx = {name: i for i, name in enumerate(self.packages)}

        # 2. Precompute distances using BFS
        for start_room in self.rooms:
            dists = {r: float('inf') for r in self.rooms}
            dists[start_room] = 0
            queue = collections.deque([start_room])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dists[v] == float('inf'):
                        dists[v] = dists[u] + 1
                        queue.append(v)
            u_idx = self.room_name_to_idx[start_room]
            for end_room, d in dists.items():
                v_idx = self.room_name_to_idx[end_room]
                self.dist_matrix[(u_idx, v_idx)] = d

        # 3. Build fast bitwise lookups
        self.rloc_ids = [None] * len(self.rooms)
        self.holding_ids = [None] * len(self.packages)
        for i in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(i)
            pred, args = self._parse_fact_name(f.name)
            if pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "rloc":
                self.rloc_ids[self.room_name_to_idx[args[0]]] = f.global_id
            elif pred == "holding":
                self.holding_ids[self.pkg_name_to_idx[args[0]]] = f.global_id
            elif pred == "in":
                p_idx = self.pkg_name_to_idx[args[0]]
                r_idx = self.room_name_to_idx[args[1]]
                self.in_ids[(p_idx, r_idx)] = f.global_id
            elif pred == "goal_in":
                p_idx = self.pkg_name_to_idx[args[0]]
                r_idx = self.room_name_to_idx[args[1]]
                self.pkg_targets[p_idx] = r_idx

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        # Determine robot location
        robot_room_idx = -1
        for i, gid in enumerate(self.rloc_ids):
            if gid is not None and (state >> gid) & 1:
                robot_room_idx = i
                break
        
        if robot_room_idx == -1: return 0

        h = 0
        undelivered_pkgs_locs = []
        holding_any = False
        
        for p_idx in range(len(self.packages)):
            target_room_idx = self.pkg_targets.get(p_idx)
            if target_room_idx is None: continue

            # Case A: Holding this package
            h_id = self.holding_ids[p_idx]
            if h_id is not None and (state >> h_id) & 1:
                h += 1 + self.dist_matrix.get((robot_room_idx, target_room_idx), 100)
                holding_any = True
                continue
            
            # Case B: Package is in a room
            current_room_idx = -1
            for r_idx in range(len(self.rooms)):
                in_id = self.in_ids.get((p_idx, r_idx))
                if in_id is not None and (state >> in_id) & 1:
                    current_room_idx = r_idx
                    break
            
            if current_room_idx != -1 and current_room_idx != target_room_idx:
                # Needs pickup (1) + move (dist) + putdown (1)
                dist = self.dist_matrix.get((current_room_idx, target_room_idx), 100)
                h += 2 + dist
                undelivered_pkgs_locs.append(current_room_idx)

        # If arm empty, robot must travel to the first package
        if not holding_any and undelivered_pkgs_locs:
            min_travel = min(self.dist_matrix.get((robot_room_idx, ploc), 100) for ploc in undelivered_pkgs_locs)
            h += min_travel
            
        # Add a small tie-breaker for task network size to favor progress in decompositions
        h += 0.001 * len(task_network)

        return int(h) if h > 0 else 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
