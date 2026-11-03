from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic15(Heuristic):
    """
    Robot Domain Heuristic
    Bottleneck: Single-carry constraint. The robot can carry only one package at a time, 
    forcing all deliveries to be serial (pickup, move, putdown, move to next).
    
    Admissibility Sketch:
    1. Action Count: Each undelivered package requires at least one pickup and one putdown (2 actions). 
       If currently held, it requires at least one putdown (1 action). This is a strict lower bound.
    2. Travel Distance: The robot must traverse the room graph. The sum of move actions between 
       each package's origin and its target room is an admissible lower bound on movement costs.
    3. Door Constraints: Any closed door on a mandatory path must be opened (1 action each).
    Total h = sum(delivery_costs) + distance_to_first_target, which remains admissible due to the serial bottleneck.
    """
    def __init__(self, robot_gemini3flash_heuristic_15="robot_gemini3flash_heuristic_15"):
        super().__init__(robot_gemini3flash_heuristic_15)
        self.package_goal_rooms = {}
        self.room_adj = collections.defaultdict(list)
        self.room_dists = []
        self.room_paths = {} # (r1, r2) -> [door_ids]
        self.fact_info = {} # fact_id -> (pred_type, arg1, arg2)
        self.armempty_id = -1
        self.rloc_facts = {} # room_idx -> fact_id
        self.pkg_ids = []
        self.room_ids = []
        self.door_ids = []

    def initialize(self, model: Model, initial_node: HTNNode):
        fact_map = {}
        pkg_names = []
        room_names = []
        door_names = []
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "package": pkg_names.append(args[0])
            elif pred == "room": room_names.append(args[0])
            elif pred == "roomdoor": door_names.append(args[0])
        
        if not pkg_names: # Fallback if types not explicit in facts
            for f in model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred == "in":
                    if args[0] not in pkg_names: pkg_names.append(args[0])
                    if args[1] not in room_names: room_names.append(args[1])
                elif pred == "door":
                    if args[0] not in room_names: room_names.append(args[0])
                    if args[1] not in room_names: room_names.append(args[1])
                    if args[2] not in door_names: door_names.append(args[2])

        p_map = {name: i for i, name in enumerate(pkg_names)}
        r_map = {name: i for i, name in enumerate(room_names)}
        d_map = {name: i for i, name in enumerate(door_names)}
        self.pkg_ids = list(range(len(pkg_names)))
        self.room_ids = list(range(len(room_names)))

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "armempty": self.armempty_id = f.global_id
            elif pred == "rloc":
                ridx = r_map[args[0]]
                self.rloc_facts[ridx] = f.global_id
                self.fact_info[f.global_id] = ("rloc", -1, ridx)
            elif pred == "in":
                pidx, ridx = p_map[args[0]], r_map[args[1]]
                self.fact_info[f.global_id] = ("in", pidx, ridx)
            elif pred == "holding":
                pidx = p_map[args[0]]
                self.fact_info[f.global_id] = ("holding", pidx, -1)
            elif pred == "closed":
                didx = d_map[args[0]]
                self.fact_info[f.global_id] = ("closed", -1, didx)
            elif pred == "door":
                r1, r2, d = r_map[args[0]], r_map[args[1]], d_map[args[2]]
                self.room_adj[r1].append((r2, d))
            elif pred == "goal_in":
                pidx, ridx = p_map[args[0]], r_map[args[1]]
                if pidx not in self.package_goal_rooms: self.package_goal_rooms[pidx] = set()
                self.package_goal_rooms[pidx].add(ridx)

        num_rooms = len(room_names)
        self.room_dists = [[1000] * num_rooms for _ in range(num_rooms)]
        for r in range(num_rooms):
            self.room_dists[r][r] = 0
            q = collections.deque([r])
            visited = {r}
            self.room_paths[(r, r)] = []
            while q:
                u = q.popleft()
                for v, d in self.room_adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.room_dists[r][v] = self.room_dists[r][u] + 1
                        self.room_paths[(r, v)] = self.room_paths.get((r, u), []) + [d]
                        q.append(v)
        
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return None, []

    def _compute(self, state, task_network):
        curr_rloc = -1
        held_pkg = -1
        is_armempty = (state >> self.armempty_id) & 1 if self.armempty_id != -1 else False
        pkg_locs = {} # pidx -> ridx
        closed_doors = set()

        temp_state = state
        while temp_state:
            bit = (temp_state & -temp_state).bit_length() - 1
            info = self.fact_info.get(bit)
            if info:
                f_type, p_idx, r_idx = info
                if f_type == "rloc": curr_rloc = r_idx
                elif f_type == "holding": held_pkg = p_idx
                elif f_type == "in": pkg_locs[p_idx] = r_idx
                elif f_type == "closed": closed_doors.add(r_idx) # r_idx stores d_idx here
            temp_state &= temp_state - 1

        h = 0
        min_dist_to_pkg = 1000
        undelivered_found = False

        for p_idx in self.pkg_ids:
            goals = self.package_goal_rooms.get(p_idx, set())
            if p_idx == held_pkg:
                undelivered_found = True
                best_g_dist = 1000
                best_g_idx = -1
                for g_idx in goals:
                    d = self.room_dists[curr_rloc][g_idx]
                    if d < best_g_dist:
                        best_g_dist = d
                        best_g_idx = g_idx
                h += 1 + best_g_dist # putdown + moves
                if best_g_idx != -1:
                    for d_idx in self.room_paths.get((curr_rloc, best_g_idx), []):
                        if d_idx in closed_doors: h += 1
            else:
                loc = pkg_locs.get(p_idx, -1)
                if loc != -1 and loc not in goals:
                    undelivered_found = True
                    best_g_dist = 1000
                    best_g_idx = -1
                    for g_idx in goals:
                        d = self.room_dists[loc][g_idx]
                        if d < best_g_dist:
                            best_g_dist = d
                            best_g_idx = g_idx
                    h += 2 + best_g_dist # pickup + putdown + moves
                    if best_g_idx != -1:
                        for d_idx in self.room_paths.get((loc, best_g_idx), []):
                            if d_idx in closed_doors: h += 1
                    
                    if is_armempty:
                        d_to_p = self.room_dists[curr_rloc][loc]
                        if d_to_p < min_dist_to_pkg:
                            min_dist_to_pkg = d_to_p
                            # Add open actions for path to origin
                            door_penalty = sum(1 for d_idx in self.room_paths.get((curr_rloc, loc), []) if d_idx in closed_doors)
                            min_dist_to_pkg += door_penalty

        if is_armempty and undelivered_found and min_dist_to_pkg < 1000:
            h += min_dist_to_pkg
            
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
