from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic13(Heuristic):
    """
    Bottleneck Analysis: 
    The single robot arm constraint serializes all package deliveries (only one package can be carried at a time).

    Admissibility Sketches:
    1. Package Handling: Each package not at its goal room requires at least a 'pickup' and a 'putdown' action (cost 2). 
       If the robot is currently holding a package, that package only needs a 'putdown' (cost 1).
    2. Travel Distance: Each package must be moved from its current location to its goal location via 'move' actions. 
       The minimum number of moves is the shortest-path distance between the rooms in the door-adjacency graph.
    3. Robot Access: If the robot is empty, it must move to at least one package's location before picking it up. 
       The distance to the nearest pending package is a lower bound on this movement.
    Because the robot can only carry one package, these costs (handling + travel per package + initial robot travel) 
    represent a summation of sequential actions that must occur, making it a strong admissible lower bound.
    """
    def __init__(self, robot_gemini3flash_heuristic_13="RobotGemini3FlashHeuristic13"):
        super().__init__(robot_gemini3flash_heuristic_13)
        self.package_goals = {} # pkg_idx -> room_idx
        self.room_ids = {} # room_name -> int
        self.pkg_ids = {} # pkg_name -> int
        self.room_list = []
        self.pkg_list = []
        self.dist_matrix = {}
        self.fact_to_info = {}

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
        self.model = model
        rooms = set()
        packages = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred in ['rloc', 'in', 'goal_in', 'door']:
                for a in args: 
                    if a.startswith('r') or a == 'c': rooms.add(a)
                    elif a.startswith('o'): packages.add(a)
        
        self.room_list = sorted(list(rooms))
        self.room_ids = {name: i for i, name in enumerate(self.room_list)}
        self.pkg_list = sorted(list(packages))
        self.pkg_ids = {name: i for i, name in enumerate(self.pkg_list)}

        adj = {i: set() for i in range(len(self.room_list))}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == 'rloc':
                self.fact_to_info[f.global_id] = ('rloc', self.room_ids[args[0]])
            elif pred == 'in':
                self.fact_to_info[f.global_id] = ('in', (self.pkg_ids[args[0]], self.room_ids[args[1]]))
            elif pred == 'holding':
                self.fact_to_info[f.global_id] = ('holding', self.pkg_ids[args[0]])
            elif pred == 'goal_in':
                self.package_goals[self.pkg_ids[args[0]]] = self.room_ids[args[1]]
            elif pred == 'door':
                r1, r2 = self.room_ids[args[0]], self.room_ids[args[1]]
                adj[r1].add(r2)
                adj[r2].add(r1)

        num_rooms = len(self.room_list)
        for start in range(num_rooms):
            self.dist_matrix[(start, start)] = 0
            q = [(start, 0)]
            visited = {start}
            while q:
                u, d = q.pop(0)
                self.dist_matrix[(start, u)] = d
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append((v, d + 1))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        robot_room = -1
        holding_pkg = -1
        pkg_locs = {}
        
        bits = state
        while bits:
            lsb = bits & -bits
            g_id = lsb.bit_length() - 1
            if g_id in self.fact_to_info:
                type, val = self.fact_to_info[g_id]
                if type == 'rloc': robot_room = val
                elif type == 'holding': holding_pkg = val
                elif type == 'in': pkg_locs[val[0]] = val[1]
            bits ^= lsb

        if robot_room == -1: return 0

        total_h = 0
        pending_pkgs = []
        
        for p_idx in range(len(self.pkg_list)):
            goal_room = self.package_goals.get(p_idx)
            if goal_room is None: continue
            if p_idx == holding_pkg:
                total_h += 1 + self.dist_matrix.get((robot_room, goal_room), 0)
            elif p_idx in pkg_locs and pkg_locs[p_idx] != goal_room:
                total_h += 2 + self.dist_matrix.get((pkg_locs[p_idx], goal_room), 0)
                pending_pkgs.append(pkg_locs[p_idx])

        if holding_pkg == -1 and pending_pkgs:
            total_h += min(self.dist_matrix.get((robot_room, loc), 0) for loc in pending_pkgs)

        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
