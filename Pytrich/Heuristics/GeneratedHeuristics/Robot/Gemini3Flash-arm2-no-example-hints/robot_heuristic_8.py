from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic8(Heuristic):
    """
    Domain-Dependent Heuristic for the Robot HDDL Domain.

    BOTTLENECK ANALYSIS:
    The primary bottleneck in this domain is the single-carry constraint. The robot can only hold one package
    at a time, forcing a strictly serial execution of deliveries. This means the total cost is the sum of 
    individual delivery costs (pickup + travel to target + putdown) plus the travel cost to the next pickup.

    ADMISSIBLE LOWER BOUNDS:
    1. Delivery Actions: Each undelivered package requires at least one pickup and one putdown (2 actions).
       If a package is already being held, it requires at least one putdown (1 action).
    2. Travel Distance: The robot must move from its current location to the target of the held package,
       and for every other undelivered package, it must travel from its starting room to its target room.
    3. Navigation Connectivity: The shortest path (BFS) distance between rooms is an admissible lower bound 
       on the number of 'move' actions required.
    """

    def __init__(self, robot_gemini3flash_heuristic_8="robot_gemini3flash_heuristic_8"):
        super().__init__(robot_gemini3flash_heuristic_8)
        self.target_rooms = {}  # pkg -> room_name
        self.rloc_fact_to_room = {} # fact_id -> room_name
        self.holding_fact_to_pkg = {} # fact_id -> pkg_name
        self.in_fact_to_pkg_room = {} # fact_id -> (pkg_name, room_name)
        self.dist_matrix = {} # room -> room -> int
        self.room_names = []

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
        # 1. Map predicates and build distance matrix
        rooms = set()
        adj = {}
        
        # Parse all facts for mapping
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                rooms.add(args[0])
                self.rloc_fact_to_room[fact.global_id] = args[0]
            elif pred == "holding":
                self.holding_fact_to_pkg[fact.global_id] = args[0]
            elif pred == "in":
                self.in_fact_to_pkg_room[fact.global_id] = (args[0], args[1])
            elif pred == "door":
                r1, r2, d = args
                if r1 not in adj: adj[r1] = []
                adj[r1].append(r2)

        self.room_names = list(rooms)
        # BFS for all-pairs shortest paths (unweighted)
        for r_start in self.room_names:
            self.dist_matrix[r_start] = {r_end: 1000 for r_end in self.room_names}
            self.dist_matrix[r_start][r_start] = 0
            queue = [(r_start, 0)]
            visited = {r_start}
            while queue:
                curr, d = queue.pop(0)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist_matrix[r_start][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        # 2. Identify package goals from initial state (goal_in is semi-static)
        init_state = initial_node.state
        for fact in model.facts:
            if (init_state >> fact.global_id) & 1:
                pred, args = self._parse_fact_name(fact.name)
                if pred == "goal_in":
                    self.target_rooms[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        if not task_network and (state & self.model.goals == self.model.goals):
            return 0

        rloc_room = None
        holding_pkg = None
        wrong_pkgs = [] # list of (pkg_name, current_room_name)

        # Efficient state scanning using bit manipulation
        temp_state = state
        while temp_state:
            bit_idx = (temp_state & -temp_state).bit_length() - 1
            if bit_idx in self.rloc_fact_to_room:
                rloc_room = self.rloc_fact_to_room[bit_idx]
            elif bit_idx in self.holding_fact_to_pkg:
                holding_pkg = self.holding_fact_to_pkg[bit_idx]
            elif bit_idx in self.in_fact_to_pkg_room:
                p, r = self.in_fact_to_pkg_room[bit_idx]
                target = self.target_rooms.get(p)
                if target and r != target:
                    wrong_pkgs.append((p, r))
            temp_state ^= (1 << bit_idx)

        # Base actions: pickups (1 each) and putdowns (1 each)
        h = (2 * len(wrong_pkgs))
        if holding_pkg:
            h += 1

        # Movement costs
        if rloc_room:
            # 1. Travel for the currently held package
            if holding_pkg:
                target = self.target_rooms.get(holding_pkg)
                if target:
                    h += self.dist_matrix[rloc_room].get(target, 0)
                    # After dropping off current, must travel to the next wrong package
                    if wrong_pkgs:
                        h += min(self.dist_matrix[target].get(p_loc, 0) for p_name, p_loc in wrong_pkgs)
            # 2. If empty, travel to the first wrong package
            elif wrong_pkgs:
                h += min(self.dist_matrix[rloc_room].get(p_loc, 0) for p_name, p_loc in wrong_pkgs)

            # 3. Sum of travel for all packages from their start to their target
            for p_name, p_loc in wrong_pkgs:
                target = self.target_rooms.get(p_name)
                if target:
                    h += self.dist_matrix[p_loc].get(target, 0)

        # Tie-breaker: small penalty for task network size to encourage decomposition
        return int(h) + (1 if task_network else 0)
