from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic17(Heuristic):
    """
    Heuristic for the Robot HDDL domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the sequential nature of transport. The robot can carry only one package at a time
    and must physically move between rooms to pick up and drop off packages. Every package not at its 
    goal destination requires a minimum of one pickup and one putdown action. Travel between rooms 
    adds a variable number of move actions.

    Admissibility Sketches:
    - Term 1 (Transport Cost): Every package currently not at its goal room requires a pickup and a putdown (2 actions).
      If the robot is already holding the package, it only requires a putdown (1 action). These are disjoint
      operator applications (pickup/putdown) and thus the sum is a valid lower bound.
    - Term 2 (Travel Cost): To perform the next necessary transport action, the robot must move to the location 
      of a package (for pickup) or to the destination of the held package (for putdown). BFS distance in the 
      room-door graph provides an admissible lower bound on the number of 'move' operators required.
    """

    def __init__(self, robot_gemini3flash_heuristic_17="robot_heuristic_17"):
        super().__init__(name=robot_gemini3flash_heuristic_17)
        self.package_goals = {}  # pkg -> target_room
        self.dist = {}           # room1 -> room2 -> min_moves
        self.dynamic_fact_mask = 0
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
        # 1. Parse static structure
        rooms = set()
        adj = {}
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "goal_in":
                self.package_goals[args[0]] = args[1]
            elif pred == "door":
                r1, r2 = args[0], args[1]
                rooms.add(r1)
                rooms.add(r2)
                if r1 not in adj: adj[r1] = set()
                if r2 not in adj: adj[r2] = set()
                adj[r1].add(r2)
                adj[r2].add(r1)
            elif pred == "rloc":
                rooms.add(args[0])

        self.room_names = list(rooms)
        
        # 2. Precompute all-pairs BFS distances (moves)
        self.dist = {r: {r2: 999999 for r2 in self.room_names} for r in self.room_names}
        for r in self.room_names:
            self.dist[r][r] = 0
            if r in adj:
                queue = [(r, 0)]
                visited = {r}
                while queue:
                    curr, d = queue.pop(0)
                    self.dist[r][curr] = d
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

        # 3. Define dynamic mask for fast __call__ bit-walking
        self.dynamic_fact_mask = 0
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred in ["rloc", "in", "holding", "closed", "armempty"]:
                self.dynamic_fact_mask |= (1 << f.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Efficient bit-walking over dynamic state
        dynamic_bits = state & self.dynamic_fact_mask
        curr_room = None
        held_pkg = None
        packages_to_move = {} # pkg -> current_loc

        while dynamic_bits:
            bit = dynamic_bits & -dynamic_bits
            idx = bit.bit_length() - 1
            dynamic_bits ^= bit
            
            fact_name = self.model.facts[idx].name
            pred, args = self._parse_fact_name(fact_name)
            
            if pred == "rloc":
                curr_room = args[0]
            elif pred == "holding":
                held_pkg = args[0]
            elif pred == "in":
                pkg, loc = args[0], args[1]
                # Only track if not already at goal
                if pkg in self.package_goals and loc != self.package_goals[pkg]:
                    packages_to_move[pkg] = loc

        # Term 1: Transport cost (2 per pkg to move, 1 if holding)
        h_transport = 2 * len(packages_to_move)
        if held_pkg:
            h_transport += 1

        # Term 2: Travel cost (moves to next location)
        h_dist = 0
        if curr_room:
            if held_pkg:
                # Must move to goal room of held package
                target = self.package_goals.get(held_pkg)
                if target and curr_room in self.dist and target in self.dist[curr_room]:
                    h_dist = self.dist[curr_room][target]
            elif packages_to_move:
                # Must move to the nearest package pickup
                min_d = 999999
                for loc in packages_to_move.values():
                    if curr_room in self.dist and loc in self.dist[curr_room]:
                        min_d = min(min_d, self.dist[curr_room][loc])
                h_dist = min_d if min_d != 999999 else 0

        # Total admissible estimate + tie-breaker for task resolution progress
        h_total = h_transport + h_dist
        h_total += 0.001 * len(task_network)
        
        return int(h_total) if h_total < 1000000 else 1000000

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
