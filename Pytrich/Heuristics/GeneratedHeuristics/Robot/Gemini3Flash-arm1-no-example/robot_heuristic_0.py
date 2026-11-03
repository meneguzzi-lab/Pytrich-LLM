from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic0(Heuristic):
    """
    Heuristic for the Robot HDDL domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the robot's single arm. 
    The robot can carry at most one package at a time. This means that 
    delivering multiple packages is a strictly sequential process.
    
    Admissibility Sketches:
    1. Action Cost Bound: Every package not at its goal room requires at 
       least a 'putdown' action (1 cost). If it's not currently held, it 
       also requires a 'pickup' action (1 cost). Because the robot is 
       serial, these costs are additive across all missing packages.
    2. Movement Bound: Every package must be moved from its current room 
       to its target room. The number of 'move' actions required is at 
       least the shortest-path distance between the rooms. These distances 
       are also additive because the robot cannot transport multiple 
       packages simultaneously.
    3. Initial Move: If the robot is not currently holding a package, it 
       must move to at least one package to start working.
    """

    def __init__(self, robot_gemini3flash_heuristic_0="robot_heuristic_0"):
        super().__init__(robot_gemini3flash_heuristic_0)
        self.targets = {} # pkg -> target_room
        self.distances = {} # (r1, r2) -> min_moves
        self.pkg_names = []
        self.room_names = []
        self.rloc_ids = {} # room -> global_id
        self.in_fact_ids = {} # pkg -> room -> global_id
        self.holding_ids = {} # pkg -> global_id
        self.armempty_id = None

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to identify rooms and packages
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                r = args[0]
                if r not in self.room_names: self.room_names.append(r)
                self.rloc_ids[r] = f.global_id
            elif pred == "holding":
                p = args[0]
                if p not in self.pkg_names: self.pkg_names.append(p)
                self.holding_ids[p] = f.global_id
            elif pred == "in":
                p, r = args[0], args[1]
                if p not in self.in_fact_ids: self.in_fact_ids[p] = {}
                self.in_fact_ids[p][r] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id

        # 2. Identify target locations from model goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "in":
                self.targets[args[0]] = args[1]

        # 3. Build room adjacency and compute all-pairs shortest paths
        room_adj = {r: [] for r in self.room_names}
        init_facts = model.state_explicit_repr(initial_node.state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "door":
                r1, r2 = args[0], args[1]
                if r2 not in room_adj[r1]: room_adj[r1].append(r2)
                if r1 not in room_adj[r2]: room_adj[r2].append(r1)

        for start_r in self.room_names:
            self.distances[(start_r, start_r)] = 0
            queue = [(start_r, 0)]
            visited = {start_r}
            while queue:
                curr, d = queue.pop(0)
                for neighbor in room_adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_r, neighbor)] = d + 1
                        queue.append((neighbor, d + 1))

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # A state with no goals is 0 cost
        if not self.targets: return 0

        # 1. Identify robot location
        rob_room = None
        for r in self.room_names:
            if (state >> self.rloc_ids[r]) & 1:
                rob_room = r
                break
        if rob_room is None: return 0

        total_cost = 0
        missing_pkgs = []
        holding_any_needed = False

        # 2. Calculate sequential delivery cost for every package not at its goal
        for p, target in self.targets.items():
            # Check if package is at its target
            at_goal = (state >> self.in_fact_ids[p][target]) & 1
            if at_goal:
                continue
            
            missing_pkgs.append(p)
            # Case A: Robot is holding the package
            if (state >> self.holding_ids[p]) & 1:
                holding_any_needed = True
                # Cost = putdown + distance to target room
                total_cost += 1 + self.distances.get((rob_room, target), 10000)
            else:
                # Case B: Package is in some room (not the target)
                p_room = None
                for r in self.room_names:
                    if (state >> self.in_fact_ids[p][r]) & 1:
                        p_room = r
                        break
                if p_room:
                    # Cost = pickup + putdown + distance from p_room to target
                    total_cost += 2 + self.distances.get((p_room, target), 10000)

        # 3. If the robot isn't holding any necessary package, it must move to the first one
        if missing_pkgs and not holding_any_needed:
            min_dist_to_p = 10000
            for p in missing_pkgs:
                # Find where p is currently located to calculate distance from robot
                for r in self.room_names:
                    if (state >> self.in_fact_ids[p][r]) & 1:
                        min_dist_to_p = min(min_dist_to_p, self.distances.get((rob_room, r), 10000))
                        break
            if min_dist_to_p < 10000:
                total_cost += min_dist_to_p

        # 4. Tie-breaker: small penalty for task network complexity to break symmetries
        return int(total_cost) + (0.001 * len(task_network))

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
            return name, []
        return None, []
