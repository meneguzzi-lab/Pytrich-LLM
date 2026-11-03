import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic16(Heuristic):
    """
    Domain-dependent heuristic for the Robot planning domain.

    Bottleneck Analysis:
    The primary bottleneck in this domain is the single-package capacity of the robot's arm (armempty/holding).
    This serializes the delivery of all packages. Every undelivered package requires a specific sequence of 
    actions: moving to its location, picking it up, moving to the goal, and putting it down. Since only one 
    package can be carried at a time, the total plan cost is at least the sum of these individual delivery costs.

    Admissibility Sketches:
    1. Delivery Cost: Each package currently not at its goal requires at least dist(start, goal) moves plus 
       one pickup and one putdown action (2 actions). This is admissible because these actions are mandatory 
       and cannot be shared between packages due to the single-arm constraint.
    2. Travel Cost: The robot must travel to the first package's location. The distance from the robot's 
       current location to the nearest pickup or the goal of a held package is a required lower bound.
    3. Door Opening: Every closed door on a mandatory path adds at least one 'open' action. (Simplified here 
       to travel distance for speed, but distance is still admissible).
    """

    def __init__(self, robot_gemini3flash_heuristic_16=None, name="RobotGemini3FlashHeuristic16"):
        super().__init__(name=name)
        self.goal_rooms = {}  # pkg -> target_room
        self.dist_matrix = defaultdict(lambda: defaultdict(lambda: 1000))
        self.fact_info = {}   # global_id -> (type, arg1, arg2...)
        self.packages = set()
        self.rooms = set()
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # 1. Map facts and extract static information
        adj = defaultdict(list)
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == 'rloc':
                self.fact_info[f.global_id] = ('rloc', args[0])
                self.rooms.add(args[0])
            elif pred == 'in':
                self.fact_info[f.global_id] = ('in', args[0], args[1])
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == 'holding':
                self.fact_info[f.global_id] = ('holding', args[0])
                self.packages.add(args[0])
            elif pred == 'door' and len(args) >= 2:
                r1, r2 = args[0], args[1]
                adj[r1].append(r2)
                self.rooms.add(r1)
                self.rooms.add(r2)

        # 2. Extract goals (from initial state properties and model goals)
        # goal_in is often a static property in Robot domain problems
        init_facts_names = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'goal_in' and len(args) >= 2:
                self.goal_rooms[args[0]] = args[1]

        # Check model.goals as well in case they are explicit in the goal bitset
        if hasattr(model, 'goals') and model.goals:
            goal_facts_names = model.state_explicit_repr(model.goals)
            for f_name in goal_facts_names:
                pred, args = self._parse_fact_name(f_name)
                if pred == 'in' and len(args) >= 2:
                    self.goal_rooms[args[0]] = args[1]
                elif pred == 'goal_in' and len(args) >= 2:
                    self.goal_rooms[args[0]] = args[1]

        # 3. Precompute all-pairs shortest paths for rooms
        for start_node in self.rooms:
            self.dist_matrix[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                curr_d = self.dist_matrix[start_node][u]
                for v in adj[u]:
                    if self.dist_matrix[start_node][v] == 1000:
                        self.dist_matrix[start_node][v] = curr_d + 1
                        queue.append(v)

        self.preprocessing_time = time.time() - start_time
        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

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

    def _compute(self, state, task_network):
        # Extract state info via bit scanning
        robot_loc = None
        package_locs = {} # pkg -> current_room
        held_pkg = None

        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            fid = lsb.bit_length() - 1
            info = self.fact_info.get(fid)
            if info:
                if info[0] == 'rloc':
                    robot_loc = info[1]
                elif info[0] == 'in':
                    package_locs[info[1]] = info[2]
                elif info[0] == 'holding':
                    held_pkg = info[1]
            temp_state ^= lsb

        h = 0
        undelivered = []
        for pkg in self.packages:
            if pkg == held_pkg:
                continue
            curr_loc = package_locs.get(pkg)
            goal_loc = self.goal_rooms.get(pkg)
            if curr_loc and goal_loc and curr_loc != goal_loc:
                undelivered.append((pkg, curr_loc, goal_loc))

        if held_pkg:
            goal = self.goal_rooms.get(held_pkg)
            if goal and robot_loc:
                h += self.dist_matrix[robot_loc][goal] + 1
            
            if undelivered:
                # Robot must finish current delivery, then move to others
                h += sum(self.dist_matrix[loc][gl] + 2 for p, loc, gl in undelivered)
                if goal:
                    h += min(self.dist_matrix[goal][loc] for p, loc, gl in undelivered)
        else:
            if not undelivered:
                return 0.001 * len(task_network)
            
            # Sum costs of picking up and delivering every remaining package
            h += sum(self.dist_matrix[loc][gl] + 2 for p, loc, gl in undelivered)
            # Add robot travel to the first package
            if robot_loc:
                h += min(self.dist_matrix[robot_loc][loc] for p, loc, gl in undelivered)

        # Standard tie-breaker for HTN
        return h + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return int(h_val)

    def __repr__(self):
        return f"RobotGemini3FlashHeuristic16(pkgs={len(self.packages)}, rooms={len(self.rooms)})"
