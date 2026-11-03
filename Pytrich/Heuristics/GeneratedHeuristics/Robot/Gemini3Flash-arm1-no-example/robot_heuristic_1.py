from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic1(Heuristic):
    """
    Robot Domain Heuristic Analysis:
    1. Bottleneck: The single robot arm. Only one package can be carried at a time.
       Work is serialized around this arm, meaning transport tasks must be performed sequentially.
    
    2. Admissible Lower Bounds:
       - T1: Total Transport Cost. Sum of (dist(start, goal) + 1 pickup + 1 putdown) for every 
         package not currently at its goal location. This is admissible because one robot must 
         perform all these distinct actions.
       - T2: Navigation Cost. If the arm is empty, the robot must travel to the nearest 
         available package to begin work.
       - T3: Door Opening Costs. If a door is closed, it adds at least +1 (open action) 
         to any path passing through it. For simplicity and to maintain admissibility while 
         providing guidance, we add a sub-unit penalty for all closed doors.
    
    3. Preprocessing: BFS to precompute all-pairs room distances.
    """

    def __init__(self, robot_gemini3flash_heuristic_1="robot_gemini3flash_heuristic_1"):
        super().__init__(robot_gemini3flash_heuristic_1)
        self.rloc_ids = {} # room_name -> global_id
        self.pkg_in_ids = {} # (pkg_name, room_name) -> global_id
        self.pkg_holding_ids = {} # pkg_name -> global_id
        self.armempty_id = None
        self.closed_door_ids = []
        self.room_list = []
        self.pkg_list = []
        self.goal_pkg_to_room = {}
        self.dist = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to identify relevant objects and their relationships
        adj = {}
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                room = args[0]
                self.rloc_ids[room] = fact.global_id
                if room not in self.room_list: self.room_list.append(room)
            elif pred == "in":
                pkg, room = args[0], args[1]
                self.pkg_in_ids[(pkg, room)] = fact.global_id
                if pkg not in self.pkg_list: self.pkg_list.append(pkg)
            elif pred == "holding":
                pkg = args[0]
                self.pkg_holding_ids[pkg] = fact.global_id
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "closed":
                self.closed_door_ids.append(fact.global_id)
            elif pred == "door":
                r1, r2, _ = args
                if r1 not in adj: adj[r1] = []
                adj[r1].append(r2)

        # 2. Identify target goal locations for packages
        # Use goal state first, fallback to goal_in metadata facts
        goal_names = model.state_explicit_repr(model.goals)
        for f_name in goal_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == "in":
                self.goal_pkg_to_room[args[0]] = args[1]
        
        if not self.goal_pkg_to_room:
            for fact in model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if pred == "goal_in":
                    self.goal_pkg_to_room[args[0]] = args[1]

        # 3. BFS for all-pairs room distances
        for start_r in self.room_list:
            self.dist[start_r] = {r: 1000 for r in self.room_list}
            self.dist[start_r][start_r] = 0
            if start_r not in adj: continue
            queue = deque([(start_r, 0)])
            visited = {start_r}
            while queue:
                curr, d = queue.popleft()
                self.dist[start_r][curr] = d
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state: int, task_network) -> int:
        # Identify robot room
        robot_room = None
        for r, gid in self.rloc_ids.items():
            if (state >> gid) & 1:
                robot_room = r
                break
        
        if not robot_room:
            return 1000

        h = 0
        undone_packages_at_room = []
        holding_pkg = None

        # Calculate transport costs
        for pkg, target_room in self.goal_pkg_to_room.items():
            # Check if held
            if (state >> self.pkg_holding_ids[pkg]) & 1:
                holding_pkg = pkg
                h += self.dist[robot_room][target_room] + 1 # move + putdown
            else:
                # Check if at its goal
                at_goal_id = self.pkg_in_ids.get((pkg, target_room))
                if at_goal_id is not None and (state >> at_goal_id) & 1:
                    continue # Package finished
                else:
                    # Not at goal, not held: find where it is
                    for r in self.room_list:
                        in_id = self.pkg_in_ids.get((pkg, r))
                        if in_id is not None and (state >> in_id) & 1:
                            undone_packages_at_room.append(r)
                            h += self.dist[r][target_room] + 2 # pickup + move + putdown
                            break

        # Navigation cost to next package if arm is free
        if not holding_pkg and undone_packages_at_room:
            h += min(self.dist[robot_room][r] for r in undone_packages_at_room)

        # Tie-breaker 1: Closed doors in the domain
        closed_doors = 0
        for d_id in self.closed_door_ids:
            if (state >> d_id) & 1:
                closed_doors += 1
        h_final = float(h) + (0.1 * closed_doors)

        # Tie-breaker 2: Task network length
        h_final += 0.01 * len(task_network)

        return int(h_final)

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
