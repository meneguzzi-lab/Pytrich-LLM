from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic12(Heuristic):
    """
    Heuristic for the Robot domain.

    1. Bottleneck: Single-carry constraint. The robot can only hold one package at a time.
       Every delivery requires a disjoint sequence of (Move -> Pickup -> Move -> Putdown).

    2. Admissible lower bounds:
       - Action Term: Each undelivered package requires at least 2 actions (pickup/putdown),
         or 1 if already held.
       - Carry Term: Each undelivered package requires the robot to move from its starting
         location to its goal location (dist(start, goal) moves).
       - Navigation Term: The robot must move from its current location to the first target
         (either the pickup point of the next package or the goal of the held package).

    Summation of these terms is admissible because the actions for each package delivery
    and the initial navigation are disjoint in the plan.
    """

    def __init__(self, robot_gemini3flash_heuristic_12="robot_gemini3flash_heuristic_12"):
        super().__init__(robot_gemini3flash_heuristic_12)
        self.package_goal = {}  # pkg_name -> room_name
        self.package_start = {} # pkg_name -> room_name
        self.room_dist = {}     # (r1, r2) -> int
        
        self.fact_in_goal = {}  # pkg_name -> fact_id (in pkg goal_room)
        self.fact_holding = {}  # pkg_name -> fact_id
        self.fact_in_any = collections.defaultdict(list) # pkg_name -> [(room, fact_id)]
        self.fact_rloc = {}     # room_name -> fact_id
        self.fact_armempty = None

        self.packages = []
        self.rooms = []

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
        # 1. Map predicates to IDs
        temp_adj = collections.defaultdict(list)
        initial_state = initial_node.state

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == "goal_in":
                self.package_goal[args[0]] = args[1]
            elif pred == "in":
                self.fact_in_any[args[0]].append((args[1], f.global_id))
                # Find start location from initial state
                if (initial_state >> f.global_id) & 1:
                    self.package_start[args[0]] = args[1]
            elif pred == "holding":
                self.fact_holding[args[0]] = f.global_id
            elif pred == "rloc":
                self.fact_rloc[args[1]] = f.global_id
            elif pred == "armempty":
                self.fact_armempty = f.global_id
            elif pred == "door":
                temp_adj[args[0]].append(args[1])

        self.packages = list(self.package_goal.keys())
        self.rooms = list(self.fact_rloc.keys())

        # 2. Precompute distances using BFS
        for start_node in self.rooms:
            dists = {start_node: 0}
            queue = collections.deque([start_node])
            while queue:
                u = queue.popleft()
                for v in temp_adj[u]:
                    if v not in dists:
                        dists[v] = dists[u] + 1
                        queue.append(v)
            for end_node, d in dists.items():
                self.room_dist[(start_node, end_node)] = d

        # 3. Cache 'in_at_goal' IDs
        for pkg, g_room in self.package_goal.items():
            for room, fid in self.fact_in_any[pkg]:
                if room == g_room:
                    self.fact_in_goal[pkg] = fid
                    break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network) -> int:
        if not task_network:
            return 0

        # Find current robot location
        curr_rloc = None
        for rname, fid in self.fact_rloc.items():
            if (state >> fid) & 1:
                curr_rloc = rname
                break
        if not curr_rloc:
            return 0

        h = 0
        undelivered = []
        held_pkg = None

        # Check each package
        for p in self.packages:
            # If in goal room and robot not holding it, it's done
            is_held = (state >> self.fact_holding[p]) & 1
            if is_held:
                held_pkg = p
                continue
            
            is_in_goal = (state >> self.fact_in_goal[p]) & 1
            if not is_in_goal:
                undelivered.append(p)

        # Calculate costs
        # Case 1: Holding a package
        if held_pkg:
            g_room = self.package_goal[held_pkg]
            h += 1 + self.room_dist.get((curr_rloc, g_room), 0)
            # All other undelivered packages must start from their current location
            for p in undelivered:
                # Find where it is (usually start_loc in this domain structure)
                p_loc = self.package_start.get(p)
                # Find where it's actually in case it's not at start
                for room, fid in self.fact_in_any[p]:
                    if (state >> fid) & 1:
                        p_loc = room
                        break
                h += 2 + self.room_dist.get((p_loc, self.package_goal[p]), 0)
        
        # Case 2: Not holding anything
        else:
            min_nav = 0
            first = True
            for p in undelivered:
                p_loc = self.package_start.get(p)
                for room, fid in self.fact_in_any[p]:
                    if (state >> fid) & 1:
                        p_loc = room
                        break
                
                delivery_cost = 2 + self.room_dist.get((p_loc, self.package_goal[p]), 0)
                h += delivery_cost
                
                nav_to_p = self.room_dist.get((curr_rloc, p_loc), 0)
                if first or nav_to_p < min_nav:
                    min_nav = nav_to_p
                    first = False
            h += min_nav

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
