import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Robot planning domain.

BOTTLENECK ANALYSIS:
The robot has a single arm (armempty), meaning it can only carry one package at a time.
This serializes the entire task: for every package not at its goal, the robot must:
1. Move to the package's current location.
2. Pickup the package.
3. Move to the goal location.
4. Putdown the package.
Because of the single-arm constraint, these action sequences are largely disjoint and additive.

ADMISSIBILITY SKETCH:
1. Package Transport Cost (h_sum): Each unmoved package requires at least 1 pickup, N moves (shortest path), 
   and 1 putdown. If the package is already held, it requires N moves and 1 putdown. 
   Since the robot can only move one package at a time, the sum of these costs is a lower bound.
2. Robot Travel Cost (h_travel): The robot must move from its current location to the first package's 
   starting location or the current held package's goal. This adds unavoidable move actions.
3. Door Opening Cost: Although not explicitly added here to ensure strict admissibility across 
   complex room layouts, every closed door on a mandatory path adds at least one 'open' action.

Total H = h_sum + h_travel
"""

class RobotGemini3FlashHeuristic13(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_13=None, name="RobotGemini3FlashHeuristic13"):
        super().__init__(name=name)
        self.goal_rooms = {}      # package -> goal room
        self.pkg_in_facts = {}    # package -> {room -> fact_id}
        self.pkg_holding_facts = {} # package -> fact_id
        self.room_rloc_facts = {}  # room -> fact_id
        self.armempty_fact_id = None
        self.dist_matrix = {}
        self.all_rooms = []
        self.all_packages = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

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

    def _preprocess(self):
        # 1. Identify goal rooms for packages
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for gn in goal_names:
                pred, args = self._parse_fact_name(gn)
                if pred == "in" and len(args) == 2:
                    self.goal_rooms[args[0]] = args[1]

        adj = defaultdict(list)
        pkgs = set()
        rooms = set()

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "goal_in":
                self.goal_rooms[args[0]] = args[1]
            elif pred == "door":
                r1, r2, d = args
                adj[r1].append(r2)
                rooms.add(r1)
                rooms.add(r2)
            elif pred == "in":
                p, r = args
                pkgs.add(p)
                if p not in self.pkg_in_facts: self.pkg_in_facts[p] = {}
                self.pkg_in_facts[p][r] = fact.global_id
            elif pred == "holding":
                p = args[0]
                pkgs.add(p)
                self.pkg_holding_facts[p] = fact.global_id
            elif pred == "rloc":
                r = args[0]
                rooms.add(r)
                self.room_rloc_facts[r] = fact.global_id
            elif pred == "armempty":
                self.armempty_fact_id = fact.global_id

        self.all_rooms = list(rooms)
        self.all_packages = list(pkgs)

        # 2. Precompute All-Pairs Shortest Paths (BFS)
        for r_start in self.all_rooms:
            self.dist_matrix[r_start] = {r: 1000 for r in self.all_rooms}
            self.dist_matrix[r_start][r_start] = 0
            queue = deque([r_start])
            while queue:
                u = queue.popleft()
                d = self.dist_matrix[r_start][u]
                for v in adj[u]:
                    if self.dist_matrix[r_start][v] == 1000:
                        self.dist_matrix[r_start][v] = d + 1
                        queue.append(v)

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        # Current Robot Location
        current_room = None
        for r, fid in self.room_rloc_facts.items():
            if (state_bitwise >> fid) & 1:
                current_room = r
                break

        if current_room is None:
            return 0

        # Analyze packages
        held_pkg = None
        unmoved = [] # (package, current_room)

        for p in self.all_packages:
            # Is it held?
            if p in self.pkg_holding_facts and (state_bitwise >> self.pkg_holding_facts[p]) & 1:
                held_pkg = p
                continue
            
            # Is it at its goal?
            goal = self.goal_rooms.get(p)
            at_goal = False
            if goal and p in self.pkg_in_facts and goal in self.pkg_in_facts[p]:
                if (state_bitwise >> self.pkg_in_facts[p][goal]) & 1:
                    at_goal = True
            
            if not at_goal:
                # Find where it is
                for r, fid in self.pkg_in_facts.get(p, {}).items():
                    if (state_bitwise >> fid) & 1:
                        unmoved.append((p, r))
                        break

        h = 0
        # Cost for held package
        if held_pkg:
            g_room = self.goal_rooms.get(held_pkg)
            if g_room:
                h += 1 # putdown
                h += self.dist_matrix[current_room].get(g_room, 0)
        
        # Cost for others
        for p, r in unmoved:
            g_room = self.goal_rooms.get(p)
            if g_room:
                h += 2 # pickup + putdown
                h += self.dist_matrix[r].get(g_room, 0)

        # Travel cost to first task
        if not held_pkg and unmoved:
            travel = min(self.dist_matrix[current_room].get(r, 0) for p, r in unmoved)
            h += travel
        elif held_pkg and unmoved:
            # After delivering held, robot must go to next
            g_held = self.goal_rooms.get(held_pkg)
            if g_held:
                travel = min(self.dist_matrix[g_held].get(r, 0) for p, r in unmoved)
                h += travel

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
