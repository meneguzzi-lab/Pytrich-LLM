import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Robot transport domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the robot's single-arm capacity. It can only 
    carry one package at a time. This serializes all deliveries. Additionally, the robot 
    must travel between rooms to pick up and drop off packages, and potentially open doors.

    Admissibility Sketches:
    1. Action Cost (Term 1): Every package not yet at its goal room requires at least 
       one 'putdown' action. If the robot is not currently holding the package, it also 
       requires at least one 'pickup' action. These actions are distinct from movement.
       Lower Bound = sum(1 if held else 2 for p in unsettled_packages).
    2. Travel Cost (Term 2): The robot must traverse a sequence of rooms to deliver packages.
       A lower bound on movement actions is the distance to finish the 'closest' available 
       package delivery (Distance to package + Distance to goal). Since every 'move' is an 
       action, the shortest path distance in the room-door graph is a lower bound.
    3. Total Cost: Since pickup/putdown and moves are disjoint sets of actions, 
       Term 1 + Term 2 is an admissible lower bound on the total plan length.
    """

    def __init__(self, robot_gemini3flash_heuristic_5=None, name="RobotGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.goal_locations = {} # package -> room
        self.package_in_fact_ids = {} # package -> {room: global_id}
        self.holding_fact_ids = {} # package -> global_id
        self.rloc_fact_ids = {} # room -> global_id
        self.armempty_id = -1
        self.distances = {} # r1 -> {r2: dist}
        self.rooms = []
        self.packages = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map predicates to IDs
        adj = {}
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "rloc" and len(args) == 1:
                room = args[0]
                self.rloc_fact_ids[room] = fact.global_id
                if room not in adj: adj[room] = set()
            elif pred == "in" and len(args) == 2:
                p, r = args[0], args[1]
                if p not in self.package_in_fact_ids: self.package_in_fact_ids[p] = {}
                self.package_in_fact_ids[p][r] = fact.global_id
            elif pred == "holding" and len(args) == 1:
                p = args[0]
                self.holding_fact_ids[p] = fact.global_id
            elif pred == "door" and len(args) == 3:
                r1, r2 = args[0], args[1]
                if r1 not in adj: adj[r1] = set()
                if r2 not in adj: adj[r2] = set()
                adj[r1].add(r2)
                adj[r2].add(r1)

        self.rooms = list(self.rloc_fact_ids.keys())
        self.packages = list(self.holding_fact_ids.keys())

        # 2. Extract Goals (from model.goals bitmask or goal_in facts)
        if hasattr(model, 'goals') and model.goals:
            goal_names = model.state_explicit_repr(model.goals)
            for gname in goal_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "in" and len(args) == 2:
                    self.goal_locations[args[0]] = args[1]
        
        # Fallback to static goal_in facts if goal section is task-driven
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "goal_in" and len(args) == 2:
                if (initial_node.state >> fact.global_id) & 1:
                    self.goal_locations[args[0]] = args[1]

        # 3. Precompute All-Pairs Shortest Paths (Moves)
        for r in self.rooms:
            self.distances[r] = {r: 0}
            queue = deque([r])
            while queue:
                curr = queue.popleft()
                for neighbor in adj.get(curr, []):
                    if neighbor not in self.distances[r]:
                        self.distances[r][neighbor] = self.distances[r][curr] + 1
                        queue.append(neighbor)

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _get_robot_loc(self, state):
        for r, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1: return r
        return None

    def _get_held_package(self, state):
        if self.armempty_id != -1 and (state >> self.armempty_id) & 1: return None
        for p, fid in self.holding_fact_ids.items():
            if (state >> fid) & 1: return p
        return None

    def _compute(self, state):
        rloc = self._get_robot_loc(state)
        held_p = self._get_held_package(state)
        
        unsettled = []
        for p in self.packages:
            goal_r = self.goal_locations.get(p)
            if not goal_r: continue
            in_goal_fid = self.package_in_fact_ids.get(p, {}).get(goal_r)
            if in_goal_fid is not None:
                if not ((state >> in_goal_fid) & 1):
                    unsettled.append(p)

        if not unsettled: return 0

        # Term 1: Action Count (Pickup/Putdown)
        # 2 per non-held unsettled package, 1 for held package
        action_cost = 2 * len(unsettled) - (1 if held_p else 0)

        # Term 2: Travel Count (Moves)
        # Distance to finish at least one package delivery
        min_travel = float('inf')
        if rloc:
            for p in unsettled:
                goal_r = self.goal_locations[p]
                if p == held_p:
                    dist = self.distances.get(rloc, {}).get(goal_r, 0)
                else:
                    # Find current room of p
                    p_loc = None
                    for r, fid in self.package_in_fact_ids.get(p, {}).items():
                        if (state >> fid) & 1:
                            p_loc = r
                            break
                    if p_loc:
                        dist = self.distances.get(rloc, {}).get(p_loc, 0) + self.distances.get(p_loc, {}).get(goal_r, 0)
                    else:
                        dist = 0 # Should not happen in reachable states
                if dist < min_travel: min_travel = dist

        return int(action_cost + (min_travel if min_travel != float('inf') else 0))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
