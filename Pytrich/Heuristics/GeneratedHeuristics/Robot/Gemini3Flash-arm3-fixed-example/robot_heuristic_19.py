"""
Domain-dependent heuristic for the Robot planning domain.

Bottleneck Analysis:
- The robot has a single arm (armempty predicate). This serializes all package delivery tasks.
- Every package must be moved from its source to its destination individually.
- The total cost is the sum of (travel to package + pickup + travel to goal + putdown) for all packages.

Admissibility Sketch:
1. Action Count Lower Bound: Every undelivered package requires at least a 'pickup' and a 'putdown' (2 actions). 
   If a package is already being held, it only requires a 'putdown' (1 action).
2. Travel Distance Lower Bound: The robot must traverse at least the distance between each package's 
   source and destination. Since the robot can only carry one package at a time, these distances are additive. 
   We also add the distance from the robot's current location to the nearest package/goal.
3. Door Penalty: Each closed door on a necessary path requires an 'open' action.

Term used: h = Sum_over_packages(action_cost + travel_cost) + min_dist_to_first_task.
"""

import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic19(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_19=None, name="RobotGemini3FlashHeuristic19"):
        super().__init__(name=name)
        self.package_goals = {}  # pkg -> room_name
        self.rooms = set()
        self.packages = set()
        self.dist = defaultdict(lambda: defaultdict(lambda: float('inf')))
        self.rloc_ids = {}      # room -> global_id
        self.in_ids = defaultdict(dict)  # pkg -> room -> global_id
        self.holding_ids = {}    # pkg -> global_id
        self.armempty_id = None
        self.initialized = False

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess(model)
        self.initialized = True
        h_value = self._compute(initial_node.state, initial_node.task_network)
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
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _preprocess(self, model: Model):
        adj = defaultdict(set)
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "door":
                r1, r2 = args[0], args[1]
                adj[r1].add(r2)
                adj[r2].add(r1)
                self.rooms.add(r1)
                self.rooms.add(r2)
            elif pred == "rloc":
                self.rloc_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "in":
                self.in_ids[args[0]][args[1]] = fact.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "holding":
                self.holding_ids[args[0]] = fact.global_id
                self.packages.add(args[0])
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "goal_in":
                # goal_in is defined in init state and is static
                if (model.initial_state >> fact.global_id) & 1:
                    self.package_goals[args[0]] = args[1]

        # BFS for all-pairs shortest paths
        for start_node in self.rooms:
            self.dist[start_node][start_node] = 0
            q = deque([start_node])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if self.dist[start_node][v] == float('inf'):
                        self.dist[start_node][v] = self.dist[start_node][u] + 1
                        q.append(v)

    def _compute(self, state, task_network):
        if not task_network and (self.model.goals & state) == self.model.goals:
            return 0

        h = 0
        curr_r = None
        for r, r_id in self.rloc_ids.items():
            if (state >> r_id) & 1:
                curr_r = r
                break
        
        held_p = None
        for p, h_id in self.holding_ids.items():
            if (state >> h_id) & 1:
                held_p = p
                break

        undelivered_locs = {}
        for p in self.packages:
            if p == held_p: continue
            target = self.package_goals.get(p)
            if not target: continue
            # Check if package is already at target
            target_fact_id = self.in_ids[p].get(target)
            if target_fact_id is not None and (state >> target_fact_id) & 1:
                continue
            # Package is not at target, find where it is
            for r, in_id in self.in_ids[p].items():
                if (state >> in_id) & 1:
                    undelivered_locs[p] = r
                    break

        # Action and Travel Cost Sum
        for p, loc in undelivered_locs.items():
            target = self.package_goals[p]
            h += 2 # Pickup + Putdown
            h += self.dist[loc][target]

        if held_p:
            target = self.package_goals.get(held_p)
            if target:
                h += 1 # Putdown
                h += self.dist[curr_r][target]
        elif undelivered_locs and curr_r:
            # Add distance to get to the first package
            h += min(self.dist[curr_r][loc] for loc in undelivered_locs.values())

        # Tie-breaker for progress
        h += 0.001 * len(task_network)
        return int(h) if h != float('inf') else 1000

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
