import sys
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic16(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_13, model):
        super().__init__(robot_gemini3pro_heuristic_13, model)
        self.model = model
        self.max_rooms = 0
        self.room_to_idx = {}
        self.idx_to_room = {}
        self.dist_matrix = []
        self.pkg_goals = {}  # pkg -> target_room
        self.facts_rloc = [] # (id, room)
        self.facts_holding = {} # pkg -> id
        self.facts_in = {} # pkg -> [(id, room)]
        
        self._initialize()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _initialize(self):
        rooms = set()
        doors = []

        # Parse static goal_in facts and other topology info from model.facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == 'rloc':
                rooms.add(args[0])
                self.facts_rloc.append((fact.global_id, args[0]))
            elif pred == 'in':
                # +in[pkg, loc]
                if len(args) == 2:
                    pkg, loc = args[0], args[1]
                    rooms.add(loc)
                    if pkg not in self.facts_in:
                        self.facts_in[pkg] = []
                    self.facts_in[pkg].append((fact.global_id, loc))
            elif pred == 'holding':
                # +holding[pkg]
                if len(args) == 1:
                    self.facts_holding[args[0]] = fact.global_id
            elif pred == 'door':
                # +door[r1,r2,d]
                if len(args) == 3:
                    rooms.add(args[0])
                    rooms.add(args[1])
                    doors.append((args[0], args[1]))
            elif pred == 'goal_in':
                # +goal_in[pkg, loc]
                if len(args) == 2:
                    self.pkg_goals[args[0]] = args[1]
                    rooms.add(args[1])
        
        # Build Distance Matrix (All-Pairs Shortest Path)
        sorted_rooms = sorted(list(rooms))
        self.room_to_idx = {r: i for i, r in enumerate(sorted_rooms)}
        self.idx_to_room = {i: r for i, r in enumerate(sorted_rooms)}
        n = len(rooms)
        self.max_rooms = n
        
        # Init Matrix with infinity
        INF = 9999
        self.dist_matrix = [[INF] * n for _ in range(n)]
        for i in range(n):
            self.dist_matrix[i][i] = 0
            
        for r1, r2 in doors:
            if r1 in self.room_to_idx and r2 in self.room_to_idx:
                u, v = self.room_to_idx[r1], self.room_to_idx[r2]
                self.dist_matrix[u][v] = 1
                self.dist_matrix[v][u] = 1

        # Floyd-Warshall to precompute static topological distances
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.dist_matrix[i][j] > self.dist_matrix[i][k] + self.dist_matrix[k][j]:
                        self.dist_matrix[i][j] = self.dist_matrix[i][k] + self.dist_matrix[k][j]

    def compute(self, state, task_network):
        # 1. Determine Robot Location
        r_loc_idx = -1
        for fid, r_name in self.facts_rloc:
            if (state >> fid) & 1:
                r_loc_idx = self.room_to_idx.get(r_name, -1)
                break
        
        if r_loc_idx == -1:
            return float('inf')

        total_cost = 0.0
        pending_starts = []
        
        holding_relevant = False
        held_goal_idx = -1
        
        # 2. Iterate over all defined goals to calculate package costs
        for pkg, target_room in self.pkg_goals.items():
            target_idx = self.room_to_idx.get(target_room)
            if target_idx is None: continue

            # Check if already at goal
            is_done = False
            if pkg in self.facts_in:
                for fid, loc in self.facts_in[pkg]:
                    if loc == target_room and ((state >> fid) & 1):
                        is_done = True
                        break
            if is_done:
                continue

            # Check if holding
            is_held = False
            if pkg in self.facts_holding:
                fid = self.facts_holding[pkg]
                if (state >> fid) & 1:
                    is_held = True
            
            if is_held:
                # Robot is holding this package
                dist = self.dist_matrix[r_loc_idx][target_idx]
                # If unreachable, return inf
                if dist >= 9999: return float('inf')
                # Cost: Move to goal + Putdown (1)
                total_cost += dist + 1
                holding_relevant = True
                held_goal_idx = target_idx
            else:
                # Package is somewhere else
                pkg_loc_idx = -1
                if pkg in self.facts_in:
                    for fid, loc in self.facts_in[pkg]:
                        if (state >> fid) & 1:
                            pkg_loc_idx = self.room_to_idx.get(loc, -1)
                            break
                
                if pkg_loc_idx != -1:
                    # Cost: Transport (Loc -> Goal) + Pickup (1) + Putdown (1)
                    dist = self.dist_matrix[pkg_loc_idx][target_idx]
                    if dist >= 9999: return float('inf')
                    total_cost += dist + 2
                    pending_starts.append(pkg_loc_idx)
        
        # 3. Add Empty Transit Cost
        # If robot is currently holding a relevant package, it must go to that goal first.
        # From there, it must go to the start of the next package (if any).
        # We take the minimum transit to any pending package.
        
        if pending_starts:
            min_transit = 9999
            
            start_node = r_loc_idx
            if holding_relevant:
                start_node = held_goal_idx
                
            for p_start in pending_starts:
                d = self.dist_matrix[start_node][p_start]
                if d < min_transit:
                    min_transit = d
            
            if min_transit < 9999:
                total_cost += min_transit
            else:
                return float('inf')

        return total_cost