from Pytrich.Heuristics.heuristic import Heuristic
import sys

class RobotGemini3ProHeuristic14(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_11):
        super().__init__(robot_gemini3pro_heuristic_11)
        self.model = robot_gemini3pro_heuristic_11.model
        self.inf = float('inf')
        
        # Mappings
        self.room_to_id = {}
        self.id_to_room = []
        self.pkg_to_id = {}
        self.id_to_pkg = []
        
        # Fact Indices for fast lookup
        self.fact_rloc = {} # room_id -> global_id
        self.fact_holding = {} # pkg_id -> global_id
        self.fact_in = {} # pkg_id -> {room_id -> global_id}
        self.fact_armempty = -1
        
        # Graph
        self.dist_matrix = []
        
        self._initialize()

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

    def _initialize(self):
        # 1. Identify Objects and Types
        rooms = set()
        packages = set()
        
        # First pass to find types from static declarations or init facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'rloc':
                rooms.add(args[0])
            elif pred == 'in':
                packages.add(args[0])
                rooms.add(args[1])
            elif pred == 'holding':
                packages.add(args[0])
            elif pred == 'door':
                rooms.add(args[0])
                rooms.add(args[1])
        
        self.id_to_room = sorted(list(rooms))
        self.room_to_id = {r: i for i, r in enumerate(self.id_to_room)}
        
        self.id_to_pkg = sorted(list(packages))
        self.pkg_to_id = {p: i for i, p in enumerate(self.id_to_pkg)}
        
        n_rooms = len(self.id_to_room)
        self.dist_matrix = [[self.inf] * n_rooms for _ in range(n_rooms)]
        for i in range(n_rooms):
            self.dist_matrix[i][i] = 0
            
        # 2. Build Graph and Indices
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'door':
                u = self.room_to_id.get(args[0])
                v = self.room_to_id.get(args[1])
                if u is not None and v is not None:
                    self.dist_matrix[u][v] = 1
                    # Note: We rely on facts for symmetry. If domain defines both directions, we get both edges.
            
            elif pred == 'rloc':
                rid = self.room_to_id.get(args[0])
                if rid is not None:
                    self.fact_rloc[rid] = fact.global_id
                    
            elif pred == 'holding':
                pid = self.pkg_to_id.get(args[0])
                if pid is not None:
                    self.fact_holding[pid] = fact.global_id
            
            elif pred == 'in':
                pid = self.pkg_to_id.get(args[0])
                rid = self.room_to_id.get(args[1])
                if pid is not None and rid is not None:
                    if pid not in self.fact_in:
                        self.fact_in[pid] = {}
                    self.fact_in[pid][rid] = fact.global_id
            
            elif pred == 'armempty':
                self.fact_armempty = fact.global_id

        # 3. Floyd-Warshall Algorithm for APSP
        for k in range(n_rooms):
            for i in range(n_rooms):
                for j in range(n_rooms):
                    if self.dist_matrix[i][j] > self.dist_matrix[i][k] + self.dist_matrix[k][j]:
                        self.dist_matrix[i][j] = self.dist_matrix[i][k] + self.dist_matrix[k][j]

        # 4. Parse Goals
        self.goal_list = [] # List of (pkg_id, room_id, fact_global_id)
        goal_mask = self.model.goals
        
        if isinstance(goal_mask, int):
            for i in range(goal_mask.bit_length()):
                if (goal_mask >> i) & 1:
                    if i >= len(self.model.facts): continue
                    fact = self.model.facts[i]
                    pred, args = self._parse_fact_name(fact.name)
                    # The goal state typically consists of 'in' predicates
                    if pred == 'in':
                        pid = self.pkg_to_id.get(args[0])
                        rid = self.room_to_id.get(args[1])
                        if pid is not None and rid is not None:
                            self.goal_list.append((pid, rid, i))

    def h(self, node):
        state = node.state
        cost = 0.0
        
        # 1. Get Robot Location
        robot_loc = -1
        for rid, fid in self.fact_rloc.items():
            if (state >> fid) & 1:
                robot_loc = rid
                break
        
        if robot_loc == -1: return self.inf
        
        # 2. Check held object
        held_pkg = -1
        is_arm_empty = False
        if self.fact_armempty != -1:
            if (state >> self.fact_armempty) & 1:
                is_arm_empty = True
        
        # If armempty fact is not present or false, check holding
        if not is_arm_empty:
            for pid, fid in self.fact_holding.items():
                if (state >> fid) & 1:
                    held_pkg = pid
                    break
        
        # 3. Calculate costs for pending goals
        pending_locations = []
        
        for pid, goal_rid, goal_fid in self.goal_list:
            # Check if satisfied
            if (state >> goal_fid) & 1:
                continue
            
            # Goal not satisfied
            if held_pkg == pid:
                # Robot is holding the package
                # Cost: Move to goal + Putdown
                dist = self.dist_matrix[robot_loc][goal_rid]
                if dist == self.inf: return self.inf
                cost += dist + 1
            else:
                # Package is on the ground (or held by another? No, single robot)
                pkg_loc = -1
                if pid in self.fact_in:
                    for rid, fid in self.fact_in[pid].items():
                        if (state >> fid) & 1:
                            pkg_loc = rid
                            break
                
                if pkg_loc != -1:
                    # Cost: Pickup + Carry + Putdown
                    # We count the transport cost from pkg location to goal
                    d_carry = self.dist_matrix[pkg_loc][goal_rid]
                    if d_carry == self.inf: return self.inf
                    
                    # Cost = Pickup(1) + Move(d_carry) + Putdown(1)
                    task_cost = d_carry + 2
                    cost += task_cost
                    pending_locations.append(pkg_loc)
        
        # 4. Add approach cost
        # If robot is free (not holding anything), it must travel to a pending package.
        # We add the minimum distance to any pending package.
        if held_pkg == -1 and pending_locations:
            min_dist = self.inf
            for loc in pending_locations:
                d = self.dist_matrix[robot_loc][loc]
                if d < min_dist:
                    min_dist = d
            
            if min_dist != self.inf:
                cost += min_dist
            
        return float(cost)
