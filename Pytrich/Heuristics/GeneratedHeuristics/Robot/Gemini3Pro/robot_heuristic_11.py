from pytrich.heuristics.heuristic import Heuristic
import sys

class RobotGemini3ProHeuristic11(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_8=None):
        super().__init__(model)
        self.inf = 100000
        self.graph = {}
        self.rooms = set()
        self.goal_map = {}
        self.dist = {}
        
        # Precompute static structures
        self._parse_domain_structure()
        self._compute_distances()
        self._identify_goals()
        self._precompute_fact_indices()

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

    def _parse_domain_structure(self):
        # Extract connectivity from 'door' facts and other static info
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            if pred == 'door':
                # args: [loc1, loc2, d]
                if len(args) >= 2:
                    u, v = args[0], args[1]
                    self.rooms.add(u)
                    self.rooms.add(v)
                    if u not in self.graph: self.graph[u] = []
                    if v not in self.graph: self.graph[v] = []
                    self.graph[u].append(v)
                    self.graph[v].append(u)
            elif pred == 'rloc':
                self.rooms.add(args[0])
            elif pred == 'in':
                self.rooms.add(args[1])
        
        # Deduplicate neighbors
        for r in self.graph:
            self.graph[r] = list(set(self.graph[r]))

    def _compute_distances(self):
        # Initialize distances with infinity
        all_rooms = list(self.rooms)
        self.dist = {r: {r2: self.inf for r2 in all_rooms} for r in all_rooms}
        
        for r in all_rooms:
            self.dist[r][r] = 0
            if r in self.graph:
                for neighbor in self.graph[r]:
                    self.dist[r][neighbor] = 1
        
        # Floyd-Warshall Algorithm for All-Pairs Shortest Path
        # Efficient enough for domain size (max ~100 rooms)
        for k in all_rooms:
            for i in all_rooms:
                if self.dist[i][k] == self.inf:
                    continue
                for j in all_rooms:
                    if self.dist[k][j] == self.inf:
                        continue
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]

    def _identify_goals(self):
        # Parse goals from model.goals bitmask
        # We only care about final 'in' requirements for packages
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_facts:
            pred, args = self._parse_fact_name(fact_name)
            if pred == 'in':
                obj, loc = args[0], args[1]
                self.goal_map[obj] = loc

    def _precompute_fact_indices(self):
        # Cache global_ids for fast state lookup
        self.rloc_indices = []
        self.in_indices = {}
        self.holding_indices = {}
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'rloc':
                self.rloc_indices.append((fact.global_id, args[0]))
            elif pred == 'in':
                obj, loc = args[0], args[1]
                if obj not in self.in_indices: self.in_indices[obj] = []
                self.in_indices[obj].append((fact.global_id, loc))
            elif pred == 'holding':
                obj = args[0]
                self.holding_indices[obj] = fact.global_id

    def h(self, node):
        state = node.state
        
        # 1. Identify Robot Location
        robot_loc = None
        for idx, room in self.rloc_indices:
            if (state >> idx) & 1:
                robot_loc = room
                break
        
        if robot_loc is None:
            return self.inf

        total_cost = 0
        min_pickup_dist = self.inf
        is_holding_anything = False
        
        # 2. Evaluate Cost for Each Goal
        for obj, goal_loc in self.goal_map.items():
            # Check if goal is already satisfied
            is_satisfied = False
            obj_current_loc = None
            
            if obj in self.in_indices:
                for idx, loc in self.in_indices[obj]:
                    if (state >> idx) & 1:
                        obj_current_loc = loc
                        if loc == goal_loc:
                            is_satisfied = True
                        break
            
            if is_satisfied:
                continue
                
            # Check if robot is holding the object
            is_held = False
            if obj in self.holding_indices:
                idx = self.holding_indices[obj]
                if (state >> idx) & 1:
                    is_held = True
                    is_holding_anything = True
            
            if is_held:
                # Cost: Distance to goal + Putdown
                # This ignores 'open' actions cost beyond distance, keeping it admissible
                dist = self.dist[robot_loc].get(goal_loc, self.inf)
                total_cost += dist + 1
            else:
                # Cost: Pickup + Transport + Putdown
                if obj_current_loc:
                    # Transport cost
                    transit_dist = self.dist[obj_current_loc].get(goal_loc, self.inf)
                    total_cost += transit_dist + 2 # +1 pickup, +1 putdown
                    
                    # Approach cost (to be minimized later)
                    approach_dist = self.dist[robot_loc].get(obj_current_loc, self.inf)
                    if approach_dist < min_pickup_dist:
                        min_pickup_dist = approach_dist
                else:
                    # Object location unknown or lost (should not happen)
                    total_cost += self.inf

        # 3. Add Approach Cost
        # If we are holding something, we are en-route (approach cost included in held item transport logic or effectively 0 relative to next step)
        # If not holding, we must travel to the closest pickup first.
        if not is_holding_anything and min_pickup_dist != self.inf:
            total_cost += min_pickup_dist
            
        return total_cost
