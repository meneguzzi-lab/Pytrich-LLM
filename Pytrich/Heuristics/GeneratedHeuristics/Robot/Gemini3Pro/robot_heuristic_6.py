import sys
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic6(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_3=None):
        super().__init__(model)
        self.robot_gemini3pro_heuristic_3 = robot_gemini3pro_heuristic_3
        self.infinity = 1000000
        
        # Caches for fast state decoding
        self.rloc_facts = {}  # global_id -> room_name
        self.holding_facts = {} # global_id -> pkg_name
        self.pkg_in_facts = defaultdict(dict) # pkg_name -> {room_name: global_id}
        
        # Goal tracking
        self.goal_targets = {} # pkg_name -> (room_name, global_id)
        
        # Graph data
        self.adjacency = defaultdict(set)
        self.distances = {} # room -> {room -> dist}
        
        self._initialize_domain()

    def _parse_fact_name(self, name):
        # Helper to parse +predicate[arg1,arg2] format
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

    def _initialize_domain(self):
        # 1. Parse Facts to build Maps and Graph
        rooms = set()
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'door':
                # door[r1,r2,d]
                r1, r2 = args[0], args[1]
                self.adjacency[r1].add(r2)
                self.adjacency[r2].add(r1)
                rooms.add(r1)
                rooms.add(r2)
                
            elif pred == 'rloc':
                self.rloc_facts[fact.global_id] = args[0]
                rooms.add(args[0])
                
            elif pred == 'in':
                pkg, room = args[0], args[1]
                self.pkg_in_facts[pkg][room] = fact.global_id
                
            elif pred == 'holding':
                self.holding_facts[fact.global_id] = args[0]

        # 2. Precompute All-Pairs Shortest Paths (BFS)
        all_rooms = list(rooms)
        for r in all_rooms:
            self.distances[r] = self._bfs(r, all_rooms)

        # 3. Parse Goals from the model's bitmask representation
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'in':
                pkg, target_room = args[0], args[1]
                # Find the global_id for this specific goal fact
                fid = self.pkg_in_facts.get(pkg, {}).get(target_room)
                if fid is not None:
                    self.goal_targets[pkg] = (target_room, fid)

    def _bfs(self, start, all_nodes):
        dists = {node: self.infinity for node in all_nodes}
        dists[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            d = dists[u]
            for v in self.adjacency[u]:
                if dists[v] == self.infinity:
                    dists[v] = d + 1
                    queue.append(v)
        return dists

    def h(self, task_network, state_bitwise):
        # 1. Identify Robot Location
        current_rloc = None
        for fid, rname in self.rloc_facts.items():
            if (state_bitwise >> fid) & 1:
                current_rloc = rname
                break
        
        if current_rloc is None:
            return self.infinity

        # 2. Identify what is being held
        held_pkg = None
        for fid, pkg in self.holding_facts.items():
            if (state_bitwise >> fid) & 1:
                held_pkg = pkg
                break

        # 3. Calculate Cost
        total_cost = 0
        min_access_cost = self.infinity
        pending_packages_on_floor = False

        # Evaluate each goal
        for pkg, (target, target_fid) in self.goal_targets.items():
            # Check if goal already satisfied
            if (state_bitwise >> target_fid) & 1:
                continue
            
            # Goal not satisfied
            if pkg == held_pkg:
                # Robot has it. Cost = distance to target + putdown (1)
                # We assume doors are open (lower bound on distance)
                d = self.distances[current_rloc].get(target, self.infinity)
                total_cost += d + 1
            else:
                # Robot doesn't have it. Find where it is.
                pkg_loc = None
                locations = self.pkg_in_facts.get(pkg, {})
                for room, fid in locations.items():
                    if (state_bitwise >> fid) & 1:
                        pkg_loc = room
                        break
                
                if pkg_loc:
                    # Delivery Cost = dist(pkg_loc, target) + pickup(1) + putdown(1)
                    d_deliver = self.distances[pkg_loc].get(target, self.infinity)
                    total_cost += d_deliver + 2
                    
                    # Access Cost: Robot must reach the package
                    # If robot is not holding anything, we add the distance to the nearest required package
                    if held_pkg is None:
                        d_access = self.distances[current_rloc].get(pkg_loc, self.infinity)
                        if d_access < min_access_cost:
                            min_access_cost = d_access
                        pending_packages_on_floor = True
                else:
                    # Fallback if package location is unknown (should not happen in valid state)
                    pass

        # Add access cost if necessary
        if held_pkg is None and pending_packages_on_floor:
            if min_access_cost < self.infinity:
                total_cost += min_access_cost
        
        return total_cost
