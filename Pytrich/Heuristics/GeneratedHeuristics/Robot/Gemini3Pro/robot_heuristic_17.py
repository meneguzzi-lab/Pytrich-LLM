import collections
from Pytrich.core.heuristic import Heuristic

class RobotGemini3ProHeuristic17(Heuristic):
    def __init__(self, start_node, model, robot_gemini3pro_heuristic_14=None):
        super().__init__(start_node, model)
        self.model = model
        self.inf = 999999
        self.adj = collections.defaultdict(list)
        self.dist = {}
        self.goal_map = {} 
        self.rloc_facts = [] 
        self.in_facts = [] 
        self.holding_facts = [] 
        
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
        init_facts = self.model.state_explicit_repr(self.model.initial_state)
        rooms = set()
        
        # 1. Build Connectivity Graph from Initial State
        for name in init_facts:
            pred, args = self._parse_fact_name(name)
            if pred == 'door' and len(args) == 3:
                u, v, d = args
                self.adj[u].append(v)
                # Note: door facts in this domain are usually symmetric in definition,
                # but we add edges based on observed facts.
                rooms.add(u)
                rooms.add(v)
        
        # 2. Index Facts and Collect All Rooms
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'rloc': 
                rooms.add(args[0])
                self.rloc_facts.append((args[0], f.global_id))
            elif pred == 'in': 
                rooms.add(args[1])
                self.in_facts.append((args[0], args[1], f.global_id))
            elif pred == 'holding':
                self.holding_facts.append((args[0], f.global_id))

        # 3. Compute All-Pairs Shortest Paths (BFS)
        for r in rooms:
            self.dist[r] = {}
            for r2 in rooms:
                self.dist[r][r2] = self.inf
            self.dist[r][r] = 0
            
            queue = collections.deque([(r, 0)])
            visited = {r}
            while queue:
                curr, d = queue.popleft()
                self.dist[r][curr] = d
                for nxt in self.adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, d+1))

        # 4. Parse Goals
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        for name in goal_facts:
            pred, args = self._parse_fact_name(name)
            if pred == 'in':
                self.goal_map[args[0]] = args[1]

    def __call__(self, node):
        state = node.state
        cost = 0
        
        # 1. Determine Robot Location
        rloc = None
        for room, gid in self.rloc_facts:
            if (state >> gid) & 1:
                rloc = room
                break
        
        if rloc is None: 
            return self.inf
            
        # 2. Determine Held Object
        held_obj = None
        for obj, gid in self.holding_facts:
            if (state >> gid) & 1:
                held_obj = obj
                break
        
        # 3. Map Object Locations
        obj_locs = {}
        for obj, room, gid in self.in_facts:
            if (state >> gid) & 1:
                obj_locs[obj] = room
        
        # 4. Calculate Transport Costs for Pending Goals
        pickup_locs = []
        
        for obj, goal in self.goal_map.items():
            current_loc = obj_locs.get(obj)
            
            # If object is already at goal, no cost
            if current_loc == goal:
                continue
            
            if obj == held_obj:
                # Robot is holding the object -> Cost to move to goal + putdown
                d = self.dist.get(rloc, {}).get(goal, self.inf)
                cost += d + 1 # move + putdown
            else:
                # Object needs to be picked up -> Cost to move P->Goal + pickup + putdown
                if current_loc:
                    d = self.dist.get(current_loc, {}).get(goal, self.inf)
                    cost += d + 2 # pickup + move + putdown
                    pickup_locs.append(current_loc)
                # If location unknown, we ignore (admissible under-approximation)
        
        # 5. Robot Transit Cost
        # If robot is not holding a goal object, it must move to a pickup location
        if held_obj is None and pickup_locs:
            min_d = self.inf
            for pl in pickup_locs:
                d = self.dist.get(rloc, {}).get(pl, self.inf)
                if d < min_d:
                    min_d = d
            if min_d < self.inf:
                cost += min_d
                
        return cost