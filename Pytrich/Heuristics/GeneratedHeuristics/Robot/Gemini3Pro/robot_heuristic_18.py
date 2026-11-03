from Pytrich.Heuristics.heuristic import Heuristic
import math

class RobotGemini3ProHeuristic18(Heuristic):
    def __init__(self, initial_state, model, robot_gemini3pro_heuristic_15=None):
        super().__init__(initial_state, model)
        self.inf = float('inf')
        
        # 1. Identify all rooms and build ID mappings
        rooms = set()
        temp_facts = []
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            temp_facts.append((fact, pred, args))
            
            if pred == 'rloc':
                rooms.add(args[0])
            elif pred == 'in':
                # args: obj, loc
                rooms.add(args[1])
            elif pred == 'door':
                # args: loc1, loc2, d
                rooms.add(args[0])
                rooms.add(args[1])
        
        sorted_rooms = sorted(list(rooms))
        self.room_to_id = {r: i for i, r in enumerate(sorted_rooms)}
        self.num_rooms = len(rooms)
        
        # 2. Build lookups and static connectivity graph
        self.rloc_facts = []    # (fact_id, room_id)
        self.in_facts = []      # (fact_id, obj_name, room_id)
        self.holding_facts = [] # (fact_id, obj_name)
        
        adj = [[] for _ in range(self.num_rooms)]
        
        # Check for static door facts in the initial state
        # model.initial_state is a set of global_ids for facts true at start
        init_state_set = self.model.initial_state
        
        for fact, pred, args in temp_facts:
            fid = fact.global_id
            
            if pred == 'rloc':
                if args[0] in self.room_to_id:
                    self.rloc_facts.append((fid, self.room_to_id[args[0]]))
            
            elif pred == 'in':
                if args[1] in self.room_to_id:
                    self.in_facts.append((fid, args[0], self.room_to_id[args[1]]))
                    
            elif pred == 'holding':
                self.holding_facts.append((fid, args[0]))
                
            elif pred == 'door':
                # Check if this door fact is true in the initial state
                # Doors are static rigid predicates in this domain
                if fid in init_state_set:
                    if args[0] in self.room_to_id and args[1] in self.room_to_id:
                        u, v = self.room_to_id[args[0]], self.room_to_id[args[1]]
                        adj[u].append(v)
                        # Note: 'door' is symmetric in connectivity but defined as directed triples in facts.
                        # We process each fact. If (door a b d) and (door b a d) exist, both edges are added.
        
        # 3. Compute All-Pairs Shortest Paths (APSP)
        # We assume move cost = 1. We ignore 'closed' status for heuristic relaxation.
        self.distances = self._compute_apsp(adj)
        
        # 4. Parse Goals
        # model.goals is a bitwise integer representing the goal state
        self.goals = {} # obj_name -> room_id
        goal_mask = self.model.goals
        
        if goal_mask:
            for fact, pred, args in temp_facts:
                if (goal_mask >> fact.global_id) & 1:
                    if pred == 'in':
                        obj, room = args[0], args[1]
                        if room in self.room_to_id:
                            self.goals[obj] = self.room_to_id[room]

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _compute_apsp(self, adj):
        n = self.num_rooms
        dist = [[self.inf] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        for u in range(n):
            for v in adj[u]:
                dist[u][v] = 1
                
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def h(self, task_network, state):
        cost = 0
        
        # 1. Find Robot Location
        robot_loc = -1
        for fid, rid in self.rloc_facts:
            if (state >> fid) & 1:
                robot_loc = rid
                break
        
        if robot_loc == -1:
            return self.inf

        # 2. Identify Held Object
        held_obj = None
        for fid, obj in self.holding_facts:
            if (state >> fid) & 1:
                held_obj = obj
                break

        # 3. Identify Object Locations (if not held)
        # We only look up objects relevant to goals if needed, or scan all active in-facts.
        # Scanning active in-facts is reasonably fast.
        obj_locations = {}
        for fid, obj, rid in self.in_facts:
            if (state >> fid) & 1:
                obj_locations[obj] = rid

        # 4. Calculate Costs
        pending_pickups = []
        
        # Virtual robot location tracks where the robot ends up after handling the held item
        robot_virtual_loc = robot_loc
        
        for obj, goal_rid in self.goals.items():
            # Case A: Object is held by robot
            if held_obj == obj:
                dist_to_goal = self.distances[robot_loc][goal_rid]
                if dist_to_goal == self.inf: return self.inf
                
                # Cost: Move to goal + Putdown
                cost += dist_to_goal + 1
                
                # Robot is now effectively at the goal of this package
                robot_virtual_loc = goal_rid
            
            # Case B: Object is at goal
            elif obj in obj_locations and obj_locations[obj] == goal_rid:
                continue
                
            # Case C: Object is elsewhere (needs transport)
            elif obj in obj_locations:
                curr_rid = obj_locations[obj]
                dist_pkg_to_goal = self.distances[curr_rid][goal_rid]
                
                if dist_pkg_to_goal == self.inf: return self.inf
                
                # Cost: Pickup + Move + Putdown
                # We do NOT add robot travel to pickup here; we do it in batch below
                cost += 1 + dist_pkg_to_goal + 1
                
                pending_pickups.append(curr_rid)
            
            # Case D: Object location unknown (should not happen in valid state) or unreachable
            else:
                pass

        # 5. Add Robot Transit Cost
        # The robot must travel from its virtual location (after delivering held item)
        # to the nearest pending pickup.
        if pending_pickups:
            min_transit = self.inf
            for pickup_loc in pending_pickups:
                d = self.distances[robot_virtual_loc][pickup_loc]
                if d < min_transit:
                    min_transit = d
            
            if min_transit != self.inf:
                cost += min_transit
            else:
                return self.inf
                
        return cost
