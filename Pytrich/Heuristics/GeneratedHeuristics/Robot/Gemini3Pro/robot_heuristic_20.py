from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic20(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_17=None):
        super().__init__(model)
        self.robot_gemini3pro_heuristic_17 = robot_gemini3pro_heuristic_17
        
        # Data structures for fast lookup of fact IDs
        self.rloc_facts = []  # List of (room, global_id)
        self.in_facts = {}    # pkg -> List of (room, global_id)
        self.holding_facts = [] # List of (pkg, global_id)
        self.closed_facts = {}  # door -> global_id
        
        self.rooms = set()
        self.goal_map = {} # pkg -> target_room
        
        # 1. Parse all facts to map predicates to IDs and categorize them
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'rloc':
                self.rloc_facts.append((args[0], fact.global_id))
                self.rooms.add(args[0])
            elif pred == 'in':
                pkg, room = args
                if pkg not in self.in_facts: self.in_facts[pkg] = []
                self.in_facts[pkg].append((room, fact.global_id))
                self.rooms.add(room)
            elif pred == 'holding':
                self.holding_facts.append((args[0], fact.global_id))
            elif pred == 'closed':
                self.closed_facts[args[0]] = fact.global_id

        # 2. Build Connectivity Graph (Topology)
        # Scan initial state for static 'door' facts to build the room graph
        # We assume door topology is static (doors exist/don't exist is fixed)
        adj = {}
        init_repr = self.model.state_explicit_repr(self.model.initial_state)
        for name in init_repr:
            pred, args = self._parse_fact_name(name)
            if pred == 'door':
                r1, r2, d = args
                if r1 not in adj: adj[r1] = []
                adj[r1].append((r2, d))
                self.rooms.add(r1)
                self.rooms.add(r2)
        
        # 3. Precompute All-Pairs Shortest Paths (Static)
        # We use BFS to find the shortest path in terms of steps (hops).
        # We store the base distance and the list of doors on the path.
        # This allows checking for closed doors efficiently during heuristic evaluation.
        self.dist_matrix = {}
        all_rooms = list(self.rooms)
        
        # Initialize self-distances
        for r in all_rooms:
            self.dist_matrix[(r, r)] = (0, [])
            
        for start_node in all_rooms:
            queue = [(start_node, 0, [])] # (current, dist, path_doors)
            visited = {start_node}
            
            idx = 0
            while idx < len(queue):
                curr, dist, doors = queue[idx]
                idx += 1
                
                if curr in adj:
                    for neighbor, door in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_doors = doors + [door]
                            self.dist_matrix[(start_node, neighbor)] = (dist + 1, new_doors)
                            queue.append((neighbor, dist + 1, new_doors))
                            
        # 4. Parse Goals
        # model.goals is a bitmask of facts that must be true in the final state
        goal_repr = self.model.state_explicit_repr(self.model.goals)
        for name in goal_repr:
            pred, args = self._parse_fact_name(name)
            if pred == 'in':
                self.goal_map[args[0]] = args[1]
                
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

    def h(self, node, task_network, state_bitwise):
        # 1. Determine Robot Location
        r_loc = None
        for r, fid in self.rloc_facts:
            if (state_bitwise >> fid) & 1:
                r_loc = r
                break
        if r_loc is None: return float('inf')
        
        # 2. Determine Held Package
        held_pkg = None
        for p, fid in self.holding_facts:
            if (state_bitwise >> fid) & 1:
                held_pkg = p
                break
                
        total_cost = 0
        pending_pkgs = False
        min_start_dist = float('inf')
        
        # 3. Iterate Goals and Calculate Costs
        for pkg, target_room in self.goal_map.items():
            # Check if package is already at goal
            at_goal = False
            if pkg in self.in_facts:
                for loc, fid in self.in_facts[pkg]:
                    if loc == target_room and ((state_bitwise >> fid) & 1):
                        at_goal = True
                        break
            if at_goal: continue
            
            pending_pkgs = True
            
            if pkg == held_pkg:
                # Robot holds this package. Must deliver.
                # Cost = Path(robot, target) + Putdown(1) + Doors check
                dist, doors = self._get_path_info(r_loc, target_room)
                door_cost = self._count_closed_doors(doors, state_bitwise)
                total_cost += dist + door_cost + 1
                
                # If we are holding a package, we are essentially 'in' a task.
                # The distance to start the 'next' task is effectively 0 from the heuristic's perspective 
                # (or relative to the drop-off point, but we simplify to 0 to prefer delivering).
                min_start_dist = 0
            else:
                # Package is on ground (not held, not at goal).
                # Find current location
                curr_loc = None
                if pkg in self.in_facts:
                    for loc, fid in self.in_facts[pkg]:
                        if (state_bitwise >> fid) & 1:
                            curr_loc = loc
                            break
                
                if curr_loc:
                    # Task Cost = Pickup(1) + Delivery(curr, target) + Putdown(1)
                    #           + Routing(robot, curr) [handled via min_start_dist]
                    
                    # Delivery leg cost
                    d_del, doors_del = self._get_path_info(curr_loc, target_room)
                    c_del = d_del + self._count_closed_doors(doors_del, state_bitwise)
                    
                    # Add task execution cost to total
                    total_cost += 1 + c_del + 1
                    
                    # Pickup leg cost (Routing)
                    d_pick, doors_pick = self._get_path_info(r_loc, curr_loc)
                    c_pick = d_pick + self._count_closed_doors(doors_pick, state_bitwise)
                    
                    if c_pick < min_start_dist:
                        min_start_dist = c_pick
                        
        # If we have pending tasks and are not holding anything, 
        # add the travel cost to the nearest task start location.
        # This approximates the routing cost.
        if pending_pkgs and held_pkg is None and min_start_dist != float('inf'):
            total_cost += min_start_dist
            
        return total_cost

    def _get_path_info(self, r1, r2):
        if (r1, r2) in self.dist_matrix:
            return self.dist_matrix[(r1, r2)]
        return 9999, []
        
    def _count_closed_doors(self, doors, state_bitwise):
        cost = 0
        for d in doors:
            if d in self.closed_facts:
                fid = self.closed_facts[d]
                if (state_bitwise >> fid) & 1:
                    cost += 1 # +1 cost because if closed: Open(1) + Move(1). Move is in base dist.
        return cost