from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RoverGTOHPGemini3FlashHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP planning domain.
    
    Bottleneck Analysis:
    The core bottleneck in the Rover domain is the navigation of rovers between waypoints 
    to collect data (soil, rock, image) and then to waypoints visible to landers for communication.
    Since navigation steps are expanded into multiple actions (visit, navigate, unvisit), the 
    movement cost is significantly higher than simple action counts.

    Admissible Lower Bounds:
    1. Goal Max Navigation Cost: For each unfulfilled goal, the minimum cost is the distance from 
       the best-placed rover to the sample source, plus the distance to a communication point, 
       multiplied by the action-per-hop factor (3). This is admissible as these moves are unavoidable.
    2. Data Acquisition: Every goal requires at least a sampling action (+1) and a 
       communication action (+1). Calibration and store-clearing add more if needed.
    """

    def __init__(self, rover_gemini3flash_heuristic_8="RoverGTOHPGemini3FlashHeuristic8"):
        super().__init__(rover_gemini3flash_heuristic_8)
        self.wp_to_idx = {}
        self.idx_to_wp = []
        self.rover_to_idx = {}
        self.idx_to_rover = []
        self.dist = []
        self.min_dist_to_comm = []
        self.comm_wps = []
        self.image_sources = {}
        self.goals_data = []
        self.rover_at_ids = {}
        self.rover_has_soil_ids = {}
        self.rover_has_rock_ids = {}
        self.rover_has_image_ids = {}
        self.rover_empty_ids = {}
        self.rover_calibrated_ids = {}
        self.rover_can_soil = set()
        self.rover_can_rock = set()
        self.rover_can_image = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _parse_item(self, name):
        name = name.replace(" ", "")
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        if "[" not in name: return name, []
        parts = name.split("[")
        pred = parts[0]
        args = parts[1][:-1].split(",")
        return pred, args

    def _preprocess(self):
        waypoint_names = set()
        rover_names = set()
        adj = {}

        # 1. Identify rovers, waypoints, and navigation edges from operators
        for op in self.model.operators:
            pred, args = self._parse_item(op.name)
            if pred == "navigate":
                r, w1, w2 = args
                rover_names.add(r)
                waypoint_names.add(w1)
                waypoint_names.add(w2)
                if r not in adj: adj[r] = {}
                if w1 not in adj[r]: adj[r][w1] = []
                adj[r][w1].append(w2)
            elif pred == "sample_soil":
                self.rover_can_soil.add(args[0])
                rover_names.add(args[0])
            elif pred == "sample_rock":
                self.rover_can_rock.add(args[0])
                rover_names.add(args[0])
            elif pred == "take_image":
                self.rover_can_image.add(args[0])
                rover_names.add(args[0])
                obj, mode, wp = args[2], args[4], args[1]
                if (obj, mode) not in self.image_sources: self.image_sources[(obj, mode)] = set()
                self.image_sources[(obj, mode)].add(wp)

        self.idx_to_wp = sorted(list(waypoint_names))
        self.wp_to_idx = {name: i for i, name in enumerate(self.idx_to_wp)}
        self.idx_to_rover = sorted(list(rover_names))
        self.rover_to_idx = {name: i for i, name in enumerate(self.idx_to_rover)}
        num_wps = len(self.idx_to_wp)
        num_rovers = len(self.idx_to_rover)

        # 2. Compute Shortest Paths (BFS) per rover
        self.dist = [[[float('inf')] * num_wps for _ in range(num_wps)] for _ in range(num_rovers)]
        for r_name, r_idx in self.rover_to_idx.items():
            r_adj = adj.get(r_name, {})
            for start_wp in self.idx_to_wp:
                s_idx = self.wp_to_idx[start_wp]
                self.dist[r_idx][s_idx][s_idx] = 0
                q = deque([start_wp])
                while q:
                    u = q.popleft()
                    u_idx = self.wp_to_idx[u]
                    for v in r_adj.get(u, []):
                        v_idx = self.wp_to_idx[v]
                        if self.dist[r_idx][s_idx][v_idx] == float('inf'):
                            self.dist[r_idx][s_idx][v_idx] = self.dist[r_idx][s_idx][u_idx] + 1
                            q.append(v)

        # 3. Identify Communication points
        self.comm_wps = [set() for _ in range(num_rovers)]
        for op in self.model.operators:
            if op.name.startswith("communicate"):
                pred, args = self._parse_item(op.name)
                # In all grounded comm ops, args[0] is rover, args[3] or args[4] is usually comm_wp
                # Safer: find the waypoint in the args where the rover is 'at' in precons
                for bit_pos in range(op.pos_precons.bit_length()):
                    if (op.pos_precons >> bit_pos) & 1:
                        f_pred, f_args = self._parse_item(self.model.facts[bit_pos].name)
                        if f_pred == "at" and f_args[0] == args[0]:
                            if f_args[0] in self.rover_to_idx and f_args[1] in self.wp_to_idx:
                                self.comm_wps[self.rover_to_idx[f_args[0]]].add(self.wp_to_idx[f_args[1]])

        self.min_dist_to_comm = [[float('inf')] * num_wps for _ in range(num_rovers)]
        for r in range(num_rovers):
            for w in range(num_wps):
                if self.comm_wps[r]:
                    self.min_dist_to_comm[r][w] = min([self.dist[r][w][c] for c in self.comm_wps[r]] + [float('inf')])

        # 4. Map Fact IDs for faster state access
        self.rover_at_ids = {r: {} for r in range(num_rovers)}
        self.rover_has_soil_ids = {r: {} for r in range(num_rovers)}
        self.rover_has_rock_ids = {r: {} for r in range(num_rovers)}
        self.rover_has_image_ids = {r: {} for r in range(num_rovers)}
        self.rover_empty_ids = [None] * num_rovers
        self.rover_calibrated_ids = [[] for _ in range(num_rovers)]

        rover_store = {}
        for op in self.model.operators:
            pred, args = self._parse_item(op.name)
            if pred == "drop": rover_store[args[0]] = args[1]

        for fact in self.model.facts:
            pred, args = self._parse_item(fact.name)
            if pred == "at" and args[0] in self.rover_to_idx:
                self.rover_at_ids[self.rover_to_idx[args[0]]][self.wp_to_idx[args[1]]] = fact.global_id
            elif pred == "have_soil_analysis" and args[0] in self.rover_to_idx:
                self.rover_has_soil_ids[self.rover_to_idx[args[0]]][self.wp_to_idx[args[1]]] = fact.global_id
            elif pred == "have_rock_analysis" and args[0] in self.rover_to_idx:
                self.rover_has_rock_ids[self.rover_to_idx[args[0]]][self.wp_to_idx[args[1]]] = fact.global_id
            elif pred == "have_image" and args[0] in self.rover_to_idx:
                self.rover_has_image_ids[self.rover_to_idx[args[0]]][(args[1], args[2])] = fact.global_id
            elif pred == "empty":
                for r_name, s_name in rover_store.items():
                    if args[0] == s_name: self.rover_empty_ids[self.rover_to_idx[r_name]] = fact.global_id
            elif pred == "calibrated":
                if args[1] in self.rover_to_idx: self.rover_calibrated_ids[self.rover_to_idx[args[1]]].append(fact.global_id)

        # 5. Extract goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_names:
            pred, args = self._parse_item(g_name)
            fid = -1
            for fact in self.model.facts:
                if fact.name == g_name: 
                    fid = fact.global_id
                    break
            if fid == -1: continue
            
            if pred == "communicated_soil_data":
                self.goals_data.append(('soil', self.wp_to_idx[args[0]], None, None, fid))
            elif pred == "communicated_rock_data":
                self.goals_data.append(('rock', self.wp_to_idx[args[0]], None, None, fid))
            elif pred == "communicated_image_data":
                self.goals_data.append(('image', None, args[0], args[1], fid))

    def _get_rover_pos(self, state, r_idx):
        for wp_idx, fid in self.rover_at_ids[r_idx].items():
            if (state >> fid) & 1: return wp_idx
        return None

    def _compute(self, state, task_network):
        total_h = 0
        max_h = 0
        unmet = 0

        for g_type, src_wp, obj, mode, g_fid in self.goals_data:
            if (state >> g_fid) & 1: continue
            
            unmet += 1
            best_rover_cost = 1000000
            
            for r_idx in range(len(self.idx_to_rover)):
                r_name = self.idx_to_rover[r_idx]
                # Check capability
                if g_type == 'soil' and r_name not in self.rover_can_soil: continue
                if g_type == 'rock' and r_name not in self.rover_can_rock: continue
                if g_type == 'image' and r_name not in self.rover_can_image: continue
                
                r_pos = self._get_rover_pos(state, r_idx)
                if r_pos is None: continue
                
                # Check if data already held
                has_data = False
                if g_type == 'soil':
                    fid = self.rover_has_soil_ids[r_idx].get(src_wp)
                    if fid is not None and (state >> fid) & 1: has_data = True
                elif g_type == 'rock':
                    fid = self.rover_has_rock_ids[r_idx].get(src_wp)
                    if fid is not None and (state >> fid) & 1: has_data = True
                elif g_type == 'image':
                    fid = self.rover_has_image_ids[r_idx].get((obj, mode))
                    if fid is not None and (state >> fid) & 1: has_data = True

                if has_data:
                    move_cost = self.min_dist_to_comm[r_idx][r_pos]
                    cost = move_cost * 3 + 1 # Comm cost
                else:
                    if g_type == 'image':
                        sources = self.image_sources.get((obj, mode), [])
                        min_move = min([self.dist[r_idx][r_pos][s] + self.min_dist_to_comm[r_idx][s] for s in self.wp_to_idx.values() if self.idx_to_wp[s] in sources] + [float('inf')])
                        cost = min_move * 3 + 3 # Calibrate + Take + Comm
                    else:
                        move_cost = self.dist[r_idx][r_pos][src_wp] + self.min_dist_to_comm[r_idx][src_wp]
                        cost = move_cost * 3 + 2 # Sample + Comm
                
                # Store full penalty
                if g_type in ('soil', 'rock'):
                    empty_fid = self.rover_empty_ids[r_idx]
                    if empty_fid is not None and not ((state >> empty_fid) & 1):
                        cost += 1
                
                if cost < best_rover_cost: best_rover_cost = cost
            
            total_h += best_rover_cost
            if best_rover_cost > max_h: max_h = best_rover_cost

        # Weighted blend of max and sum for guidance
        # Dividing by rover count allows sum-term to remain more conservative (parallelism)
        num_r = max(1, len(self.idx_to_rover))
        h = max_h + (total_h / num_r)
        
        # Small tie-breakers for network size
        h += 0.01 * unmet + 0.001 * len(task_network)
        return int(h)
