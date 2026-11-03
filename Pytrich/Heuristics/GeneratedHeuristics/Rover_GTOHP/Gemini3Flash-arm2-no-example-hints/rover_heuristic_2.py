from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

# Bottleneck: Rover Navigation serialization. Rovers can only be in one place at a time,
# and moving between waypoints to collect data (soil, rock, image) and then to waypoints
# visible to a lander to transmit that data requires sequential travel over a traversal graph.

# Admissible Bounds:
# 1. Action Count Lower Bound: Every unachieved goal requires at least a sampling/imaging action
#    and a communication action. Soil/Rock = 2 actions, Image = 3 actions (calibrate, take, send).
# 2. Navigation Distance Bound (Individual): For each goal, the cost is the shortest path
#    from a rover's current location to the target waypoint, plus the distance from the target
#    to a waypoint capable of communicating with a lander.
# 3. Store Constraint: If a rover must sample but its store is currently full, it must perform
#    a 'drop' action (+1 cost).

class RoverGTOHPGemini3FlashHeuristic2(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_2="rover_gemini3flash_heuristic_2"):
        super().__init__(rover_gemini3flash_heuristic_2)
        self.dist = {} # rover -> wp1 -> wp2 -> dist
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.rover_stores = {} # rover -> store_id
        self.at_lander_wps = set()
        self.visible_comm = collections.defaultdict(set) # wp1 -> {wp2, ...}
        self.visible_from_obj = collections.defaultdict(set) # obj -> {wp, ...}
        
        self.goal_soil_wps = {} # goal_id -> wp
        self.goal_rock_wps = {} # goal_id -> wp
        self.goal_image_data = {} # goal_id -> (obj, mode)
        
        self.at_ids = {} # rover -> wp -> global_id
        self.have_soil_ids = {} # rover -> wp -> global_id
        self.have_rock_ids = {} # rover -> wp -> global_id
        self.have_image_ids = {} # rover -> obj -> mode -> global_id
        self.full_ids = {} # store -> global_id
        
        self.waypoints = []
        self.rovers = []

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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse Entities and State IDs
        rover_set = set()
        wp_set = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                r, wp = args[0], args[1]
                rover_set.add(r)
                wp_set.add(wp)
                if r not in self.at_ids: self.at_ids[r] = {}
                self.at_ids[r][wp] = fact.global_id
            elif pred == "have_soil_analysis":
                r, wp = args[0], args[1]
                if r not in self.have_soil_ids: self.have_soil_ids[r] = {}
                self.have_soil_ids[r][wp] = fact.global_id
            elif pred == "have_rock_analysis":
                r, wp = args[0], args[1]
                if r not in self.have_rock_ids: self.have_rock_ids[r] = {}
                self.have_rock_ids[r][wp] = fact.global_id
            elif pred == "have_image":
                r, obj, mode = args[0], args[1], args[2]
                if r not in self.have_image_ids: self.have_image_ids[r] = {}
                if obj not in self.have_image_ids[r]: self.have_image_ids[r][obj] = {}
                self.have_image_ids[r][obj][mode] = fact.global_id
            elif pred == "full":
                self.full_ids[args[0]] = fact.global_id

        self.rovers = list(rover_set)
        self.waypoints = list(wp_set)

        # 2. Parse Capabilities from Operators (static info is stripped from state)
        adj = {r: collections.defaultdict(lambda: collections.defaultdict(lambda: 1000000)) for r in self.rovers}
        for r in self.rovers: 
            for wp in self.waypoints: adj[r][wp][wp] = 0
            
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == "navigate":
                r, fr, to = args[0], args[1], args[2]
                adj[r][fr][to] = 1
            elif pred == "communicate_soil_data1" or pred == "communicate_rock_data1":
                self.visible_comm[args[3]].add(args[4])
                self.at_lander_wps.add(args[4])
            elif pred == "communicate_soil_data2" or pred == "communicate_rock_data2":
                self.visible_comm[args[2]].add(args[3])
                self.at_lander_wps.add(args[3])
            elif pred == "sample_soil":
                self.equipped_soil.add(args[0])
                self.rover_stores[args[0]] = args[1]
            elif pred == "sample_rock":
                self.equipped_rock.add(args[0])
                self.rover_stores[args[0]] = args[1]
            elif pred == "take_image":
                self.equipped_image.add(args[0])
                self.visible_from_obj[args[2]].add(args[1])
            elif pred == "calibrate":
                self.visible_from_obj[args[2]].add(args[3])

        # 3. Floyd-Warshall all-pairs shortest paths per rover
        for r in self.rovers:
            r_adj = adj[r]
            for k in self.waypoints:
                for i in self.waypoints:
                    if r_adj[i][k] >= 1000000: continue
                    for j in self.waypoints:
                        if r_adj[i][j] > r_adj[i][k] + r_adj[k][j]:
                            r_adj[i][j] = r_adj[i][k] + r_adj[k][j]
            self.dist[r] = r_adj

        # 4. Identify Goal Identities
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            fid = next(f.global_id for f in model.facts if f.name == gn)
            if pred == "communicated_soil_data":
                self.goal_soil_wps[fid] = args[0]
            elif pred == "communicated_rock_data":
                self.goal_rock_wps[fid] = args[0]
            elif pred == "communicated_image_data":
                self.goal_image_data[fid] = (args[0], args[1])

        # 5. Precompute Comm Target Min Distances
        self.wp_comm_targets = {wp for wp in self.waypoints if any(v in self.at_lander_wps for v in self.visible_comm[wp])}
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Fast bitwise tests
        rover_pos = {}
        for r in self.rovers:
            for wp, fid in self.at_ids.get(r, {}).items():
                if (state >> fid) & 1:
                    rover_pos[r] = wp
                    break
        
        unachieved_costs = []
        
        # Soil Goals
        for fid, wp in self.goal_soil_wps.items():
            if not (state >> fid) & 1:
                best_g = 1000000
                for r in self.equipped_soil:
                    curr = rover_pos.get(r)
                    if not curr: continue
                    # Check if analysis already held
                    r_has = (state >> self.have_soil_ids[r][wp]) & 1 if wp in self.have_soil_ids.get(r, {}) else 0
                    m_cost = 0
                    if r_has:
                        m_cost = min((self.dist[r][curr][c_wp] for c_wp in self.wp_comm_targets), default=50)
                        best_g = min(best_g, m_cost * 3 + 1)
                    else:
                        move_to = self.dist[r][curr][wp]
                        move_to_comm = min((self.dist[r][wp][c_wp] for c_wp in self.wp_comm_targets), default=50)
                        store_penalty = 1 if (state >> self.full_ids[self.rover_stores[r]]) & 1 else 0
                        best_g = min(best_g, (move_to + move_to_comm) * 3 + 2 + store_penalty)
                unachieved_costs.append(best_g if best_g < 1000000 else 20)

        # Rock Goals
        for fid, wp in self.goal_rock_wps.items():
            if not (state >> fid) & 1:
                best_g = 1000000
                for r in self.equipped_rock:
                    curr = rover_pos.get(r)
                    if not curr: continue
                    r_has = (state >> self.have_rock_ids[r][wp]) & 1 if wp in self.have_rock_ids.get(r, {}) else 0
                    m_cost = 0
                    if r_has:
                        m_cost = min((self.dist[r][curr][c_wp] for c_wp in self.wp_comm_targets), default=50)
                        best_g = min(best_g, m_cost * 3 + 1)
                    else:
                        move_to = self.dist[r][curr][wp]
                        move_to_comm = min((self.dist[r][wp][c_wp] for c_wp in self.wp_comm_targets), default=50)
                        store_penalty = 1 if (state >> self.full_ids[self.rover_stores[r]]) & 1 else 0
                        best_g = min(best_g, (move_to + move_to_comm) * 3 + 2 + store_penalty)
                unachieved_costs.append(best_g if best_g < 1000000 else 20)

        # Image Goals
        for fid, (obj, mode) in self.goal_image_data.items():
            if not (state >> fid) & 1:
                best_g = 1000000
                for r in self.equipped_image:
                    curr = rover_pos.get(r)
                    if not curr: continue
                    r_has = (state >> self.have_image_ids[r][obj][mode]) & 1 if (r in self.have_image_ids and obj in self.have_image_ids[r] and mode in self.have_image_ids[r][obj]) else 0
                    if r_has:
                        m_cost = min((self.dist[r][curr][c_wp] for c_wp in self.wp_comm_targets), default=50)
                        best_g = min(best_g, m_cost * 3 + 1)
                    else:
                        vis_wps = self.visible_from_obj.get(obj, [])
                        if not vis_wps: continue
                        move_to_vis = min((self.dist[r][curr][v_wp] for v_wp in vis_wps), default=50)
                        move_to_comm = min((self.dist[r][v_wp][c_wp] for v_wp in vis_wps for c_wp in self.wp_comm_targets), default=50)
                        best_g = min(best_g, (move_to_vis + move_to_comm) * 3 + 3)
                unachieved_costs.append(best_g if best_g < 1000000 else 30)

        if not unachieved_costs:
            return 0
            
        # strictly admissible: max cost of any single goal or basic action count
        h1 = max(unachieved_costs)
        h2 = len(unachieved_costs) * 2
        return max(h1, h2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
