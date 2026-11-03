from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

# BOTTLENECK ANALYSIS:
# The primary serialization constraint in the Rover domain is navigation. Every data collection task 
# requires moving to a specific waypoint, potentially clearing a storage unit, and communicating 
# the data (often requiring further travel to a waypoint within range of a lander).
# Plan length is dominated by the '3d + 1' cost structure of the HTN navigation decomposition 
# (visit + navigate + unvisit per hop).

# ADMISSIBLE LOWER BOUNDS:
# 1. Achievement Cost: Each unsatisfied soil/rock/image goal requires a minimum of 2-3 actions 
#    (sample + communicate, or calibrate + take + communicate). This is a trivial lower bound on actions.
# 2. Navigation Distance: For every task in the network, the rover must be at certain waypoints. 
#    The shortest path (BFS) distance d between these waypoints multiplied by 3 (HTN cost per hop) 
#    is a lower bound on unavoidable navigation actions.
# 3. Store Management: If a sampling task is pending and the rover's store is full, a 'drop' action (+1) 
#    is strictly required.

class RoverGTOHPGemini3FlashHeuristic13(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_13="RoverGTOHPGemini3FlashHeuristic13"):
        super().__init__(rover_gemini3flash_heuristic_13)
        self.at_ids = {} # rover -> wp -> global_id
        self.full_ids = {} # rover -> global_id
        self.have_soil_ids = {} # rover -> wp -> id
        self.have_rock_ids = {} # rover -> wp -> id
        self.have_image_ids = {} # rover -> obj -> mode -> id
        
        self.rovers = set()
        self.waypoints = set()
        self.distances = {} # rover -> wp1 -> wp2 -> dist
        self.trans_wps = {} # rover -> set of waypoints with lander visibility
        self.visible_wps = {} # obj -> set of waypoints where obj is visible
        self.eligible_soil = set()
        self.eligible_rock = set()
        self.eligible_image = {} # (obj, mode) -> set of rovers

    def initialize(self, model: Model, initial_node: HTNNode):
        fact_map = {}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "at": 
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                if args[0] not in self.at_ids: self.at_ids[args[0]] = {}
                self.at_ids[args[0]][args[1]] = f.global_id
            elif pred == "full":
                # full takes store, but store is associated with rover
                fact_map[f.name] = f.global_id
            elif pred == "have_soil_analysis":
                if args[0] not in self.have_soil_ids: self.have_soil_ids[args[0]] = {}
                self.have_soil_ids[args[0]][args[1]] = f.global_id
            elif pred == "have_rock_analysis":
                if args[0] not in self.have_rock_ids: self.have_rock_ids[args[0]] = {}
                self.have_rock_ids[args[0]][args[1]] = f.global_id
            elif pred == "have_image":
                r, obj, m = args
                if r not in self.have_image_ids: self.have_image_ids[r] = {}
                if obj not in self.have_image_ids[r]: self.have_image_ids[r][obj] = {}
                self.have_image_ids[r][obj][m] = f.global_id

        # Extract static info from grounded operators
        rover_adj = {r: {wp: set() for wp in self.waypoints} for r in self.rovers}
        self.trans_wps = {r: set() for r in self.rovers}
        self.visible_wps = {}
        rover_to_store = {}

        for op in model.operators:
            p, args = self._parse_fact_name(op.name)
            if p == "navigate":
                rover_adj[args[0]][args[1]].add(args[2])
            elif p == "sample_soil":
                self.eligible_soil.add(args[0])
                rover_to_store[args[0]] = args[1]
            elif p == "sample_rock":
                self.eligible_rock.add(args[0])
                rover_to_store[args[0]] = args[1]
            elif p == "take_image":
                r, wp, obj, cam, m = args
                if (obj, m) not in self.eligible_image: self.eligible_image[(obj, m)] = set()
                self.eligible_image[(obj, m)].add(r)
                if obj not in self.visible_wps: self.visible_wps[obj] = set()
                self.visible_wps[obj].add(wp)
            elif "communicate_soil" in op.name or "communicate_rock" in op.name:
                r = args[0]
                # comm1: r, l, p1, p2, p3 where at(r, p2). p2 is index 3.
                # comm2: r, l, p1, p2 where at(r, p1). p1 is index 2.
                wp_idx = 3 if "1" in op.name else 2
                self.trans_wps[r].add(args[wp_idx])
            elif "communicate_image" in op.name:
                # comm: r, l, o, m, p1, p2 where at(r, p1). p1 is index 4.
                self.trans_wps[args[0]].add(args[4])
            elif p == "calibrate":
                obj, wp = args[2], args[3]
                if obj not in self.visible_wps: self.visible_wps[obj] = set()
                self.visible_wps[obj].add(wp)

        # Map store to rover for 'full' checks
        for r, s in rover_to_store.items():
            fname = f"+full[{s}]"
            if fname in fact_map:
                self.full_ids[r] = fact_map[fname]

        # All-pairs shortest paths per rover
        self.distances = {r: {w1: {w2: 999 for w2 in self.waypoints} for w1 in self.waypoints} for r in self.rovers}
        for r in self.rovers:
            for start in self.waypoints:
                self.distances[r][start][start] = 0
                queue = deque([(start, 0)])
                visited = {start}
                while queue:
                    curr, d = queue.popleft()
                    for neighbor in rover_adj[r][curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[r][start][neighbor] = d + 1
                            queue.append((neighbor, d + 1))

        # Precompute transmission/visibility helpers
        self.min_dist_to_trans = {r: {wp: min([self.distances[r][wp][tw] for tw in self.trans_wps[r]] + [999]) for wp in self.waypoints} for r in self.rovers}

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network: return 0
        
        # 1. Find rover positions
        rover_pos = {}
        for r in self.rovers:
            for wp, fid in self.at_ids.get(r, {}).items():
                if (state >> fid) & 1:
                    rover_pos[r] = wp
                    break
        
        total_h = 0
        for task in task_network:
            t_name = task.name
            args = self._get_args(t_name)
            
            if "get_soil_data" in t_name:
                wp = args[0]
                best = 999
                for r in self.eligible_soil:
                    pos = rover_pos.get(r)
                    if pos:
                        cost = 2 + 3 * (self.distances[r][pos][wp] + self.min_dist_to_trans[r][wp])
                        if r in self.full_ids and (state >> self.full_ids[r]) & 1: cost += 1
                        if cost < best: best = cost
                total_h += best

            elif "get_rock_data" in t_name:
                wp = args[0]
                best = 999
                for r in self.eligible_rock:
                    pos = rover_pos.get(r)
                    if pos:
                        cost = 2 + 3 * (self.distances[r][pos][wp] + self.min_dist_to_trans[r][wp])
                        if r in self.full_ids and (state >> self.full_ids[r]) & 1: cost += 1
                        if cost < best: best = cost
                total_h += best

            elif "get_image_data" in t_name:
                obj, mode = args[0], args[1]
                best = 999
                for r in self.eligible_image.get((obj, mode), []):
                    pos = rover_pos.get(r)
                    if pos and obj in self.visible_wps:
                        min_nav = min(self.distances[r][pos][vw] + self.min_dist_to_trans[r][vw] for vw in self.visible_wps[obj])
                        cost = 3 + 3 * min_nav
                        if cost < best: best = cost
                total_h += best

            elif "do_navigate" in t_name:
                # Args: x, to or x, from, to
                r = args[0]
                target_wp = args[-1]
                pos = rover_pos.get(r, args[1] if len(args) == 3 else None)
                if pos: total_h += 3 * self.distances[r][pos][target_wp]

            elif "send_soil_data" in t_name or "send_rock_data" in t_name:
                r = args[0]
                pos = rover_pos.get(r)
                if pos: total_h += 1 + 3 * self.min_dist_to_trans[r][pos]
            
            elif "send_image_data" in t_name:
                r = args[0]
                pos = rover_pos.get(r)
                if pos: total_h += 1 + 3 * self.min_dist_to_trans[r][pos]

            elif "do_calibrate" in t_name:
                r = args[0]
                # Logic: need to move to a visibility point for the camera's target
                # This is hard to get exactly without camera->obj map, so use a heuristic constant
                total_h += 2

            elif "empty_store" in t_name:
                r = args[1]
                if r in self.full_ids and (state >> self.full_ids[r]) & 1: total_h += 1

            elif t_name == "nop": pass
            else:
                # Primitives or unknown abstract tasks
                total_h += 1

        return int(total_h)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            predicate, args_str = name.split("[")
            args = args_str[:-1].split(",")
            return predicate, args
        return None, []

    def _get_args(self, name: str):
        if "[" in name:
            return name.split("[")[1][:-1].split(",")
        return []
