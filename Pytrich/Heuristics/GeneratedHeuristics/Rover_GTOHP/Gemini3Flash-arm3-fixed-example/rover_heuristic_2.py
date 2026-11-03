from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque
import time

class RoverGTOHPGemini3FlashHeuristic2(Heuristic):
    """
    Domain-Dependent Heuristic for Rover GTOHP.

    BOTTLENECK ANALYSIS:
    The primary bottleneck in the Rover domain is navigation and transmission. 
    Rovers must be at specific locations to perform sampling and imaging, and they 
    must then reach waypoints visible from a lander to transmit the data. Since 
    data cannot be shared directly, each bit of information must be independently 
    acquired and communicated.

    ADMISSIBLE LOWER BOUNDS:
    1. Communication/Action Cost: Every unmet 'communicated' goal requires at least 
       one 'communicate' action. If the rover does not yet 'have' the data, it also 
       requires a 'sample' or 'take_image' action. If it is an image, it also requires 
       'calibrate'. This counts mandatory non-navigation actions.
    2. Navigation Cost: A rover must travel to its nearest required sampling or 
       communication waypoint. The shortest path distance (number of navigate actions) 
       is an admissible lower bound on navigation cost.
    3. Store Cost: If a rover needs to sample but its store is full, an additional 
       'drop' action is required.
    """

    def __init__(self, rover_gemini3flash_heuristic_2=None, name="RoverGTOHPGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.rovers = []
        self.waypoints = []
        self.landers = []
        self.objectives = []
        self.modes = []
        self.stores = []

        # Map: Rover -> Waypoint -> Waypoint -> Shortest Path Distance
        self.dist_maps = {}
        # Map: objective -> visible waypoints
        self.objective_visibility = {}
        # Map: lander -> waypoint
        self.lander_positions = {}
        # Waypoints from which communication is possible (can see a lander waypoint)
        self.comm_waypoints = []

        # Predicate ID maps for O(1) checking
        self.at_ids = {} # (rover, wp) -> id
        self.have_soil_ids = {} # (rover, wp) -> id
        self.have_rock_ids = {} # (rover, wp) -> id
        self.have_image_ids = {} # (rover, obj, mode) -> id
        self.comm_soil_ids = {} # wp -> id
        self.comm_rock_ids = {} # wp -> id
        self.comm_image_ids = {} # (obj, mode) -> id
        self.store_full_ids = {} # store -> id
        self.store_of_ids = {} # rover -> store

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Extract objects
        objs = set()
        for f in model.facts:
            _, args = self._parse_fact_name(f.name)
            objs.update(args)

        # In Pytrich, we can't easily iterate all types directly, so we map from facts
        visibility_edges = []
        traversal_edges = {}

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.rovers: self.rovers.append(args[0])
                if args[1] not in self.waypoints: self.waypoints.append(args[1])
            elif pred == "visible":
                visibility_edges.append((args[0], args[1]))
            elif pred == "can_traverse":
                r = args[0]
                if r not in traversal_edges: traversal_edges[r] = []
                traversal_edges[r].append((args[1], args[2]))
            elif pred == "at_lander":
                self.lander_positions[args[0]] = args[1]
                if args[0] not in self.landers: self.landers.append(args[0])
            elif pred == "visible_from":
                if args[0] not in self.objective_visibility: self.objective_visibility[args[0]] = []
                self.objective_visibility[args[0]].append(args[1])
            elif pred == "store_of":
                self.store_of_ids[args[1]] = args[0]
            elif pred == "full":
                self.store_full_ids[args[0]] = f.global_id
            elif pred == "have_soil_analysis":
                self.have_soil_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis":
                self.have_rock_ids[(args[0], args[1])] = f.global_id
            elif pred == "have_image":
                self.have_image_ids[(args[0], args[1], args[2])] = f.global_id
            elif pred == "communicated_soil_data":
                self.comm_soil_ids[args[0]] = f.global_id
            elif pred == "communicated_rock_data":
                self.comm_rock_ids[args[0]] = f.global_id
            elif pred == "communicated_image_data":
                self.comm_image_ids[(args[0], args[1])] = f.global_id

        # Determine Communication-capable waypoints
        lander_wps = list(self.lander_positions.values())
        for w in self.waypoints:
            for lw in lander_wps:
                if (w, lw) in visibility_edges or (lw, w) in visibility_edges or w == lw:
                    if w not in self.comm_waypoints: self.comm_waypoints.append(w)
                    break

        # BFS for all-pairs shortest paths per rover
        for r in self.rovers:
            self.dist_maps[r] = {w1: {w2: 999 for w2 in self.waypoints} for w1 in self.waypoints}
            adj = {w: [] for w in self.waypoints}
            for (u, v) in traversal_edges.get(r, []):
                if (u, v) in visibility_edges or (v, u) in visibility_edges:
                    adj[u].append(v)
                    adj[v].append(u)
            
            for start_node in self.waypoints:
                self.dist_maps[r][start_node][start_node] = 0
                q = deque([(start_node, 0)])
                visited = {start_node}
                while q:
                    curr, d = q.popleft()
                    self.dist_maps[r][start_node][curr] = d
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d+1))

        h_init = self._compute_h(initial_node.state)
        return super().initialize(model, h_init)

    def _get_rover_at(self, state, rover):
        for wp in self.waypoints:
            if (state >> self.at_ids.get((rover, wp), 0)) & 1:
                return wp
        return None

    def _compute_h(self, state):
        h = 0
        # We calculate the sum of non-navigational actions needed for all goals
        # Goals are extracted from model.goals bitwise
        if not self.model.goals: return 0

        # Goal statuses
        unmet_soil = [wp for wp, gid in self.comm_soil_ids.items() if (self.model.goals >> gid) & 1 and not ((state >> gid) & 1)]
        unmet_rock = [wp for wp, gid in self.comm_rock_ids.items() if (self.model.goals >> gid) & 1 and not ((state >> gid) & 1)]
        unmet_img = [om for om, gid in self.comm_image_ids.items() if (self.model.goals >> gid) & 1 and not ((state >> gid) & 1)]

        total_action_cost = 0
        nav_required = []

        # Soil Goals
        for wp in unmet_soil:
            total_action_cost += 1 # communicate
            is_held = False
            for r in self.rovers:
                if (state >> self.have_soil_ids.get((r, wp), 0)) & 1:
                    is_held = True
                    break
            if not is_held:
                total_action_cost += 1 # sample

        # Rock Goals
        for wp in unmet_rock:
            total_action_cost += 1 # communicate
            is_held = False
            for r in self.rovers:
                if (state >> self.have_rock_ids.get((r, wp), 0)) & 1:
                    is_held = True
                    break
            if not is_held:
                total_action_cost += 1 # sample

        # Image Goals
        for om in unmet_img:
            obj, mode = om
            total_action_cost += 1 # communicate
            is_held = False
            for r in self.rovers:
                if (state >> self.have_image_ids.get((r, obj, mode), 0)) & 1:
                    is_held = True
                    break
            if not is_held:
                total_action_cost += 2 # calibrate + take_image

        # Rover Navigation Lower Bound
        rover_navs = []
        for r in self.rovers:
            at_wp = self._get_rover_at(state, r)
            if not at_wp: continue

            # Min distance to any relevant target for this rover
            min_d = 0
            targets = []
            # Waypoints where samples are needed
            targets.extend([wp for wp in unmet_soil])
            targets.extend([wp for wp in unmet_rock])
            # Waypoints where images can be taken
            for obj, mode in unmet_img:
                targets.extend(self.objective_visibility.get(obj, []))
            # Waypoints where communication can occur
            targets.extend(self.comm_waypoints)

            if targets:
                min_d = min([self.dist_maps[r][at_wp].get(t, 99) for t in targets] + [99])
                if min_d < 99: rover_navs.append(min_d)

        nav_cost = max(rover_navs) if rover_navs else 0
        
        return total_action_cost + nav_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute_h(node.state)
        # Add a small task network penalty to guide tie-breaking in search
        h_val += 0.1 * len(node.task_network)
        super().update_info(int(h_val))
        return int(h_val)
