import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    BOTTLENECK: The primary bottleneck in this domain is navigation across the waypoint graph 
    to synchronize data acquisition (sampling/imaging) and data transmission (communication to lander).

    ADMISSIBILITY SKETCHES:
    1. Communication Term: Each uncommunicated data goal requires at least one 'communicate' action. 
       This is a lower bound because it's a mandatory final step for every goal fact.
    2. Acquisition Term: Each item of data not currently held by any rover requires at least one 
       'sample' or 'take_image' action. This is a lower bound as these are prerequisites for communication.
    3. Navigation Term: The shortest-path distance (BFS) from a rover's current location to its 
       next objective waypoint. This is a lower bound on the number of 'navigate' actions required.
    """

    def __init__(self, rover_gemini3flash_heuristic_9=None, name="RoverGTOHPGemini3FlashHeuristic9"):
        super().__init__(name=name)
        self.facts_map = {}  # (pred, args) -> global_id
        self.rovers = []
        self.waypoints = []
        self.landers = []
        self.adj = {}        # rover -> waypoint -> list of neighbors
        self.distances = {}  # rover -> waypoint -> waypoint -> distance
        self.comm_waypoints = {} # lander -> set of waypoints that can see it
        self.goals_data = [] # List of (type, target) goals

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, tuple(args)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].split()
            if inner: return inner[0], tuple(inner[1:])
        return None, ()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Extract objects and map facts
        lander_at = {} # lander -> waypoint
        visibility = [] # (w1, w2)
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            self.facts_map[(pred, args)] = f.global_id
            
            if pred == "at_lander" and len(args) == 2:
                lander_at[args[0]] = args[1]
                if args[0] not in self.landers: self.landers.append(args[0])
            elif pred == "at" and args[0].startswith("rover"):
                if args[0] not in self.rovers: self.rovers.append(args[0])
            elif pred == "visible" and len(args) == 2:
                visibility.append(args)
            elif pred == "can_traverse" and len(args) == 3:
                rover, w1, w2 = args
                if rover not in self.adj: self.adj[rover] = {}
                if w1 not in self.adj[rover]: self.adj[rover][w1] = []
                self.adj[rover][w1].append(w2)

        # 2. Precompute distances using BFS for each rover
        for r in self.rovers:
            self.distances[r] = {}
            r_adj = self.adj.get(r, {})
            all_w = set(r_adj.keys())
            for start_w in all_w:
                dists = {start_w: 0}
                queue = deque([start_w])
                while queue:
                    curr = queue.popleft()
                    for neighbor in r_adj.get(curr, []):
                        if neighbor not in dists:
                            dists[neighbor] = dists[curr] + 1
                            queue.append(neighbor)
                self.distances[r][start_w] = dists

        # 3. Precompute communication waypoints (where to be to talk to lander)
        # A waypoint w can talk to lander L if w sees L's waypoint w_lander
        for L, w_lander in lander_at.items():
            self.comm_waypoints[L] = {w_lander}
            for w1, w2 in visibility:
                if w2 == w_lander: self.comm_waypoints[L].add(w1)
                if w1 == w_lander: self.comm_waypoints[L].add(w2)

        # 4. Parse goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred in ["communicated_soil_data", "communicated_rock_data", "communicated_image_data"]:
                self.goals_data.append((pred, args))

        h_val = self._compute(initial_node.state)
        return super().initialize(model, h_val)

    def _get_f(self, state, pred, args):
        fid = self.facts_map.get((pred, args))
        if fid is None: return 0
        return (state >> fid) & 1

    def _compute(self, state):
        total_cost = 0
        
        # Track rover info
        rover_at = {}
        for r in self.rovers:
            for w in self.adj.get(r, {}):
                if self._get_f(state, "at", (r, w)):
                    rover_at[r] = w
                    break

        # Term 1 & 2: Data Goal completion
        active_goals_cost = 0
        for pred, args in self.goals_data:
            if self._get_f(state, pred, args): continue
            
            # Map goal type to acquisition fact
            if pred == "communicated_soil_data":
                target_w = args[0]
                has_pred, acq_args = "have_soil_analysis", (target_w,)
            elif pred == "communicated_rock_data":
                target_w = args[0]
                has_pred, acq_args = "have_rock_analysis", (target_w,)
            else: # image
                obj, mode = args
                has_pred, acq_args = "have_image", (obj, mode)
                target_w = None # Visibility logic needed for images

            # Check if any rover has it
            any_has = False
            for r in self.rovers:
                if pred == "communicated_image_data":
                    if self._get_f(state, has_pred, (r, args[0], args[1])): any_has = True; break
                else:
                    if self._get_f(state, has_pred, (r, target_w)): any_has = True; break
            
            if any_has:
                active_goals_cost += 1 # Only need to communicate
            else:
                active_goals_cost += 2 # Need to sample AND communicate
        
        # Term 3: Navigation Cost (sum of min distances to required waypoints)
        # Note: Sum is used here for guidance; for strict admissibility use max
        nav_cost = 0
        for pred, args in self.goals_data:
            if self._get_f(state, pred, args): continue
            
            # Find required waypoints for this goal
            if pred == "communicated_image_data":
                # For image, we need a waypoint where the objective is visible
                obj = args[0]
                req_ws = [w for w in self.adj.get(self.rovers[0], {}) if self._get_f(state, "visible_from", (obj, w))]
            else:
                req_ws = [args[0]]
            
            # Distance to acquisition point
            min_dist = 999
            for r in self.rovers:
                curr_w = rover_at.get(r)
                if not curr_w: continue
                for rw in req_ws:
                    d = self.distances[r].get(curr_w, {}).get(rw, 5)
                    if d < min_dist: min_dist = d
            
            if min_dist < 999: nav_cost += min_dist

        return active_goals_cost + nav_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state)
        # Add small penalty for task network size to guide decomposition
        h_val += 0.1 * len(node.task_network)
        super().update_info(int(h_val))
        return int(h_val)
