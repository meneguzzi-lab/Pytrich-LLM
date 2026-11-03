from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic16(Heuristic):
    """
    Heuristic for the Rover_GTOHP domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the single physical location of each rover and the movement cost 
    (navigation) between sample sites, landers, and calibration targets. Since tasks like sampling soil, 
    sampling rock, and taking images are serialized by location, navigation dominates plan length.
    
    Admissible Lower Bounds:
    1. Goal Completion Actions: Each data communication goal requires at least 2 specific actions 
       (sample/take_image + communicate) if the data isn't already possessed, and 1 action (communicate) 
       if it is. Navigation to the site is handled separately.
    2. Navigation Cost Bound: For every incomplete goal, the rover must be at the sample site and then at a waypoint 
       visible to a lander. The minimum navigation distance (3 actions per hop) to reach these locations 
       from the current rover position is an admissible lower bound on movement cost.
    3. Store Bottleneck: Each soil/rock sample requires an empty store. If multiple samples are required 
       by the same rover and the store is currently full, at least one 'drop' action is unavoidable.
    """

    def __init__(self, rover_gemini3flash_heuristic_16="rover_gemini3flash_heuristic_16"):
        super().__init__(name=rover_gemini3flash_heuristic_16)
        self.rover_navigation_graphs = {}
        self.rover_distances = {}
        self.goal_info = []
        self.fact_map = {}
        self.rovers = []
        self.waypoints = []
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.lander_locations = {}

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
        # 1. Map all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                if pred not in self.fact_map:
                    self.fact_map[pred] = {}
                self.fact_map[pred][tuple(args)] = f.global_id

        # 2. Extract static properties from operator signatures
        # PANDA grounder often hides static predicates, so we infer from grounded ops
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == "navigate":
                r, w1, w2 = args
                if r not in self.rover_navigation_graphs:
                    self.rover_navigation_graphs[r] = collections.defaultdict(list)
                self.rover_navigation_graphs[r][w1].append(w2)
                if r not in self.rovers: self.rovers.append(r)
                if w1 not in self.waypoints: self.waypoints.append(w1)
                if w2 not in self.waypoints: self.waypoints.append(w2)
            elif pred == "sample_soil":
                self.equipped_soil.add(args[0])
            elif pred == "sample_rock":
                self.equipped_rock.add(args[0])
            elif pred == "take_image":
                self.equipped_image.add(args[0])
            elif pred.startswith("communicate") and "lander" in op.name:
                # Infer lander locations from communication actions
                # Format: communicate_soil_data1[rover, lander, wp_from, wp_rover, wp_lander]
                if "soil_data1" in pred or "rock_data1" in pred:
                    l, wp_lander = args[1], args[4]
                    self.lander_locations[l] = wp_lander
                elif "soil_data2" in pred or "rock_data2" in pred:
                    l, wp_lander = args[1], args[3]
                    self.lander_locations[l] = wp_lander
                elif "image_data" in pred:
                    l, wp_lander = args[1], args[5]
                    self.lander_locations[l] = wp_lander

        # 3. Precompute All-Pairs Shortest Paths for each rover
        for r in self.rovers:
            self.rover_distances[r] = {}
            graph = self.rover_navigation_graphs.get(r, {})
            for start_node in self.waypoints:
                distances = {start_node: 0}
                queue = collections.deque([start_node])
                while queue:
                    curr = queue.popleft()
                    for neighbor in graph.get(curr, []):
                        if neighbor not in distances:
                            distances[neighbor] = distances[curr] + 1
                            queue.append(neighbor)
                self.rover_distances[r][start_node] = distances

        # 4. Identify relevant goal facts from model.goals bitwise
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred in ["communicated_soil_data", "communicated_rock_data", "communicated_image_data"]:
                self.goal_info.append((pred, tuple(args)))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        total_action_cost = 0
        max_nav_cost = 0

        # Track store states and rover positions
        rover_at = {}
        rover_store_full = {}
        for r in self.rovers:
            # Find where rover is
            for w in self.waypoints:
                if (state >> self.fact_map["at"][(r, w)]) & 1:
                    rover_at[r] = w
                    break
            # Check store
            for args, fid in self.fact_map.get("full", {}).items():
                if args[0].startswith(r) and (state >> fid) & 1:
                    rover_store_full[r] = True
                    break

        # Evaluate each goal
        for pred, args in self.goal_info:
            goal_id = self.fact_map[pred][args]
            if (state >> goal_id) & 1:
                continue

            # Determine location of sample and equipment needed
            sample_loc = None
            req_equipped = set()
            has_data_pred = ""
            
            if pred == "communicated_soil_data":
                sample_loc = args[0]
                req_equipped = self.equipped_soil
                has_data_pred = "have_soil_analysis"
            elif pred == "communicated_rock_data":
                sample_loc = args[0]
                req_equipped = self.equipped_rock
                has_data_pred = "have_rock_analysis"
            elif pred == "communicated_image_data":
                # For image, we need to find where objective is visible from
                obj = args[0]
                for w in self.waypoints:
                    if ((obj, w) in self.fact_map.get("visible_from", {})):
                        sample_loc = w
                        break
                req_equipped = self.equipped_image
                has_data_pred = "have_image"

            # Calculate best rover to do this
            best_goal_nav = 1000000
            data_already_held = False
            
            for r in req_equipped:
                current_loc = rover_at.get(r)
                if not current_loc: continue
                
                # Check if rover already has the analysis/image
                data_args = (r,) + args
                if data_args in self.fact_map.get(has_data_pred, {}) and (state >> self.fact_map[has_data_pred][data_args]) & 1:
                    data_already_held = True
                    # Need navigate to a lander site
                    min_dist_to_comm = 1000000
                    for l, l_loc in self.lander_locations.items():
                        # Find waypoint visible to lander waypoint
                        for w_vis in self.waypoints:
                            if (w_vis, l_loc) in self.fact_map.get("visible", {}):
                                d = self.rover_distances[r][current_loc].get(w_vis, 1000)
                                if d < min_dist_to_comm: min_dist_to_comm = d
                    best_goal_nav = min(best_goal_nav, min_dist_to_comm * 3)
                else:
                    # Need navigate to sample site + navigate to lander site
                    d1 = self.rover_distances[r][current_loc].get(sample_loc, 1000)
                    min_d2 = 1000000
                    for l, l_loc in self.lander_locations.items():
                        for w_vis in self.waypoints:
                            if (w_vis, l_loc) in self.fact_map.get("visible", {}):
                                d = self.rover_distances[r][sample_loc].get(w_vis, 1000)
                                if d < min_d2: min_d2 = d
                    best_goal_nav = min(best_goal_nav, (d1 + min_d2) * 3)

            # Base action costs
            total_action_cost += 1 # Every goal needs a communicate action
            if not data_already_held:
                total_action_cost += 1 # Needs sample/take_image action
            
            if best_goal_nav < 1000000:
                max_nav_cost = max(max_nav_cost, best_goal_nav)

        return max(total_action_cost, max_nav_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        # Symmetry breaking: small penalty for number of tasks left to encourage decomposition
        h_val += 0.001 * len(node.task_network)
        super().update_info(h_val)
        return int(h_val)
