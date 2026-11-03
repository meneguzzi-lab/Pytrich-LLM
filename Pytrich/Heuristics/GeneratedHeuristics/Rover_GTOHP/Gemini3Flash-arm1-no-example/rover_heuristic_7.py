from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RoverGTOHPGemini3FlashHeuristic7(Heuristic):
    """
    Rover_GTOHP Bottleneck Analysis:
    The primary bottleneck is the sequential movement of rovers through a shared waypoint graph
    to perform data-collection tasks (sampling/imaging) and then communicating that data 
    under visibility constraints with a lander. Navigation actions make up the majority of the plan length.

    Admissible Bounds:
    1. Data Goal actions: For every data goal (soil, rock, image) not yet communicated, 
       at least 1 'communicate' action and 1 'sample'/'take_image' action (if not collected) 
       are required. These are individual, unavoidable actions for each datum.
    2. Navigation distance: For any unfinished task in the network targetting a specific waypoint,
       a rover must traverse at least the BFS distance from its current location to that waypoint.
       This is a lower bound on 'navigate' actions.
    """

    def __init__(self, rover_gemini3flash_heuristic_7="rover_gemini3flash_heuristic_7"):
        super().__init__(name=rover_gemini3flash_heuristic_7)
        self.at_fact_ids = {} # (rover, wp) -> global_id
        self.rover_at_list = {} # rover -> list of global_ids for at
        self.have_soil_ids = {} # wp -> [global_ids per rover]
        self.have_rock_ids = {} # wp -> [global_ids per rover]
        self.have_image_ids = {} # (obj, mode) -> [global_ids per rover]
        self.comm_soil_ids = {} # wp -> global_id
        self.comm_rock_ids = {} # wp -> global_id
        self.comm_image_ids = {} # (obj, mode) -> global_id
        self.equipped_soil = [] # [rover]
        self.equipped_rock = [] # [rover]
        self.equipped_image = [] # [rover]
        self.bfs_dist = {} # rover -> wp_from -> wp_to -> int

        self.goal_soil = [] # [wp]
        self.goal_rock = [] # [wp]
        self.goal_image = [] # [(obj, mode)]

        self.rovers = []
        self.waypoints = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse objects
        all_rovers = set()
        all_wps = set()
        
        # 2. Map facts
        can_traverse_map = {} # rover -> wp -> [adj_wp]

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "at":
                r, wp = args[0], args[1]
                all_rovers.add(r)
                all_wps.add(wp)
                self.at_fact_ids[(r, wp)] = fact.global_id
                if r not in self.rover_at_list: self.rover_at_list[r] = []
                self.rover_at_list[r].append(fact.global_id)

            elif pred == "have_soil_analysis":
                r, wp = args[0], args[1]
                if wp not in self.have_soil_ids: self.have_soil_ids[wp] = []
                self.have_soil_ids[wp].append(fact.global_id)
            elif pred == "have_rock_analysis":
                r, wp = args[0], args[1]
                if wp not in self.have_rock_ids: self.have_rock_ids[wp] = []
                self.have_rock_ids[wp].append(fact.global_id)
            elif pred == "have_image":
                r, obj, mode = args[0], args[1], args[2]
                key = (obj, mode)
                if key not in self.have_image_ids: self.have_image_ids[key] = []
                self.have_image_ids[key].append(fact.global_id)

            elif pred == "communicated_soil_data":
                self.comm_soil_ids[args[0]] = fact.global_id
            elif pred == "communicated_rock_data":
                self.comm_rock_ids[args[0]] = fact.global_id
            elif pred == "communicated_image_data":
                self.comm_image_ids[(args[0], args[1])] = fact.global_id

            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.append(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.append(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_image.append(args[0])

            elif pred == "can_traverse":
                r, w1, w2 = args[0], args[1], args[2]
                if r not in can_traverse_map: can_traverse_map[r] = {}
                if w1 not in can_traverse_map[r]: can_traverse_map[r][w1] = []
                can_traverse_map[r][w1].append(w2)

        self.rovers = list(all_rovers)
        self.waypoints = list(all_wps)

        # 3. BFS Navigation distance
        for r in self.rovers:
            self.bfs_dist[r] = {}
            for start_node in self.waypoints:
                dists = {start_node: 0}
                queue = deque([start_node])
                while queue:
                    curr = queue.popleft()
                    for neighbor in can_traverse_map.get(r, {}).get(curr, []):
                        if neighbor not in dists:
                            dists[neighbor] = dists[curr] + 1
                            queue.append(neighbor)
                self.bfs_dist[r][start_node] = dists

        # 4. Filter Goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "communicated_soil_data": self.goal_soil.append(args[0])
            elif pred == "communicated_rock_data": self.goal_rock.append(args[1])
            elif pred == "communicated_image_data": self.goal_image.append((args[0], args[1]))

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        h = 0
        
        # 1. Action counts for missing goals
        for wp in self.goal_soil:
            if not ((state >> self.comm_soil_ids.get(wp, 0)) & 1):
                h += 1 # Communicate
                already_sampled = False
                for f_id in self.have_soil_ids.get(wp, []):
                    if (state >> f_id) & 1: already_sampled = True; break
                if not already_sampled: h += 1 # Sample

        for wp in self.goal_rock:
            if not ((state >> self.comm_rock_ids.get(wp, 0)) & 1):
                h += 1 # Communicate
                already_sampled = False
                for f_id in self.have_rock_ids.get(wp, []):
                    if (state >> f_id) & 1: already_sampled = True; break
                if not already_sampled: h += 1 # Sample

        for key in self.goal_image:
            if not ((state >> self.comm_image_ids.get(key, 0)) & 1):
                h += 1 # Communicate
                already_sampled = False
                for f_id in self.have_image_ids.get(key, []):
                    if (state >> f_id) & 1: already_sampled = True; break
                if not already_sampled: h += 1 # Take image (+1 calibrate ignored for simplicity)

        # 2. Navigation distance for current tasks
        # Map rovers to current position
        rover_locs = {}
        for r in self.rovers:
            for f_id in self.rover_at_list.get(r, []):
                if (state >> f_id) & 1:
                    # Extract wp from fact indexing logic isn't clean here, so we lookup in mapping
                    # But we only need the wp name.
                    wp_name = self.model.facts[f_id].name.split(",")[1].replace("]", "")
                    rover_locs[r] = wp_name
                    break

        # For each task in network, find min distance to its destination
        processed_targets = set()
        for task in task_network:
            t_name, t_args = self._parse_task_name(task.name)
            if not t_args: continue
            target_wp = t_args[-1] # Usually the last arg in get/navigate/send tasks
            if target_wp in self.waypoints and target_wp not in processed_targets:
                processed_targets.add(target_wp)
                # Min dist from any rover to this destination
                min_d = 1000
                for r, loc in rover_locs.items():
                    d = self.bfs_dist.get(r, {}).get(loc, {}).get(target_wp, 1000)
                    if d < min_d: min_d = d
                if min_d < 1000:
                    h += min_d

        return h

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []
