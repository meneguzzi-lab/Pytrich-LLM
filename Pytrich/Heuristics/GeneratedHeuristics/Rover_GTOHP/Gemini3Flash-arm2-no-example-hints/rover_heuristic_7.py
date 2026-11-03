from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic7(Heuristic):
    """
    Domain-Dependent Heuristic for ROVER GTOHP.
    
    Bottleneck:
    The serialization of navigation and the restricted store capacity (one item at a time).
    A rover must navigate to a sample site, collect it, potentially navigate to a communication 
    site, and transmit it before the store is free for another sample.

    Admissible Lower Bounds:
    1. Navigation Cost (3*d): Each hop in do_navigate1 costs exactly 3 primitive actions 
       (visit + navigate + unvisit). This is a direct lower bound based on distance.
    2. Data Action Cost (2 actions per goal): Each unsatiated goal (communicated_data) requires 
       at least one sampling/imaging action and one communication action.
    3. Store Clearing Cost (1 action): If a rover needs to sample but its store is full, 
       a 'drop' action is unavoidable.
    """

    def __init__(self, rover_gemini3flash_heuristic_7="rover_gemini3flash_heuristic_7"):
        super().__init__(rover_gemini3flash_heuristic_7)
        self.rover_id_map = {}
        self.wp_id_map = {}
        self.at_fact_list = []
        self.at_fact_to_coords = {}
        self.empty_fact_id = {}
        self.dist = {}
        self.goal_info = []
        self.capable_rovers_soil = []
        self.capable_rovers_rock = []
        self.capable_rovers_image = collections.defaultdict(list)
        self.rover_to_store_id = {}
        self.num_rovers = 0
        self.num_wps = 0

    def _parse_name(self, name: str):
        name = name.strip().replace("+", "").replace("-", "")
        if "[" in name:
            parts = name.split("[")
            pred = parts[0]
            args = parts[1].replace("]", "").split(",")
            return pred, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map constants
        rovers = sorted(list(set(args[0] for op in model.operators if "[" in op.name and (self._parse_name(op.name)[0] in ["navigate", "sample_soil", "sample_rock"]) for args in [self._parse_name(op.name)[1]] if args)))
        waypoints = sorted(list(set(args[i] for op in model.operators if "[" in op.name for i in (1, 2) if i < len(self._parse_name(op.name)[1]) for args in [self._parse_name(op.name)[1]] if "waypoint" in args[i])))
        
        self.num_rovers = len(rovers)
        self.num_wps = len(waypoints)
        self.rover_id_map = {name: i for i, name in enumerate(rovers)}
        self.wp_id_map = {name: i for i, name in enumerate(waypoints)}

        # 2. Build Navigation Graph for each rover
        adj = [collections.defaultdict(set) for _ in range(self.num_rovers)]
        for op in model.operators:
            pred, args = self._parse_name(op.name)
            if pred == "navigate":
                r_idx = self.rover_id_map.get(args[0])
                w1_idx = self.wp_id_map.get(args[1])
                w2_idx = self.wp_id_map.get(args[2])
                if r_idx is not None and w1_idx is not None and w2_idx is not None:
                    adj[r_idx][w1_idx].add(w2_idx)

        # 3. BFS for all-pairs distances per rover
        self.dist = [[[999 for _ in range(self.num_wps)] for _ in range(self.num_wps)] for _ in range(self.num_rovers)]
        for r in range(self.num_rovers):
            for start in range(self.num_wps):
                self.dist[r][start][start] = 0
                queue = collections.deque([start])
                while queue:
                    curr = queue.popleft()
                    for neighbor in adj[r][curr]:
                        if self.dist[r][start][neighbor] == 999:
                            self.dist[r][start][neighbor] = self.dist[r][start][curr] + 1
                            queue.append(neighbor)

        # 4. Map facts for O(1) lookups
        store_mapping = {}
        for op in model.operators:
            pred, args = self._parse_name(op.name)
            if pred in ["sample_soil", "sample_rock"]:
                store_mapping[args[1]] = self.rover_id_map[args[0]]
                if pred == "sample_soil": self.capable_rovers_soil.append(self.rover_id_map[args[0]])
                else: self.capable_rovers_rock.append(self.rover_id_map[args[0]])
            if pred == "take_image":
                r_idx = self.rover_id_map[args[0]]
                mode = args[4]
                self.capable_rovers_image[mode].append(r_idx)

        self.capable_rovers_soil = list(set(self.capable_rovers_soil))
        self.capable_rovers_rock = list(set(self.capable_rovers_rock))

        for fact in model.facts:
            pred, args = self._parse_name(fact.name)
            if pred == "at":
                r_idx, w_idx = self.rover_id_map.get(args[0]), self.wp_id_map.get(args[1])
                if r_idx is not None and w_idx is not None:
                    self.at_fact_list.append(fact.global_id)
                    self.at_fact_to_coords[fact.global_id] = (r_idx, w_idx)
            elif pred == "empty":
                s_name = args[0]
                if s_name in store_mapping:
                    self.empty_fact_id[store_mapping[s_name]] = fact.global_id

        # 5. Goal Indexing
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_name(g_name)
            f_id = -1
            for fact in model.facts:
                if fact.name == g_name: f_id = fact.global_id; break
            if f_id != -1:
                if pred == "communicated_soil_data": self.goal_info.append((f_id, "soil", args[0]))
                elif pred == "communicated_rock_data": self.goal_info.append((f_id, "rock", args[0]))
                elif pred == "communicated_image_data": self.goal_info.append((f_id, "image", f"{args[0]}_{args[1]}"))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h = 0
        rover_pos = [None] * self.num_rovers
        for f_id in self.at_fact_list:
            if (state >> f_id) & 1:
                r_idx, w_idx = self.at_fact_to_coords[f_id]
                rover_pos[r_idx] = w_idx

        tn_goals = set()
        for task in task_network:
            pred, args = self._parse_name(task.name)
            if "soil" in pred:
                wp_idx = self.wp_id_map.get(args[-1])
                tn_goals.add(("soil", args[-1]))
                h += self._est_task_cost(self.capable_rovers_soil, rover_pos, wp_idx, 2)
            elif "rock" in pred:
                wp_idx = self.wp_id_map.get(args[-1])
                tn_goals.add(("rock", args[-1]))
                h += self._est_task_cost(self.capable_rovers_rock, rover_pos, wp_idx, 2)
            elif "image" in pred:
                tn_goals.add(("image", f"{args[0]}_{args[1]}"))
                h += 3 # Baseline image actions
            elif pred == "do_navigate1":
                r_idx, w_idx = self.rover_id_map.get(args[0]), self.wp_id_map.get(args[1])
                if r_idx is not None and w_idx is not None and rover_pos[r_idx] is not None:
                    h += 3 * self.dist[r_idx][rover_pos[r_idx]][w_idx]
            elif pred in ["sample_soil", "sample_rock", "take_image", "communicate_soil_data1", "communicate_rock_data1", "communicate_image_data", "drop", "calibrate"]: 
                h += 1

        # Unaccounted goals
        for g_id, g_type, g_key in self.goal_info:
            if not ((state >> g_id) & 1):
                if (g_type, g_key) not in tn_goals:
                    h += 2

        # Store capacity penalty
        for r_idx in range(self.num_rovers):
            if r_idx in self.empty_fact_id:
                if not ((state >> self.empty_fact_id[r_idx]) & 1):
                    h += 1

        return int(h)

    def _est_task_cost(self, capable_rovers, rover_pos, target_wp, base_actions):
        if target_wp is None: return base_actions
        min_nav = 99
        for r_idx in capable_rovers:
            curr = rover_pos[r_idx]
            if curr is not None:
                d = self.dist[r_idx][curr][target_wp]
                if d < min_nav: min_nav = d
        return (3 * min_nav if min_nav < 99 else 9) + base_actions
