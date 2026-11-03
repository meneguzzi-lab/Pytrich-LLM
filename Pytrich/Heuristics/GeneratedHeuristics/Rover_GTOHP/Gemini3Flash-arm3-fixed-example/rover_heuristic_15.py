from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections
import time

class RoverGTOHPGemini3FlashHeuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    1. Name the bottleneck:
    The primary bottleneck is the rigid data lifecycle (Navigation -> Storage Management -> Sampling -> Navigation -> Communication),
    where every piece of requested data requires at least one sampling action and one communication action, 
    separated by navigation steps.

    2. Admissible lower bounds:
    - Term 1 (h_actions): Each goal fact (communicated_data) not present in the current state requires at least 
      one 'communicate' action. If the corresponding 'have_data' analysis is not held by any rover, it 
      also requires at least one 'sample' or 'take_image' action. Admissibility: These actions are distinct 
      per piece of data and cannot be shared.
    - Term 2 (h_nav): For every active navigation task in the task network, the rover must perform at least 
      'dist(current_pos, target_pos)' navigate actions. Admissibility: Shortest-path distance in the 
      connectivity graph is a strict lower bound on move actions.
    """

    def __init__(self, rover_gemini3flash_heuristic_15=None, name="RoverGTOHPGemini3FlashHeuristic15"):
        super().__init__(name=name)
        self.waypoint_map = {}  # name -> index
        self.rover_map = {}     # name -> index
        self.dist = {}          # rover_idx -> 2D array of distances
        self.goal_requirements = [] # list of (type, target_arg, goal_fact_id)
        self.have_fact_ids = collections.defaultdict(lambda: collections.defaultdict(list)) # type -> arg -> [global_ids]
        self.at_fact_ids = collections.defaultdict(dict) # rover_idx -> waypoint_idx -> global_id
        self.comm_task_ids = set()
        self.sample_task_ids = set()
        self.nav_task_ids = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute_h(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            bracket_idx = name.find("[")
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _preprocess(self):
        waypoints = set()
        rovers = set()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at" and len(args) == 2:
                rovers.add(args[0])
                waypoints.add(args[1])
        
        sorted_w = sorted(list(waypoints))
        self.waypoint_map = {name: i for i, name in enumerate(sorted_w)}
        sorted_r = sorted(list(rovers))
        self.rover_map = {name: i for i, name in enumerate(sorted_r)}

        # Build adjacency per rover
        adj = collections.defaultdict(lambda: collections.defaultdict(list))
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "can_traverse" and len(args) == 3:
                r, w1, w2 = args
                if r in self.rover_map and w1 in self.waypoint_map and w2 in self.waypoint_map:
                    adj[self.rover_map[r]][self.waypoint_map[w1]].append(self.waypoint_map[w2])

        for r_idx in range(len(self.rover_map)):
            self.dist[r_idx] = self._compute_bfs(len(self.waypoint_map), adj[r_idx])

        # Map fact IDs
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at" and len(args) == 2:
                r_idx = self.rover_map.get(args[0])
                w_idx = self.waypoint_map.get(args[1])
                if r_idx is not None and w_idx is not None:
                    self.at_fact_ids[r_idx][w_idx] = fact.global_id
            elif pred in ["have_soil_analysis", "have_rock_analysis"] and len(args) == 2:
                self.have_fact_ids[pred][args[1]].append(fact.global_id)
            elif pred == "have_image" and len(args) == 3:
                self.have_fact_ids[pred][(args[1], args[2])].append(fact.global_id)

        # Map goals from bitwise integer
        if self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_names:
                pred, args = self._parse_fact_name(g_name)
                f_id = -1
                for f in self.model.facts:
                    if f.name == g_name: f_id = f.global_id; break
                if pred == "communicated_soil_data":
                    self.goal_requirements.append(("soil", args[0], f_id))
                elif pred == "communicated_rock_data":
                    self.goal_requirements.append(("rock", args[0], f_id))
                elif pred == "communicated_image_data":
                    self.goal_requirements.append(("image", (args[0], args[1]), f_id))

    def _compute_bfs(self, n, adj_map):
        dists = [[1000] * n for _ in range(n)]
        for start in range(n):
            dists[start][start] = 0
            queue = collections.deque([start])
            while queue:
                u = queue.popleft()
                for v in adj_map[u]:
                    if dists[start][v] == 1000:
                        dists[start][v] = dists[start][u] + 1
                        queue.append(v)
        return dists

    def _get_rover_pos(self, state, r_idx):
        for w_idx, f_id in self.at_fact_ids[r_idx].items():
            if (state >> f_id) & 1: return w_idx
        return None

    def _compute_h(self, state, task_network):
        h = 0
        # Term 1: Required sampling and communication actions
        for g_type, g_arg, g_id in self.goal_requirements:
            if not ((state >> g_id) & 1):
                h += 1 # Communication required
                have_pred = {"soil": "have_soil_analysis", "rock": "have_rock_analysis", "image": "have_image"}[g_type]
                held = False
                for h_id in self.have_fact_ids[have_pred][g_arg]:
                    if (state >> h_id) & 1:
                        held = True; break
                if not held:
                    h += 1 # Sampling required

        # Term 2: Navigation committed in the current task network
        r_nav = {}
        for t_id in task_network:
            task = self.model.get_component(t_id)
            pred, args = self._parse_fact_name(task.name)
            if pred in ["navigate", "do_navigate1", "do_navigate2"] and len(args) >= 2:
                r_name = args[0]
                dest_name = args[-1]
                if r_name in self.rover_map and dest_name in self.waypoint_map:
                    r_idx = self.rover_map[r_name]
                    dest_idx = self.waypoint_map[dest_name]
                    curr_w = self._get_rover_pos(state, r_idx)
                    if curr_w is not None:
                        d = self.dist[r_idx][curr_w][dest_idx]
                        if d < 1000:
                            r_nav[r_idx] = max(r_nav.get(r_idx, 0), d)
        
        h += sum(r_nav.values())
        
        # Sub-unit tie-breaker: prefer smaller task networks
        return h + (len(task_network) * 0.001)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute_h(node.state, node.task_network)
        self.update_info(h_val)
        return int(h_val)
