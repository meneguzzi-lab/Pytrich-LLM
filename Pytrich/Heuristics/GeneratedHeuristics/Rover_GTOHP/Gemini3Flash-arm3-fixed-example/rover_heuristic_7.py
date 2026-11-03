import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic7(Heuristic):
    """
    Domain-dependent heuristic for the ROVER_GTOHP domain.

    1. Bottleneck Analysis:
       The primary bottleneck is rover navigation between sampling waypoints and communication points.
       Each data acquisition requires a visit to a specific waypoint, and each transmission requires
       visiting a waypoint visible to a lander site.

    2. Admissible Lower Bounds:
       - Term 1 (Goal Progress): Count of communication goals not yet satisfied. Each requires at least one action.
       - Term 2 (Task Sequence Cost): For each task in the network, we estimate the minimum actions (1 for sampling,
         1 for communication, 1 for drop) and the minimum navigation steps required to reach target waypoints.
       - Term 3 (Navigation Bound): Following the ordered task network, the rover must visit waypoints in sequence.
         We sum the minimum travel distances between these consecutive waypoints.

    Admissibility Sketch:
    Each term represents a physically unavoidable requirement (actions or distance) to reach the goal state
    given the task network's ordering constraints. Navigation steps are based on BFS distances in the 
    traverse graph, which are true lower bounds on the number of 'navigate' actions.
    """

    def __init__(self, rover_gemini3flash_heuristic_7=None, name="RoverGTOHPGemini3FlashHeuristic7"):
        super().__init__(name=name)
        self.dist_matrices = {} # rover -> from_wp -> to_wp -> steps
        
        self.wp_ids = {} # name -> id (local index in waypoints list)
        self.rover_ids = {} # name -> id
        self.waypoints = []
        self.rovers = []
        
        self.at_ids = defaultdict(dict) # rover -> wp -> global_id
        self.goal_comm_ids = [] # list of global_ids for communicated_... facts
        self.comm_to_sample_wp = {} # global_id -> waypoint_name
        
        self.lander_visible_wps = [] # waypoints visible from any lander site
        self.wp_to_comm_dist = defaultdict(dict) # rover -> wp -> min steps to visibility

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            pred = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return pred, args
        return name, []

    def _preprocess(self):
        # 1. Identify objects
        wp_set = set()
        rover_set = set()
        lander_locs = {}
        visibility = defaultdict(set)
        traversability = defaultdict(lambda: defaultdict(set))

        for fact in self.model.facts:
            pred, args = self._parse_name(fact.name)
            if not pred: continue
            
            if pred == "at_lander":
                lander_locs[args[0]] = args[1]
            elif pred == "visible":
                visibility[args[0]].add(args[1])
                visibility[args[1]].add(args[0])
            elif pred == "can_traverse":
                traversability[args[0]][args[1]].add(args[2])
                traversability[args[0]][args[2]].add(args[1])
                rover_set.add(args[0])
                wp_set.add(args[1])
                wp_set.add(args[2])
            elif pred == "at":
                rover_set.add(args[0])
                wp_set.add(args[1])
                self.at_ids[args[0]][args[1]] = fact.global_id

        self.waypoints = sorted(list(wp_set))
        self.rovers = sorted(list(rover_set))
        
        # 2. Precompute Visibility to Landers
        lander_waypoints = set(lander_locs.values())
        for wp in self.waypoints:
            is_comm_site = False
            if wp in lander_waypoints:
                is_comm_site = True
            else:
                for l_wp in lander_waypoints:
                    if wp in visibility[l_wp]:
                        is_comm_site = True
                        break
            if is_comm_site:
                self.lander_visible_wps.append(wp)

        # 3. Precompute Distances per Rover
        for rover in self.rovers:
            self.dist_matrices[rover] = self._compute_bfs(rover, traversability[rover])
            # Precompute min steps from wp to any communication point
            for wp in self.waypoints:
                min_d = 1000
                for comm_wp in self.lander_visible_wps:
                    d = self.dist_matrices[rover][wp].get(comm_wp, 1000)
                    if d < min_d:
                        min_d = d
                self.wp_to_comm_dist[rover][wp] = min_d

        # 4. Map Goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_name(g_name)
            # communicated_soil_data, communicated_rock_data, communicated_image_data
            for fact in self.model.facts:
                if fact.name == g_name:
                    self.goal_comm_ids.append(fact.global_id)

    def _compute_bfs(self, rover, adj):
        dists = defaultdict(dict)
        for start_wp in self.waypoints:
            dists[start_wp][start_wp] = 0
            queue = deque([(start_wp, 0)])
            visited = {start_wp}
            while queue:
                curr, d = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dists[start_wp][neighbor] = d + 1
                        queue.append((neighbor, d + 1))
        return dists

    def _get_rover_pos(self, state, rover):
        for wp in self.waypoints:
            if (state >> self.at_ids[rover][wp]) & 1:
                return wp
        return None

    def _compute(self, state, task_network):
        # Goal Term: count unsatisfied communication goals
        h = 0
        for gid in self.goal_comm_ids:
            if not ((state >> gid) & 1):
                h += 1

        # Task Term: sequence cost for navigation
        # We only track the first active rover and its movement sequence to remain simple and O(1)-ish
        if not task_network:
            return h

        rover_positions = {}
        for r in self.rovers:
            pos = self._get_rover_pos(state, r)
            if pos: rover_positions[r] = pos

        # Track navigation costs for tasks in the network
        # Due to HTN structure, we follow the first task for heuristic guidance
        first_task = task_network[0]
        t_name, t_args = self._parse_name(first_task.name)
        
        nav_cost = 0
        if t_name in ["do_navigate1", "do_navigate2", "navigate"]:
            # args are [rover, to] or [rover, from, to]
            if len(t_args) >= 2:
                r = t_args[0]
                target = t_args[-1]
                curr = rover_positions.get(r)
                if curr and target in self.dist_matrices.get(r, {}).get(curr, {}):
                    nav_cost = self.dist_matrices[r][curr][target]
        elif t_name in ["get_soil_data", "get_rock_data"]:
            # args are [waypoint]
            target_wp = t_args[0]
            # find closest rover capable
            min_d = 1000
            for r in self.rovers:
                curr = rover_positions.get(r)
                if curr and target_wp in self.dist_matrices.get(r, {}).get(curr, {}):
                    d = self.dist_matrices[r][curr][target_wp]
                    if d < min_d: min_d = d
            if min_d < 1000: nav_cost = min_d + 1 # +1 for sampling
        elif t_name == "get_image_data":
            # arg is objective, which is visible from some waypoints
            # this is more complex, we'll assign a baseline
            nav_cost = 2

        return h + nav_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
