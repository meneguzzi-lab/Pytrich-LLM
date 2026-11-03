from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic15(Heuristic):
    """
    Domain-Dependent Heuristic for Rover_GTOHP.

    1. Bottleneck: Navigation. The primary serialization in this domain is the movement of rovers 
       between sampling locations and communication-ready waypoints. 

    2. Admissible Lower Bounds:
       - h_comm: Each uncommunicated goal requires at least 1 action (the communication action itself).
       - h_work: For each task in the network, we estimate the sum of its necessary component actions 
         (navigate, sample, calibrate, communicate) based on precomputed shortest paths.
       - Admissibility: Both are lower bounds because they assume no obstacles and minimum sequence costs.
    """

    def __init__(self, rover_gemini3flash_heuristic_15="rover_gemini3flash_heuristic_15"):
        super().__init__(rover_gemini3flash_heuristic_15)
        self.rover_at_facts = {}  # rover_name -> {wp_name -> global_id}
        self.dist = {}            # rover_name -> {wp_from -> {wp_to -> cost}}
        self.comm_points = set() # waypoints visible to a lander
        self.rover_to_store = {}  # rover -> store
        self.rover_to_cameras = collections.defaultdict(set)
        self.goal_facts = []

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
        # 1. Map predicates to IDs
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                rover, wp = args[0], args[1]
                if rover not in self.rover_at_facts:
                    self.rover_at_facts[rover] = {}
                self.rover_at_facts[rover][wp] = f.global_id

        # 2. Extract static info from operators (since PANDA strips static facts)
        # We build the navigation graph and comm points from op names
        rover_adj = collections.defaultdict(lambda: collections.defaultdict(set))
        all_waypoints = set()
        
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == "navigate":
                # args: [rover, from, to]
                rover_adj[args[0]][args[1]].add(args[2])
                all_waypoints.add(args[1])
                all_waypoints.add(args[2])
            elif pred.startswith("communicate"):
                # args: [rover, lander, target, rover_wp, lander_wp, ...]
                # communicate_soil_data1 has 5 args, communicate_soil_data2 has 4
                # rover_wp is index 3 for 1, and index 2 for 2. 
                # But a safer bet is just checking all args for common waypoint names.
                # Or looking at common patterns: communicate_soil_data1[r,l,s,w1,w2] -> w1 is comm spot
                # communicate_soil_data2[r,l,s,w1] -> s is also wp? No, w1 is comm spot.
                if pred.endswith("data1"):
                    self.comm_points.add(args[3])
                elif pred.endswith("data2"):
                    self.comm_points.add(args[3])
                else: # communicate_image_data
                    self.comm_points.add(args[4])
            elif pred == "sample_soil" or pred == "sample_rock":
                self.rover_to_store[args[0]] = args[1]

        # 3. BFS for All-Pairs Shortest Paths (Hops)
        # Cost per hop = 3 (visit + navigate + unvisit)
        for rover in self.rover_at_facts:
            self.dist[rover] = {w: {w2: 999 for w2 in all_waypoints} for w in all_waypoints}
            for w in all_waypoints:
                self.dist[rover][w][w] = 0
                q = collections.deque([(w, 0)])
                visited = {w}
                while q:
                    curr, d = q.popleft()
                    self.dist[rover][w][curr] = d * 3
                    for neighbor in rover_adj[rover][curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # 4. Process Goals
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            for f in model.facts:
                if f.name == name:
                    self.goal_facts.append(f.global_id)
                    break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_rover_at(self, state, rover):
        if rover not in self.rover_at_facts:
            return None
        for wp, gid in self.rover_at_facts[rover].items():
            if (state >> gid) & 1:
                return wp
        return None

    def _compute(self, state, task_network):
        # Term 1: Unachieved goals (admissible bound 1)
        h_comm = 0
        for gid in self.goal_facts:
            if not ((state >> gid) & 1):
                h_comm += 1

        # Term 2: Task work (admissible bound 2)
        h_work = 0
        for task in task_network:
            pred, args = self._parse_fact_name(task.name)
            if not pred: continue
            
            if pred == "do_navigate1":
                rover, to_wp = args[0], args[1]
                curr_wp = self._get_rover_at(state, rover)
                if curr_wp and to_wp in self.dist[rover][curr_wp]:
                    h_work += self.dist[rover][curr_wp][to_wp]
            
            elif pred == "get_soil_data" or pred == "get_rock_data":
                # Rough lower bound for the entire sequence: dist to sample + sample + dist to comm + comm
                target_wp = args[0]
                # Since we don't know the rover, we use a global optimistic estimate (min over rovers)
                min_task_cost = 999
                for rover in self.dist:
                    curr_wp = self._get_rover_at(state, rover)
                    if curr_wp and target_wp in self.dist[rover][curr_wp]:
                        d1 = self.dist[rover][curr_wp][target_wp]
                        d2 = min([self.dist[rover][target_wp][cp] for cp in self.comm_points if cp in self.dist[rover][target_wp]] or [0])
                        min_task_cost = min(min_task_cost, d1 + 1 + d2 + 1)
                h_work += (min_task_cost if min_task_cost < 999 else 2)

            elif pred == "send_soil_data" or pred == "send_rock_data" or pred == "send_image_data":
                rover = args[0]
                curr_wp = self._get_rover_at(state, rover)
                if curr_wp:
                    d_comm = min([self.dist[rover][curr_wp][cp] for cp in self.comm_points if cp in self.dist[rover][curr_wp]] or [0])
                    h_work += d_comm + 1

            elif pred == "do_calibrate":
                rover, cam = args[0], args[1]
                # Calibration usually happens at a visible_from waypoint. Simplified here.
                h_work += 2 

        return max(h_comm, h_work)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
