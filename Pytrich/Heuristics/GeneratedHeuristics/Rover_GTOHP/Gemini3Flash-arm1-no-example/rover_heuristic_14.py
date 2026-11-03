from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    Bottleneck Analysis:
    The bottleneck in the Rover domain is the sample-storage-communication lifecycle and the navigation 
    required between sampling sites and waypoints visible to the lander for data transmission.

    Admissibility Sketches:
    - Navigation Lower Bound: The BFS distance between waypoints for a specific rover provides an absolute 
      minimum number of 'navigate' actions required to reach a target, making it a valid admissible lower bound.
    - Task Progress Lower Bound: Each high-level data task (get_soil, get_rock, get_image) must decompose 
      into at least a sampling/imaging action and a communication action. Assigning costs (e.g., 2 for soil/rock,
      3 for image) represents the unavoidable primitive actions required, ensuring admissibility.
    """

    def __init__(self, rover_gemini3flash_heuristic_14="RoverGTOHPGemini3FlashHeuristic14"):
        super().__init__(rover_gemini3flash_heuristic_14)
        self.dist_matrix = {} # rover_name -> start_wp -> {end_wp: dist}
        self.rover_at_fids = {} # rover_name -> list of (global_id, waypoint_name)
        self.full_fact_ids = {} # store_name -> global_id
        self.rover_names = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to identify rovers, waypoints, and connectivity
        self.rover_names = set()
        waypoint_names = set()
        adj = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> wp1 -> set(wp2)

        for fid in range(len(model.facts)):
            fact = model.facts[fid]
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "at" and len(args) == 2:
                rover, wp = args[0], args[1]
                self.rover_names.add(rover)
                waypoint_names.add(wp)
                if rover not in self.rover_at_fids: self.rover_at_fids[rover] = []
                self.rover_at_fids[rover].append((fid, wp))
            
            if pred == "full" and len(args) == 1:
                self.full_fact_ids[args[0]] = fid

            # Static connectivity check (can_traverse usually set in init state)
            if pred == "can_traverse" and len(args) == 3:
                rover, f, t = args[0], args[1], args[2]
                adj[rover][f].add(t)

        # 2. Precompute BFS Distance Matrix for each rover
        for rover in self.rover_names:
            self.dist_matrix[rover] = {}
            for start_wp in waypoint_names:
                self.dist_matrix[rover][start_wp] = {start_wp: 0}
                queue = collections.deque([(start_wp, 0)])
                visited = {start_wp}
                while queue:
                    curr, d = queue.popleft()
                    for neighbor in adj[rover].get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dist_matrix[rover][start_wp][neighbor] = d + 1
                            queue.append((neighbor, d + 1))

        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network:
            return 0

        total_h = 0
        # Get current rover positions efficiently
        rover_pos = {}
        for r_name, fids in self.rover_at_fids.items():
            for fid, wp_name in fids:
                if (state >> fid) & 1:
                    rover_pos[r_name] = wp_name
                    break

        for task in task_network:
            task_base, args = self._parse_task_and_args(task.name)
            
            # 1. Skip non-cost tasks
            if task_base in ["nop", "visit", "unvisit"]:
                continue

            # 2. Additive costs for Abstract Tasks based on decomposition minimums
            if task_base == "get_soil_data" or task_base == "get_rock_data":
                total_h += 2 # sample + communicate
            elif task_base == "get_image_data":
                total_h += 3 # calibrate + take + communicate
            elif task_base == "send_soil_data" or task_base == "send_rock_data" or task_base == "send_image_data":
                total_h += 1
            elif task_base == "do_calibrate":
                total_h += 1
            elif task_base == "empty_store":
                # args: [store, rover]
                store_name = args[0] if len(args) > 0 else ""
                fid = self.full_fact_ids.get(store_name)
                if fid is not None and (state >> fid) & 1:
                    total_h += 1 # Needs a drop action
            
            # 3. Dynamic Navigation Costs (provides smooth guidance)
            elif task_base == "do_navigate1":
                # args: [rover, target_wp]
                if len(args) == 2:
                    r, target = args[0], args[1]
                    curr = rover_pos.get(r)
                    if curr:
                        total_h += self.dist_matrix.get(r, {}).get(curr, {}).get(target, 1)
            elif task_base == "do_navigate2":
                # args: [rover, from, to]
                if len(args) == 3:
                    r, f, t = args[0], args[1], args[2]
                    total_h += self.dist_matrix.get(r, {}).get(f, {}).get(t, 1)

            # 4. Handle primitive actions (Operators)
            elif hasattr(task, 'cost'):
                total_h += task.cost
            else:
                total_h += 1 # Default base cost for unknown tasks

        return total_h

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

    def _parse_task_and_args(self, task_name):
        bracket_idx = task_name.find("[")
        if bracket_idx > 0 and task_name.endswith("]"):
            base = task_name[:bracket_idx]
            args_str = task_name[bracket_idx+1:-1]
            return base, args_str.split(",") if args_str else []
        return task_name, []
