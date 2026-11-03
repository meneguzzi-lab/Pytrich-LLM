import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic11(Heuristic):
    """
    Bottleneck Analysis: 
    The primary bottleneck in the Rover domain is the physical navigation of rovers between waypoints 
    to perform sampling and reach lander-visible locations for data transmission, 
    constrained by the single-item capacity of the rover's storage.

    Admissibility Sketches:
    1. Goal-State Distance: Each 'communicated' goal not yet true in the state requires at least 2 actions 
       (one to acquire the sample/image and one to communicate it).
    2. Navigation Lower Bound: A 'do_navigate' task or any sampling task requires navigation 
       equal to the shortest path in the traverse graph.
    3. Task Decomposition cost: Every high-level task like 'get_soil_data' represents at least 4 actions 
       (nav, empty, sample, send) in its decomposition tree.
    """

    def __init__(self, rover_gemini3flash_heuristic_11=None, name="RoverGTOHPGemini3FlashHeuristic11"):
        super().__init__(name=name)
        self.dist_maps = {}  # rover -> {w1: {w2: dist}}
        self.waypoint_ids = {}
        self.goal_bits = []
        self.rover_at_bits = {} # rover -> {waypoint: bit}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute_h(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if "[" in name else []
                return predicate, args
        return None, []

    def _preprocess(self):
        rovers = set()
        waypoints = set()
        adj = {} # rover -> {w1: [w2]}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "at":
                r, w = args[0], args[1]
                rovers.add(r)
                waypoints.add(w)
                if r not in self.rover_at_bits: self.rover_at_bits[r] = {}
                self.rover_at_bits[r][w] = fact.global_id
            elif pred == "can_traverse":
                r, w1, w2 = args[0], args[1], args[2]
                if r not in adj: adj[r] = {}
                if w1 not in adj[r]: adj[r][w1] = []
                adj[r][w1].append(w2)
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])

        # Shortest Paths per Rover
        for r in rovers:
            self.dist_maps[r] = {}
            rover_adj = adj.get(r, {})
            for start_w in waypoints:
                dists = {start_w: 0}
                queue = deque([start_w])
                while queue:
                    curr = queue.popleft()
                    for neighbor in rover_adj.get(curr, []):
                        if neighbor not in dists:
                            dists[neighbor] = dists[curr] + 1
                            queue.append(neighbor)
                self.dist_maps[r][start_w] = dists

        # Goal bit indexing
        if hasattr(self.model, 'goals'):
            for bit_pos in range(self.model.goals.bit_length()):
                if (self.model.goals >> bit_pos) & 1:
                    self.goal_bits.append(bit_pos)

    def _get_rover_loc(self, r, state):
        if r in self.rover_at_bits:
            for w, bit in self.rover_at_bits[r].items():
                if (state >> bit) & 1: return w
        return None

    def _compute_h(self, state, task_network):
        # 1. Remaining Goal Cost
        h_goals = 0
        for bit in self.goal_bits:
            if not ((state >> bit) & 1):
                h_goals += 2 # Minimum 2 actions to acquire and communicate data

        # 2. Task Network Work
        h_tasks = 0
        for task in task_network:
            # Try parsing task name for logic
            name = task.name
            # Basic decomposition weights based on domain complexity
            if "get_soil_data" in name or "get_rock_data" in name or "get_image_data" in name:
                h_tasks += 4
            elif "do_navigate" in name:
                h_tasks += 1
            elif "communicate" in name:
                h_tasks += 1
            elif "sample" in name:
                h_tasks += 1
            elif "take_image" in name:
                h_tasks += 1
            elif "calibrate" in name:
                h_tasks += 1
            elif "drop" in name:
                h_tasks += 1
            
        # 3. Navigation distance from first relevant task (Tie-breaking guidance)
        h_nav = 0
        if task_network:
            first_task = task_network[0]
            # Extract arguments roughly
            args = first_task.name.split('[')[-1].split(']')[0].split(',')
            if len(args) >= 2:
                r = args[0]
                to_w = args[1]
                loc = self._get_rover_loc(r, state)
                if loc and r in self.dist_maps and loc in self.dist_maps[r]:
                    h_nav = self.dist_maps[r][loc].get(to_w, 5) # Default 5 if unreachable/complex

        return max(h_goals, h_tasks) + (0.01 * h_nav)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute_h(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
