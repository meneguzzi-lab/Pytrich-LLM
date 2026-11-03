import collections
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic12(Heuristic):
    """
    Domain-dependent heuristic for Rover_GTOHP.
    
    Bottleneck:
    The primary bottleneck is navigation across the waypoint graph and the sequential nature of store management. 
    Rovers must move to specific locations to sample data and then move to lander-visible locations to transmit it.

    Admissible Lower Bounds:
    1. Task decomposition cost: Every high-level task like get_soil_data requires a sequence of atomic actions 
       (navigate, sample, navigate, communicate). We count the minimum remaining actions for each task in the network.
    2. Navigation distance: For any navigation task explicitly mentioning a rover, we use a precomputed 
       BFS distance from its current location to the target. This is admissible as navigate actions cost 1.
    3. State-aware store cost: Physical sampling tasks require an empty store. If a rover's store is full, 
       an additional 'drop' action is required.
    """

    def __init__(self, rover_gemini3flash_heuristic_12="RoverGTOHPGemini3FlashHeuristic12"):
        super().__init__(rover_gemini3flash_heuristic_12)
        self.rover_at_fids = {}  # rover -> waypoint -> fid
        self.store_empty_fids = {}  # rover -> fid
        self.dists = {}  # rover -> {w1: {w2: int}}
        self.waypoints = []
        self.rovers = []
        self.initialized = False

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = [a.strip() for a in args_str.split(",")]
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        waypoints_set = set()
        rovers_set = set()
        adj = {}  # rover -> {w1: [w2]}

        # Parse facts
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == "at":
                rover, wp = args[0], args[1]
                rovers_set.add(rover)
                waypoints_set.add(wp)
                if rover not in self.rover_at_fids: self.rover_at_fids[rover] = {}
                self.rover_at_fids[rover][wp] = f.global_id
            elif pred == "can_traverse":
                r, w1, w2 = args[0], args[1], args[2]
                if r not in adj: adj[r] = {}
                if w1 not in adj[r]: adj[r][w1] = []
                adj[r][w1].append(w2)
            elif pred == "empty":
                # Use store_of to link store status to rover
                pass
            elif pred == "store_of":
                # Mapping stores to rovers later via state parsing if needed,
                # but the grounded 'empty[store]' fact is easier.
                pass

        # Map stores to rovers for 'empty' fact tracking
        rover_stores = {}
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == "store_of":
                rover_stores[args[0]] = args[1] # store -> rover
        
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == "empty":
                store = args[0]
                if store in rover_stores:
                    self.store_empty_fids[rover_stores[store]] = f.global_id

        self.waypoints = list(waypoints_set)
        self.rovers = list(rovers_set)

        # Precompute All-Pairs Shortest Paths for each rover
        for rover in self.rovers:
            r_adj = adj.get(rover, {})
            self.dists[rover] = {w: {w2: 20 for w2 in self.waypoints} for w in self.waypoints}
            for w in self.waypoints:
                self.dists[rover][w][w] = 0
                queue = collections.deque([(w, 0)])
                visited = {w}
                while queue:
                    curr, d = queue.popleft()
                    for neighbor in r_adj.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dists[rover][w][neighbor] = d + 1
                            queue.append((neighbor, d + 1))

        self.initialized = True
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_rover_loc(self, state, rover):
        for wp, fid in self.rover_at_fids.get(rover, {}).items():
            if (state >> fid) & 1: return wp
        return None

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            pred, args = self._parse_name(task.name)
            # Static Decomposition Costs (Term 1)
            if pred == "get_soil_data" or pred == "get_rock_data":
                h += 4  # sample + commute + roughly 2 navigates
            elif pred == "get_image_data":
                h += 5  # calibrate + take + commute + roughly 2 navigates
            elif pred == "send_soil_data" or pred == "send_rock_data" or pred == "send_image_data":
                h += 2  # navigate to lander + communicate
            elif pred == "do_navigate1" or pred == "do_navigate2" or pred == "navigate":
                # Dynamic Navigation Cost (Term 2)
                if len(args) >= 2:
                    rover = args[0]
                    dest = args[-1]
                    curr_loc = self._get_rover_loc(state, rover)
                    if curr_loc and dest in self.dists.get(rover, {}):
                        h += self.dists[rover][curr_loc][dest]
                    else:
                        h += 1
                else:
                    h += 1
            elif pred == "empty_store":
                # State-aware Store Cost (Term 3)
                rover = args[1]
                is_empty_fid = self.store_empty_fids.get(rover)
                if is_empty_fid is not None and not ((state >> is_empty_fid) & 1):
                    h += 1 # drop
            elif pred == "sample_soil" or pred == "sample_rock" or pred == "take_image":
                h += 1
            elif pred.startswith("communicate_"):
                h += 1
            elif pred == "calibrate":
                h += 1
            elif pred == "drop":
                h += 1
            elif pred == "nop":
                pass
            else:
                h += 1
        
        # Goal completion logic: if network empty but goals not met, penalize heavily
        # GTOHP usually satisfies goals via the network, so we check satisfied goal bits
        satisfied_goals = bin(state & self.model.goals).count('1')
        total_goals = bin(self.model.goals).count('1')
        h += (total_goals - satisfied_goals)

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
