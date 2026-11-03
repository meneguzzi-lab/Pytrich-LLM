from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.
    
    Bottleneck Analysis:
    The main bottleneck is rover navigation. Every task (soil, rock, image) requires a rover 
    to move to specific waypoints, perform an action, and typically move again to communicate 
    the data to a lander. Since rovers are serialized in their actions and limited by 
    connectivity, the navigation cost (3 actions per hop: visit, navigate, unvisit) 
    dominates the plan length.

    Admissibility Sketches:
    1. h_tasks: Each task in the task network has a minimum decomposition cost based on 
       the number of required primitive actions (e.g., 3 * distance + 1 for navigation, 
       +1 for sampling, +1 for communicating). Summing these for an ordered network is admissible.
    2. h_goals: Each unsatisfied goal not addressed by the current task network requires 
       at least 2 actions (sample/take + communicate). This is a safe lower bound.
    """

    def __init__(self, rover_gemini3flash_heuristic_14="rover_gemini3flash_heuristic_14"):
        super().__init__(name=rover_gemini3flash_heuristic_14)
        self.rover_at_wp_id = {} # (rover, wp) -> global_id
        self.goal_ids = []
        self.rover_graphs = collections.defaultdict(lambda: collections.defaultdict(list))
        self.rover_dists = {} # rover -> (wp1, wp2) -> int
        self.rovers = set()
        self.waypoints = set()
        self.task_info = {} # global_id -> (name, args)
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.target_waypoints = {} # obj -> wp (for imaging)

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
        # 1. Map facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                self.rover_at_wp_id[(args[0], args[1])] = fact.global_id
                self.rovers.add(args[0])
                self.waypoints.add(args[1])

        # 2. Extract Goal IDs
        bitwise_goals = model.goals
        for i in range(bitwise_goals.bit_length()):
            if (bitwise_goals >> i) & 1:
                self.goal_ids.append(i)

        # 3. Extract Static capabilities and graph from Operators
        # Grounder strips static facts, so we infer from actions
        for op in model.operators:
            self.task_info[op.global_id] = self._parse_fact_name(op.name)
            pred, args = self.task_info[op.global_id]
            if pred == "navigate":
                # args: [rover, from, to]
                self.rover_graphs[args[0]][args[1]].append(args[2])
            elif pred == "sample_soil":
                self.equipped_soil.add(args[0])
            elif pred == "sample_rock":
                self.equipped_rock.add(args[0])
            elif pred == "take_image":
                self.equipped_image.add(args[0])
            elif pred == "calibrate":
                # calibrate(rover, camera, objective, waypoint)
                self.target_waypoints[args[2]] = args[3]

        # Precompute BFS distances per rover
        for r in self.rovers:
            self.rover_dists[r] = {}
            for start_node in self.waypoints:
                q = collections.deque([(start_node, 0)])
                visited = {start_node}
                while q:
                    curr, d = q.popleft()
                    self.rover_dists[r][(start_node, curr)] = d
                    for neighbor in self.rover_graphs[r].get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # 4. Map Abstract Tasks
        for abt in model.abstract_tasks:
            self.task_info[abt.global_id] = self._parse_fact_name(abt.name)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise, task_network):
        # Get current rover positions
        curr_rover_pos = {}
        for r in self.rovers:
            for wp in self.waypoints:
                fact_id = self.rover_at_wp_id.get((r, wp))
                if fact_id is not None and (state_bitwise >> fact_id) & 1:
                    curr_rover_pos[r] = wp
                    break

        h = 0
        addressed_goals = set()

        # Term 1: Cost of tasks in the network
        # In GTOHP, the decomposition of high level tasks requires moves
        # visit(wp), navigate(r, from, to), unvisit(wp) -> 3 actions per hop
        for task in task_network:
            pred, args = self.task_info.get(task.global_id, (None, []))
            if not pred: continue

            task_cost = 1
            # Add navigation overhead if we can estimate the rover
            if pred in ["get_soil_data", "get_rock_data"]:
                target_wp = args[0]
                potential_rovers = self.equipped_soil if pred == "get_soil_data" else self.equipped_rock
                min_nav = 1000
                for r in potential_rovers:
                    pos = curr_rover_pos.get(r)
                    dist = self.rover_dists[r].get((pos, target_wp), 10) if pos else 5
                    min_nav = min(min_nav, 3 * dist)
                task_cost = min_nav + 2 # sample + communicate
            elif pred == "get_image_data":
                target_wp = self.target_waypoints.get(args[0], "")
                min_nav = 1000
                for r in self.equipped_image:
                    pos = curr_rover_pos.get(r)
                    dist = self.rover_dists[r].get((pos, target_wp), 10) if pos and target_wp else 5
                    min_nav = min(min_nav, 3 * dist)
                task_cost = min_nav + 3 # calibrate + take + communicate
            elif pred == "do_navigate1":
                rover, target_wp = args[0], args[1]
                pos = curr_rover_pos.get(rover)
                dist = self.rover_dists[rover].get((pos, target_wp), 0) if pos else 0
                task_cost = 3 * dist if dist > 0 else 1
            
            h += task_cost

        # Term 2: Goals not addressed
        for goal_id in self.goal_ids:
            if not ((state_bitwise >> goal_id) & 1):
                # Check if this specific goal (e.g. communicated_soil_data[wp]) is addressed
                # This is simplified: count goals as 2 actions each if not in network
                # This provides guidance to pursue goals.
                h += 2

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
