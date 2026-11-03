import collections
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic9(Heuristic):
    """
    Bottleneck: The robot's single arm makes the task of moving packages strictly serial.
    Only one package can be handled at a time, requiring a sequence of pickup, travel, and putdown.

    Admissible Lower Bounds:
    1. Transport Cost: Every package not at its goal room requires at least 2 actions (pickup/putdown),
       plus the number of moves to travel from its current location to its target.
    2. Sequence Cost: If holding a package, the robot must first move to its goal. If not holding,
       it must first move to the location of an undelivered package.
    """

    def __init__(self, robot_gemini3flash_heuristic_9="RobotGemini3FlashHeuristic9"):
        super().__init__(robot_gemini3flash_heuristic_9)
        self.pkg_targets = {}
        self.fact_ids_in = collections.defaultdict(dict)
        self.fact_ids_holding = {}
        self.fact_ids_rloc = {}
        self.dist_matrix = collections.defaultdict(lambda: collections.defaultdict(lambda: 9999))
        self.rooms = []

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
        self.model = model
        adj = collections.defaultdict(list)
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                self.fact_ids_in[args[0]][args[1]] = f.global_id
                if args[1] not in self.rooms: self.rooms.append(args[1])
            elif pred == "holding":
                self.fact_ids_holding[args[0]] = f.global_id
            elif pred == "rloc":
                self.fact_ids_rloc[args[0]] = f.global_id
                if args[0] not in self.rooms: self.rooms.append(args[0])
            elif pred == "goal_in":
                if (initial_node.state >> f.global_id) & 1:
                    self.pkg_targets[args[0]] = args[1]
            elif pred == "door":
                adj[args[0]].append(args[1])
                adj[args[1]].append(args[0])

        for r in self.rooms:
            self.dist_matrix[r][r] = 0
            queue = collections.deque([r])
            visited = {r}
            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist_matrix[r][neighbor] = self.dist_matrix[r][curr] + 1
                        queue.append(neighbor)

        initial_h = self._compute(initial_node.state)
        return super().initialize(model, initial_h)

    def _compute(self, state: int) -> int:
        robot_room = None
        for r_name, f_id in self.fact_ids_rloc.items():
            if (state >> f_id) & 1:
                robot_room = r_name
                break
        if robot_room is None: return 0

        held_pkg = None
        for p_name, f_id in self.fact_ids_holding.items():
            if (state >> f_id) & 1:
                held_pkg = p_name
                break

        h = 0
        undelivered_start_rooms = []
        current_pos = robot_room

        if held_pkg:
            target = self.pkg_targets.get(held_pkg)
            if target:
                h += 1 + self.dist_matrix[robot_room][target]
                current_pos = target

        for p_name, target in self.pkg_targets.items():
            if p_name == held_pkg: continue
            target_f_id = self.fact_ids_in[p_name].get(target)
            if target_f_id is not None and (state >> target_f_id) & 1:
                continue
            
            h += 2
            p_loc = None
            for r_name, f_id in self.fact_ids_in[p_name].items():
                if (state >> f_id) & 1:
                    p_loc = r_name
                    break
            if p_loc:
                h += self.dist_matrix[p_loc][target]
                undelivered_start_rooms.append(p_loc)

        if undelivered_start_rooms:
            h += min(self.dist_matrix[current_pos][loc] for loc in undelivered_start_rooms)

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
