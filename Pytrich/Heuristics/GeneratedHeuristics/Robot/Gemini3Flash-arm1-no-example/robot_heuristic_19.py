from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic19(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Robot domain is the single-arm capacity (can only hold one package)
    and the sequential nature of movement. Each package must be picked up, transported to its
    destination, and put down. Doors act as additional sequential obstacles. 

    Admissibility Sketch:
    1. Every package not at its goal requires at least a pickup and a putdown (2 actions).
    2. Every package not at its goal requires movement to its current location (if not holding)
       and movement to its destination.
    3. Since there is only one robot, these actions cannot be done in parallel. The sum of
       distances and interaction costs (pickup/putdown) for all pending packages provides
       a strong lower bound on the total plan length.
    """
    def __init__(self, robot_gemini3flash_heuristic_19="robot_gemini3flash_heuristic_19"):
        super().__init__(robot_gemini3flash_heuristic_19)
        self.package_goals = {}  # pkg_name -> room_name
        self.dist_matrix = {}    # (room1, room2) -> int
        self.in_fact_map = {}    # (pkg, room) -> global_id
        self.holding_fact_map = {} # pkg -> global_id
        self.rloc_fact_map = {}  # room -> global_id
        self.closed_fact_map = {} # door -> global_id
        self.pkg_list = []
        self.rooms_list = []
        self.pkg_in_fact_ids = collections.defaultdict(list)

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
        rooms = set()
        packages = set()
        door_info = []

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                self.rloc_fact_map[args[0]] = fact.global_id
                rooms.add(args[0])
            elif pred == "in":
                self.in_fact_map[(args[0], args[1])] = fact.global_id
                self.pkg_in_fact_ids[args[0]].append((args[1], fact.global_id))
                packages.add(args[0])
            elif pred == "holding":
                self.holding_fact_map[args[0]] = fact.global_id
                packages.add(args[0])
            elif pred == "closed":
                self.closed_fact_map[args[0]] = fact.global_id
            elif pred == "door":
                door_info.append((args[0], args[1]))
            elif pred == "goal_in":
                self.package_goals[args[0]] = args[1]

        adj = collections.defaultdict(list)
        for r1, r2 in door_info:
            adj[r1].append(r2)
            adj[r2].append(r1)

        self.rooms_list = list(rooms)
        for start in self.rooms_list:
            self.dist_matrix[(start, start)] = 0
            queue = collections.deque([(start, 0)])
            visited = {start}
            while queue:
                curr, d = queue.popleft()
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        self.dist_matrix[(start, nxt)] = d + 1
                        queue.append((nxt, d + 1))

        self.pkg_list = list(packages)
        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        robot_loc = None
        for r_name, f_id in self.rloc_fact_map.items():
            if (state >> f_id) & 1:
                robot_loc = r_name
                break
        
        if not robot_loc:
            return 100
            
        h_sum = 0
        for pkg in self.pkg_list:
            target = self.package_goals.get(pkg)
            if not target:
                continue
            
            in_target_fid = self.in_fact_map.get((pkg, target))
            if in_target_fid is not None and (state >> in_target_fid) & 1:
                continue
            
            holding_fid = self.holding_fact_map.get(pkg)
            if holding_fid is not None and (state >> holding_fid) & 1:
                h_sum += self.dist_matrix.get((robot_loc, target), 10) + 1
            else:
                pkg_loc = None
                for r_name, f_id in self.pkg_in_fact_ids[pkg]:
                    if (state >> f_id) & 1:
                        pkg_loc = r_name
                        break
                
                if pkg_loc:
                    h_sum += self.dist_matrix.get((robot_loc, pkg_loc), 10) + 1 + self.dist_matrix.get((pkg_loc, target), 10) + 1
                else:
                    h_sum += 1
        
        closed_penalty = 0
        for f_id in self.closed_fact_map.values():
            if (state >> f_id) & 1:
                closed_penalty += 1
                
        h_tn = len(task_network)
        return h_sum + closed_penalty + h_tn

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
