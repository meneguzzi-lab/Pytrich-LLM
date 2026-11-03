from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
:
        def __init__(self, model):
            self.model = model

class RobotGemini3ProHeuristic4(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_1):
        super().__init__(robot_gemini3pro_heuristic_1)
        self.model = robot_gemini3pro_heuristic_1
        self.INF = 1000000
        self.rooms = set()
        self.goal_map = {}
        self.graph = defaultdict(list)
        self.rloc_facts = []
        self.holding_facts = []
        self.in_facts = defaultdict(list)
        self._parse_domain_structure()
        self.dist = self._compute_apsp()

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

    def _parse_domain_structure(self):
        init_state = self.model.initial_state
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == 'rloc':
                self.rloc_facts.append((args[0], fact.global_id))
                self.rooms.add(args[0])
            elif pred == 'holding':
                self.holding_facts.append((args[0], fact.global_id))
            elif pred == 'in':
                self.in_facts[args[0]].append((args[1], fact.global_id))
                self.rooms.add(args[1])
            elif pred == 'door':
                if (init_state >> fact.global_id) & 1:
                    u, v = args[0], args[1]
                    self.graph[u].append(v)
                    self.graph[v].append(u)
                    self.rooms.add(u)
                    self.rooms.add(v)
            elif pred == 'goal_in':
                if (init_state >> fact.global_id) & 1:
                    self.goal_map[args[0]] = args[1]
                    self.rooms.add(args[1])

    def _compute_apsp(self):
        dist = defaultdict(lambda: defaultdict(lambda: self.INF))
        all_rooms = list(self.rooms)
        for r in all_rooms:
            dist[r][r] = 0
        for u in self.graph:
            for v in self.graph[u]:
                dist[u][v] = 1
        for k in all_rooms:
            for i in all_rooms:
                if dist[i][k] == self.INF:
                    continue
                for j in all_rooms:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def h(self, node):
        state = node.state
        robot_loc = None
        for r, fid in self.rloc_facts:
            if (state >> fid) & 1:
                robot_loc = r
                break
        if robot_loc is None:
            return self.INF
        held_obj = None
        for obj, fid in self.holding_facts:
            if (state >> fid) & 1:
                held_obj = obj
                break
        cost = 0
        misplaced_targets = []
        for obj, goal in self.goal_map.items():
            if held_obj == obj:
                d = self.dist[robot_loc][goal]
                if d == self.INF:
                    return self.INF
                cost += d + 1
            else:
                obj_loc = None
                for r, fid in self.in_facts[obj]:
                    if (state >> fid) & 1:
                        obj_loc = r
                        break
                if obj_loc and obj_loc != goal:
                    d = self.dist[obj_loc][goal]
                    if d == self.INF:
                        return self.INF
                    cost += d + 2
                    misplaced_targets.append(obj_loc)
        if held_obj is None and misplaced_targets:
            min_dist = self.INF
            for t in misplaced_targets:
                d = self.dist[robot_loc][t]
                if d < min_dist:
                    min_dist = d
            if min_dist != self.INF:
                cost += min_dist
        return cost