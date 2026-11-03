import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The primary bottleneck in the Towers domain is the recursive disk-moving constraint: 
# a disk cannot move unless it is the 'towerTop', and it can only be placed on an 
# object (tower or ring) that is 'smallerThan' it. This serializes all moves, 
# as moving a large disk requires clearing all smaller disks above it onto a single temporary tower.

# Admissibility Sketches:
# 1. Goal Displacement (State-based): Any ring not currently on its goal position must be moved 
#    at least once. This is a standard admissible lower bound for movement/placement tasks.
# 2. Dependency Chain (State-based): If a ring 'r' is on its goal object 'o', but 'o' itself 
#    is a ring that is misplaced, 'r' must eventually be moved to allow 'o' to reach its 
#    goal. This is admissible as it counts unavoidable future moves.
# 3. Task Network Weight (Task-based): Every move_abstract task in the network corresponds to 
#    at least one future move action. Summing these is admissible as they are disjoint requirements.

class TowersGemini3FlashHeuristic11(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_11="towers_gemini3flash_heuristic_11"):
        super().__init__(name=towers_gemini3flash_heuristic_11)
        self.ring_ids = []
        self.tower_ids = []
        self.on_map = {} # (ring, obj) -> global_id
        self.goal_on_map = {} # ring -> target_obj
        self.smaller_than = {} # r -> set of rings smaller than r
        self.sorted_rings = [] # Rings sorted by rank (largest to smallest)
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse objects
        temp_rings = []
        temp_towers = []
        
        # 2. Parse all facts to identify rings, towers, and relations
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                self.on_map[(args[0], args[1])] = fact.global_id
                if args[0] not in temp_rings: temp_rings.append(args[0])
            elif pred == "smallerThan":
                r, o = args[0], args[1]
                if r not in self.smaller_than: self.smaller_than[r] = set()
                self.smaller_than[r].add(o) # Note: Domain says smallerThan ring object

        self.ring_ids = temp_rings
        self.num_rings = len(self.ring_ids)

        # 3. Determine Ring Hierarchy (Rank)
        # Rank = how many rings are smaller than this ring.
        ranks = {}
        for r in self.ring_ids:
            count = 0
            for other in self.ring_ids:
                # Is 'other' smaller than 'r'? 
                # In this domain, smallerThan(r, o) means r is smaller than o.
                # So we check if smallerThan(other, r) exists.
                if (other, r) in self.on_map or any(f.name == f"+smallerThan[{other},{r}]" for f in model.facts):
                    # Actually we should look at the smallerThan facts explicitly
                    pass
            
            # Accurate rank based on the smallerThan predicate count
            rank = 0
            for f in model.facts:
                p, a = self._parse_fact_name(f.name)
                if p == "smallerThan" and a[1] == r and a[0] in self.ring_ids:
                    rank += 1
            ranks[r] = rank

        # Sort rings from largest rank to smallest rank
        self.sorted_rings = sorted(self.ring_ids, key=lambda x: ranks.get(x, 0), reverse=True)

        # 4. Identify Goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gn in goal_names:
            p, a = self._parse_fact_name(gn)
            if p == "on":
                self.goal_on_map[a[0]] = a[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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

    def _get_current_on(self, ring, state):
        # Brute force check for which object the ring is on
        # In initialize we can optimize this by mapping potential objects
        for obj in self.ring_ids + ["t1", "t2", "t3"]: # Check rings and tower constants
            fid = self.on_map.get((ring, obj))
            if fid is not None and (state >> fid) & 1:
                return obj
        return None

    def _compute(self, state, task_network):
        # Term 1: State-based misplacement cost
        misplaced_count = 0
        is_bad = {}

        # Process rings from largest to smallest
        # If a large ring is misplaced, it's 'bad'. 
        # If a small ring is on its goal, but that goal is a 'bad' ring, the small ring is also 'bad'.
        for r in self.sorted_rings:
            goal_obj = self.goal_on_map.get(r)
            curr_obj = self._get_current_on(r, state)
            
            if curr_obj != goal_obj or is_bad.get(goal_obj, False):
                is_bad[r] = True
                misplaced_count += 1
            else:
                is_bad[r] = False

        # Term 2: Task-based cost
        # Each move_abstract or shiftTower represents progress
        task_cost = 0
        for t in task_network:
            if hasattr(t, 'name'):
                name = t.name.lower()
                if "move" in name: task_cost += 1
                elif "shift" in name: task_cost += self.num_rings
                elif "rotate" in name: task_cost += 1
                else: task_cost += 0.1
            else:
                task_cost += 1

        return max(misplaced_count, int(task_cost))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
