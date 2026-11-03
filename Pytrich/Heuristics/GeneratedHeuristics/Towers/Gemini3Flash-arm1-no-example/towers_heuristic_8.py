from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

class TowersGemini3FlashHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The central bottleneck in the Towers domain is the exponential growth of required moves (2^n - 1) 
    to transfer a stack of n rings. HTN planning in this domain recursively decomposes 
    the shiftTower and rotateTower tasks, mirroring this exponential structure.

    Admissibility Sketches:
    1. Task Network Move Cost (h_tn): Each grounded task in the network represents a part of the 
       Hanoi sequence. A task like shiftTower(t1) or selectDirection(r) for a stack of height h 
       requires at least 2^h - 1 primitive moves. Summing these costs across the active task 
       network provides a realistic lower bound of the remaining effort.
    2. State-Based Misplacement (h_state): A ring is misplaced if it is not on its goal parent, 
       or if its parent is misplaced. Each unstable ring requires at least one move. This is 
       a much weaker bound but provides sub-unit guidance.
    """

    def __init__(self, towers_gemini3flash_heuristic_8="TowersGemini3FlashHeuristic8"):
        super().__init__(towers_gemini3flash_heuristic_8)
        self.rank = {}  # Object ID -> Rank (0=smallest ring, N-1=largest)
        self.tower_top_fact_ids = {}  # tower_name -> List[(obj_name, global_id)]
        self.ring_ids = set()
        self.total_rings = 0
        self.goal_on_map = {}  # ring_name -> target_obj_name
        self.on_fact_ids = {}  # (ring_name, obj_name) -> global_id

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

    def _parse_component_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            base_name = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return base_name, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map all facts and identify rings
        smaller_than_relations = []
        all_objs = set()
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.ring_ids.add(args[0])
                self.on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "smallerThan":
                smaller_than_relations.append((args[0], args[1]))
            elif pred == "towerTop":
                tower = args[1]
                if tower not in self.tower_top_fact_ids:
                    self.tower_top_fact_ids[tower] = []
                self.tower_top_fact_ids[tower].append((args[0], f.global_id))

        # 2. Determine Ring Ranks (Smallest to Largest)
        # rank = number of rings larger than this ring
        for r in self.ring_ids:
            # count x where x is a ring and smallerThan(r, x)
            r_rank = 0
            for x in self.ring_ids:
                if (r, x) in smaller_than_relations:
                    r_rank += 1
            self.rank[r] = r_rank
        
        self.total_rings = len(self.ring_ids)

        # 3. Goal Analysis
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on_map[args[0]] = args[1]

        towers_gemini3flash_heuristic_8_init_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, towers_gemini3flash_heuristic_8_init_h)

    def _compute(self, state, task_network):
        h_tn = 0
        
        # Process Task Network Costs
        for task in task_network:
            task_name, args = self._parse_component_name(task.name)
            
            if task_name in ["move", "move_abstract", "exchange"]:
                h_tn += 1
            elif task_name in ["shiftTower", "rotateTower"]:
                # Source tower is args[0]
                src_tower = args[0]
                if src_tower in self.tower_top_fact_ids:
                    for obj, f_id in self.tower_top_fact_ids[src_tower]:
                        if (state >> f_id) & 1:
                            if obj in self.rank:
                                # Height of stack = rank + 1
                                h_tn += (2 ** (self.rank[obj] + 1)) - 1
                            break
            elif task_name == "selectDirection":
                # args[0] is the top ring of the sub-problem
                ring = args[0]
                if ring in self.rank:
                    h_tn += (2 ** (self.rank[ring] + 1)) - 1

        # Tie-breaker: Stability check (misplaced rings)
        h_state = 0
        unstable = set()
        # Sort rings largest to smallest (rank ascending 0 to N-1)
        sorted_rings = sorted(list(self.ring_ids), key=lambda x: self.rank[x])
        for r in sorted_rings:
            target_o = self.goal_on_map.get(r)
            on_target_fid = self.on_fact_ids.get((r, target_o))
            if target_o is None or on_target_fid is None or not ((state >> on_target_fid) & 1) or target_o in unstable:
                unstable.add(r)
                h_state += 1

        # The TN cost is primary; h_state provides minor guidance for same-task nodes
        return h_tn + (0.01 * h_state)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        towers_gemini3flash_heuristic_8_val = self._compute(node.state, node.task_network)
        super().update_info(towers_gemini3flash_heuristic_8_val)
        return towers_gemini3flash_heuristic_8_val
