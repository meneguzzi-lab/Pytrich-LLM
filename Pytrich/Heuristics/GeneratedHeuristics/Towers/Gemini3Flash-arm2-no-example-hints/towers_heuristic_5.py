from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic5(Heuristic):
    """
    Bottleneck Analysis: The Tower of Hanoi recursive structure. Moving the largest misplaced ring k 
    requires clearing all rings 1..k-1 to a different peg, moving k, and then moving 1..k-1 back. 
    This work is fundamentally serial.
    
    Admissibility Sketches:
    - Term 1 (State-based Hanoi cost): 2^k - 1, where k is the 1-based size of the largest ring 
      not currently on its goal tower. This is the exact optimal cost to move a stack of k rings 
      and remains a valid lower bound for any state where the largest misplaced ring is at size k.
    - Term 2 (Tie-breaking): 0.001 * len(task_network). This provides a tiny cost for every 
      task remaining in the network, helping the search choose deeper (more decomposed) nodes 
      when state values are identical.
    """
    def __init__(self, towers_gemini3flash_heuristic_5="towers_gemini3flash_heuristic_5"):
        super().__init__(name=towers_gemini3flash_heuristic_5)
        self.ring_names = []
        self.tower_names = []
        self.ring_to_size = {}
        self.goal_pegs = {}
        self.on_facts_by_target = {}
        self.obj_to_id = {}

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
        all_objs = set()
        on_relations = []
        smaller_than_relations = []
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            for a in args: all_objs.add(a)
            if pred == "on":
                on_relations.append((args[0], args[1], f.global_id))
            elif pred == "smallerThan":
                smaller_than_relations.append((args[0], args[1]))
        
        first_args = set(r[0] for r in on_relations)
        self.ring_names = list(first_args)
        self.tower_names = list(all_objs - first_args)
        
        ring_smaller_counts = {r: 0 for r in self.ring_names}
        for r_sub, r_obj in smaller_than_relations:
            if r_sub in self.ring_names and r_obj in self.ring_names:
                ring_smaller_counts[r_sub] += 1
        
        sorted_rings = sorted(self.ring_names, key=lambda r: ring_smaller_counts[r], reverse=True)
        for i, r_name in enumerate(sorted_rings):
            self.ring_to_size[r_name] = i + 1
            
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        goal_adj = {}
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" or pred == "goal_on":
                goal_adj[args[0]] = args[1]
        
        for ring in self.ring_names:
            curr = ring
            visited = set()
            while curr in goal_adj and curr not in self.tower_names:
                if curr in visited: break
                visited.add(curr)
                curr = goal_adj[curr]
            self.goal_pegs[ring] = curr

        obj_to_id = {}
        for obj_name in all_objs: obj_to_id[obj_name] = len(obj_to_id)
        self.obj_to_id = obj_to_id
        
        for r_name, t_name, fid in on_relations:
            tid = obj_to_id[t_name]
            if tid not in self.on_facts_by_target: self.on_facts_by_target[tid] = []
            self.on_facts_by_target[tid].append((r_name, fid))
            
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        if not task_network and (state & self.model.goals == self.model.goals):
            return 0
        
        current_pegs = {}
        for t_name in self.tower_names:
            tid = self.obj_to_id[t_name]
            curr_obj_id = tid
            stack_depth = 0
            while stack_depth < len(self.ring_names):
                found_next = False
                for r_name, fid in self.on_facts_by_target.get(curr_obj_id, []):
                    if (state >> fid) & 1:
                        current_pegs[r_name] = t_name
                        curr_obj_id = self.obj_to_id[r_name]
                        found_next = True
                        stack_depth += 1
                        break
                if not found_next: break
        
        max_size = 0
        for ring_name, goal_peg in self.goal_pegs.items():
            curr_peg = current_pegs.get(ring_name)
            if curr_peg != goal_peg:
                size = self.ring_to_size.get(ring_name, 0)
                if size > max_size:
                    max_size = size
        
        h = (2 ** max_size) - 1 if max_size > 0 else 0
        h += 0.001 * len(task_network)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
