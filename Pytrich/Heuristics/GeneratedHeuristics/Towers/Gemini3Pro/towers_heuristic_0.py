from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType
import re

class TowersGemini3ProHeuristic0(TaskDecompositionHeuristic):
    """
    Initial state-aware heuristic for the Towers domain (v0).
    Penalizes optimistic assumptions about:
    1. Valid Moves (Ensures smaller on larger).
    2. Source Availability (Ensures source tower is not empty).
    """
    def __init__(self, name="TowersGemini3ProHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}
        self.ring_size = {} # r1 -> 1, r2 -> 2

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
            predicate = parts[0]
            args = tuple(parts[1:])
            key = (predicate, args)
            self.fact_map[key] = fact.global_id
            
            # Infer ring sizes from fact arguments? 
            # Or just process all seen "rX" later.
            for arg in args:
                if arg.startswith('r') and arg[1:].isdigit():
                    self.ring_size[arg] = int(arg[1:])

        def get_fact_id(pred, args_tuple):
            key_std = (pred, args_tuple)
            key_plus = ("+" + pred, args_tuple)
            if key_plus in self.fact_map:
                return self.fact_map[key_plus]
            if key_std in self.fact_map:
                return self.fact_map[key_std]
            return None
            
        def get_top_rings(state, tower):
            # Find loop over all rings? 
            # We need to know which fact "towerTop[r, tower]" is true.
            # This requires iterating facts or known potential fact IDs for this tower.
            pass

        # Since we need to check "towerTop[r, t]" dynamically, let's precompute list of possible (ring, fact_id) for each tower.
        self.tower_tops = {} # tower -> list of (ring, fid)
        
        # Scan facts again to populate tower_tops
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) > 2 and "towertop" in parts[0]:
                # towerTop r1 t1
                ring = parts[1]
                tower = parts[2]
                if tower not in self.tower_tops:
                    self.tower_tops[tower] = []
                self.tower_tops[tower].append( (ring, fact.global_id) )
                
                # Also capture size if missed
                if ring.startswith('r') and ring[1:].isdigit():
                    self.ring_size[ring] = int(ring[1:])

        # 2. Identify Optimistic Tasks
        for node in self.and_or_graph.nodes:
            if node and node.content_type == ContentType.ABSTRACT_TASK:
                task_name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                parts = task_name_clean.split()
                if not parts:
                    continue
                
                task_type = parts[0]
                
                base_penalty = 0
                cheapest = node.value
                alternative = float('inf')
                for child in node.successors:
                    if child.value > cheapest:
                        alternative = min(alternative, child.value)
                
                if alternative < float('inf'):
                    base_penalty = alternative - cheapest

                # --- MOVE(from, to) ---
                # Check for "move" or "move_abstract"
                # PSAS: "move_abstract[t1,t3]"
                if "move" in task_type and len(parts) > 2:
                    # Parts: [move_abstract, t1, t3]
                    # Sometimes args are weird. Check logic.
                    # Primitives have many args: move[r1,t1,t2,t1,t3]. (ring, from_tower, to_tower, from_support, to_support?)
                    # Abstract task usually move_abstract[from, to].
                    
                    # Assume last 2 args are [from, to] if size 2?
                    # Or check for tower names.
                    
                    src = parts[1]
                    dst = parts[2]
                    
                    # Only proceed if they look like towers "tX"
                    if src.startswith('t') and dst.startswith('t'):
                        
                        self.optimistic_tasks[node.ID] = {
                            "type": "move_check",
                            "penalty": base_penalty if base_penalty > 0 else 1.0,
                            "src": src,
                            "dst": dst,
                            "extra_penalty": 10.0 # High penalty for illegal move
                        }

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                type_ = info["type"]
                
                if type_ == "move_check":
                    src = info["src"]
                    dst = info["dst"]
                    
                    # 1. Find Top Ring of Src
                    src_ring = None
                    src_size = float('inf')
                    
                    # Check which towerTop fact is true for src
                    possible_tops_src = self.tower_tops.get(src, [])
                    for r, fid in possible_tops_src:
                        if (state & (1 << fid)):
                            src_ring = r
                            src_size = self.ring_size.get(r, 999)
                            break
                    
                    if src_ring is None:
                        # Source is empty! Cannot move from empty.
                        h_value += info["extra_penalty"]
                        continue
                        
                    # 2. Find Top Ring of Dst
                    dst_ring = None
                    dst_size = float('inf') # Empty allows any size (infinite cap?) No, empty allows max size.
                    # Standard Hanoi rules: Any ring can go on empty peg.
                    # Constraint: Smaller on Larger.
                    # So if Dst is empty, effective size is INFINITY (accepts anything).
                    # If Dst has ring r, effective size is size(r).
                    # We must move SMALLER size onto LARGER size.
                    # src_size < dst_size.
                    
                    possible_tops_dst = self.tower_tops.get(dst, [])
                    is_dst_empty = True
                    for r, fid in possible_tops_dst:
                        if (state & (1 << fid)):
                            dst_ring = r
                            dst_size = self.ring_size.get(r, 0) # 0 if unknown
                            is_dst_empty = False
                            break
                    
                    if is_dst_empty:
                        # Allowed.
                        continue
                    
                    # Check Constraint: Src < Dst
                    # Warning: r1 is usually smallest. rN is largest?
                    # "Smaller on Larger".
                    # If r1 is smallest (size 1), and r3 is largest (size 3).
                    # on[r1, r3] is VALID.
                    # r1 < r3.
                    # So we check: src_size < dst_size.
                    
                    if not (src_size < dst_size):
                        # Illegal Move (Larger on Smaller)
                        h_value += info["extra_penalty"]

        self.update_info(h_value)
        return h_value
