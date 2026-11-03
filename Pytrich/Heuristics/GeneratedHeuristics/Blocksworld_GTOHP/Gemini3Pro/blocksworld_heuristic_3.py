import sys
from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic3(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_0):
        self.model = blocksworld_gemini3pro_heuristic_0.model
        # Increase recursion limit just in case, though we use iterative BFS
        sys.setrecursionlimit(5000)
        
        # Parse facts to find global IDs for relevant predicates
        self.on_fact_ids = {}      # global_id -> (top_block, bottom_block)
        self.ontable_fact_ids = {} # global_id -> block
        self.holding_fact_ids = {} # global_id -> block
        self.block_names = set()
        
        # Parse all fact names once during initialization for efficiency
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 2:
                self.on_fact_ids[fact.global_id] = (args[0], args[1])
                self.block_names.add(args[0])
                self.block_names.add(args[1])
            elif predicate == "ontable" and len(args) == 1:
                self.ontable_fact_ids[fact.global_id] = args[0]
                self.block_names.add(args[0])
            elif predicate == "holding" and len(args) == 1:
                self.holding_fact_ids[fact.global_id] = args[0]
                self.block_names.add(args[0])
        
        # Parse goals to determine target positions
        # We assume goals are static for the problem instance
        self.goal_targets = {} # block -> target_block (or 'TABLE')
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        for name in goal_facts:
            predicate, args = self._parse_fact_name(name)
            if predicate == "on" and len(args) == 2:
                self.goal_targets[args[0]] = args[1]
            elif predicate == "ontable" and len(args) == 1:
                self.goal_targets[args[0]] = "TABLE"

    def _parse_fact_name(self, name: str):
        """Helper to parse grounded fact names like +on[b1,b2]"""
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

    def compute(self, state, task_network):
        """
        Domain-dependent heuristic for Blocksworld.
        
        Strategy:
        1. Reconstruct the current configuration (stacks) from the bitwise state.
        2. Determine which blocks are "stable". A block is stable if:
           - It is on the Table and the goal allows it (target is Table or unspecified).
           - It is on a stable block Y, and the goal requires it to be on Y (or unspecified).
        3. Count "unstable" blocks. Every unstable block must be moved at least once.
        4. Add penalties for blocks that must move but their target is not yet ready (also unstable),
           implying they must move to Table first (temporary storage).
           
        Estimated Cost Calculation:
        - Basic move (Unstack+Stack or Pickup+Putdown) = 2 actions.
        - Indirect move (Unstack -> Putdown -> Pickup -> Stack) = 4 actions.
        """
        
        # 1. Reconstruct state from bitwise integer
        current_on = {} # top -> bottom
        on_table = set()
        holding = None
        
        # Bitwise checks are O(1) per fact, iterating pre-filtered lists
        for fid, (top, bot) in self.on_fact_ids.items():
            if (state >> fid) & 1:
                current_on[top] = bot
        
        for fid, blk in self.ontable_fact_ids.items():
            if (state >> fid) & 1:
                on_table.add(blk)

        for fid, blk in self.holding_fact_ids.items():
            if (state >> fid) & 1:
                holding = blk
                break # Can only hold one
        
        # 2. Build Adjacency for BFS (bottom -> tops)
        supports = {}
        for top, bot in current_on.items():
            if bot not in supports: supports[bot] = []
            supports[bot].append(top)
            
        # 3. Determine Stability (Bottom-Up)
        is_stable = {}
        queue = list(on_table)
        
        # Initial pass for blocks on table
        for b in on_table:
            g = self.goal_targets.get(b)
            # Stable if goal matches current (Table) or goal is indifferent
            if g == "TABLE" or g is None:
                is_stable[b] = True
            else:
                is_stable[b] = False
        
        # Held block is inherently unstable (unless goal is to hold it, but standard BW goal is 'on')
        if holding:
            is_stable[holding] = False
        
        # Propagate stability upwards
        idx = 0
        while idx < len(queue):
            base = queue[idx]
            idx += 1
            
            if base in supports:
                for top in supports[base]:
                    base_stable = is_stable.get(base, False)
                    g = self.goal_targets.get(top)
                    
                    # Block matches goal if target is base or unspecified
                    # Block is stable if base is stable AND block matches goal
                    if base_stable and (g == base or g is None):
                        is_stable[top] = True
                    else:
                        is_stable[top] = False
                    
                    queue.append(top)
        
        # Also handle stacks on top of held block (though rare in std BW, logically unstable)
        if holding:
            q_hold = [holding]
            h_idx = 0
            while h_idx < len(q_hold):
                base = q_hold[h_idx]
                h_idx += 1
                if base in supports:
                    for top in supports[base]:
                        is_stable[top] = False
                        q_hold.append(top)

        # 4. Calculate Heuristic Value
        h_val = 0
        # Identify all unstable blocks
        U_set = set()
        for b in self.block_names:
            if not is_stable.get(b, False):
                U_set.add(b)
        
        for b in U_set:
            is_held = (b == holding)
            is_on_table = (b in on_table)
            
            # Check if target is "ready" (stable). 
            # If target is None/Table, it is always ready.
            g = self.goal_targets.get(b)
            target_ready = True
            if g and g != "TABLE":
                # Target is a block. It is not ready if it is currently unstable.
                if g in U_set:
                    target_ready = False
            
            if is_on_table:
                # Block on table needs PickUp + Stack = 2 actions.
                # Even if target not ready, it waits on table (0 cost) then moves (2 cost).
                h_val += 2
            elif is_held:
                # Block in hand needs Stack (1 action).
                # If target not ready: PutDown(1) + (Wait) + PickUp(1) + Stack(1) = 3.
                if target_ready:
                    h_val += 1
                else:
                    h_val += 3
            else:
                # Block on another block needs Unstack + Stack = 2 actions.
                # If target not ready: Unstack(1) + PutDown(1) + (Wait) + PickUp(1) + Stack(1) = 4.
                if target_ready:
                    h_val += 2
                else:
                    h_val += 4
                    
        return float(h_val)