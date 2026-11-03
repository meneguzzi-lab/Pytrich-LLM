from Pytrich.core.heuristic import Heuristic
import sys

class BlocksworldGemini3ProHeuristic18(Heuristic):
    def __init__(self, model, blocksworld_gemini3pro_heuristic_15=None):
        super().__init__(model)
        self.model = model
        
        # 1. Map Block Names to Integers for efficient indexing
        self.block_to_id = {}
        self.id_to_block = []
        
        # Infer objects from facts
        objects = set()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred in ["on", "ontable", "clear", "holding", "handempty"]:
                for arg in args:
                    objects.add(arg)
        
        sorted_objs = sorted(list(objects))
        for i, obj in enumerate(sorted_objs):
            self.block_to_id[obj] = i
            self.id_to_block.append(obj)
            
        self.num_blocks = len(self.block_to_id)
        
        # 2. Precompute Goal Configuration
        # goal_on[u] = v implies goal is on(u, v)
        self.goal_on = {}
        self.goal_ontable = set()
        self.goal_clear = set()
        
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on':
                u, v = args[0], args[1]
                if u in self.block_to_id and v in self.block_to_id:
                    self.goal_on[self.block_to_id[u]] = self.block_to_id[v]
            elif pred == 'ontable':
                u = args[0]
                if u in self.block_to_id:
                    self.goal_ontable.add(self.block_to_id[u])
            elif pred == 'clear':
                u = args[0]
                if u in self.block_to_id:
                    self.goal_clear.add(self.block_to_id[u])

        # 3. Precompute Fact Indices for Fast State Reconstruction
        # We map global_id to (type, u_id, v_id)
        # Type: 0=ON, 1=ONTABLE, 2=HOLDING
        self.relevant_fact_indices = []
        self.fact_data = {}
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on':
                u, v = args[0], args[1]
                if u in self.block_to_id and v in self.block_to_id:
                    gid = fact.global_id
                    self.relevant_fact_indices.append(gid)
                    self.fact_data[gid] = (0, self.block_to_id[u], self.block_to_id[v])
            elif pred == 'ontable':
                u = args[0]
                if u in self.block_to_id:
                    gid = fact.global_id
                    self.relevant_fact_indices.append(gid)
                    self.fact_data[gid] = (1, self.block_to_id[u], -1)
            elif pred == 'holding':
                u = args[0]
                if u in self.block_to_id:
                    gid = fact.global_id
                    self.relevant_fact_indices.append(gid)
                    self.fact_data[gid] = (2, self.block_to_id[u], -2)

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
        return name, []

    def __call__(self, model, state, task_network=[]):
        # 1. Reconstruct Current State
        # current_on[u] = v  (where v is block_id, -1=Table, -2=Held, -3=Unknown)
        current_on = [-3] * self.num_blocks
        
        # Using bitwise check on relevant facts only
        for gid in self.relevant_fact_indices:
            if (state >> gid) & 1:
                ftype, u, v = self.fact_data[gid]
                if ftype == 0:   # on(u, v)
                    current_on[u] = v
                elif ftype == 1: # ontable(u)
                    current_on[u] = -1
                elif ftype == 2: # holding(u)
                    current_on[u] = -2
        
        # 2. Build Support Structure (children map)
        # children_map[v] = list of u such that on(u, v)
        # Only needed for actual blocks (0..N-1), not Table/Held
        children_map = [[] for _ in range(self.num_blocks)]
        for u in range(self.num_blocks):
            target = current_on[u]
            if target >= 0:
                children_map[target].append(u)
        
        # 3. Identify Initial Unstable Blocks
        # A block is unstable if it is not in its goal position, or obstructing a goal position.
        needs_move = [False] * self.num_blocks
        propagation_queue = []
        
        for u in range(self.num_blocks):
            target = current_on[u]
            is_unstable = False
            
            # Case A: Block is Held
            if target == -2:
                is_unstable = True
            
            # Case B: Explicit Goal Mismatch
            elif u in self.goal_on:
                goal_target = self.goal_on[u]
                if target != goal_target:
                    is_unstable = True
                else:
                    # Even if matches goal, if it's obstructing someone else? 
                    # (Handled by Case D below on the *other* block)
                    pass
                    
            elif u in self.goal_ontable:
                if target != -1:
                    is_unstable = True
                    
            # Case C: Clear Goal Mismatch
            # If goal is clear(u), but u has children, those children must move.
            # This effectively makes children unstable. We handle this by making u 'unstable' source?
            # No, 'needs_move' applies to the block that must move. 
            # If clear(u) and on(k, u), k must move. 
            # We check this via children_map.
            if not is_unstable and u in self.goal_clear:
                if len(children_map[u]) > 0:
                    # The children are blocking.
                    # We will catch this in the propagation phase or obstruction check?
                    # Actually, obstruction check below is safer.
                    for child in children_map[u]:
                        if not needs_move[child]:
                            needs_move[child] = True
                            propagation_queue.append(child)
            
            # Case D: Target Obstruction
            # If u wants to be on V, but K is currently on V (and K != u), K must move.
            if u in self.goal_on:
                wanted_v = self.goal_on[u]
                # If the wanted base is a block (not table)
                if wanted_v >= 0:
                    # Check who is currently on wanted_v
                    for occupant in children_map[wanted_v]:
                        if occupant != u:
                            # occupant is blocking u's spot
                            if not needs_move[occupant]:
                                needs_move[occupant] = True
                                propagation_queue.append(occupant)

            if is_unstable and not needs_move[u]:
                needs_move[u] = True
                propagation_queue.append(u)
        
        # 4. Propagate Instability Upwards
        # If block V needs to move, anything on top of V (U) must also move.
        head = 0
        while head < len(propagation_queue):
            v = propagation_queue[head]
            head += 1
            
            # Check all blocks u sitting on v
            for u in children_map[v]:
                if not needs_move[u]:
                    needs_move[u] = True
                    propagation_queue.append(u)
        
        # 5. Calculate Cost
        # Each block that needs to move contributes to the cost.
        # This is a relaxation (1 move) but admissible for stack/unstack (min cost 1).
        return sum(needs_move)