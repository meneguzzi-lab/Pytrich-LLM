from Pytrich.core.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic16(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_13):
        super().__init__(blocksworld_gemini3pro_heuristic_13)
        self.model = blocksworld_gemini3pro_heuristic_13
        
        # Indexing for fast state lookup
        # on_facts_by_arg2: y -> [(global_id, x)] where on(x, y)
        self.on_facts_by_arg2 = {}
        # on_facts_by_arg1: x -> [(global_id, y)] where on(x, y)
        self.on_facts_by_arg1 = {}
        # ontable_facts: x -> global_id where ontable(x)
        self.ontable_facts = {}
        
        self._index_facts()

    def _index_facts(self):
        """
        Pre-process domain facts to map grounded predicates to efficient lookups.
        Facts format: +predicate[arg1,arg2] or +predicate[arg1]
        """
        for fact in self.model.facts:
            name = fact.name.strip()
            if not name.startswith("+"):
                continue
                
            # Remove '+' prefix
            core_name = name[1:]
            
            # Parse based on brackets
            bracket_start = core_name.find("[")
            bracket_end = core_name.find("]")
            
            if bracket_start == -1 or bracket_end == -1:
                continue
                
            predicate = core_name[:bracket_start]
            args_str = core_name[bracket_start+1:bracket_end]
            args = args_str.split(",") if args_str else []
            
            if predicate == "on" and len(args) == 2:
                x, y = args[0].strip(), args[1].strip()
                
                if y not in self.on_facts_by_arg2:
                    self.on_facts_by_arg2[y] = []
                self.on_facts_by_arg2[y].append((fact.global_id, x))
                
                if x not in self.on_facts_by_arg1:
                    self.on_facts_by_arg1[x] = []
                self.on_facts_by_arg1[x].append((fact.global_id, y))
                
            elif predicate == "ontable" and len(args) == 1:
                x = args[0].strip()
                self.ontable_facts[x] = fact.global_id

    def get_block_on_top_of(self, state, block_y):
        """Return the block x that is currently on block_y, or None."""
        if block_y in self.on_facts_by_arg2:
            for fid, block_x in self.on_facts_by_arg2[block_y]:
                if (state >> fid) & 1:
                    return block_x
        return None

    def is_ontable(self, state, block_x):
        """Check if block_x is on the table."""
        if block_x in self.ontable_facts:
            fid = self.ontable_facts[block_x]
            return (state >> fid) & 1
        return False
        
    def is_on(self, state, x, y):
        """Check if on(x, y) is true."""
        if x in self.on_facts_by_arg1:
            for fid, target_y in self.on_facts_by_arg1[x]:
                if target_y == y and ((state >> fid) & 1):
                    return True
        return False

    def count_above(self, state, block):
        """Count how many blocks are stacked above the given block."""
        count = 0
        curr = block
        visited = set()
        
        while True:
            # Cycle detection (should not happen in valid states)
            if curr in visited:
                break
            visited.add(curr)
            
            above = self.get_block_on_top_of(state, curr)
            if above:
                count += 1
                curr = above
            else:
                break
        return count

    def h(self, node):
        """
        Estimate the cost to complete the task network from the current state.
        Based on the decomposition strategy: 
        do_put_on(x, y) -> clear x, clear y, put y on table, move x to y.
        """
        state = node.state
        total_cost = 0
        
        for task in node.tn:
            # Task names are typically 'task_name arg1 arg2 ...'
            # Clean up the name to handle potential formatting variations
            parts = task.name.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
            if not parts:
                continue
                
            task_type = parts[0]
            
            if task_type == "do_put_on":
                if len(parts) >= 3:
                    x, y = parts[1], parts[2]
                    
                    # If x is already on y, cost is 0 (m0)
                    if self.is_on(state, x, y):
                        continue
                        
                    # Cost breakdown based on m1 decomposition:
                    # 1. Clear x: 2 actions per block above x
                    total_cost += 2 * self.count_above(state, x)
                    
                    # 2. Clear y: 2 actions per block above y
                    total_cost += 2 * self.count_above(state, y)
                    
                    # 3. Put y on table: 2 actions if not already there
                    # (Assuming m1 enforces this base stability)
                    if not self.is_ontable(state, y):
                        total_cost += 2
                        
                    # 4. Move x to y: 2 actions (pick/unstack + stack)
                    total_cost += 2
            
            elif task_type == "do_on_table":
                if len(parts) >= 2:
                    x = parts[1]
                    if not self.is_ontable(state, x):
                        # Cost to move to table = 2 actions
                        # Note: clearing cost is handled by do_clear tasks if present,
                        # or implicitly here if we assume relaxed validity.
                        total_cost += 2
                        
            elif task_type == "do_clear":
                if len(parts) >= 2:
                    x = parts[1]
                    # Cost to clear = 2 actions per block above
                    total_cost += 2 * self.count_above(state, x)
            
            elif task_type == "do_move":
                # Primitive-like compound task: pick/unstack + stack
                # Assumes preconditions (clear x, clear y) are met or charged elsewhere
                total_cost += 2
                
            elif task_type in ["pick-up", "put-down", "stack", "unstack"]:
                total_cost += 1
                
            elif task_type == "nop":
                total_cost += 0
                
            else:
                # Default fallback for unknown tasks
                total_cost += 1
                
        return total_cost
