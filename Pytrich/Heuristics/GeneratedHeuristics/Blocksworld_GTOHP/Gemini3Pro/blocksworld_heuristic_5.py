from Pytrich.core.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic5(Heuristic):
    def __init__(self, model, blocksworld_gemini3pro_heuristic_2=None):
        super().__init__(model)
        self.fact_lookup = {}
        # Precompute parsed fact names for efficient lookup during search
        for fact in self.model.facts:
            self.fact_lookup[fact.global_id] = self._parse_name(fact.name)

    def _parse_name(self, name):
        name = name.strip()
        # Remove +/- prefix if present (grounded facts)
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        # Handle bracket format: predicate[arg1,arg2]
        if "[" in name and name.endswith("]"):
            idx = name.find("[")
            pred = name[:idx]
            args = name[idx+1:-1].split(",")
            return pred, [a.strip() for a in args]
            
        # Handle space format: predicate arg1 arg2
        parts = name.split()
        if parts:
            return parts[0], parts[1:]
        return name, []

    def _get_state_topology(self, state_int):
        """
        Reconstructs the block configuration from the bitwise state.
        Returns:
            above: dict mapping block_under -> block_over (representing on(over, under))
            ontable: set of blocks currently on the table
            holding: the block currently being held, or None
        """
        above = {}
        ontable = set()
        holding = None
        
        temp = state_int
        # Efficiently iterate only set bits
        while temp:
            lowest = temp & -temp
            idx = lowest.bit_length() - 1
            temp ^= lowest # Clear the processed bit
            
            parsed = self.fact_lookup.get(idx)
            if parsed:
                pred, args = parsed
                if pred == "on" and len(args) == 2:
                    # on(top, bottom) -> bottom has top on it
                    above[args[1]] = args[0]
                elif pred == "ontable" and len(args) == 1:
                    ontable.add(args[0])
                elif pred == "holding" and len(args) == 1:
                    holding = args[0]
                    
        return above, ontable, holding

    def _depth_to_clear(self, block, above_map):
        """
        Calculates the number of blocks stacked above the given block.
        Used to estimate the cost of 'do_clear'.
        """
        c = 0
        curr = block
        # Limit loop to prevent infinite recursion in case of cycles (though unlikely in valid states)
        for _ in range(1000):
            if curr not in above_map:
                break
            curr = above_map[curr]
            c += 1
        return c

    def __call__(self, node):
        state = node.state
        tn = node.task_network
        
        # 1. Parse current state topology
        above_map, ontable_set, holding_block = self._get_state_topology(state)
        
        total = 0.0
        
        # Penalty if hand is not empty, as most high-level tasks require handempty
        # This estimates the cost of a 'put-down' action
        if holding_block is not None:
            total += 1.0

        # 2. Sum costs of all tasks in the network
        for task in tn:
            # Primitive Operator: Base cost of 1
            if hasattr(task, "cost"):
                total += 1
                continue
            
            # Abstract Task: Estimate refinement cost based on domain logic
            pred, args = self._parse_name(task.name)
            
            if pred == "do_put_on" and len(args) == 2:
                x, y = args
                # m0: If x is already on y, cost is 0
                if above_map.get(y) == x:
                    total += 0
                else:
                    # m1 strategy: clear x, clear y, put y on table, move x to y
                    
                    # Cost to clear x (unstack + put-down per block above)
                    cx = self._depth_to_clear(x, above_map) * 2
                    
                    # Cost to clear y
                    cy = self._depth_to_clear(y, above_map) * 2
                    
                    # Cost to put y on table (if not already)
                    # Method m1 forces y to table. If y is on z, we need to unstack it (cost ~2)
                    # If y is on table, cost is 0
                    cty = 0 if y in ontable_set else 2
                    
                    # Cost to move x to y (pick-up + stack)
                    cm = 2
                    
                    total += cx + cy + cty + cm

            elif pred == "do_on_table" and len(args) == 1:
                x = args[0]
                if x in ontable_set:
                    total += 0
                else:
                    # Cost: clear x + move to table (unstack + put-down)
                    cx = self._depth_to_clear(x, above_map) * 2
                    total += cx + 2

            elif pred == "do_clear" and len(args) == 1:
                x = args[0]
                # Cost: 2 actions for every block above x
                total += self._depth_to_clear(x, above_map) * 2

            elif pred == "do_move" and len(args) >= 2:
                # Primitive decomposition: pick/unstack + stack -> cost 2
                total += 2
                
            elif pred == "nop":
                total += 0
                
            else:
                # Fallback for unknown tasks
                total += 1
                
        return total
