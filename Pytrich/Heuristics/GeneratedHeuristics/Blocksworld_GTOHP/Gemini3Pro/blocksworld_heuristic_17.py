from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic17(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_14, strategy=None):
        super().__init__(blocksworld_gemini3pro_heuristic_14)
        self.model = blocksworld_gemini3pro_heuristic_14
        
        # Maps for rapid state querying
        # on_map: { bottom_block: [(top_block, fact_id), ...] }
        self.on_map = {}
        # clear_map: { block: fact_id }
        self.clear_map = {}
        # ontable_map: { block: fact_id }
        self.ontable_map = {}
        # holding_map: { block: fact_id }
        self.holding_map = {}
        
        # Pre-process facts to build lookup maps
        for fact in self.model.facts:
            name = fact.name
            # Parse grounded fact name: +predicate[arg1,arg2]
            predicate, args = self._parse_fact_name(name)
            
            if predicate == "on" and len(args) == 2:
                top, bottom = args[0], args[1]
                if bottom not in self.on_map:
                    self.on_map[bottom] = []
                self.on_map[bottom].append((top, fact.global_id))
            
            elif predicate == "clear" and len(args) == 1:
                self.clear_map[args[0]] = fact.global_id
                
            elif predicate == "ontable" and len(args) == 1:
                self.ontable_map[args[0]] = fact.global_id
                
            elif predicate == "holding" and len(args) == 1:
                self.holding_map[args[0]] = fact.global_id

    def _parse_fact_name(self, name: str):
        """
        Parses names like +on[b1,b2] into ('on', ['b1', 'b2'])
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove sign
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return None, []

    def _is_true(self, state, fact_id):
        """Check if bit at fact_id is set in the state bitmask."""
        return (state >> fact_id) & 1

    def _get_tower_clearing_cost(self, state, block, cache):
        """
        Recursive function to estimate cost to clear a block.
        Cost = number of blocks stacked above it (each requires an unstack).
        """
        if block in cache:
            return cache[block]
            
        # If clear(block) is true, cost is 0
        if block in self.clear_map and self._is_true(state, self.clear_map[block]):
            cache[block] = 0.0
            return 0.0
            
        # Find the block currently on top of 'block'
        block_above = None
        if block in self.on_map:
            for top, fid in self.on_map[block]:
                if self._is_true(state, fid):
                    block_above = top
                    break
        
        if block_above:
            # Cost is 1 (unstack action) + cost to clear the block above
            cost = 1.0 + self._get_tower_clearing_cost(state, block_above, cache)
            cache[block] = cost
            return cost
            
        # Fallback if state seems inconsistent or block is implicitly clear (should be covered by clear_map)
        cache[block] = 0.0
        return 0.0

    def h(self, node):
        """
        Heuristic estimates the remaining cost by summing the estimated cost 
        of performing each task in the task network.
        """
        state = node.state
        tasks = node.task_network
        total_cost = 0.0
        
        # Cache tower calculations to avoid O(N^2) in deep recursions for the same node
        clearing_cache = {}
        
        for task in tasks:
            # Parse task name. Format typically: "task_name arg1 arg2"
            t_name = task.name
            # Clean up potential formatting (brackets, parens, commas)
            clean_name = t_name.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace(',', ' ')
            parts = clean_name.split()
            if not parts:
                continue
                
            task_type = parts[0]
            args = parts[1:]
            
            if "do_put_on" in task_type and len(args) >= 2:
                x, y = args[0], args[1]
                
                # Check if x is already on y
                already_done = False
                if y in self.on_map:
                    for top, fid in self.on_map[y]:
                        if top == x and self._is_true(state, fid):
                            already_done = True
                            break
                
                if already_done:
                    continue
                
                # Cost estimation for do_put_on(x, y):
                # 1. Clear x
                # 2. Clear y
                # 3. Stack x on y (1 action)
                # 4. If x is on another block, Unstack x (1 action)
                
                cost = 0.0
                cost += self._get_tower_clearing_cost(state, x, clearing_cache)
                cost += self._get_tower_clearing_cost(state, y, clearing_cache)
                cost += 1.0 # The 'stack' action itself
                
                # Check if we need to pick up x (if not holding it)
                is_holding = x in self.holding_map and self._is_true(state, self.holding_map[x])
                if not is_holding:
                    cost += 1.0 # Pick-up or Unstack action
                
                total_cost += cost
            
            elif "do_on_table" in task_type and len(args) >= 1:
                x = args[0]
                
                # Check if already on table
                is_ontable = x in self.ontable_map and self._is_true(state, self.ontable_map[x])
                if is_ontable:
                    continue
                    
                # Cost: Clear x + Putdown (1) + Unstack (1)
                cost = 0.0
                cost += self._get_tower_clearing_cost(state, x, clearing_cache)
                cost += 1.0 # Put-down action
                
                is_holding = x in self.holding_map and self._is_true(state, self.holding_map[x])
                if not is_holding:
                    cost += 1.0 # Unstack/Pick-up
                
                total_cost += cost
                
            elif "do_clear" in task_type and len(args) >= 1:
                x = args[0]
                # Just the cost to clear the tower above x
                total_cost += self._get_tower_clearing_cost(state, x, clearing_cache)
                
            elif "do_move" in task_type:
                # Primitive-like move helper: implies moving x to y or table
                # Assumes minimum 1 action cost
                total_cost += 1.0
                
        return total_cost
