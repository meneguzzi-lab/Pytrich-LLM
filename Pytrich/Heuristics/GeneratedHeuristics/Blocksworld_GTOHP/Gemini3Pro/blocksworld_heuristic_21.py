from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic21(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_18):
        super().__init__(blocksworld_gemini3pro_heuristic_18)
        self.model = blocksworld_gemini3pro_heuristic_18
        
        self.TABLE = -1
        self.HOLDING = -2
        
        # Mappings
        self.block_map = {}
        self.fact_map = {} # global_id -> (type, arg1, arg2)
        self.task_map = {} # global_id -> (type, arg1, arg2)
        
        self._initialize_mappings()

    def _initialize_mappings(self):
        # Helper to generate block IDs
        next_id = 0
        def get_blk_id(n):
            nonlocal next_id
            if n not in self.block_map:
                self.block_map[n] = next_id
                next_id += 1
            return self.block_map[n]

        # 1. Process Facts to identify static info and block names
        for f in self.model.facts:
            name = f.name.strip()
            # Handle grounded format +predicate[args]
            if name.startswith("+") or name.startswith("-"):
                name = name[1:]
            
            pred = None
            args = []
            if "[" in name:
                pred = name.split("[")[0]
                args_str = name.split("[")[1].rstrip("]")
                args = [x.strip() for x in args_str.split(",") if x.strip()]
            else:
                parts = name.split()
                pred = parts[0]
                args = parts[1:]
            
            # Map fact types: 0=on, 1=ontable, 2=clear
            ftype = -1
            if pred == "on": ftype = 0
            elif pred == "ontable": ftype = 1
            elif pred == "clear": ftype = 2
            
            if ftype != -1:
                a1 = get_blk_id(args[0]) if len(args) > 0 else -1
                a2 = get_blk_id(args[1]) if len(args) > 1 else -1
                self.fact_map[f.global_id] = (ftype, a1, a2)

        # 2. Process Tasks to identify abstract tasks and operators
        all_tasks = self.model.operators + self.model.abstract_tasks
        for task in all_tasks:
            name = task.name.strip()
            
            t_name = None
            args = []
            if "[" in name:
                t_name = name.split("[")[0]
                args_str = name.split("[")[1].rstrip("]")
                args = [x.strip() for x in args_str.split(",") if x.strip()]
            else:
                parts = name.split()
                t_name = parts[0]
                args = parts[1:]
            
            # Identify task types
            # 10: do_put_on, 11: do_on_table, 12: do_move, 13: do_clear
            # 20: stack, 21: unstack, 22: pick-up, 23: put-down
            ttype = -1
            if "do_put_on" in t_name: ttype = 10
            elif "do_on_table" in t_name: ttype = 11
            elif "do_move" in t_name: ttype = 12
            elif "do_clear" in t_name: ttype = 13
            elif "unstack" in t_name: ttype = 21
            elif "stack" in t_name: ttype = 20
            elif "pick-up" in t_name: ttype = 22
            elif "put-down" in t_name: ttype = 23
            
            if ttype != -1:
                a1 = self.block_map.get(args[0], -1) if len(args) > 0 else -1
                a2 = self.block_map.get(args[1], -1) if len(args) > 1 else -1
                self.task_map[task.global_id] = (ttype, a1, a2)

    def h(self, node):
        state = node.state
        # Reconstruct graph state from bitmask
        # parent[x] = y  => x is on y
        # on_top[y] = x  => y has x on it
        parent = {}
        on_top = {}
        
        # Efficiently iterate over set bits in the state integer
        temp_state = state
        while temp_state:
            # Get index of lowest set bit
            idx = (temp_state & -temp_state).bit_length() - 1
            # Clear that bit
            temp_state &= ~(1 << idx)
            
            info = self.fact_map.get(idx)
            if info:
                ft, a1, a2 = info
                if ft == 0: # on(a1, a2)
                    parent[a1] = a2
                    on_top[a2] = a1
                elif ft == 1: # ontable(a1)
                    parent[a1] = self.TABLE

        cost = 0.0
        
        # Simulation Helper: Calculate cost to clear a block and update virtual state
        def clear_cost(b):
            c = 0
            curr = b
            stack = []
            # Traverse up to find all blocks above b
            seen = set()
            while curr in on_top:
                curr = on_top[curr]
                if curr in seen: break # Cycle protection
                seen.add(curr)
                stack.append(curr)
            
            if not stack:
                return 0
            
            # Cost is roughly 1 per block to unstack/move away
            c = len(stack)
            
            # Update virtual state: Move all these blocks to TABLE
            for blk in stack:
                p = parent.get(blk)
                if p is not None and p in on_top and on_top[p] == blk:
                    del on_top[p]
                parent[blk] = self.TABLE
                
            return c

        # Iterate through tasks to estimate remaining effort
        # This simulation accounts for interactions sequentially as ordered in the HTN
        for task in node.task_network:
            info = self.task_map.get(task.global_id)
            if not info:
                continue
                
            ttype, a1, a2 = info
            
            # High-level Tasks
            if ttype == 10: # do_put_on(x, y)
                # If already in place, cost is 0 (approx)
                if parent.get(a1) == a2:
                    continue
                
                # We need x clear and y clear
                cost += clear_cost(a1)
                cost += clear_cost(a2)
                cost += 1 # The move action itself
                
                # Update state: x is now on y
                p = parent.get(a1)
                if p is not None and p != self.TABLE and p != self.HOLDING:
                    if on_top.get(p) == a1:
                        del on_top[p]
                parent[a1] = a2
                on_top[a2] = a1
                
            elif ttype == 11: # do_on_table(x)
                if parent.get(a1) == self.TABLE:
                    continue
                cost += clear_cost(a1)
                cost += 1
                
                p = parent.get(a1)
                if p is not None and p != self.TABLE and p != self.HOLDING:
                    if on_top.get(p) == a1:
                        del on_top[p]
                parent[a1] = self.TABLE
                
            elif ttype == 12: # do_move(x, y)
                # Similar to put_on but primitive wrapper
                if parent.get(a1) == a2:
                    continue
                cost += clear_cost(a1)
                cost += clear_cost(a2)
                cost += 1
                
                p = parent.get(a1)
                if p is not None and p != self.TABLE and p != self.HOLDING:
                    if on_top.get(p) == a1:
                        del on_top[p]
                parent[a1] = a2
                on_top[a2] = a1
                
            elif ttype == 13: # do_clear(x)
                cost += clear_cost(a1)
                # state updated inside clear_cost
                
            # Primitives
            elif ttype >= 20:
                cost += 1
                # Apply primitive effects to keep simulation consistent
                if ttype == 20: # stack x y
                    parent[a1] = a2
                    on_top[a2] = a1
                elif ttype == 21: # unstack x y
                    parent[a1] = self.HOLDING
                    if on_top.get(a2) == a1:
                        del on_top[a2]
                elif ttype == 22: # pick-up x
                    parent[a1] = self.HOLDING
                    # if it was on table, no on_top update needed
                elif ttype == 23: # put-down x
                    parent[a1] = self.TABLE

        return float(cost)
