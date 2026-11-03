from pytrich.heuristics.heuristic import Heuristic
import sys

class BlocksworldGemini3ProHeuristic20(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_17, model):
        super().__init__(blocksworld_gemini3pro_heuristic_17, model)
        self.model = model
        
        # Efficient Lookups using Bitmasks for large instance scaling
        # y -> bitmask where the bit at global_id of on(x,y) is 1
        self.masks_above = {}
        # global_id -> x_name (inverse lookup for the block x in on(x,y))
        self.fid_to_block = {}
        
        self.ontable_mask = {}
        self.clear_mask = {}
        self.holding_mask = {}
        
        # For specific fact checking
        self.on_facts_check = {}  # y -> [(x, global_id)]
        
        self.handempty_id = -1
        
        self._preprocess_facts()

    def _preprocess_facts(self):
        """
        Pre-processes all facts to build bitmasks and maps for O(1) or O(1)-like access 
        during heuristic evaluation. This is crucial for N=1000 instances.
        """
        for fact in self.model.facts:
            name = fact.name
            # Expecting format +pred[arg1,arg2] or similar
            if not name.startswith("+"): continue
            
            content = name[1:]
            if "[" in content:
                pred_part, args_part = content.split("[", 1)
                pred = pred_part
                if args_part.endswith("]"):
                    args = args_part[:-1].split(",")
                    args = [a.strip() for a in args if a.strip()]
                else:
                    args = []
            else:
                pred = content
                args = []
                
            fid = fact.global_id
            
            if pred == "on" and len(args) == 2:
                x, y = args
                
                # Build mask for finding what is above y
                if y not in self.masks_above:
                    self.masks_above[y] = 0
                self.masks_above[y] |= (1 << fid)
                
                self.fid_to_block[fid] = x
                
                if y not in self.on_facts_check:
                    self.on_facts_check[y] = []
                self.on_facts_check[y].append((x, fid))
                
            elif pred == "ontable" and len(args) == 1:
                x = args[0]
                self.ontable_mask[x] = (1 << fid)
                
            elif pred == "clear" and len(args) == 1:
                x = args[0]
                self.clear_mask[x] = (1 << fid)
                
            elif pred == "holding" and len(args) == 1:
                x = args[0]
                self.holding_mask[x] = (1 << fid)
                
            elif pred == "handempty":
                self.handempty_id = fid

    def _is_clear(self, x, state_int):
        if x in self.clear_mask:
            return (state_int & self.clear_mask[x]) != 0
        return False

    def _is_ontable(self, x, state_int):
        if x in self.ontable_mask:
            return (state_int & self.ontable_mask[x]) != 0
        return False
        
    def _is_holding(self, x, state_int):
        if x in self.holding_mask:
            return (state_int & self.holding_mask[x]) != 0
        return False

    def _get_block_on(self, y, state_int):
        """
        Finds the block x such that on(x, y) is true using bit manipulation.
        This avoids iterating over lists of potential blocks.
        """
        if y not in self.masks_above:
            return None
            
        # Intersect state with mask of all facts that put something on y
        active = state_int & self.masks_above[y]
        
        if not active:
            return None
            
        # In a valid blocksworld state, only one block can be on y.
        # active should be a power of 2. We find the index of the set bit.
        fid = active.bit_length() - 1
        return self.fid_to_block.get(fid)

    def _calculate_clear_cost(self, start_block, state_int, memo):
        """
        Calculates the number of blocks stacked above start_block.
        Uses iterative approach to avoid recursion depth limits on large instances.
        """
        if start_block in memo:
            return memo[start_block]
            
        stack = []
        curr = start_block
        base_cost = 0
        
        # 1. Traverse up the stack
        while True:
            if curr in memo:
                base_cost = memo[curr]
                break
            
            # If clear or holding, no cost to clear
            if self._is_clear(curr, state_int) or self._is_holding(curr, state_int):
                base_cost = 0
                memo[curr] = 0
                break
            
            # Find what is above
            above = self._get_block_on(curr, state_int)
            if above is None:
                # Should be clear if nothing above, assuming consistent state
                base_cost = 0
                memo[curr] = 0
                break
                
            if curr in stack:
                # Cycle detected (should not happen)
                base_cost = 9999
                break
                
            stack.append(curr)
            curr = above
        
        # 2. Unwind and propagate costs
        # cost(x) = 1 + cost(above)
        cost = base_cost
        while stack:
            blk = stack.pop()
            cost += 1
            memo[blk] = cost
            
        return memo[start_block]

    def h(self, task_network, state_int):
        total_cost = 0
        memo_clear = {}
        
        # Determine hand state
        handempty = True
        if self.handempty_id != -1:
            handempty = ((state_int >> self.handempty_id) & 1) == 1
            
        # Helper check for already satisfied put_on goals
        def is_on(x, y):
            if y in self.on_facts_check:
                for bx, fid in self.on_facts_check[y]:
                    if bx == x and ((state_int >> fid) & 1):
                        return True
            return False

        for task in task_network:
            name = task.name
            parts = name.split()
            if not parts: continue
            
            ttype = parts[0]
            
            if ttype == "do_put_on" and len(parts) >= 3:
                x, y = parts[1], parts[2]
                
                # If already achieved, cost 0
                if is_on(x, y):
                    continue
                
                # Estimate cost based on standard decomposition:
                # To put x on y: 
                # 1. Clear x
                # 2. Clear y
                # 3. If y not on table, unstack y (Domain Specific Method m1 puts y on table)
                # 4. Move x to y
                
                c_x = self._calculate_clear_cost(x, state_int, memo_clear)
                c_y = self._calculate_clear_cost(y, state_int, memo_clear)
                
                # Domain specific penalty: method m1 often forces the base (y) to be on the table
                c_y_table = 0
                if not self._is_ontable(y, state_int):
                    if not self._is_holding(y, state_int):
                         # Need to unstack y from wherever it is
                         c_y_table = 1 
                
                # Hand mechanics cost
                c_hand = 0
                holding_x = self._is_holding(x, state_int)
                
                if holding_x:
                    c_hand = 0
                elif handempty:
                    c_hand = 1 # Pick up x
                else:
                    c_hand = 2 # Put down whatever held + Pick up x
                
                # Stack action itself
                c_stack = 1
                
                total_cost += c_x + c_y + c_y_table + c_hand + c_stack

            elif ttype == "do_on_table" and len(parts) >= 2:
                x = parts[1]
                if self._is_ontable(x, state_int):
                    continue
                
                c_x = self._calculate_clear_cost(x, state_int, memo_clear)
                
                c_move = 0
                if self._is_holding(x, state_int):
                    c_move = 1 # put-down
                else:
                    c_move = 2 # unstack + put-down
                
                total_cost += c_x + c_move

            elif ttype == "do_clear" and len(parts) >= 2:
                x = parts[1]
                total_cost += self._calculate_clear_cost(x, state_int, memo_clear)
                
            elif ttype == "do_move" and len(parts) >= 3:
                # Primitive wrapper usually pick + stack
                total_cost += 2 
            
            elif ttype in ["pick-up", "put-down", "stack", "unstack"]:
                total_cost += 1
                
        return total_cost
