from pytrich.heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic12(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_9):
        super().__init__(blocksworld_gemini3pro_heuristic_9)
        self.model = blocksworld_gemini3pro_heuristic_9
        self.fact_id_map = {}
        
        # Precompute fact name to ID mapping for O(1) lookup during heuristic evaluation
        # The heuristic needs to check facts like +on[b1,b2] or +clear[b1]
        for fact in self.model.facts:
            self.fact_id_map[fact.name] = fact.global_id

    def _get_fact_id(self, predicate, args):
        """
        Constructs the grounded fact name string and retrieves its global ID.
        Format: +predicate[arg1,arg2,...]
        """
        args_str = ",".join(args)
        key = f"+{predicate}[{args_str}]"
        return self.fact_id_map.get(key)

    def _is_fact_true(self, state_bitwise, predicate, args):
        """
        Checks if a specific fact is true in the bitwise state.
        Returns False if the fact does not exist in the model.
        """
        fid = self._get_fact_id(predicate, args)
        if fid is None:
            return False
        return (state_bitwise >> fid) & 1

    def h(self, node):
        """
        Domain-dependent heuristic for Blocksworld HTN.
        Estimates the cost of completing the Task Network based on the 
        specific Blocksworld hierarchy strategies (clearing blocks, moving to table, etc.).
        """
        cost = 0
        state = node.state
        
        # Iterate through the tasks in the network to estimate remaining effort.
        # This leverages the Total-Order nature of the problem.
        for task in node.task_network:
            # If the task is a primitive Operator, it has a fixed cost (usually 1)
            if hasattr(task, 'cost'): 
                cost += task.cost
                continue
            
            # If it is an AbstractTask, we analyze it based on domain semantics.
            # We first parse the task name to identify the type and arguments.
            # Task names are typically "task_name arg1 arg2 ..."
            clean_name = task.name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = clean_name.split()
            if not parts:
                continue
                
            task_type = parts[0]
            args = parts[1:]
            
            # Heuristic logic for do_put_on(x, y)
            # The domain method m1 enforces: clear(x), clear(y), on_table(y), move(x,y).
            if "do_put_on" in task_type and len(args) >= 2:
                block_x = args[0]
                block_y = args[1]
                
                # If the goal condition is already met, cost is 0 (method m0)
                if self._is_fact_true(state, "on", [block_x, block_y]):
                    continue
                
                # Base cost for the final move action (stack)
                cost += 1
                
                # Cost to clear block X if obstructed
                if not self._is_fact_true(state, "clear", [block_x]):
                    cost += 1
                
                # Cost to clear block Y if obstructed
                if not self._is_fact_true(state, "clear", [block_y]):
                    cost += 1
                
                # The domain specifically often requires Y to be on the table before stacking X on it
                # (See method m1_do_put_on). If Y is not on table, add cost to move it there.
                if not self._is_fact_true(state, "ontable", [block_y]):
                    cost += 1
                
                # If X is not accessible (on table or holding), we might need an unstack
                if not self._is_fact_true(state, "ontable", [block_x]) and \
                   not self._is_fact_true(state, "holding", [block_x]):
                    cost += 1

            # Heuristic logic for do_on_table(x)
            elif "do_on_table" in task_type and len(args) >= 1:
                block_x = args[0]
                
                # If already on table, 0 cost
                if self._is_fact_true(state, "ontable", [block_x]):
                    continue
                    
                # Cost to put down
                cost += 1
                
                # If we are not holding it, we need to get it (pick-up or unstack)
                if not self._is_fact_true(state, "holding", [block_x]):
                    # If covered, need to clear
                    if not self._is_fact_true(state, "clear", [block_x]):
                        cost += 1
                    # Implicit unstack cost if not holding
                    cost += 1

            # Heuristic logic for do_clear(x)
            elif "do_clear" in task_type and len(args) >= 1:
                block_x = args[0]
                # If already clear, 0 cost
                if self._is_fact_true(state, "clear", [block_x]):
                    continue
                # At least one unstack action required to clear it
                cost += 1
            
            # Heuristic logic for do_move(x, y)
            elif "do_move" in task_type:
                # do_move involves stacking. Assuming preconditions (clear) are handled by prior tasks.
                cost += 1

            else:
                # Fallback for unknown tasks
                cost += 1
                
        return cost