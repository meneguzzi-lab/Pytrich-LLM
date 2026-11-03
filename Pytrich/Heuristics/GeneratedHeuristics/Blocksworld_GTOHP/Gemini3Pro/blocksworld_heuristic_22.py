from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic22(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_19):
        super().__init__(blocksworld_gemini3pro_heuristic_19)
        self.model = blocksworld_gemini3pro_heuristic_19
        
        # Pre-compute fact mappings for O(1) state lookups
        self.fact_ids_on = {}
        self.fact_ids_ontable = {}
        self.fact_ids_clear = {}
        self.fact_ids_holding = {}
        
        # Parse grounded facts to map (predicate, args) -> global_id
        for fact in self.model.facts:
            name = fact.name
            # Strip sign if present
            if name.startswith("+") or name.startswith("-"):
                name = name[1:]
            
            # Expected format: predicate[arg1,arg2]
            if "[" in name and name.endswith("]"):
                idx = name.find("[")
                pred = name[:idx]
                args = name[idx+1:-1].split(",")
                
                if pred == "on" and len(args) == 2:
                    self.fact_ids_on[(args[0], args[1])] = fact.global_id
                elif pred == "ontable" and len(args) == 1:
                    self.fact_ids_ontable[args[0]] = fact.global_id
                elif pred == "clear" and len(args) == 1:
                    self.fact_ids_clear[args[0]] = fact.global_id
                elif pred == "holding" and len(args) == 1:
                    self.fact_ids_holding[args[0]] = fact.global_id

    def _check(self, state, mapping, key):
        """Helper to check a fact bit in the state integer."""
        fid = mapping.get(key)
        if fid is not None:
            return (state >> fid) & 1
        return 0

    def h(self, node, state, task_network):
        """
        Domain-dependent heuristic for Blocksworld HTN.
        Estimates cost by summing costs of pending tasks in the network.
        Uses O(1) bitwise checks to penalize tasks based on current state (e.g., clearing blocks).
        """
        total_cost = 0
        
        for task in task_network:
            # Normalize task name: "do_put_on b1 b2" or "do_put_on(b1, b2)"
            name = task.name.replace("(", " ").replace(")", " ").replace(",", " ")
            parts = name.split()
            if not parts:
                continue
            
            task_type = parts[0]
            
            if task_type == "do_put_on" and len(parts) >= 3:
                x, y = parts[1], parts[2]
                
                # If goal is already satisfied, cost is 0
                if self._check(state, self.fact_ids_on, (x, y)):
                    continue
                
                # Base cost for the move operation itself
                cost = 1
                
                # Penalties for unsatisfied preconditions (heuristic estimate of subtasks)
                # If destination y is not clear, we must clear it
                if not self._check(state, self.fact_ids_clear, y):
                    cost += 1
                
                # If source x is not clear, we must clear it
                if not self._check(state, self.fact_ids_clear, x):
                    cost += 1
                
                # If x is not on table and not held, it is on something else, requiring unstacking
                if not self._check(state, self.fact_ids_ontable, x) and not self._check(state, self.fact_ids_holding, x):
                    cost += 1
                
                total_cost += cost

            elif task_type == "do_on_table" and len(parts) >= 2:
                x = parts[1]
                if self._check(state, self.fact_ids_ontable, x):
                    continue
                
                cost = 1
                # Clearing cost
                if not self._check(state, self.fact_ids_clear, x):
                    cost += 1
                # Unstacking cost if currently on another block
                if not self._check(state, self.fact_ids_holding, x) and not self._check(state, self.fact_ids_ontable, x):
                    cost += 1
                    
                total_cost += cost
            
            elif task_type == "do_clear" and len(parts) >= 2:
                x = parts[1]
                # If not clear, we must remove at least one block above it
                if not self._check(state, self.fact_ids_clear, x):
                    total_cost += 1
            
            elif task_type == "do_move":
                # Primitive abstract move: implies pickup + stack
                total_cost += 1
        
        return total_cost