from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic14(Heuristic):
    def __init__(self, model, blocksworld_gemini3pro_heuristic_11=None):
        super().__init__(model)
        self.blocksworld_gemini3pro_heuristic_11 = blocksworld_gemini3pro_heuristic_11
        
        # Maps for O(1) checks from (args) -> global_id
        self.on_map = {}
        self.ontable_map = {}
        self.clear_map = {}
        self.holding_map = {}
        self.handempty_id = -1

        # Cache parsed task info to avoid string parsing during search
        # global_id -> (task_type, [arg1, arg2, ...])
        self.task_info = {}

        self._preprocess_facts()
        self._preprocess_tasks()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess_facts(self):
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            if pred == 'on' and len(args) == 2:
                self.on_map[(args[0], args[1])] = fact.global_id
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_map[args[0]] = fact.global_id
            elif pred == 'clear' and len(args) == 1:
                self.clear_map[args[0]] = fact.global_id
            elif pred == 'holding' and len(args) == 1:
                self.holding_map[args[0]] = fact.global_id
            elif pred == 'handempty':
                self.handempty_id = fact.global_id

    def _preprocess_tasks(self):
        # Helper to parse "task_name arg1 arg2 ..."
        def parse_and_store(item):
            parts = item.name.replace('(', '').replace(')', '').split()
            if not parts:
                return
            t_type = parts[0]
            args = parts[1:]
            self.task_info[item.global_id] = (t_type, args)

        for task in self.model.abstract_tasks:
            parse_and_store(task)
        for op in self.model.operators:
            parse_and_store(op)

    def _is_true(self, state, fid):
        if fid == -1: return False
        return (state >> fid) & 1

    def __call__(self, state, task_network):
        h_val = 0
        
        # We iterate over the task network to estimate cost.
        # This avoids iterating the large state bitmask directly.
        for task in task_network:
            info = self.task_info.get(task.global_id)
            if not info:
                continue
                
            t_type, args = info
            
            if t_type == 'do_put_on' and len(args) >= 2:
                x, y = args[0], args[1]
                
                # If x is already on y, cost is 0 (m0 applies)
                if self._is_true(state, self.on_map.get((x, y), -1)):
                    continue
                
                # Base cost to perform the move/stack action
                cost = 1
                
                # Analyze Source X
                is_holding_x = self._is_true(state, self.holding_map.get(x, -1))
                if not is_holding_x:
                    # If we aren't holding X, we must pick/unstack it.
                    cost += 1
                    # If X is not clear, we must unstack blocks above it.
                    # Exact depth calculation is expensive (O(N)), so we use a constant penalty
                    # relying on 'clear' predicate which is O(1).
                    if not self._is_true(state, self.clear_map.get(x, -1)):
                        cost += 1
                
                # Analyze Target Y
                # Y must be clear to receive X.
                if not self._is_true(state, self.clear_map.get(y, -1)):
                    cost += 1
                
                h_val += cost

            elif t_type == 'do_on_table' and len(args) >= 1:
                x = args[0]
                
                # If already on table, cost 0 (m3 applies if clear)
                # Checking ontable explicitly
                if self._is_true(state, self.ontable_map.get(x, -1)):
                    continue
                
                # Cost to put down
                cost = 1
                
                is_holding_x = self._is_true(state, self.holding_map.get(x, -1))
                if not is_holding_x:
                    cost += 1 # Unstack/Pick
                    if not self._is_true(state, self.clear_map.get(x, -1)):
                        cost += 1 # Obstruction penalty
                
                h_val += cost

            elif t_type == 'do_clear' and len(args) >= 1:
                x = args[0]
                if self._is_true(state, self.clear_map.get(x, -1)):
                    continue
                # If not clear, we must unstack at least one block
                h_val += 1

            elif t_type == 'do_move' and len(args) >= 2:
                # Primitive-like move: stack x on y
                x, y = args[0], args[1]
                cost = 1
                
                # Preconditions: clear x (if not held), clear y
                is_holding_x = self._is_true(state, self.holding_map.get(x, -1))
                if not is_holding_x:
                     if not self._is_true(state, self.clear_map.get(x, -1)): 
                         cost += 1
                     cost += 1 # Pick up
                
                if not self._is_true(state, self.clear_map.get(y, -1)): 
                    cost += 1
                    
                h_val += cost
                
            elif t_type in ['stack', 'unstack', 'pick-up', 'put-down']:
                # Primitive actions have unit cost
                h_val += 1
                
        return h_val
