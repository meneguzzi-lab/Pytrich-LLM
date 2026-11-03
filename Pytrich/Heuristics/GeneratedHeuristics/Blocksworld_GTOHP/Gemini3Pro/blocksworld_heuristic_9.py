from pytrich.heuristics.base import Heuristic

class BlocksworldGemini3ProHeuristic9(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_6):
        super().__init__(blocksworld_gemini3pro_heuristic_6)
        self.model = blocksworld_gemini3pro_heuristic_6
        
        # Precompute maps for fast state lookups
        self.ontable_map = {}
        self.clear_map = {}
        self.holding_map = {}
        self.supported_by_map = {} # y -> [(x, fid)] i.e. x is on y
        self.on_top_of_map = {}    # x -> [(y, fid)] i.e. x is on y
        
        for fact in self.model.facts:
            name = fact.name.strip()
            # Handle grounded format +predicate[arg1,arg2] and standard PDDL
            clean = name
            if clean.startswith("+"): clean = clean[1:]
            elif clean.startswith("-"): clean = clean[1:]
            
            pred = None
            args = []
            
            if "[" in clean and clean.endswith("]"):
                idx = clean.find("[")
                pred = clean[:idx]
                args_str = clean[idx+1:-1]
                args = args_str.split(",") if args_str else []
            else:
                parts = clean.replace("(", " ").replace(")", " ").split()
                if parts:
                    pred = parts[0]
                    args = parts[1:]
            
            if not pred:
                continue
                
            fid = fact.global_id
            
            if pred == "on" and len(args) == 2:
                x, y = args
                if y not in self.supported_by_map: self.supported_by_map[y] = []
                self.supported_by_map[y].append((x, fid))
                if x not in self.on_top_of_map: self.on_top_of_map[x] = []
                self.on_top_of_map[x].append((y, fid))
            elif pred == "ontable" and len(args) == 1:
                self.ontable_map[args[0]] = fid
            elif pred == "clear" and len(args) == 1:
                self.clear_map[args[0]] = fid
            elif pred == "holding" and len(args) == 1:
                self.holding_map[args[0]] = fid

    def _is_true(self, state, fid):
        return (state >> fid) & 1

    def _get_above(self, state, block):
        """Returns the name of the block directly on top of 'block', or None."""
        if block in self.supported_by_map:
            for top, fid in self.supported_by_map[block]:
                if self._is_true(state, fid):
                    return top
        return None

    def _is_ontable(self, state, block):
        fid = self.ontable_map.get(block)
        return fid is not None and self._is_true(state, fid)

    def _is_holding(self, state, block):
        fid = self.holding_map.get(block)
        return fid is not None and self._is_true(state, fid)

    def _pile_height_cost(self, state, block):
        """Estimates cost to clear a block (2 actions per block above: unstack + putdown)."""
        cost = 0
        current = block
        visited = {block}
        while True:
            above = self._get_above(state, current)
            if above is None or above in visited:
                break
            visited.add(above)
            cost += 2
            current = above
        return cost

    def eval(self, node):
        state = node.state
        tasks = node.task_network
        if not tasks:
            return 0
        
        total_cost = 0
        
        for task in tasks:
            name = task.name.strip()
            parts = name.replace("(", " ").replace(")", " ").split()
            if not parts: continue
            task_type = parts[0]
            args = parts[1:]
            
            # Cost for primitive actions
            if task_type in ["pick-up", "put-down", "stack", "unstack"]:
                total_cost += 1
            elif task_type == "nop":
                pass
            
            # Domain-specific estimation for Abstract Tasks
            elif task_type == "do_put_on" and len(args) >= 2:
                x, y = args[0], args[1]
                
                # Check if x is already on y
                is_on = False
                if x in self.on_top_of_map:
                    for bottom, fid in self.on_top_of_map[x]:
                        if bottom == y and self._is_true(state, fid):
                            is_on = True
                            break
                if is_on:
                    continue
                
                # Calculate cost components
                cost_x_clear = 0
                move_cost = 0
                
                # Cost to access X
                if self._is_holding(state, x):
                    move_cost = 1 # Just stack
                else:
                    cost_x_clear = self._pile_height_cost(state, x)
                    move_cost = 2 # Pick/Unstack + Stack

                # Cost to access Y
                cost_y_clear = self._pile_height_cost(state, y)
                
                # Domain specific: method m1 enforces y must be on table
                # If y is not on table, we must unstack it and put it down
                cost_y_table = 0
                if not self._is_ontable(state, y):
                    if self._is_holding(state, y):
                        cost_y_table = 1 # put-down
                    else:
                        # We already paid to clear y above. Now move y to table.
                        cost_y_table = 2 # unstack + put-down
                
                total_cost += cost_x_clear + cost_y_clear + cost_y_table + move_cost

            elif task_type == "do_on_table" and len(args) >= 1:
                x = args[0]
                if self._is_ontable(state, x):
                    continue
                if self._is_holding(state, x):
                    total_cost += 1
                else:
                    # Clear x + unstack + putdown
                    total_cost += self._pile_height_cost(state, x) + 2

            elif task_type == "do_clear" and len(args) >= 1:
                x = args[0]
                total_cost += self._pile_height_cost(state, x)

            elif task_type == "do_move":
                # Assumes preconditions met or small wrapper
                total_cost += 2
                
            else:
                # Default small cost for unknown tasks
                total_cost += 1
                
        return total_cost