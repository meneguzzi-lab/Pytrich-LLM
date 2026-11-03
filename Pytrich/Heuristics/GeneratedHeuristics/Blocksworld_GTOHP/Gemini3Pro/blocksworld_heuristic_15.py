from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic15(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_12, model):
        super().__init__(blocksworld_gemini3pro_heuristic_12, model)
        self.model = model
        self.obj_to_id = {}
        self.on_facts = {}
        self.clear_facts = {}
        self.ontable_facts = {}
        self.holding_facts = {}
        self.handempty_id = -1
        self.supporters = {}
        self.task_args_cache = {}

        self._initialize_mappings()

    def _initialize_mappings(self):
        # Parse all facts to build efficient lookups
        for fact in self.model.facts:
            name = fact.name.strip()
            if name.startswith("+"): name = name[1:]
            
            pred = None
            args = []
            
            # Extract predicate and arguments from format like predicate[arg1,arg2]
            if "[" in name and name.endswith("]"):
                pred, args_str = name[:-1].split("[", 1)
                args = args_str.split(",") if args_str else []
            else:
                pred = name
            
            # Map string arguments to integer IDs
            arg_ids = []
            for a in args:
                if a not in self.obj_to_id:
                    self.obj_to_id[a] = len(self.obj_to_id)
                arg_ids.append(self.obj_to_id[a])
                
            gid = fact.global_id
            
            if pred == "on" and len(arg_ids) == 2:
                top, bot = arg_ids
                self.on_facts[(top, bot)] = gid
                if bot not in self.supporters: 
                    self.supporters[bot] = []
                self.supporters[bot].append((top, gid))
                
            elif pred == "clear" and len(arg_ids) == 1:
                self.clear_facts[arg_ids[0]] = gid
                
            elif pred == "ontable" and len(arg_ids) == 1:
                self.ontable_facts[arg_ids[0]] = gid
                
            elif pred == "holding" and len(arg_ids) == 1:
                self.holding_facts[arg_ids[0]] = gid
                
            elif pred == "handempty":
                self.handempty_id = gid

    def get_task_args(self, task_name):
        # Cache parsed task arguments to avoid repeated string processing
        if task_name in self.task_args_cache:
            return self.task_args_cache[task_name]
        
        # Clean and parse: "(task_name arg1 arg2)" -> type, [arg_ids]
        clean = task_name.replace("(", " ").replace(")", " ").strip()
        parts = clean.split()
        if not parts:
            return None, []
        
        t_type = parts[0]
        arg_ids = []
        for p in parts[1:]:
            if p in self.obj_to_id:
                arg_ids.append(self.obj_to_id[p])
        
        self.task_args_cache[task_name] = (t_type, arg_ids)
        return t_type, arg_ids

    def get_blocks_above_count(self, block_id, state):
        # Returns the number of blocks stacked above block_id
        # Uses iterative traversal to avoid recursion limits on deep stacks
        count = 0
        curr = block_id
        visited = {curr}
        
        while True:
            # If block is clear, no more blocks above
            if curr in self.clear_facts:
                if (state >> self.clear_facts[curr]) & 1:
                    break
            
            # Find the block directly on top of curr
            found = False
            if curr in self.supporters:
                for top, gid in self.supporters[curr]:
                    if (state >> gid) & 1:
                        if top in visited: 
                            break # Cycle protection
                        visited.add(top)
                        count += 1
                        curr = top
                        found = True
                        break
            if not found:
                break
                
        return count

    def compute(self, node, task_network, **kwargs):
        state = node.state
        cost = 0
        
        # Estimate cost for each task in the network
        for task in task_network:
            t_type, args = self.get_task_args(task.name)
            
            if t_type == "do_put_on" and len(args) == 2:
                x, y = args
                # If already on(x, y), cost is 0
                if (x, y) in self.on_facts and ((state >> self.on_facts[(x, y)]) & 1):
                    continue
                
                # Cost to make X held
                is_holding = (x in self.holding_facts) and ((state >> self.holding_facts[x]) & 1)
                if not is_holding:
                    # Clear X (unstack + putdown = 2 ops per block above)
                    cost += self.get_blocks_above_count(x, state) * 2 
                    # Pick up X (1 op)
                    cost += 1
                
                # Cost to clear Y (unstack + putdown per block)
                cost += self.get_blocks_above_count(y, state) * 2
                
                # Stack X on Y (1 op)
                cost += 1
                
            elif t_type == "do_on_table" and len(args) == 1:
                x = args[0]
                if (x in self.ontable_facts) and ((state >> self.ontable_facts[x]) & 1):
                    continue
                
                is_holding = (x in self.holding_facts) and ((state >> self.holding_facts[x]) & 1)
                if not is_holding:
                    cost += self.get_blocks_above_count(x, state) * 2
                    cost += 1 # Unstack/Pick-up
                
                cost += 1 # Put-down
                
            elif t_type == "do_clear" and len(args) == 1:
                x = args[0]
                # Just clear above X
                cost += self.get_blocks_above_count(x, state) * 2
            
            elif t_type == "do_move" and len(args) == 2:
                # Assumes do_move x y requires both x and y clear
                x, y = args
                is_holding = (x in self.holding_facts) and ((state >> self.holding_facts[x]) & 1)
                if not is_holding:
                     cost += self.get_blocks_above_count(x, state) * 2
                     cost += 1
                cost += self.get_blocks_above_count(y, state) * 2
                cost += 1
            
            # Fallback for primitive actions if they appear in the task network
            elif t_type in ["stack", "unstack", "pick-up", "put-down"]:
                cost += 1
                
        return cost
