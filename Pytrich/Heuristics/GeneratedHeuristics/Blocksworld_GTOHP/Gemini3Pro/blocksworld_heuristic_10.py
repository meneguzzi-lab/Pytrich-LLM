from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic10(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_7, model):
        super().__init__(blocksworld_gemini3pro_heuristic_7, model)
        
        # 1. Parse Facts and build Indices for fast O(1) lookups
        self.on_ids = {}      # (top, bottom) -> global_id
        self.ontable_ids = {} # block -> global_id
        self.clear_ids = {}   # block -> global_id
        self.holding_ids = {} # block -> global_id
        self.all_blocks = set()

        for fact in self.model.facts:
            pred, args = self._parse_fact_str(fact.name)
            if pred == 'on' and len(args) == 2:
                self.on_ids[(args[0], args[1])] = fact.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_ids[args[0]] = fact.global_id
                self.all_blocks.add(args[0])
            elif pred == 'clear' and len(args) == 1:
                self.clear_ids[args[0]] = fact.global_id
                self.all_blocks.add(args[0])
            elif pred == 'holding' and len(args) == 1:
                self.holding_ids[args[0]] = fact.global_id
                self.all_blocks.add(args[0])

        # 2. Parse Goal State
        # The goal is stored as a bitmask in model.goals
        # We need to parse this once to establish the target hierarchy
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self.goal_reqs = {} # block -> target (target is block_name or "TABLE")
        self.goal_blocks = set()
        
        for name in goal_fact_names:
            pred, args = self._parse_fact_str(name)
            if pred == 'on' and len(args) == 2:
                self.goal_reqs[args[0]] = args[1]
                self.goal_blocks.add(args[0])
            elif pred == 'ontable' and len(args) == 1:
                self.goal_reqs[args[0]] = "TABLE"
                self.goal_blocks.add(args[0])

        # 3. Build Evaluation Order (Topological Sort)
        # We want to process blocks bottom-up. If Goal is on(A, B), 
        # we must check B's status before A's. A depends on B.
        
        adj = {b: [] for b in self.goal_blocks}
        for b, tgt in self.goal_reqs.items():
            if tgt in self.goal_blocks:
                # tgt must be processed before b
                adj[tgt].append(b)
        
        # We want an order where dependencies come first. 
        # Actually, in the evaluation loop, we check: "Is `tgt` well placed?".
        # So we need to iterate `tgt` before `b`.
        # This corresponds to a topological sort where edges are Tgt -> B.
        # adj[tgt] contains B. This is correct.
        
        # Identify roots (blocks that want to be on TABLE or on non-goal blocks)
        in_degree = {b: 0 for b in self.goal_blocks}
        for u in adj:
            for v in adj[u]:
                in_degree[v] += 1
        
        queue = [b for b in self.goal_blocks if in_degree[b] == 0]
        self.eval_list = []
        
        while queue:
            u = queue.pop(0)
            self.eval_list.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # Handle any cycles or disconnected components (cycles shouldn't exist in valid BW goals)
        # If goal blocks remain, just append them (heuristic will just penalize them)
        for b in self.goal_blocks:
            if b not in self.eval_list:
                self.eval_list.append(b)
        
        # 4. Pre-compile check tuples for fast iteration in __call__
        self.compiled_checks = []
        for b in self.eval_list:
            target = self.goal_reqs[b]
            
            # Determine the specific bit ID to check for "is b on target?"
            check_id = -1
            if target == "TABLE":
                check_id = self.ontable_ids.get(b, -1)
            else:
                check_id = self.on_ids.get((b, target), -1)
            
            clear_id = self.clear_ids.get(b, -1)
            holding_id = self.holding_ids.get(b, -1)
            
            self.compiled_checks.append({
                'b': b,
                'tgt': target,
                'check_id': check_id,
                'clear_id': clear_id,
                'holding_id': holding_id
            })

        # Initial well_placed set
        # "TABLE" is always well placed.
        # Blocks NOT in the goal set are considered "well placed" foundations for the purpose
        # of admissibility (optimistic assumption that they don't need to move).
        self.base_well_placed = {"TABLE"}
        for b in self.all_blocks:
            if b not in self.goal_blocks:
                self.base_well_placed.add(b)

    def _parse_fact_str(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_start = name.find("[")
        if bracket_start == -1:
            return name, []
            
        pred = name[:bracket_start]
        args_part = name[bracket_start+1:-1]
        if not args_part:
            return pred, []
        return pred, [x.strip() for x in args_part.split(",")]

    def eval(self, node) -> float:
        state = node.state
        h_val = 0.0
        
        # Start with base set of well-placed items
        well_placed = self.base_well_placed.copy()
        
        # Iterate through goal blocks bottom-up
        for item in self.compiled_checks:
            b = item['b']
            tgt = item['tgt']
            check_id = item['check_id']
            
            # Condition 1: Is the support (tgt) well placed?
            if tgt in well_placed:
                # Condition 2: Is the block actually where it should be?
                if check_id != -1 and ((state >> check_id) & 1):
                    well_placed.add(b)
                    continue # Block is correct, no cost
            
            # If we reach here, the block is NOT well placed.
            # It contributes to the cost.
            
            # Base cost to move a block is 2 (pick up + put down)
            # If we are already holding it, cost is 1 (put down)
            holding_id = item['holding_id']
            is_holding = (holding_id != -1) and ((state >> holding_id) & 1)
            
            if is_holding:
                h_val += 1.0
            else:
                h_val += 2.0
                # Additional penalty if the block is buried (not clear)
                # A buried block requires at least one block above it to move.
                clear_id = item['clear_id']
                is_clear = (clear_id != -1) and ((state >> clear_id) & 1)
                
                if not is_clear:
                    h_val += 1.0
        
        return h_val
