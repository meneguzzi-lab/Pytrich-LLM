from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BlocksworldGemini3ProHeuristic2(TaskDecompositionHeuristic):
    """
    Round 2 state-aware heuristic for Blocksworld (v2).
    Uses proportional penalties based on the depth of blocks covering a target.
    Logic:
    - For do_put_on(x, y), do_on_table(x), do_clear(x):
        - Count how many blocks are currently 'on' x and y.
        - Add a penalty proportional to the number of blocks that must be moved.
    """
    def __init__(self, name="BlocksworldGemini3ProHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} # (bottom) -> list of (top, fid)
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & On-Relation Map
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
            predicate = parts[0]
            args = tuple(parts[1:])
            key = (predicate, args)
            self.fact_map[key] = fact.global_id
            
            if predicate == "on" and len(args) == 2:
                top, bottom = args
                if bottom not in self.on_facts:
                    self.on_facts[bottom] = []
                self.on_facts[bottom].append((top, fact.global_id))

        def get_fact_id(pred, args_tuple):
            key_std = (pred, args_tuple)
            key_plus = ("+" + pred, args_tuple)
            if key_plus in self.fact_map:
                return self.fact_map[key_plus]
            if key_std in self.fact_map:
                return self.fact_map[key_std]
            return None

        # 2. Identify Optimistic Tasks
        for node in self.and_or_graph.nodes:
            if node and node.content_type == ContentType.ABSTRACT_TASK:
                task_name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                parts = task_name_clean.split()
                if len(parts) < 2:
                    continue
                
                task_type = parts[0]
                
                # We target abstract tasks: do_put_on, do_on_table, do_clear
                if "do_put_on" in task_type and len(parts) > 2:
                    x, y = parts[1], parts[2]
                    self.optimistic_tasks[node.ID] = {
                        "type": "put_on",
                        "x": x,
                        "y": y,
                        "fid_holding_x": get_fact_id("holding", (x,)),
                        "fid_clear_y": get_fact_id("clear", (y,))
                    }
                elif "do_on_table" in task_type:
                    x = parts[1]
                    self.optimistic_tasks[node.ID] = {
                        "type": "on_table",
                        "x": x,
                        "fid_holding_x": get_fact_id("holding", (x,))
                    }
                elif "do_clear" in task_type:
                    x = parts[1]
                    self.optimistic_tasks[node.ID] = {
                        "type": "clear",
                        "x": x
                    }

        return h_value

    def get_stack_depth(self, block, state):
        """Recursively count how many blocks are on top of 'block' in current 'state'."""
        depth = 0
        current = block
        visited = {block}
        
        while True:
            found_on = False
            # Check if any block is on 'current'
            possible_tops = self.on_facts.get(current, [])
            for top, fid in possible_tops:
                if state & (1 << fid):
                    if top in visited: # Cycle protection
                        break
                    depth += 1
                    current = top
                    visited.add(top)
                    found_on = True
                    break
            if not found_on:
                break
        return depth

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                t_type = info["type"]
                
                if t_type == "put_on":
                    # Extra cost to clear x and y, and pickup x if not held
                    x, y = info["x"], info["y"]
                    
                    # 1. Depth of x (blocks above x)
                    depth_x = self.get_stack_depth(x, state)
                    # 2. Depth of y (blocks above y)
                    depth_y = self.get_stack_depth(y, state)
                    
                    # 3. Holding check
                    fid_holding = info["fid_holding_x"]
                    is_holding = fid_holding is not None and (state & (1 << fid_holding))
                    
                    h_value += (depth_x + depth_y) * 2.0 # Each block above needs move (unstack + putdown/stack)
                    if not is_holding:
                        h_value += 1.0 # Pickup cost
                        
                elif t_type == "on_table" or t_type == "clear":
                    x = info["x"]
                    depth_x = self.get_stack_depth(x, state)
                    h_value += depth_x * 2.0
                    
                    if t_type == "on_table":
                        fid_holding = info.get("fid_holding_x")
                        is_holding = fid_holding is not None and (state & (1 << fid_holding))
                        if not is_holding:
                            h_value += 1.0

        self.update_info(h_value)
        return h_value
