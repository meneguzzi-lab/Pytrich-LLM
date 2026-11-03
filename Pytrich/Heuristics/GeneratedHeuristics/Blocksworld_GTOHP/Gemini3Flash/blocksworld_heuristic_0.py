from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BlocksworldGemini3FlashHeuristic0(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v0 heuristic for Blocksworld.
    Uses proportional stack depth penalties for unstacking/clearing.
    """
    def __init__(self, name="BlocksworldGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} # bottom -> list of (top, fid)
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & On-Relation Map
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'):
                predicate = predicate[1:]
                
            args = tuple(parts[1:])
            fid = fact.global_id
            self.fact_map[(predicate, args)] = fid
            
            if predicate == "on" and len(args) == 2:
                top, bottom = args
                if bottom not in self.on_facts:
                    self.on_facts[bottom] = []
                self.on_facts[bottom].append((top, fid))

        # 2. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            
            if "do_put_on" in task_type and len(parts) >= 3:
                # do_put_on x y
                self.optimistic_tasks[node.ID] = {
                    "type": "put_on",
                    "x": parts[1],
                    "y": parts[2]
                }
            elif "do_clear" in task_type and len(parts) >= 2:
                # do_clear x
                self.optimistic_tasks[node.ID] = {
                    "type": "clear",
                    "x": parts[1]
                }
            elif "do_on_table" in task_type and len(parts) >= 2:
                # do_on_table x
                self.optimistic_tasks[node.ID] = {
                    "type": "on_table",
                    "x": parts[1]
                }

        return h_value

    def get_stack_depth(self, block, state):
        """Recursively count how many blocks are on top of 'block' in current 'state'."""
        depth = 0
        current = block
        visited = {block}
        while True:
            found_on = False
            possible_tops = self.on_facts.get(current, [])
            for top, fid in possible_tops:
                if state & (1 << fid):
                    if top in visited: break
                    depth += 1
                    current = top
                    visited.add(top)
                    found_on = True
                    break
            if not found_on: break
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
                    depth_x = self.get_stack_depth(info["x"], state)
                    depth_y = self.get_stack_depth(info["y"], state)
                    h_value += (depth_x + depth_y) * 2.0
                elif t_type == "clear":
                    depth_x = self.get_stack_depth(info["x"], state)
                    h_value += depth_x * 2.0
                elif t_type == "on_table":
                    depth_x = self.get_stack_depth(info["x"], state)
                    h_value += depth_x * 2.0

        self.update_info(h_value)
        return h_value
