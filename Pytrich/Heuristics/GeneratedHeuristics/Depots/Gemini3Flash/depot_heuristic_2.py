from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class DepotGemini3FlashHeuristic2(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v2 heuristic for Depots.
    Goal-Stack Focus:
    - Identifies crates involved in 'on' goals.
    - Applies stack depth penalties primarily to goal-related crates to avoid unnecessary clearing.
    """
    def __init__(self, name="DepotGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} 
        self.goal_items = set()
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            predicate = parts[0].strip('+-')
            args = tuple(parts[1:])
            fid = fact.global_id
            self.fact_map[(predicate, args)] = fid
            
            if predicate == "on" and len(args) == 2:
                top, bottom = args
                if bottom not in self.on_facts: self.on_facts[bottom] = []
                self.on_facts[bottom].append((top, fid))

        # Extract goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                fact = next((f for f in model.facts if f.global_id == fid), None)
                if fact:
                    name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                    parts = name_clean.split()
                    if len(parts) >= 2:
                        for p in parts[1:]:
                            self.goal_items.add(p)
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) < 2: continue
            
            task_type = parts[0]
            if "do_put_on" in task_type and len(parts) > 2:
                self.optimistic_tasks[node.ID] = {"type": "put_on", "target": parts[2]}
            elif "do_lift_crate" in task_type and len(parts) > 1:
                self.optimistic_tasks[node.ID] = {"type": "lift_crate", "crate": parts[1]}

        return h_value

    def get_stack_depth(self, item, state):
        depth = 0
        current = item
        visited = {item}
        while True:
            found = False
            for top, fid in self.on_facts.get(current, []):
                if state & (1 << fid):
                    if top in visited: break
                    depth += 1
                    current = top
                    visited.add(top)
                    found = True
                    break
            if not found: break
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
                item = info.get("target") or info.get("crate")
                
                # Only apply depth penalty if the item is goal-related or contains goal items on top
                depth = self.get_stack_depth(item, state)
                if item in self.goal_items or depth > 0:
                    h_value += depth * 5.0
        
        self.update_info(h_value)
        return h_value
