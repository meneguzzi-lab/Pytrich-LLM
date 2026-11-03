from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class DepotGemini3FlashHeuristic1(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v1 heuristic for Depots.
    Increased stack depth weight (5.0) to match Blocksworld success.
    """
    def __init__(self, name="DepotGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} 
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
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
                if bottom not in self.on_facts: self.on_facts[bottom] = []
                self.on_facts[bottom].append((top, fid))

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) < 2: continue
            
            task_type = parts[0]
            if "do_put_on" in task_type and len(parts) > 2:
                self.optimistic_tasks[node.ID] = {"type": "put_on", "target": parts[2]}
            elif "do_load_truck" in task_type and len(parts) > 4:
                self.optimistic_tasks[node.ID] = {"type": "load_truck", "loc": parts[3], "truck": parts[4]}
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
                if t_type == "put_on":
                    h_value += self.get_stack_depth(info["target"], state) * 5.0 # Increased from 2.0
                elif t_type == "lift_crate":
                    h_value += self.get_stack_depth(info["crate"], state) * 5.0 # Increased from 2.0
                elif t_type == "load_truck":
                    fid_at = self.fact_map.get(("at", (info["truck"], info["loc"])))
                    if fid_at is not None and not (state & (1 << fid_at)):
                        h_value += 2.0 # Slightly increased
        self.update_info(h_value)
        return h_value
