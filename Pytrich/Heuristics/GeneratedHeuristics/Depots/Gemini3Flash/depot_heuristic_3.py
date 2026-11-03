from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class DepotGemini3FlashHeuristic3(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v3 heuristic for Depots.
    Applies the successful Blocksworld stack-depth logic:
    - Penalizes depth of both source and destination in put_on.
    - Penalizes depth of crate in lift.
    """
    def __init__(self, name="DepotGemini3FlashHeuristic3"):
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
            predicate = parts[0].strip('+-')
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
            # do_put_on[crate, surface, truck, location]
            if "do_put_on" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "put_on", "crate": parts[1], "surface": parts[2]}
            # do_lift_crate[crate, surface, location]
            elif "do_lift_crate" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "lift", "crate": parts[1], "surface": parts[2]}

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
                if info["type"] == "put_on":
                    depth_x = self.get_stack_depth(info["crate"], state)
                    depth_y = self.get_stack_depth(info["surface"], state)
                    h_value += (depth_x + depth_y) * 5.0
                elif info["type"] == "lift":
                    h_value += self.get_stack_depth(info["crate"], state) * 5.0
        self.update_info(h_value)
        return h_value
