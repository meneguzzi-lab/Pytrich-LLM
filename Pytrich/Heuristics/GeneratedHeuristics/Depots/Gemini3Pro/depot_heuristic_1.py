from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class DepotGemini3ProHeuristic1(TaskDecompositionHeuristic):
    """
    Round 2 state-aware heuristic for Depots (v1).
    Penalizes:
    1. DO_LIFT_CRATE if the crate is covered by other crates (depth penalty).
    2. DO_PUT_ON if the target surface/crate is covered.
    3. DO_LOAD_TRUCK if the truck is at a different location.
    """
    def __init__(self, name="DepotGemini3ProHeuristic1"):
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
                
                if "do_put_on" in task_type and len(parts) > 2:
                    # do_put_on crate target
                    self.optimistic_tasks[node.ID] = {
                        "type": "put_on",
                        "target": parts[2]
                    }
                elif "do_load_truck" in task_type and len(parts) > 4:
                    # do_load_truck crate surface loc truck
                    self.optimistic_tasks[node.ID] = {
                        "type": "load_truck",
                        "loc": parts[3],
                        "truck": parts[4]
                    }
                elif "do_lift_crate" in task_type and len(parts) > 1:
                    # do_lift_crate crate ...
                    self.optimistic_tasks[node.ID] = {
                        "type": "lift_crate",
                        "crate": parts[1]
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
                    depth = self.get_stack_depth(info["target"], state)
                    h_value += depth * 2.0
                elif t_type == "lift_crate":
                    depth = self.get_stack_depth(info["crate"], state)
                    h_value += depth * 2.0
                elif t_type == "load_truck":
                    fid_at = self.fact_map.get(("at", (info["truck"], info["loc"])))
                    if fid_at is not None and not (state & (1 << fid_at)):
                        h_value += 1.0
        self.update_info(h_value)
        return h_value
