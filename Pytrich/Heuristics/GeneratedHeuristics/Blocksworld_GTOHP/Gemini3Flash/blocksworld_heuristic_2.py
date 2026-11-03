from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BlocksworldGemini3FlashHeuristic2(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v2 heuristic for Blocksworld.
    Adds Dependency-Aware Unstacking:
    - Penalizes placing a block on a target that is already a goal-base 
      for a different block.
    - Also includes v1 Stability and v0 Stack-Depth logic.
    """
    def __init__(self, name="BlocksworldGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} 
        self.goal_on = {} # top -> bottom
        self.is_goal_base = set() # set of blocks that are the bottom of some goal on(x, bottom)
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & On-Relation Map
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'): predicate = predicate[1:]
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id
            if predicate == "on" and len(args) == 2:
                top, bottom = args
                if bottom not in self.on_facts: self.on_facts[bottom] = []
                self.on_facts[bottom].append((top, fact.global_id))

        # 2. Extract Goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                target_fact = next((f for f in model.facts if f.global_id == fid), None)
                if target_fact:
                    parts = target_fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ").split()
                    if len(parts) >= 3:
                        pred = parts[0]
                        if pred.startswith('+') or pred.startswith('-'): pred = pred[1:]
                        if pred == "on":
                            self.goal_on[parts[1]] = parts[2]
                            self.is_goal_base.add(parts[2])
            goals_bitwise >>= 1
            fid += 1

        # 3. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            if "do_put_on" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {
                    "type": "put_on",
                    "block": parts[1],
                    "target": parts[2]
                }
            elif "do_clear" in task_type and len(parts) >= 2:
                self.optimistic_tasks[node.ID] = {
                    "type": "clear",
                    "block": parts[1]
                }
            elif "pickup" in task_type and len(parts) >= 2:
                self.optimistic_tasks[node.ID] = {
                    "type": "move",
                    "block": parts[1]
                }

        return h_value

    def is_stable(self, block, state):
        if block not in self.goal_on: return False
        target_bottom = self.goal_on[block]
        fid_on = self.fact_map.get(("on", (block, target_bottom)))
        if fid_on is None or not (state & (1 << fid_on)): return False
        if target_bottom.startswith('t'): return True
        return self.is_stable(target_bottom, state)

    def get_stack_depth(self, block, state):
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
                block = info["block"]
                
                # 1. Stabilization (v1)
                if self.is_stable(block, state):
                    h_value += 50.0

                # 2. Dependency Awareness (v2)
                if info["type"] == "put_on":
                    target = info["target"]
                    # If target is a goal-base for something and that something is NOT 'block'
                    if target in self.is_goal_base:
                        # Find who belongs on 'target' in goal state
                        goal_top = next((k for k, v in self.goal_on.items() if v == target), None)
                        if goal_top and goal_top != block:
                            h_value += 20.0 # Penalty for burying a goal base

                # 3. Stack Depth (v0)
                depth = self.get_stack_depth(block, state)
                h_value += depth * 2.0

        self.update_info(h_value)
        return h_value
