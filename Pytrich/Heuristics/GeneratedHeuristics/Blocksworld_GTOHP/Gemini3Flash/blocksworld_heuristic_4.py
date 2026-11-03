from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BlocksworldGemini3FlashHeuristic4(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v4 heuristic for Blocksworld.
    Adds Support-Aware Efficiency and Mandatory Unstacking:
    - Penalizes any move(A, Target) if Target is NOT the goal-support.
    - Penalizes moving block A if its goal-support B is NOT yet stable.
    - Includes v3 Deadlock Detection, v2 Dependency Awareness, 
      v1 Stability, and v0 Stack-Depth logic.
    """
    def __init__(self, name="BlocksworldGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} 
        self.goal_on = {} 
        self.is_goal_base = set()
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

    def is_above(self, block_a, block_b, state):
        current = block_b
        visited = {block_b}
        while True:
            found_on = False
            possible_tops = self.on_facts.get(current, [])
            for top, fid in possible_tops:
                if state & (1 << fid):
                    if top == block_a: return True
                    if top in visited: break
                    current = top
                    visited.add(top)
                    found_on = True
                    break
            if not found_on: break
        return False

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
                    h_value += 100.0 # Increased penalty to strictly preserve goals

                # 2. Deadlock Detection (v3)
                if block in self.goal_on:
                    target_support = self.goal_on[block]
                    if not target_support.startswith('t'):
                        if self.is_above(target_support, block, state):
                            h_value += 200.0 # Circularity is very bad

                # 3. Support-Aware Efficiency (v4)
                if info["type"] == "put_on":
                    target = info["target"]
                    
                    # If we belong on target in goal, but target isn't stable, don't go there yet!
                    if block in self.goal_on and self.goal_on[block] == target:
                        if not target.startswith('t') and not self.is_stable(target, state):
                            h_value += 40.0 # Target isn't ready
                    else:
                        # Penalize arbitrary stacking (if not goal support and not table)
                        if not target.startswith('t'):
                            h_value += 30.0
                    
                    # Dependency Awareness (v2)
                    if target in self.is_goal_base:
                        goal_top = next((k for k, v in self.goal_on.items() if v == target), None)
                        if goal_top and goal_top != block:
                            h_value += 50.0 # Burial is more penalized

                # 4. Stack Depth (v0)
                depth = self.get_stack_depth(block, state)
                h_value += depth * 4.0 # Weighted more heavily to force unstacking

        self.update_info(h_value)
        return h_value
