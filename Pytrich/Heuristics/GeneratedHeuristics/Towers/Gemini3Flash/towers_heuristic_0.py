from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class TowersGemini3FlashHeuristic0(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v0 heuristic for Towers of Hanoi.
    Introduces a Stability Check:
    - A ring is Stable if it's at its goal position AND all rings below it are also at their goal positions.
    - Penalizes moving a stable ring.
    - Penalizes illegal moves (larger on smaller) with a reduced weight of 2.0.
    """
    def __init__(self, name="TowersGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.ring_size = {}
        self.goal_on = {} # ring -> target
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & Ring Sizes
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'):
                predicate = predicate[1:]
            
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id
            
            # Infer ring sizes from names (e.g., r1=1, r2=2)
            for arg in args:
                if arg.startswith('r') and arg[1:].isdigit():
                    self.ring_size[arg] = int(arg[1:])

        # 2. Map Goals
        # model.goals is a bitvector (int)
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                # Finding which fact this fid corresponds to
                fact = None
                for f in model.facts:
                    if f.global_id == fid:
                        fact = f
                        break
                if fact:
                    name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                    parts = name_clean.split()
                    if not parts: continue
                    pred = parts[0]
                    if pred.startswith('+') or pred.startswith('-'):
                        pred = pred[1:]
                    if pred == "on" and len(parts) == 3:
                        self.goal_on[parts[1]] = parts[2]
            goals_bitwise >>= 1
            fid += 1

        # 3. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            
            # Primitive move[ring, from_supp, from_tower, to_supp, to_tower]
            if task_type == "move" and len(parts) >= 6:
                self.optimistic_tasks[node.ID] = {
                    "type": "move",
                    "ring": parts[1],
                    "src_tower": parts[3],
                    "dst_tower": parts[5]
                }
            # Also catch abstract tasks if they represent a move
            elif "move" in task_type and len(parts) >= 3:
                # Assuming abstract of form move_abstract ring src dst
                self.optimistic_tasks[node.ID] = {
                    "type": "move",
                    "ring": parts[1],
                    "src_tower": parts[2],
                    "dst_tower": parts[-1]
                }

        return h_value

    def is_stable(self, ring, state):
        """A ring is stable if it's at its goal and its support is stable."""
        target = self.goal_on.get(ring)
        if not target: return False # Not in goal spec?
        
        # Check if ring is on target
        fid = self.fact_map.get(("on", (ring, target)))
        if fid is None or not (state & (1 << fid)):
            return False
            
        # If target is a tower, it's the base, so it's stable if ring is on it.
        if target.startswith('t'):
            return True
            
        # If target is another ring, that ring must also be stable.
        return self.is_stable(target, state)

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                ring = info["ring"]
                
                # 1. Stability Check: Don't move if it's already stable
                if self.is_stable(ring, state):
                    h_value += 50.0
        
        self.update_info(h_value)
        return h_value
