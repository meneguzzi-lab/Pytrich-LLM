from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class TowersGemini3FlashHeuristic2(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v2 heuristic for Towers domain.
    Strictly penalizes moving rings that are already in their goal configuration.
    """
    def __init__(self, name="TowersGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_on = {} # ring -> support
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
            self.fact_map[(predicate, args)] = fact.global_id

        # Extract goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                fact = next((f for f in model.facts if f.global_id == fid), None)
                if fact:
                    name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                    parts = name_clean.split()
                    if parts and parts[0].strip('+-') == "on" and len(parts) == 3:
                        self.goal_on[parts[1]] = parts[2]
            goals_bitwise >>= 1
            fid += 1

        # Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("[", " ").replace("]", " ").replace(",", " ")
            parts = name_clean.split()
            if "move" in parts:
                ring = None
                for p in parts:
                    if p.startswith('r') and p[1:].isdigit():
                        ring = p
                        break
                if ring:
                    self.optimistic_tasks[node.ID] = {"ring": ring}

        return h_value

    def is_stable(self, ring, state):
        current = ring
        while True:
            target_supp = self.goal_on.get(current)
            if not target_supp: return False
            fid = self.fact_map.get(("on", (current, target_supp)))
            if fid is None or not (state & (1 << fid)): return False
            if target_supp.startswith('t'): return True
            current = target_supp

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                ring = self.optimistic_tasks[task.global_id]["ring"]
                
                # Penalty if ring is already stable
                if self.is_stable(ring, state):
                    h_value += 100.0 # High penalty to prevent undoing goals

        self.update_info(h_value)
        return h_value
