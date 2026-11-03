from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class TowersGemini3FlashHeuristic1(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v1 heuristic for Towers domain.
    Includes:
    1. Stability Check (from v0).
    2. Illegal Move Check (Larger on Smaller).
    """
    def __init__(self, name="TowersGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.fact_map = {}
        self.ring_size = {} # ring name -> int
        self.goal_on = {} # ring -> support
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
            
            for arg in args:
                if arg.startswith('r') and arg[1:].isdigit():
                    self.ring_size[arg] = int(arg[1:])

        # 2. Extract Goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
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
            name_clean = node.str_name.lower().replace("[", " ").replace("]", " ").replace(",", " ")
            parts = name_clean.split()
            if "move" in parts:
                ring = None
                src_tower = None
                dst_tower = None
                # Traditional HTN name: move[ring, src_tower, dst_tower]
                for p in parts:
                    if p.startswith('r') and p[1:].isdigit(): ring = p
                    if p.startswith('t') and p[1:].isdigit():
                        if src_tower is None: src_tower = p
                        else: dst_tower = p
                
                if ring:
                    self.optimistic_tasks[node.ID] = {
                        "ring": ring,
                        "src": src_tower,
                        "dst": dst_tower
                    }

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
        
        # Pre-filter tower tops
        tower_tops = {} # tower -> ring
        for fact_key, fid in self.fact_map.items():
            if fact_key[0] == "towertop" and (state & (1 << fid)):
                item, tower = fact_key[1]
                if item.startswith('r'):
                    tower_tops[tower] = item

        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                ring = info["ring"]
                dst = info["dst"]
                
                # 1. Stability Check
                if self.is_stable(ring, state):
                    h_value += 50.0
                
                # 2. Illegal Move Check (Smaller on Larger)
                if dst in tower_tops:
                    top_ring = tower_tops[dst]
                    if self.ring_size.get(ring, 0) > self.ring_size.get(top_ring, 999):
                        h_value += 100.0

        self.update_info(h_value)
        return h_value
