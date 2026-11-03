from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class TowersGemini3FlashHeuristic5(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v5 heuristic for Towers domain.
    Adds Smallest-Disk Bottleneck Management:
    - Prioritizes moving the smallest disk (r1) to clear destination towers.
    - Penalizes moving any large disk to a tower if r1 is currently blocking it.
    - Includes Global Target Alignment (v4) and Ring-Size constraints (v1-v3).
    """
    def __init__(self, name="TowersGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.fact_map = {}
        self.ring_size = {} 
        self.goal_on = {} 
        self.goal_tower = {} 
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            predicate = parts[0].strip('+-')
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id
            for arg in args:
                if arg.startswith('r') and arg[1:].isdigit():
                    self.ring_size[arg] = int(arg[1:])

        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                target_fact = next((f for f in model.facts if f.global_id == fid), None)
                if target_fact:
                    parts = target_fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ").split()
                    if len(parts) >= 3:
                        pred = parts[0].strip('+-')
                        if pred == "on":
                            ring, support = parts[1], parts[2]
                            self.goal_on[ring] = support
                            curr = support
                            while not curr.startswith('t') and curr in self.goal_on: curr = self.goal_on[curr]
                            if curr.startswith('t'): self.goal_tower[ring] = curr
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("[", " ").replace("]", " ").replace(",", " ")
            parts = name_clean.split()
            if "move" in parts:
                ring = None; dst_tower = None
                for p in parts:
                    if p.startswith('r') and p[1:].isdigit(): ring = p
                    if p.startswith('t') and p[1:].isdigit(): dst_tower = p
                if ring: self.optimistic_tasks[node.ID] = {"ring": ring, "dst": dst_tower}
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
        
        tower_tops = {}
        r1_loc = None
        for fact_key, fid in self.fact_map.items():
            if fact_key[0] == "towertop" and (state & (1 << fid)):
                tower_tops[fact_key[1][1]] = fact_key[1][0]
                if fact_key[1][0] == "r1": r1_loc = fact_key[1][1]

        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                ring, dst = info["ring"], info["dst"]
                
                # 1. Stability (v1)
                if self.is_stable(ring, state): h_value += 70.0
                
                # 2. Ring-Size (v1)
                if dst in tower_tops:
                    top_item = tower_tops[dst]
                    if top_item.startswith('r'):
                        if self.ring_size.get(ring, 0) > self.ring_size.get(top_item, 999):
                            h_value += 120.0 

                # 3. Smallest-Disk Bottleneck (v5 Enhancement)
                # If we are moving a ring other than r1, and r1 is at the destination, it MUST move first.
                if ring != "r1" and dst == r1_loc:
                    h_value += 50.0 

                # 4. Global Target Alignment (v4)
                if ring in self.goal_tower:
                    target_tower = self.goal_tower[ring]
                    if dst != target_tower:
                        # Optimization: if r1 is currently on target tower and shouldn't be, don't penalize detour as much
                        h_value += 40.0

        self.update_info(h_value)
        return h_value
