from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class TowersGemini3FlashHeuristic4(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v4 heuristic for Towers domain.
    Adds Global Target Alignment and Greedy Selection:
    - Penalizes moves to non-goal towers if the goal tower is available.
    - Prioritizes placing rings on surfaces that are stable goal bases.
    - Includes Stability and Ring-Size constraints from v1-v3.
    """
    def __init__(self, name="TowersGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.fact_map = {}
        self.ring_size = {} 
        self.goal_on = {} 
        self.goal_tower = {} # ring -> tower name
        self.is_goal_base = set()
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & Ring Sizes
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'): predicate = predicate[1:]
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id
            for arg in args:
                if arg.startswith('r') and arg[1:].isdigit():
                    self.ring_size[arg] = int(arg[1:])

        # 2. Extract Goals and Target Towers
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
                            ring, support = parts[1], parts[2]
                            self.goal_on[ring] = support
                            self.is_goal_base.add(support)
                            # Find recursive tower
                            curr = support
                            while not curr.startswith('t') and curr in self.goal_on:
                                curr = self.goal_on[curr]
                            if curr.startswith('t'):
                                self.goal_tower[ring] = curr
            goals_bitwise >>= 1
            fid += 1

        # 3. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("[", " ").replace("]", " ").replace(",", " ")
            parts = name_clean.split()
            if "move" in parts:
                ring = None
                dst_tower = None
                for p in parts:
                    if p.startswith('r') and p[1:].isdigit(): ring = p
                    if p.startswith('t') and p[1:].isdigit(): dst_tower = p
                if ring:
                    self.optimistic_tasks[node.ID] = {"ring": ring, "dst": dst_tower}
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
        
        # Current top status
        tower_tops = {}
        for fact_key, fid in self.fact_map.items():
            if fact_key[0] == "towertop" and (state & (1 << fid)):
                tower_tops[fact_key[1][1]] = fact_key[1][0]

        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                ring = info["ring"]
                dst = info["dst"]
                
                # 1. Stability Check (v1)
                if self.is_stable(ring, state):
                    h_value += 60.0 # Preserve goals
                
                # 2. Ring-Size Consistency (v1)
                if dst in tower_tops:
                    top_item = tower_tops[dst]
                    if top_item.startswith('r'):
                        if self.ring_size.get(ring, 0) > self.ring_size.get(top_item, 999):
                            h_value += 100.0 # Force optimality/correctness

                # 3. Global Target Alignment (v4)
                if ring in self.goal_tower:
                    target_tower = self.goal_tower[ring]
                    if dst != target_tower:
                        # Only move to non-goal tower if goal tower is blocked by a SMALLER ring
                        # or if our intended foundation is not clear.
                        foundation = self.goal_on.get(ring)
                        foundation_ready = False
                        if foundation == target_tower:
                            if tower_tops.get(target_tower) == target_tower: foundation_ready = True
                        else:
                            # foundation is another ring
                            fid_foundation_top = self.fact_map.get(("onhead", (foundation,))) # actually clear[foundation]
                            # Simplified check: foundation is towertop? 
                            found_at_top = False
                            for t, r in tower_tops.items():
                                if r == foundation: found_at_top = True; break
                            if found_at_top: foundation_ready = True
                        
                        if foundation_ready:
                            h_value += 40.0 # Unnecessary detour to non-goal tower

        self.update_info(h_value)
        return h_value
