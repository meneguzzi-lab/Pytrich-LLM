from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class DepotGemini3FlashHeuristic5(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v5 heuristic for Depots.
    Adds Truck-Crate Affinity (Load Synchronization):
    - Penalizes truck moving AWAY from a location if there are goal-crates 
      available to be loaded there.
    - Penalizes truck moves if it is currently carrying a crate to its goal location.
    - Combines Recursive Stability and Stack Depth logic.
    """
    def __init__(self, name="DepotGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.fact_map = {}
        self.on_facts = {} 
        self.goal_on = {} 
        self.goal_at = {} 
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

        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                target_fact = next((f for f in model.facts if f.global_id == fid), None)
                if target_fact:
                    parts = target_fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ").split()
                    if len(parts) >= 3:
                        pred = parts[0].strip('+-')
                        if pred == "on": self.goal_on[parts[1]] = parts[2]
                        if pred == "at": self.goal_at[parts[1]] = parts[2]
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) < 2: continue
            task_type = parts[0]
            if "do_put_on" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "put_on", "crate": parts[1], "surface": parts[2]}
            elif "do_lift_crate" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "lift", "crate": parts[1], "surface": parts[2]}
            elif "do_move_truck" in task_type and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {"type": "move_truck", "truck": parts[1], "src": parts[2], "dst": parts[3]}
        return h_value

    def is_stable(self, crate, state):
        if crate not in self.goal_on: return False
        target_supp = self.goal_on[crate]
        fid_on = self.fact_map.get(("on", (crate, target_supp)))
        if fid_on is None or not (state & (1 << fid_on)): return False
        if target_supp.startswith('p') or target_supp.startswith('distributor'): return True
        return self.is_stable(target_supp, state)

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        truck_cargo = {}
        truck_at = {}
        for fact_key, fid in self.fact_map.items():
            if fact_key[0] == "in" and (state & (1 << fid)):
                truck_cargo[fact_key[1][1]] = fact_key[1][0]
            if fact_key[0] == "at" and (state & (1 << fid)) and fact_key[1][0].startswith('truck'):
                truck_at[fact_key[1][0]] = fact_key[1][1]

        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                t_type = info["type"]
                
                if t_type == "move_truck":
                    truck, src, dst = info["truck"], info["src"], info["dst"]
                    
                    # 1. Truck-Crate Affinity (v5 Enhancement)
                    # If truck is at src, and there is a crate at src that needs to go to dst (or anywhere else), stay!
                    for crate, target_loc in self.goal_at.items():
                        at_fid = self.fact_map.get(("at", (crate, src)))
                        if at_fid is not None and (state & (1 << at_fid)):
                            if target_loc != src:
                                h_value += 50.0 # Don't leave if there are crates to pick up
                                break
                    
                    # Already carrying goal?
                    if truck in truck_cargo:
                        cargo = truck_cargo[truck]
                        goal_loc = self.goal_at.get(cargo)
                        if goal_loc and dst != goal_loc: h_value += 50.0 
                        
                elif t_type == "lift":
                    if self.is_stable(info["crate"], state): h_value += 70.0

        self.update_info(h_value)
        return h_value
