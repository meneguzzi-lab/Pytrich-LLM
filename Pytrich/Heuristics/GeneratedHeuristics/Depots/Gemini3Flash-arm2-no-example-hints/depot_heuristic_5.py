from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS:
# The primary bottleneck in the Depots domain is the hierarchical stack of crates on surfaces (pallets or other crates).
# Moving a crate buried in a stack requires a sequence of 'do_clear' tasks, which translates to multiple 
# Lift/Load/Unload/Drop actions. Additionally, the spatial constraint of trucks moving between depots/distributors
# requires Drive actions, creating a Logistics-style cost layer on top of the Blocksworld-style stacking.

# ADMISSIBILITY SKETCHES:
# 1. Goal-Based Lower Bound (goal_h): For every goal (on ?c ?s) not yet satisfied, we must perform at least
#    a Lift and a Drop (cost 2). If the crate is at a different location, it requires Lift, Load, Drive, Unload, 
#    and Drop (cost 5). This is a lower bound because it ignores hoist availability and other crate interference.
# 2. Task-Network Lower Bound (tn_h): Every 'do_put_on' task in the task network represents a required movement.
#    Using the same costing logic as goal_h, we sum these minimum costs. This is admissible as these tasks
#    are explicitly required by the HTN decomposition.
# 3. Clearance Lower Bound (clear_h): If a target surface for a goal or task is not 'clear', at least 2 actions
#    (Lift and Load/Drop) are required to move the obstructing crate. 

class DepotsGemini3FlashHeuristic5(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_5="depot_gemini3flash_heuristic_5"):
        super().__init__(name=depot_gemini3flash_heuristic_5)
        self.at_map = {} # (locatable, place) -> fact_id
        self.on_map = {} # (crate, surface) -> fact_id
        self.in_map = {} # (crate, truck) -> fact_id
        self.clear_map = {} # surface -> fact_id
        self.lifting_map = {} # (hoist, crate) -> fact_id
        
        self.goal_on = [] # List of (crate, surface)
        self.surface_to_place = {}
        self.trucks = set()
        self.places = set()
        self.crates = set()
        self.hoists = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to global IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                self.at_map[(args[0], args[1])] = fact.global_id
                self.places.add(args[1])
            elif pred == "on":
                self.on_map[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
            elif pred == "in":
                self.in_map[(args[0], args[1])] = fact.global_id
                self.trucks.add(args[1])
            elif pred == "clear":
                self.clear_map[args[0]] = fact.global_id
            elif pred == "lifting":
                self.lifting_map[(args[0], args[1])] = fact.global_id
                self.hoists.add(args[0])

        # 2. Extract static surface-to-place mappings from operators
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            # Lift/Drop/Load/Unload all contain location/hoist info
            if pred in ["Lift", "Drop"]:
                # args: [hoist, crate, surface, place]
                self.surface_to_place[args[2]] = args[3]
                self.at_map[(args[0], args[3])] = self.at_map.get((args[0], args[3]), -1)

        # 3. Identify goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on.append((args[0], args[1]))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _get_place(self, locatable, state):
        # Check direct 'at'
        for p in self.places:
            fid = self.at_map.get((locatable, p))
            if fid is not None and fid != -1 and (state >> fid) & 1:
                return p
        # Check if in a truck
        for t in self.trucks:
            fid = self.in_map.get((locatable, t))
            if fid is not None and (state >> fid) & 1:
                # Crate is in truck, location is truck's location
                return self._get_place(t, state)
        # Check if being lifted
        for h in self.hoists:
            fid = self.lifting_map.get((h, locatable))
            if fid is not None and (state >> fid) & 1:
                return self._get_place(h, state)
        return None

    def _get_clear_penalty(self, surface, state):
        fid = self.clear_map.get(surface)
        if fid is not None and (state >> fid) & 1:
            return 0
        return 2 # Minimum Lift + Drop/Load to clear an obstruction

    def _compute_task_cost(self, crate, target_surface, state):
        # Target check
        fid_on = self.on_map.get((crate, target_surface))
        if fid_on is not None and (state >> fid_on) & 1:
            return 0

        place_c = self._get_place(crate, state)
        place_s = self.surface_to_place.get(target_surface)

        if place_c is None or place_s is None:
            return 2 # Default min move cost
        
        cost = 0
        if place_c == place_s:
            cost = 2 # Lift + Drop
        else:
            cost = 5 # Lift + Load + Drive + Unload + Drop
        
        return cost + self._get_clear_penalty(target_surface, state)

    def _compute(self, state, task_network):
        # Term 1: Goal-based progress
        goal_h = 0
        for crate, surface in self.goal_on:
            goal_h += self._compute_task_cost(crate, surface, state)

        # Term 2: Task-network-based requirements
        tn_h = 0
        for task in task_network:
            if hasattr(task, 'name'):
                # Format often: do_put_on[crate, surface]
                name = task.name
                if "do_put_on" in name:
                    # Heuristic parsing of task string
                    parts = name.replace("[", ",").replace("]", "").split(",")
                    if len(parts) >= 3:
                        c, s = parts[1].strip(), parts[2].strip()
                        tn_h += self._compute_task_cost(c, s, state)
                elif "do_clear" in name:
                    parts = name.replace("[", ",").replace("]", "").split(",")
                    if len(parts) >= 2:
                        s = parts[1].strip()
                        tn_h += self._get_clear_penalty(s, state)

        # Return max for admissibility across different components
        return max(goal_h, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
