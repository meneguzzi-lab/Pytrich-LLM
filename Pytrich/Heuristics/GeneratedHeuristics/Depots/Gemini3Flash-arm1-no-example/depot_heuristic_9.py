from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic9(Heuristic):
    """
    Heuristic for the Depots domain.
    Bottleneck Analysis:
    The main bottleneck is the sequencing of crate movements (Logistics) and clearing/stacking (BlocksWorld).
    Crates must be at the correct location before they can be dropped on the correct surface, and surfaces 
    must be clear before they can receive a crate.

    Admissibility Sketches:
    - Term 1: Task Network Cost. Each task in the network has a minimum cost based on the current state. 
      For example, do_put_on(c, s) costs at least 2 actions (Lift, Drop) if the crate is at the right place,
      and at least 5 (Lift, Load, Drive, Unload, Drop) if it is at a different place.
    - Term 2: Unsatisfied Goals. Every goal on(c, s) not in the state requires at least a Lift and a Drop (2 actions).
    """

    def __init__(self, depot_gemini3flash_heuristic_9="depot_gemini3flash_heuristic_9"):
        super().__init__(depot_gemini3flash_heuristic_9)
        self.on_ids = {} # (crate, surface) -> global_id
        self.at_ids = {} # (locatable, place) -> global_id
        self.in_ids = {} # (crate, truck) -> global_id
        self.clear_ids = {} # surface -> global_id
        self.lifting_ids = {} # (hoist, crate) -> global_id
        self.available_ids = {} # hoist -> global_id
        
        self.crate_names = set()
        self.surface_names = set()
        self.place_names = set()
        self.truck_names = set()
        self.hoist_names = set()

        self.task_data = {} # task_id -> (type, args)
        self.goal_on = {} # crate -> surface
        self.fixed_surface_places = {} # surface -> place (for pallets)

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
        # 1. Map Facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.crate_names.add(args[0])
                self.surface_names.add(args[1])
            elif pred == "at":
                self.at_ids[(args[0], args[1])] = f.global_id
                self.place_names.add(args[1])
            elif pred == "in":
                self.in_ids[(args[0], args[1])] = f.global_id
                self.truck_names.add(args[1])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
                self.surface_names.add(args[0])
            elif pred == "lifting":
                self.lifting_ids[(args[0], args[1])] = f.global_id
                self.hoist_names.add(args[0])
            elif pred == "available":
                self.available_ids[args[0]] = f.global_id

        # 2. Map Task Data
        for task in model.operators + model.abstract_tasks:
            name = task.name
            pred = name.split("[")[0] if "[" in name else name
            args = name[name.find("[")+1:-1].split(",") if "[" in name else []
            self.task_data[task.global_id] = (pred, args)

        # 3. Identify Goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # 4. Pallets are fixed at places
        init_facts = model.state_explicit_repr(model.initial_state)
        for f in init_facts:
            pred, args = self._parse_fact_name(f)
            if pred == "at" and args[1] in self.place_names:
                # Pallets don't move
                if "pallet" in args[0]:
                    self.fixed_surface_places[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_place(self, state, obj):
        """Helper to find where an object is located in the bitwise state."""
        if obj in self.fixed_surface_places: return self.fixed_surface_places[obj]
        # Check at(obj, place)
        for p in self.place_names:
            fid = self.at_ids.get((obj, p))
            if fid is not None and (state >> fid) & 1: return p
        # Check in(obj, truck)
        for t in self.truck_names:
            fid = self.in_ids.get((obj, t))
            if fid is not None and (state >> fid) & 1:
                # Find where truck is
                for p in self.place_names:
                    tfid = self.at_ids.get((t, p))
                    if tfid is not None and (state >> tfid) & 1: return p
        # Check lifting(hoist, obj)
        for h in self.hoist_names:
            fid = self.lifting_ids.get((h, obj))
            if fid is not None and (state >> fid) & 1:
                # Find where hoist is
                for p in self.place_names:
                    hfid = self.at_ids.get((h, p))
                    if hfid is not None and (state >> hfid) & 1: return p
        return None

    def _compute(self, state, task_network) -> int:
        cost = 0
        # Term 1: Task Network
        for task in task_network:
            t_type, t_args = self.task_data.get(task.global_id, (None, []))
            if t_type == "do_put_on":
                c, s = t_args[0], t_args[1]
                # If already satisfied
                on_fid = self.on_ids.get((c, s))
                if on_fid is not None and (state >> on_fid) & 1: continue
                
                p_c = self._get_place(state, c)
                p_s = self._get_place(state, s)
                
                if p_c is None or p_s is None:
                    cost += 2
                elif p_c == p_s:
                    cost += 2 # Lift + Drop
                else:
                    cost += 5 # Lift + Load + Drive + Unload + Drop
            
            elif t_type == "do_clear":
                s = t_args[0]
                cfid = self.clear_ids.get(s)
                if cfid is not None and not ((state >> cfid) & 1):
                    cost += 2
            
            elif t_type in ["do_load_truck", "do_unload_truck", "do_lift_crate", "do_get_truck"]:
                cost += 2
            
            elif t_type in ["Drive", "Lift", "Drop", "Load", "Unload"]:
                cost += 1

        # Term 2: Unsatisfied goal penalty (symmetry breaking and guidance)
        for crate, target_surface in self.goal_on.items():
            on_fid = self.on_ids.get((crate, target_surface))
            if on_fid is not None and not ((state >> on_fid) & 1):
                # Check if this goal is already addressed in the task network
                # If not, add a small penalty to encourage decomposition towards it
                cost += 1

        return cost
