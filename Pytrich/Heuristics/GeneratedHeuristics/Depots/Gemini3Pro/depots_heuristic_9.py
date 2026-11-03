from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic9(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_9=None):
        super().__init__(model)
        self.depots_gemini3pro_heuristic_9 = depots_gemini3pro_heuristic_9

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

    def _get_object_state(self, state_repr):
        loc = {}
        supports = {}
        on_surface = {}
        in_truck = {}
        lifting = {}

        for fact in state_repr:
            pred, args = self._parse_fact_name(fact)
            if not pred:
                continue
            
            if pred == 'at':
                # at[obj, place]
                if len(args) == 2:
                    loc[args[0]] = args[1]
            elif pred == 'on':
                # on[crate, surface]
                if len(args) == 2:
                    c, s = args[0], args[1]
                    on_surface[c] = s
                    if s not in supports:
                        supports[s] = []
                    supports[s].append(c)
            elif pred == 'in':
                # in[crate, truck]
                if len(args) == 2:
                    in_truck[args[0]] = args[1]
            elif pred == 'lifting':
                # lifting[hoist, crate]
                if len(args) == 2:
                    lifting[args[0]] = args[1]
        
        return loc, supports, on_surface, in_truck, lifting

    def _get_real_location(self, obj, loc, in_truck, lifting):
        # 1. Directly at a place
        if obj in loc:
            return loc[obj]
        
        # 2. Inside a truck
        if obj in in_truck:
            truck = in_truck[obj]
            return loc.get(truck)
            
        # 3. Being lifted by a hoist
        # lifting maps hoist -> crate. Need crate -> hoist.
        # This scan is slow but safe for small lifting sets
        for h, c in lifting.items():
            if c == obj:
                return loc.get(h)
                
        return None

    def _count_above(self, obj, supports):
        # Count stack height above obj
        # In Depots, 'on' is usually 1-to-1 for crates, but supports is a list
        count = 0
        current = obj
        visited = {obj}
        
        # Simple traversal upwards
        while current in supports and supports[current]:
            # Take the first item (stack is linear)
            top = supports[current][0]
            if top in visited:
                break # Cycle protection
            visited.add(top)
            count += 1
            current = top
            
        return count

    def h(self, node):
        # 1. Reconstruct state graph from bitwise state representation
        state_repr = self.model.state_explicit_repr(node.state)
        loc, supports, on_surface, in_truck, lifting = self._get_object_state(state_repr)
        
        total_cost = 0
        
        # 2. Iterate through the task network
        for task in node.task_network:
            # Normalize task name for parsing: "do_put_on(c,s)" -> "do_put_on c s"
            clean_name = task.name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = clean_name.split()
            if not parts:
                continue
                
            task_type = parts[0]
            
            if task_type == 'do_put_on':
                # Expected: do_put_on crate surface
                if len(parts) >= 3:
                    crate = parts[1]
                    surface = parts[2]
                    
                    # Check if already satisfied
                    current_surface = on_surface.get(crate)
                    if current_surface == surface:
                        continue

                    # --- Cost Component 1: Clearing the destination surface ---
                    # If surface is a crate/pallet, we need to clear everything above it
                    cost_clear_s = self._count_above(surface, supports)
                    
                    # --- Cost Component 2: Clearing the crate itself ---
                    # If the crate is under other crates, they must be moved
                    cost_clear_c = self._count_above(crate, supports)
                    
                    # --- Cost Component 3: Transport ---
                    loc_c = self._get_real_location(crate, loc, in_truck, lifting)
                    loc_s = self._get_real_location(surface, loc, in_truck, lifting)
                    
                    transport_cost = 0
                    
                    if crate in in_truck:
                        # Already in truck: Drive (maybe) + Unload + Drop
                        # Assume minimal 2 ops (Unload, Drop)
                        transport_cost = 2
                        if loc_c and loc_s and loc_c != loc_s:
                             transport_cost += 2 # Drive cost estimate
                    elif loc_c and loc_s and loc_c != loc_s:
                        # Different locations: Load + Drive + Unload + Drop + (Clear ops)
                        # Base transport is Load(1) + Drive(1) + Unload(1) + Drop(1) = 4
                        transport_cost = 4
                    else:
                        # Same location: Lift + Drop = 2
                        transport_cost = 2

                    # Total: We multiply clearing costs by 2 (Lift + Drop to move obstacles aside)
                    # plus the transport of the target crate itself.
                    total_cost += (cost_clear_s * 2) + (cost_clear_c * 2) + transport_cost
            
            elif task_type == 'do_clear':
                # do_clear surface place
                if len(parts) >= 2:
                    surface = parts[1]
                    cost_clear = self._count_above(surface, supports)
                    total_cost += cost_clear * 2
            
            elif task_type == 'do_get_truck':
                # do_get_truck truck place
                if len(parts) >= 3:
                    truck = parts[1]
                    place = parts[2]
                    truck_loc = loc.get(truck)
                    if truck_loc and truck_loc != place:
                        total_cost += 1 # Drive
            
            elif task_type == 'do_load_truck':
                total_cost += 2 # Lift + Load roughly
            
            elif task_type == 'do_unload_truck':
                total_cost += 2 # Unload + Drop roughly
            
            elif task_type == 'do_lift_crate':
                total_cost += 1
                
            # Primitive actions cost 1
            elif task_type in ['Drive', 'Lift', 'Drop', 'Load', 'Unload']:
                total_cost += 1

        return total_cost
