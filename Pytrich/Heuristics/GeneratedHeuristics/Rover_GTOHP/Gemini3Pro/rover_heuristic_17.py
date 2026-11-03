import sys
from collections import defaultdict
import math

class Heuristic:
    def __init__(self, heuristic_name):
        self.heuristic_name = heuristic_name

    def h(self, model, state, task_network):
        raise NotImplementedError

class RoverGemini3ProHeuristic17(Heuristic):
    def __init__(self, rover_gemini3pro_heuristic_16=None):
        super().__init__("RoverGemini3ProHeuristic17")
        self.facts_by_name = {}
        self.graph = defaultdict(dict)
        self.waypoints = set()
        self.rovers = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()
        
        self.can_traverse_static = defaultdict(list)
        self.visible = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.on_board = {}
        self.calibration_target = {}
        self.supports = defaultdict(set)
        self.visible_from = defaultdict(set)
        self.lander_at = {}
        self.comm_points = set()
        
        self.task_cache = {}
        self.distances = {}
        self.initialized = False

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
        return name, []

    def init(self, model):
        for fact in model.facts:
            self.facts_by_name[fact.name] = fact
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == 'can_traverse':
                r, f, t = args
                self.rovers.add(r)
                self.waypoints.add(f)
                self.waypoints.add(t)
                self.can_traverse_static[f].append(t)
            elif pred == 'visible':
                p1, p2 = args
                self.visible.add((p1, p2))
            elif pred == 'at_lander':
                l, p = args
                self.lander_at[l] = p
                self.waypoints.add(p)
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'on_board':
                c, r = args
                self.on_board[c] = r
                self.cameras.add(c)
                self.rovers.add(r)
            elif pred == 'calibration_target':
                c, o = args
                self.calibration_target[c] = o
                self.cameras.add(c)
                self.objectives.add(o)
            elif pred == 'supports':
                c, m = args
                self.supports[c].add(m)
                self.cameras.add(c)
                self.modes.add(m)
            elif pred == 'visible_from':
                o, p = args
                self.visible_from[o].add(p)
                self.objectives.add(o)
                self.waypoints.add(p)
            elif pred == 'store_of':
                s, r = args
                self.stores.add(s)
                self.rovers.add(r)

        for l, l_loc in self.lander_at.items():
            for w in self.waypoints:
                if (w, l_loc) in self.visible or (l_loc, w) in self.visible:
                    self.comm_points.add(w)
            if (l_loc, l_loc) in self.visible:
                self.comm_points.add(l_loc)
            # Implicit visibility assumption: being at lander usually allows comms
            self.comm_points.add(l_loc)

        self._compute_distances()
        
        for task in model.abstract_tasks:
            pred, args = self._parse_fact_name(task.name)
            self.task_cache[task.global_id] = (pred, args)
            
        self.initialized = True

    def _compute_distances(self):
        wps = list(self.waypoints)
        inf = 9999
        for w1 in wps:
            self.distances[w1] = {w2: inf for w2 in wps}
            self.distances[w1][w1] = 0
            
        for f, targets in self.can_traverse_static.items():
            for t in targets:
                self.distances[f][t] = 1
        
        # Floyd-Warshall
        # Given graph size (up to ~150 nodes), O(V^3) ~ 3.3M ops is acceptable for init
        for k in wps:
            for i in wps:
                if self.distances[i][k] == inf: continue
                for j in wps:
                    if self.distances[k][j] == inf: continue
                    new_dist = self.distances[i][k] + self.distances[k][j]
                    if new_dist < self.distances[i][j]:
                        self.distances[i][j] = new_dist

    def h(self, model, state, task_network):
        if not self.initialized:
            self.init(model)
            
        rover_locs = {}
        store_full = set()
        have_soil = set()
        have_rock = set()
        have_image = set()
        calibrated = set()
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        
        true_facts = model.state_explicit_repr(state)
        
        for name in true_facts:
            p, args = self._parse_fact_name(name)
            if p == 'at':
                if len(args) == 2:
                    rover_locs[args[0]] = args[1]
            elif p == 'full':
                store_full.add(args[0])
            elif p == 'have_soil_analysis':
                have_soil.add((args[0], args[1]))
            elif p == 'have_rock_analysis':
                have_rock.add((args[0], args[1]))
            elif p == 'have_image':
                have_image.add((args[0], args[1], args[2]))
            elif p == 'calibrated':
                calibrated.add((args[0], args[1]))
            elif p == 'communicated_soil_data':
                communicated_soil.add(args[0])
            elif p == 'communicated_rock_data':
                communicated_rock.add(args[0])
            elif p == 'communicated_image_data':
                communicated_image.add((args[0], args[1]))

        total_cost = 0
        
        for task in task_network:
            if hasattr(task, 'add_effects'): # Operator
                total_cost += 1
                continue
                
            t_pred, t_args = self.task_cache.get(task.global_id, ('unknown', []))
            cost = 0
            
            if t_pred == 'do_navigate1' or t_pred == 'do_navigate2':
                if len(t_args) >= 2:
                    r, to = t_args[0], t_args[1]
                    if r in rover_locs:
                        fr = rover_locs[r]
                        dist = self.distances.get(fr, {}).get(to, 9999)
                        cost = dist
                    else:
                        cost = 100
                    
            elif t_pred == 'get_soil_data':
                wp = t_args[0]
                if wp in communicated_soil:
                    cost = 0
                else:
                    min_r_cost = 9999
                    for r in self.equipped_soil:
                        if r not in rover_locs: continue
                        curr = 0
                        if (r, wp) in have_soil:
                            d_comm = min([self.distances.get(rover_locs[r], {}).get(c, 9999) for c in self.comm_points] + [9999])
                            curr = d_comm + 1
                        else:
                            d_sample = self.distances.get(rover_locs[r], {}).get(wp, 9999)
                            d_comm = min([self.distances.get(wp, {}).get(c, 9999) for c in self.comm_points] + [9999])
                            curr = d_sample + 1 + d_comm + 1
                        if curr < min_r_cost:
                            min_r_cost = curr
                    cost = min_r_cost if min_r_cost != 9999 else 100

            elif t_pred == 'get_rock_data':
                wp = t_args[0]
                if wp in communicated_rock:
                    cost = 0
                else:
                    min_r_cost = 9999
                    for r in self.equipped_rock:
                        if r not in rover_locs: continue
                        curr = 0
                        if (r, wp) in have_rock:
                            d_comm = min([self.distances.get(rover_locs[r], {}).get(c, 9999) for c in self.comm_points] + [9999])
                            curr = d_comm + 1
                        else:
                            d_sample = self.distances.get(rover_locs[r], {}).get(wp, 9999)
                            d_comm = min([self.distances.get(wp, {}).get(c, 9999) for c in self.comm_points] + [9999])
                            curr = d_sample + 1 + d_comm + 1
                        if curr < min_r_cost:
                            min_r_cost = curr
                    cost = min_r_cost if min_r_cost != 9999 else 100

            elif t_pred == 'get_image_data':
                obj, mode = t_args[0], t_args[1]
                if (obj, mode) in communicated_image:
                    cost = 0
                else:
                    min_r_cost = 9999
                    valid_rovers = set()
                    for c, r in self.on_board.items():
                        if mode in self.supports[c]:
                            valid_rovers.add((r, c))
                    
                    for r, c in valid_rovers:
                        if r not in rover_locs: continue
                        curr = 0
                        if (r, obj, mode) in have_image:
                            d_comm = min([self.distances.get(rover_locs[r], {}).get(cp, 9999) for cp in self.comm_points] + [9999])
                            curr = d_comm + 1
                        else:
                            calib_cost = 0
                            curr_r_loc = rover_locs[r]
                            
                            if (c, r) not in calibrated:
                                target_obj = self.calibration_target.get(c)
                                if target_obj:
                                    view_wps = self.visible_from.get(target_obj, [])
                                    d_calib = min([self.distances.get(curr_r_loc, {}).get(wp, 9999) for wp in view_wps] + [9999])
                                    calib_cost = d_calib + 1 
                                else:
                                    calib_cost = 100
                            
                            view_wps = self.visible_from.get(obj, [])
                            best_path = 9999
                            for v in view_wps:
                                d_r_v = self.distances.get(curr_r_loc, {}).get(v, 9999)
                                d_v_c = min([self.distances.get(v, {}).get(cp, 9999) for cp in self.comm_points] + [9999])
                                path = d_r_v + 1 + d_v_c + 1
                                if path < best_path:
                                    best_path = path
                            
                            curr = calib_cost + best_path
                            
                        if curr < min_r_cost:
                            min_r_cost = curr
                    cost = min_r_cost if min_r_cost != 9999 else 100
            
            elif t_pred.startswith('send_'):
                if len(t_args) > 0:
                    r = t_args[0]
                    if r in rover_locs:
                        d_comm = min([self.distances.get(rover_locs[r], {}).get(c, 9999) for c in self.comm_points] + [9999])
                        cost = d_comm + 1
                    else:
                        cost = 100
            
            elif t_pred == 'do_calibrate':
                r, c = t_args[0], t_args[1]
                if (c, r) in calibrated:
                    cost = 0
                else:
                    if r in rover_locs:
                        target = self.calibration_target.get(c)
                        views = self.visible_from.get(target, [])
                        d = min([self.distances.get(rover_locs[r], {}).get(v, 9999) for v in views] + [9999])
                        cost = d + 1
                    else:
                        cost = 100

            total_cost += cost
            
        return total_cost
