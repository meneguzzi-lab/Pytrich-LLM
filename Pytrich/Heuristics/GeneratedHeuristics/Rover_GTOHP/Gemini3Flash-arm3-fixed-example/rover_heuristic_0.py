import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.

    BOTTLENECK:
    The primary bottleneck is the rover navigation and coordination between sampling locations 
    and communication points (locations visible to landers), while managing equipment requirements.

    ADMISSIBILITY SKETCHES:
    1. Navigation Lower Bound: Shortest path distance (number of navigate actions) in the traversal 
       graph required for rovers to reach target waypoints is a strict lower bound on cost.
    2. Task Action Complexity: Every data goal (soil, rock, image) requires at least one sampling/imaging 
       action and one communication action. This provides a clear minimum decomposition cost.
    3. Store/Equipment Constraints: If a rover's store is full, an additional 'drop' action is required 
       before sampling, increasing the unavoidable action count.
    """

    def __init__(self, rover_gemini3flash_heuristic_0=None, name="RoverGTOHPGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        
        self.at_fact_ids = {}            # rover -> waypoint -> global_id
        self.have_soil_ids = {}          # rover -> waypoint -> global_id
        self.have_rock_ids = {}          # rover -> waypoint -> global_id
        self.have_image_ids = {}         # rover -> (obj, mode) -> global_id
        self.store_full_ids = {}         # rover -> global_id
        
        self.goal_soil_ids = {}          # waypoint -> global_id
        self.goal_rock_ids = {}          # waypoint -> global_id
        self.goal_image_ids = {}         # (obj, mode) -> global_id

        self.adj = {}                    # rover -> waypoint -> list(waypoints)
        self.dist = {}                   # rover -> waypoint -> waypoint -> int
        self.comm_waypoints = {}         # rover -> set(waypoints)
        self.min_dist_to_comm = {}       # rover -> waypoint -> int

        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.obj_visible_from = {}       # objective -> list(waypoints)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if bracket_idx < len(name)-2 else []
                return predicate, args
        return None, []

    def _preprocess(self):
        lander_positions = {}
        visibilities = {}
        traversals = {}
        rover_stores = {}
        store_to_rover = {}

        # First pass to identify objects and static relations
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at_lander":
                self.landers.add(args[0])
                self.waypoints.add(args[1])
                lander_positions[args[0]] = args[1]
            elif pred == "at":
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "can_traverse":
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                self.waypoints.add(args[2])
                if args[0] not in traversals: traversals[args[0]] = []
                traversals[args[0]].append((args[1], args[2]))
            elif pred == "visible":
                self.waypoints.add(args[0])
                self.waypoints.add(args[1])
                if args[0] not in visibilities: visibilities[args[0]] = []
                visibilities[args[0]].append(args[1])
            elif pred == "equipped_for_soil_analysis": self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis": self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging": self.equipped_image.add(args[0])
            elif pred == "visible_from":
                if args[0] not in self.obj_visible_from: self.obj_visible_from[args[0]] = []
                self.obj_visible_from[args[0]].append(args[1])
            elif pred == "store_of":
                rover_stores[args[1]] = args[0]
                store_to_rover[args[0]] = args[1]

        # Indexing global IDs for bitmask checks
        for rover in self.rovers:
            self.at_fact_ids[rover] = {}
            self.have_soil_ids[rover] = {}
            self.have_rock_ids[rover] = {}
            self.have_image_ids[rover] = {}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at" and args[0] in self.rovers:
                self.at_fact_ids[args[0]][args[1]] = fact.global_id
            elif pred == "have_soil_analysis":
                self.have_soil_ids[args[0]][args[1]] = fact.global_id
            elif pred == "have_rock_analysis":
                self.have_rock_ids[args[0]][args[1]] = fact.global_id
            elif pred == "have_image":
                self.have_image_ids[args[0]][(args[1], args[2])] = fact.global_id
            elif pred == "full" and args[0] in store_to_rover:
                self.store_full_ids[store_to_rover[args[0]]] = fact.global_id

        # Goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            fid = next((f.global_id for f in self.model.facts if f.name == gn), -1)
            if pred == "communicated_soil_data": self.goal_soil_ids[args[0]] = fid
            elif pred == "communicated_rock_data": self.goal_rock_ids[args[0]] = fid
            elif pred == "communicated_image_data": self.goal_image_ids[(args[0], args[1])] = fid

        # Build distance matrices
        lander_locs = set(lander_positions.values())
        for r in self.rovers:
            self.adj[r] = {w: [] for w in self.waypoints}
            edges = traversals.get(r, [])
            for u, v in edges:
                if u in visibilities and v in visibilities[u]:
                    self.adj[r][u].append(v)
            
            self.dist[r] = {w: {w2: 999 for w2 in self.waypoints} for w in self.waypoints}
            for start_w in self.waypoints:
                self.dist[r][start_w][start_w] = 0
                q = deque([(start_w, 0)])
                while q:
                    curr, d = q.popleft()
                    for neighbor in self.adj[r][curr]:
                        if self.dist[r][start_w][neighbor] == 999:
                            self.dist[r][start_w][neighbor] = d + 1
                            q.append((neighbor, d + 1))
            
            self.comm_waypoints[r] = set()
            for w in self.waypoints:
                if any(v in lander_locs for v in visibilities.get(w, [])):
                    self.comm_waypoints[r].add(w)
            
            self.min_dist_to_comm[r] = {}
            for w in self.waypoints:
                d_comm = [self.dist[r][w][cw] for cw in self.comm_waypoints[r]]
                self.min_dist_to_comm[r][w] = min(d_comm) if d_comm else 999

    def _get_rover_at(self, state, rover):
        for w, fid in self.at_fact_ids.get(rover, {}).items():
            if (state >> fid) & 1: return w
        return None

    def _compute(self, state, task_network):
        if not task_network and not (self.model.goals <= state): return 0
        
        h = 0
        rover_locs = {r: self._get_rover_at(state, r) for r in self.rovers}

        # Costing tasks in the network frontier
        for task in task_network:
            parts = task.name.split()
            t_type = parts[0]
            args = parts[1:]

            if t_type == "get_soil_data":
                target_w = args[0]
                best = 999
                for r in self.equipped_soil:
                    pos = rover_locs[r]
                    if pos:
                        cost = self.dist[r][pos][target_w] + 1 + self.min_dist_to_comm[r][target_w] + 1
                        best = min(best, cost)
                h += best if best < 999 else 10
            elif t_type == "get_rock_data":
                target_w = args[0]
                best = 999
                for r in self.equipped_rock:
                    pos = rover_locs[r]
                    if pos:
                        cost = self.dist[r][pos][target_w] + 1 + self.min_dist_to_comm[r][target_w] + 1
                        best = min(best, cost)
                h += best if best < 999 else 10
            elif t_type == "get_image_data":
                obj, mode = args[0], args[1]
                best = 999
                for r in self.equipped_image:
                    pos = rover_locs[r]
                    if pos:
                        for w in self.obj_visible_from.get(obj, []):
                            cost = self.dist[r][pos][w] + 1 + self.min_dist_to_comm[r][w] + 1
                            best = min(best, cost)
                h += best if best < 999 else 10
            elif t_type == "send_soil_data" or t_type == "send_rock_data":
                r, target_w = args[0], args[1]
                pos = rover_locs.get(r)
                if pos: h += self.min_dist_to_comm[r][pos] + 1
                else: h += 2
            elif t_type == "send_image_data":
                r = args[0]
                pos = rover_locs.get(r)
                if pos: h += self.min_dist_to_comm[r][pos] + 1
                else: h += 2
            elif t_type == "do_navigate1" or t_type == "do_navigate2":
                r, to_w = args[0], args[-1]
                pos = rover_locs.get(r)
                if pos: h += self.dist[r][pos][to_w]
                else: h += 1
            elif t_type in ["sample_soil", "sample_rock", "take_image", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data", "navigate", "drop", "calibrate"]:
                h += 1
        
        # Penalty for goal stability - check if goals are actually met
        for w, fid in self.goal_soil_ids.items():
            if not ((state >> fid) & 1):
                # If not being handled by network, add cost
                if not any("soil" in t.name and w in t.name for t in task_network): h += 1
        for w, fid in self.goal_rock_ids.items():
            if not ((state >> fid) & 1):
                if not any("rock" in t.name and w in t.name for t in task_network): h += 1

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
