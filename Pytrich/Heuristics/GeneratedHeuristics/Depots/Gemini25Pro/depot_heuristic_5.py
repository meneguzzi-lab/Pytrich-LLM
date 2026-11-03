import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini25ProHeuristic5(Heuristic):
    """
    A domain-dependent heuristic for the Depot domain.

    This heuristic estimates the cost to reach the goal state by calculating a relaxed
    plan cost for each unsatisfied goal. The cost is primarily based on the number
    of manipulation and transportation actions required to move a crate to its
    target destination.

    The heuristic considers three main factors for each misplaced crate:
    1.  **Transportation Cost:** If a crate is not at its target location (depot or
        distributor), it estimates the cost of using a truck (Drive action).
    2.  **Manipulation Cost:** It counts the minimum number of actions like Lift, Load,
        Unload, and Drop required to move the crate.
    3.  **Clearing Cost:** A small penalty is added if either the crate to be moved or
        its destination surface is not clear, representing the need to move
        obstructing crates.

    To handle complex goal dependencies (e.g., stacking crates on other crates),
    the heuristic pre-computes the final destination place for every crate involved
    in a goal during initialization. It also pre-parses all facts and infers
    object types for efficient evaluation at each search node.
    """

    def __init__(self, depot_heuristic_5=None):
        """
        Initializes the DepotGemini25ProHeuristic5.

        :param depot_heuristic_5: A dictionary of parameters to tune the heuristic's behavior.
                                        Supported keys: 'drive_cost', 'manipulation_cost', 'clear_penalty'.
        """
        super().__init__(name="depot_gemini25pro_heuristic_5")
        depot_gemini25pro_heuristic_5 = depot_heuristic_5 if depot_heuristic_5 is not None else {}

        # Heuristic cost parameters based on action counts
        self.penalties = {
            'drive': depot_gemini25pro_heuristic_5.get('drive_cost', 1),
            'lift_drop': depot_gemini25pro_heuristic_5.get('lift_drop_cost', 2),      # Lift (1) + Drop (1)
            'load_unload': depot_gemini25pro_heuristic_5.get('load_unload_cost', 2),  # Load (1) + Unload (1)
            'clear': depot_gemini25pro_heuristic_5.get('clear_penalty', 1)
        }

        # Pre-computation attributes
        self.model = None
        self.parsed_facts = []
        self.object_types = {}
        self.static_surface_locations = {} # For pallets
        self.goals_to_check = []
        self.crate_goal_locations = {} # Final place for each goal crate

    def _parse_fact(self, fact_str: str):
        """Helper function to parse a fact string like '(pred obj1 obj2)' into a tuple."""
        fact_str = fact_str.strip()
        if fact_str.startswith('(') and fact_str.endswith(')'):
            parts = fact_str[1:-1].split()
            return tuple(parts)
        return tuple()

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing necessary information from the model.
        """
        self.model = model
        self.parsed_facts = [self._parse_fact(f.name) for f in self.model.facts]
        self._infer_object_types()

        initial_state_facts = [self.parsed_facts[i] for i, bit in enumerate(bin(initial_node.state)[2:][::-1]) if bit == '1']

        # Pre-compute locations of static surfaces (pallets)
        for fact in initial_state_facts:
            if fact and fact[0] == 'at' and self.object_types.get(fact[1]) == 'pallet':
                self.static_surface_locations[fact[1]] = fact[2]

        # Process goals to find final locations
        goal_facts = [self.parsed_facts[gid] for gid in self.model.goals]
        on_goals = [g for g in goal_facts if g and g[0] == 'on']
        goal_on_relations = {g[1]: g[2] for g in on_goals}

        # Recursively find the final base location for each goal crate
        memo = {}
        def find_final_location(crate):
            if crate in memo:
                return memo[crate]
            if crate not in goal_on_relations:
                memo[crate] = None
                return None
            
            target_surface = goal_on_relations[crate]
            if self.object_types.get(target_surface) in ['pallet', 'surface']:
                loc = self.static_surface_locations.get(target_surface)
                memo[crate] = loc
                return loc
            else: # Target is another crate
                loc = find_final_location(target_surface)
                memo[crate] = loc
                return loc

        all_crates = {obj for obj, type_name in self.object_types.items() if type_name == 'crate'}
        for c in all_crates:
            find_final_location(c)
        self.crate_goal_locations = memo

        # Store goals for quick checking
        for gid in self.model.goals:
            p_fact = self.parsed_facts[gid]
            if p_fact and p_fact[0] == 'on':
                self.goals_to_check.append({
                    'id': gid,
                    'crate': p_fact[1],
                    'surface': p_fact[2]
                })
        
        return super().initialize(model, self.__call__(None, initial_node))

    def _infer_object_types(self):
        """Infers object types from their usage in domain predicates."""
        for fact in self.parsed_facts:
            if not fact: continue
            pred, args = fact[0], fact[1:]
            if pred == 'at': # (at ?x - locatable ?y - place)
                if args[0] not in self.object_types: self.object_types[args[0]] = 'locatable'
                if args[1] not in self.object_types: self.object_types[args[1]] = 'place'
            elif pred == 'on': # (on ?x - crate ?y - surface)
                self.object_types[args[0]] = 'crate'
                if args[1] not in self.object_types or self.object_types[args[1]] != 'crate':
                    self.object_types[args[1]] = 'surface' 
            elif pred == 'in': # (in ?x - crate ?y - truck)
                self.object_types[args[0]] = 'crate'
                self.object_types[args[1]] = 'truck'
            elif pred == 'lifting': # (lifting ?x - hoist ?y - crate)
                self.object_types[args[0]] = 'hoist'
                self.object_types[args[1]] = 'crate'

        # Assume surfaces not identified as crates are pallets for simplicity
        for obj, type_name in self.object_types.items():
            if type_name == 'surface':
                self.object_types[obj] = 'pallet'

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """
        Calculates the heuristic value for the given node.
        """
        h = 0
        state = node.state

        # --- 1. Efficiently parse the current state --- #
        current_locations = {} # {obj: place} for locatables
        crates_in_trucks = {} # {crate: truck}
        whats_on_top = defaultdict(list) # {surface: [crates_on_top]}
        
        for fact_id in range(state.bit_length()):
            if (state >> fact_id) & 1:
                p_fact = self.parsed_facts[fact_id]
                if not p_fact: continue
                
                pred, args = p_fact[0], p_fact[1:]
                if pred == 'at':
                    current_locations[args[0]] = args[1]
                elif pred == 'in':
                    crates_in_trucks[args[0]] = args[1]
                elif pred == 'on':
                    whats_on_top[args[1]].append(args[0])

        # --- 2. Evaluate each unachieved 'on' goal --- #
        for goal in self.goals_to_check:
            if (state >> goal['id']) & 1:
                continue

            cost = 0
            c_goal, s_goal = goal['crate'], goal['surface']
            
            p_goal = self.crate_goal_locations.get(c_goal)
            if p_goal is None: continue

            # --- A. Calculate movement and manipulation cost --- #
            if c_goal in crates_in_trucks:
                truck = crates_in_trucks[c_goal]
                p_current = current_locations.get(truck)
                cost += self.penalties['load_unload'] # For Unload + Drop
                if p_current != p_goal:
                    cost += self.penalties['drive']
            else:
                p_current = current_locations.get(c_goal)
                if p_current is None: continue
                
                if p_current != p_goal:
                    cost += self.penalties['lift_drop'] + self.penalties['load_unload'] + self.penalties['drive']
                else: # Same place, wrong surface
                    cost += self.penalties['lift_drop']

            # --- B. Add clearing penalties --- #
            cost += self.penalties['clear'] * len(whats_on_top.get(s_goal, []))
            cost += self.penalties['clear'] * len(whats_on_top.get(c_goal, []))

            h += cost

        super().update_info(h)
        return h