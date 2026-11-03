from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic13(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.

    BOTTLENECK ANALYSIS:
    The primary bottleneck in the Barman domain is container management (shots and shakers) 
    and the multi-step serialized process of cocktail preparation (cleaning, filling, pouring, shaking).
    Hand availability (only two hands) and the need to swap containers create a serial workflow.

    ADMISSIBILITY SKETCH:
    1. Goal Progress: Every unsatisfied cocktail goal requires at least 6 atomic actions 
       (clean-shot, fill-shot x2, pour-shot x2, shake, pour-shaker). A cost of 10 per 
       unsatisfied goal is a safe lower bound.
    2. Task Network: Each task in the task network represents at least one future operator.
       A cost of 1 per task is a safe lower bound.
    These terms are combined using addition to estimate remaining work.

    SYMMETRY BREAKING:
    - Penalizes use of 'goal shots' (shots assigned to specific cocktails) for other tasks (~500).
    - Preferred hand (left) vs non-preferred hand (right) to reduce branching (~50).
    - Working-shot ordering: assigning increasing penalties to shots to break symmetry (~100/index).
    """

    def __init__(self, barman_gemini3flash_heuristic_13="barman_gemini3flash_heuristic_13"):
        super().__init__(name=barman_gemini3flash_heuristic_13)
        self.goal_fact_ids = []
        self.goal_map = {}  # shot -> cocktail
        self.goal_cocktail_shots = set()
        self.all_shots = set()
        self.all_shakers = set()
        self.all_hands = set()
        self.shot_to_idx = {}
        self.non_preferred_hand = None
        self.cocktail_parts = {} # cocktail -> set(ingredients)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map predicates and objects
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "shakerLevel": 
                self.all_shakers.add(args[0])
            elif pred == "handEmpty": 
                self.all_hands.add(args[0])

        # Shots are containers that are not shakers or hands
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred in ["clean", "empty", "ontable"]:
                for a in args:
                    if a not in self.all_shakers and a not in self.all_hands:
                        self.all_shots.add(a)

        # 2. Symmetry breaking setup
        sorted_shots = sorted(list(self.all_shots))
        for i, s_name in enumerate(sorted_shots):
            self.shot_to_idx[s_name] = i
        
        sorted_hands = sorted(list(self.all_hands))
        if len(sorted_hands) > 1:
            self.non_preferred_hand = sorted_hands[1]  # Prefer the first alphabetical hand

        # 3. Goal state analysis
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains" and len(args) == 2:
                if args[0] in self.all_shots:
                    self.goal_map[args[0]] = args[1]
                    self.goal_cocktail_shots.add(args[0])
            
            # Map goal fact IDs for bitwise checks
            for f in model.facts:
                if f.name == g_name:
                    self.goal_fact_ids.append(f.global_id)
                    break

        # 4. Cocktail recipe analysis
        init_facts = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred in ["cocktailPart1", "cocktailPart2"]:
                cocktail, ingredient = args[0], args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = set()
                self.cocktail_parts[cocktail].add(ingredient)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        h = 0
        
        # --- Admissible Work Estimate ---
        # Cost per unsatisfied cocktail goal
        for g_id in self.goal_fact_ids:
            if not ((state >> g_id) & 1):
                h += 20
        
        # Cost per remaining task in network
        h += len(task_network) * 10

        # --- Symmetry Breaking Penalties ---
        # Check first tasks in network to guide decomposition choices
        for i in range(min(len(task_network), 3)):
            task = task_network[i]
            tname = task.name
            args = self._get_args(tname)
            
            for arg in args:
                # Shot penalties
                if arg in self.shot_to_idx:
                    h += self.shot_to_idx[arg] * 100
                    
                    if arg in self.goal_cocktail_shots:
                        # Penalize using a goal-shot unless it's for its own cocktail
                        is_valid_use = False
                        reserved_cocktail = self.goal_map.get(arg)
                        if "AchieveContainsShotCocktail" in tname and len(args) > 1 and args[1] == reserved_cocktail:
                            is_valid_use = True
                        elif ("ShotIngredient" in tname or "fill-shot" in tname or "pour-shot" in tname) and len(args) > 1:
                            if args[1] in self.cocktail_parts.get(reserved_cocktail, set()):
                                is_valid_use = True
                        
                        if not is_valid_use:
                            h += 500
                
                # Hand preference penalty
                if arg == self.non_preferred_hand:
                    h += 50
                    
        return h

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

    def _get_args(self, grounded_name: str):
        bracket_idx = grounded_name.find("[")
        if bracket_idx > 0 and grounded_name.endswith("]"):
            return grounded_name[bracket_idx+1:-1].split(",")
        return []
