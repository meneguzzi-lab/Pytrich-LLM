from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

# Bottleneck Analysis:
# The primary bottleneck in the Barman_BDI domain is the sequential hand-usage and container management.
# Preparing a cocktail requires a shaker to be cleaned, then filled with two specific ingredients via shots,
# then shaken (requiring two hands to be available: one for the shaker, one empty), and finally poured 
# into a clean shot. This requires careful coordination of cleaning, holding, and dropping actions.
#
# Admissibility Sketches:
# 1. Task Network Cost (Term 1): Each task in the network represents a sequence of actions. We assign a 
#    minimal action count (lower bound) to each task based on its distance from completion in the hierarchy.
#    e.g., 'AchieveContainsShotCocktail' requires at least 1 pour if the cocktail is already in the shaker,
#    or many more if ingredients are missing. This is admissible as it never overestimates the actions.
# 2. Hand-Conflict Penalty (Term 2): If multiple tasks require holding different items, we add a small 
#    admissible cost for the necessary drop/grasp transitions.

class BarmanBDIGemini3FlashHeuristic3(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_3="barman_gemini3flash_heuristic_3"):
        super().__init__(name=barman_gemini3flash_heuristic_3)
        self.fact_map = {}  # (predicate, args_tuple) -> global_id
        self.goal_facts = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, tuple(args)
        return None, ()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[(pred, args)] = fact.global_id

        # Pre-identify goal facts from the bitwise model.goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred and (pred, args) in self.fact_map:
                self.goal_facts.append(((pred, args), self.fact_map[(pred, args)]))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _get_fact_val(self, state, pred, args):
        fid = self.fact_map.get((pred, args))
        if fid is None:
            return 0
        return (state >> fid) & 1

    def _compute(self, state, task_network):
        h = 0
        
        # Term 1: Task Network Progress
        for task in task_network:
            t_name = task.name
            # Clean name for parsing (e.g., "AchieveContainsShotCocktail shot1 cocktail1")
            parts = t_name.split()
            if not parts: continue
            base_task = parts[0]
            args = tuple(parts[1:])

            if base_task == "AchieveContainsShotCocktail":
                # args: (shot, cocktail)
                if not self._get_fact_val(state, "contains", args):
                    h += 5 # Minimal: pour-shaker-to-shot (1) + decomposition overhead
            elif base_task == "AchieveContainsShakerCocktail":
                # args: (shaker, cocktail)
                if not self._get_fact_val(state, "contains", args):
                    h += 4 # Minimal: shake + ingredients
            elif base_task == "AchieveContainsShakerIngredient":
                # args: (shaker, ingredient)
                if not self._get_fact_val(state, "contains", args):
                    h += 2 # Minimal: pour-shot-to-shaker
            elif base_task == "AchieveContainsShotIngredient":
                # args: (shot, ingredient)
                if not self._get_fact_val(state, "contains", args):
                    h += 2 # Minimal: fill-shot
            elif base_task == "AchieveCleanShot" or base_task == "AchieveCleanShaker":
                # args: (container)
                if not self._get_fact_val(state, "clean", args):
                    h += 2
            elif base_task == "AchieveHolding":
                # args: (hand, container)
                if not self._get_fact_val(state, "holding", args):
                    h += 1
            elif base_task == "AchieveHandEmpty":
                # args: (hand)
                if not self._get_fact_val(state, "handEmpty", args):
                    h += 1
            else:
                # Standard cost for primitive operators or unknown tasks
                h += getattr(task, 'cost', 1)

        # Term 2: Goal Stability (Check goals not covered by current task network)
        # This prevents the heuristic from dropping to 0 if the task network is empty but goals aren't met.
        # Only adding costs if the facts are actually false.
        for (pred, args), fid in self.goal_facts:
            if not ((state >> fid) & 1):
                # Check if this goal is likely already being handled by the task network
                # (Simple check: is the container/cocktail mentioned in the network?)
                in_network = False
                target_obj = args[0] if args else ""
                for task in task_network:
                    if target_obj in task.name:
                        in_network = True
                        break
                if not in_network:
                    h += 3 # Minimum cost to start a new cocktail sequence

        # Symmetry Breaking: subtle penalty for using non-canonical shots or hands
        # Encourages the planner to stick to one path when options are identical.
        symmetry_penalty = 0
        for task in task_network:
            if "shot" in task.name:
                # Penalize higher-indexed shots slightly to prefer shot1, then shot2...
                nums = re.findall(r'\d+', task.name)
                if nums:
                    symmetry_penalty += int(nums[0]) * 0.001

        return int(h + symmetry_penalty)
