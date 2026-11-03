from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Heuristic Analysis for Barman_BDI:
# 1. Bottleneck: The primary serialization bottleneck in Barman is the limited availability of shakers and hands, and the rigid sequence required to produce a cocktail (Clean -> Fill Intermediary Shot -> Pour -> Shake -> Pour). Symmetry in shot selection is a major branching factor.
# 2. Admissible Lower Bounds:
#    - Unmet Goal Count: Each unmet 'contains(shot, cocktail)' goal requires at least 7 actions (clean shaker, fill-shot 1, pour 1, fill-shot 2, pour 2, shake, pour-shaker-to-shot).
#    - Task Network Length: Each non-primitive task in the network will eventually result in at least one action.
# 3. Symmetry-Breaking Guidance (Critical for performance):
#    - Goal-shot penalty: Penalizes using a shot that is a target for a final cocktail as an ingredient intermediary (500 per task).
#    - Working-shot ordering: Canonicalizes choices by penalizing higher-indexed shots (100 * index).
#    - Hand preference: Prefers the 'left' hand to reduce hand-binding permutations (50 per task).

class BarmanBDIGemini3FlashHeuristic4(Heuristic):
	def __init__(self, barman_gemini3flash_heuristic_4="barman_gemini3flash_heuristic_4"):
		super().__init__(barman_gemini3flash_heuristic_4)
		self.goal_shots = set()
		self.shot_indices = {}
		self.all_shots = set()
		self.hands = set()
		self.preferred_hand = None

	def initialize(self, model: Model, initial_node: HTNNode):
		self.model = model
		# Identify all shots and hands from facts
		for fact in model.facts:
			pred, args = self._parse_fact_name(fact.name)
			if not pred: continue
			for arg in args:
				if "shot" in arg.lower():
					self.all_shots.add(arg)
				if pred in ["handEmpty", "holding", "handempty"]:
					self.hands.add(arg)
		
		sorted_shots = sorted(list(self.all_shots))
		self.shot_indices = {name: i for i, name in enumerate(sorted_shots)}
		
		if self.hands:
			self.preferred_hand = "left" if "left" in self.hands else sorted(list(self.hands))[0]

		# Identify goal shots from goal facts
		goal_fact_names = model.state_explicit_repr(model.goals)
		for gname in goal_fact_names:
			pred, args = self._parse_fact_name(gname)
			if pred == "contains" and args:
				# Goal is contains[shot, cocktail]
				if args[0] in self.all_shots:
					self.goal_shots.add(args[0])

		h_val = self._compute(initial_node.state, initial_node.task_network)
		return super().initialize(model, h_val)

	def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
		h_val = self._compute(node.state, node.task_network)
		super().update_info(h_val)
		return h_val

	def _compute(self, state, task_network):
		# Bitwise count of unmet goal facts
		unmet_goal_count = bin(self.model.goals & ~state).count('1')
		h_base = (unmet_goal_count * 7) + len(task_network)
		
		penalty = 0
		for task in task_network:
			tname, targs = self._parse_task_name(task.name)
			
			# Intermediary shot penalties (symmetry breaking)
			if any(kw in tname for kw in ["ShotIngredient", "fill-shot", "clean-shot", "empty-shot", "pour-shot"]):
				shot = self._get_shot_from_args(targs)
				if shot:
					if shot in self.goal_shots:
						penalty += 500
					if shot in self.shot_indices:
						penalty += self.shot_indices[shot] * 100
			
			# Hand preference penalty
			for arg in targs:
				if arg in self.hands and arg != self.preferred_hand:
					penalty += 50
					break
					
		return h_base + penalty

	def _parse_fact_name(self, name: str):
		name = name.strip()
		if name.startswith("+") or name.startswith("-"):
			name = name[1:]
		if "[" in name:
			idx = name.find("[")
			pred = name[:idx]
			args = [a.strip() for a in name[idx+1:-1].split(",")]
			return pred, args
		return None, []

	def _parse_task_name(self, name: str):
		if "(" in name:
			idx = name.find("(")
			pred = name[:idx]
			args = [a.strip() for a in name[idx+1:-1].split(",")]
			return pred, args
		return name, []

	def _get_shot_from_args(self, args):
		for arg in args:
			if arg in self.all_shots:
				return arg
		return None
