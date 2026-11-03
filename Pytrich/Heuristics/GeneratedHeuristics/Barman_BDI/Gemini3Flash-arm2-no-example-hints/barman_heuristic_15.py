from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic15(Heuristic):
	"""
	Domain-dependent heuristic for Barman_BDI planning.
	
	1. Bottleneck Analysis: Hand capacity is the primary constraint. Each cocktail requires a complex 
	   sequence of cleaning, filling ingredients into shots, pouring to shaker, shaking, and pouring to the final shot.
	2. Admissible Bounds:
	   - Task Network Cost: Sum of minimal action counts to fulfill each non-satisfied task in the current network.
	   - Goal Fact Cost: A lower bound based on missing 'contains' facts for goal cocktails (each at least 15 actions).
	3. Symmetry Breaking:
	   - Goal-shot penalty: Using a shot designated for a specific goal cocktail as an intermediary vessel (~500).
	   - Working-shot ordering: Assignment of increasing penalties to identical shots to force canonical selection (~100*i).
	   - Hand preference: Halve branching by preferring one hand over the other (~50).
	"""
	def __init__(self, barman_gemini3flash_heuristic_15="barman_gemini3flash_heuristic_15"):
		super().__init__(name=barman_gemini3flash_heuristic_15)
		self.fact_map = {}
		self.goal_map = {}
		self.goal_shots = set()
		self.all_shots = set()
		self.shot_penalty_map = {}
		self.preferred_hand = None
		self.hands = set()
		self.task_base_costs = {
			"AchieveContainsShotCocktail": 20,
			"AchieveContainsShakerCocktail": 15,
			"AchieveContainsShakerIngredient": 8,
			"AchieveContainsShotIngredient": 4,
			"AchieveCleanShot": 2,
			"AchieveCleanShaker": 2,
			"AchieveHolding": 1,
			"AchieveHandEmpty": 1,
			"AchieveOnTable": 1,
			"DoPourShakerToShot": 2
		}

	def initialize(self, model: Model, initial_node: HTNNode):
		for f in model.facts:
			pred, args = self._parse_fact_name(f.name)
			if pred:
				self.fact_map[(pred, tuple(args))] = f.global_id
				if pred == "handEmpty": self.hands.add(args[0])
				for a in args:
					if "shot" in a.lower(): self.all_shots.add(a)
		if self.hands: self.preferred_hand = sorted(list(self.hands))[0]
		sorted_shots = sorted(list(self.all_shots))
		for i, s in enumerate(sorted_shots): self.shot_penalty_map[s] = i * 100
		goal_fact_names = model.state_explicit_repr(model.goals)
		for g_name in goal_fact_names:
			pred, args = self._parse_fact_name(g_name)
			if pred == "contains" and len(args) >= 2:
				self.goal_map[args[0]] = args[1]
				self.goal_shots.add(args[0])
		h_val = self._compute(initial_node.state, initial_node.task_network)
		return super().initialize(model, h_val)

	def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
		h_val = self._compute(node.state, node.task_network)
		super().update_info(h_val)
		return h_val

	def _compute(self, state, task_network):
		task_cost = 0
		penalty = 0
		for task in task_network:
			t_name, t_args = self._parse_task_name(task.name)
			base = self.task_base_costs.get(t_name, 1)
			if t_name == "AchieveContainsShotCocktail" and len(t_args) >= 2:
				if self._check(state, self.fact_map.get(("contains", (t_args[0], t_args[1])))): base = 0
			elif t_name == "AchieveCleanShot" and len(t_args) >= 1:
				if self._check(state, self.fact_map.get(("clean", (t_args[0],)))): base = 0
			elif t_name == "AchieveHandEmpty" and len(t_args) >= 1:
				if self._check(state, self.fact_map.get(("handEmpty", (t_args[0],)))): base = 0
			task_cost += base
			for arg in t_args:
				if arg in self.hands and arg != self.preferred_hand: penalty += 50
			shot = self._extract_shot(t_args)
			if shot:
				if shot in self.goal_shots:
					if not (t_name == "AchieveContainsShotCocktail" and self.goal_map.get(shot) == t_args[1]): penalty += 500
				penalty += self.shot_penalty_map.get(shot, 0)
		goal_cost = 0
		for s, c in self.goal_map.items():
			if not self._check(state, self.fact_map.get(("contains", (s, c)))): goal_cost += 15
		return max(task_cost, goal_cost) + penalty

	def _parse_fact_name(self, name: str):
		name = name.strip()
		if name.startswith("+") or name.startswith("-"): name = name[1:]
		bracket_idx = name.find("[")
		if bracket_idx > 0 and name.endswith("]"):
			return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
		return None, []

	def _parse_task_name(self, name: str):
		name = name.strip()
		for o, c in [("[", "]"), ("(", ")")]:
			if o in name:
				parts = name.split(o)
				return parts[0], [a.strip() for a in parts[1].rstrip(c).split(",")]
		return name, []

	def _extract_shot(self, args):
		for a in args:
			if a in self.all_shots: return a
		return None

	def _check(self, state, fact_id):
		if fact_id is None: return False
		return (state >> fact_id) & 1
