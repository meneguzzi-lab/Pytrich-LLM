import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic3(Heuristic):
    def __init__(self, barman_heuristic_3=1.0, name="BarmanGemini20FlashHeuristic3"):
        super().__init__(name=name)
        self.clean_shot_cost = 1.0
        self.clean_shaker_cost = 1.5
        self.add_ingredient_shot_cost = 2.0
        self.add_ingredient_shaker_cost = 2.5
        self.pour_cost = 1.0
        self.shake_cost = 0.5
        self.holding_cost = 0.2  # Cost associated with holding an object
        self.ontable_cost = 0.1   # Cost associated with an object being on the table
        self.handempty_cost = 0.1 # Cost associated with achieving handempty
        self.number_of_shots=0
        self.number_of_cocktails=0
        self.preprocessing_time = 0
        self.method_scores = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_state = initial_node.state

        #Precompute the number of cocktails and shots to serve as parameters
        shots = set()
        cocktails = set()
        for fact in model.facts:
            if "shot" in fact.name and not "Level" in fact.name:
                shots.add(fact.name)
            if "cocktail" in fact.name:
                cocktails.add(fact.name)

        self.number_of_shots = len(shots)
        self.number_of_cocktails = len(cocktails)
        self._precompute_method_info()

        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks
            clean_count = 0
            add_ingredient_count = 0
            pour_count = 0
            shake_count = 0
            subtask_count = 0
            holding_count=0
            handempty_count=0
            ontable_count = 0

            # Get subtasks from method
            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'addingredient' in subtask_lower:
                        add_ingredient_count += 1
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'achieveholding' in subtask_lower:
                        holding_count+=1
                    if 'achievehandempty' in subtask_lower:
                        handempty_count+=1
                    if 'achieveontable' in subtask_lower:
                        ontable_count+=1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'add_ingredient_count': add_ingredient_count,
                'pour_count': pour_count,
                'shake_count': shake_count,
                'subtask_count': subtask_count,
                'holding_count':holding_count,
                'ontable_count':ontable_count,
                'handempty_count':handempty_count,
                'complexity_score': clean_count + add_ingredient_count + pour_count + shake_count
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['clean_count'] * self.clean_shot_cost
            score += info['add_ingredient_count'] * self.add_ingredient_shot_cost
            score += info['pour_count'] * self.pour_cost
            score += info['shake_count'] * self.shake_cost
            score += info['holding_count'] * self.holding_cost
            score += info['handempty_count'] * self.handempty_cost
            score += info['ontable_count'] * self.ontable_cost
            score += info['subtask_count'] * 0.01  #Added to evaluate the subtasks
            score += info['complexity_score'] * 0.05 # Small cost for the complexity of an action

        # Additional heuristic based on number of cocktails/shots needed
        score += self.number_of_cocktails*0.05
        score += self.number_of_shots*0.01

        return max(score, 0)


    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic3()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic3()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 3 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean Shot Cost: {self.clean_shot_cost}\n"
            f"\tClean Shaker Cost: {self.clean_shaker_cost}\n"
            f"\tAdd Ingredient Shot Cost: {self.add_ingredient_shot_cost}\n"
            f"\tAdd Ingredient Shaker Cost: {self.add_ingredient_shaker_cost}\n"
            f"\tPour Cost: {self.pour_cost}\n"
            f"\tShake Cost: {self.shake_cost}\n"
            f"\tHolding Cost: {self.holding_cost}\n"
            f"\tOntable Cost: {self.ontable_cost}\n"
            f"\tHandEmpty Cost: {self.handempty_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
