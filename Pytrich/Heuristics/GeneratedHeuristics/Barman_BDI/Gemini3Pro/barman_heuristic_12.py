from Pytrich.Heuristics.heuristic import Heuristic
import re

class BarmanGemini3ProHeuristic12(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_7=None):
        super().__init__(model)
        self.model = model
        self.fact_ids = {}
        self.recipes = {}
        self.shakers = set()
        self.shots = set()
        
        # Regex to parse grounded facts like "+contains[shot1,cocktail1]"
        # or "+shakerLevel[shaker1,level1]"
        self.fact_pattern = re.compile(r'^([+-])?(\w+)(?:\[(.*)\])?$')
        
        for fact in self.model.facts:
            match = self.fact_pattern.match(fact.name.strip())
            if match:
                sign = match.group(1)
                pred = match.group(2)
                args_str = match.group(3)
                args = tuple(args_str.split(',')) if args_str else ()
                
                # Map positive facts for state checking
                if sign != '-':
                    self.fact_ids[(pred,) + args] = fact.global_id
                    
                # Extract Domain Static Info
                if pred == 'cocktailPart1':
                    c, i = args
                    if c not in self.recipes: self.recipes[c] = {}
                    self.recipes[c]['p1'] = i
                elif pred == 'cocktailPart2':
                    c, i = args
                    if c not in self.recipes: self.recipes[c] = {}
                    self.recipes[c]['p2'] = i
                elif pred == 'shakerLevel':
                    # If it has a level, it is a shaker
                    if args:
                        self.shakers.add(args[0])
                elif pred == 'clean':
                    # Rough object identification if needed
                    pass

    def _get_fact_id(self, pred, *args):
        return self.fact_ids.get((pred,) + args)

    def _is_true(self, state, pred, *args):
        fid = self._get_fact_id(pred, *args)
        if fid is not None:
            return (state >> fid) & 1
        return False

    def compute(self, node):
        state = node.state
        tasks = node.task_network
        h = 0
        
        # Iterate through pending tasks to estimate remaining effort
        for task in tasks:
            # Parse task name to extract arguments
            # Expected format: TaskName[arg1,arg2] or TaskName arg1 arg2
            raw_name = task.name
            clean_name = raw_name.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            parts = clean_name.split()
            if not parts:
                continue
                
            t_type = parts[0]
            args = parts[1:]
            
            if 'AchieveContainsShotCocktail' in t_type and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                
                # 1. Check if goal is already satisfied
                if self._is_true(state, 'contains', shot, cocktail):
                    continue
                    
                # 2. Check if cocktail is already mixed in a shaker
                is_mixed_in_shaker = False
                for shaker in self.shakers:
                    if self._is_true(state, 'contains', shaker, cocktail) and \
                       self._is_true(state, 'shaked', shaker):
                        is_mixed_in_shaker = True
                        break
                
                shot_dirty = not self._is_true(state, 'clean', shot) and not self._is_true(state, 'empty', shot)
                clean_shot_cost = 3 if shot_dirty else 0
                
                if is_mixed_in_shaker:
                    # Remaining: Grasp Shaker + Pour + Drop/Clean
                    # Rough estimate: 2-3 steps
                    h += 2 + clean_shot_cost
                else:
                    # Full process: Clean Shaker? + Add Ing1 + Add Ing2 + Shake + Pour
                    # Add Ingredient = Clean Shot? + Fill + Pour to Shaker (~3-4 steps)
                    # Make Cocktail base cost ~ 10
                    h += 10 + clean_shot_cost
                    
            elif 'AchieveCleanShaker' in t_type and len(args) >= 1:
                shaker = args[0]
                if not self._is_true(state, 'clean', shaker):
                    # Empty + Clean or just Clean
                    h += 3
                    
            elif 'AchieveCleanShot' in t_type and len(args) >= 1:
                shot = args[0]
                if not self._is_true(state, 'clean', shot):
                    h += 3
                    
            elif 'AchieveHandEmpty' in t_type and len(args) >= 1:
                hand = args[0]
                if not self._is_true(state, 'handEmpty', hand):
                    h += 1
                    
            elif 'AchieveHolding' in t_type and len(args) >= 2:
                hand, obj = args[0], args[1]
                if not self._is_true(state, 'holding', hand, obj):
                    h += 1
                    
        return h
