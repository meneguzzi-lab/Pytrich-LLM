from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic5(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_0):
        super().__init__(barman_gemini3pro_heuristic_0)
        self.task_info = {}
        
        # 1. Build Fact Map and auxiliary structures
        # Maps: predicate -> args_tuple -> global_id
        fact_map = {}
        # Map: cocktail_name -> list of global_ids for facts "contains[shakerX, cocktail]"
        shaker_contains_cocktail = {}

        for f in self.model.facts:
            # Parse fact name: +pred[arg1,arg2]
            name = f.name.strip()
            if name.startswith("+") or name.startswith("-"):
                name = name[1:]
            
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                pred = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = tuple(args_str.split(",")) if args_str else ()
                
                if pred not in fact_map:
                    fact_map[pred] = {}
                fact_map[pred][args] = f.global_id
                
                # Domain Insight: Identify facts representing a cocktail in a shaker
                # This helps estimate if we are in the "Make" phase or "Pour" phase
                if pred == "contains" and len(args) == 2:
                    cont, bev = args
                    # Heuristic: objects named "shaker..." are shakers
                    if "shaker" in cont:
                        if bev not in shaker_contains_cocktail:
                            shaker_contains_cocktail[bev] = []
                        shaker_contains_cocktail[bev].append(f.global_id)

        # 2. Pre-process Abstract Tasks for efficient heuristic evaluation
        for task in self.model.abstract_tasks:
            # task.name example: "AchieveContainsShotCocktail shot1 cocktail1"
            parts = task.name.split()
            if not parts:
                continue
                
            t_name = parts[0]
            t_args = tuple(parts[1:])
            
            # Default strategy: constant cost
            info = {'type': 'const', 'cost': 1}
            
            # Domain-Specific Logic:
            if t_name == "AchieveContainsShotCocktail" and len(t_args) == 2:
                shot, cocktail = t_args
                target_fid = fact_map.get("contains", {}).get((shot, cocktail))
                shaker_fids = shaker_contains_cocktail.get(cocktail, [])
                
                info = {
                    'type': 'cocktail_goal',
                    'target_fid': target_fid,
                    'shaker_fids': shaker_fids,
                    'cost_make': 12, # High cost: ingredients + shake + pour + clean
                    'cost_pour': 4   # Medium cost: grasp + pour + clean
                }
            
            elif t_name == "AchieveCleanShot" and len(t_args) == 1:
                fid = fact_map.get("clean", {}).get(t_args)
                info = {'type': 'simple_check', 'fid': fid, 'cost': 3}
                
            elif t_name == "AchieveCleanShaker" and len(t_args) == 1:
                fid = fact_map.get("clean", {}).get(t_args)
                info = {'type': 'simple_check', 'fid': fid, 'cost': 3}
            
            elif t_name == "AchieveHandEmpty" and len(t_args) == 1:
                fid = fact_map.get("handEmpty", {}).get(t_args)
                info = {'type': 'simple_check', 'fid': fid, 'cost': 1}
                
            elif t_name == "AchieveHolding" and len(t_args) == 2:
                fid = fact_map.get("holding", {}).get(t_args)
                info = {'type': 'simple_check', 'fid': fid, 'cost': 2}
            
            elif t_name == "AchieveContainsShakerIngredient" and len(t_args) == 2:
                fid = fact_map.get("contains", {}).get(t_args)
                info = {'type': 'simple_check', 'fid': fid, 'cost': 4}
            
            elif t_name == "AchieveContainsShakerCocktail" and len(t_args) == 2:
                fid = fact_map.get("contains", {}).get(t_args)
                info = {'type': 'simple_check', 'fid': fid, 'cost': 8}

            # Map global_id to optimization info
            self.task_info[task.global_id] = info

    def __call__(self, node):
        h = 0
        state = node.state
        
        # Iterate through the Task Network
        for task in node.tn:
            # Primitive Task (Operator)
            if hasattr(task, 'add_effects'):
                h += 1
                continue
            
            # Abstract Task: Use pre-computed heuristic logic
            info = self.task_info.get(task.global_id)
            if not info:
                h += 1
                continue
            
            t_type = info['type']
            
            if t_type == 'simple_check':
                # Check if the goal predicate is already true in the state
                fid = info['fid']
                if fid is not None and ((state >> fid) & 1):
                    pass # Goal satisfied, cost 0
                else:
                    h += info['cost']
            
            elif t_type == 'cocktail_goal':
                target_fid = info['target_fid']
                # 1. Check if shot already has the cocktail
                if target_fid is not None and ((state >> target_fid) & 1):
                    continue # Done
                
                # 2. Check if the cocktail is prepared in any shaker
                in_shaker = False
                for sfid in info['shaker_fids']:
                    if (state >> sfid) & 1:
                        in_shaker = True
                        break
                
                if in_shaker:
                    h += info['cost_pour'] # Just need to pour (and maybe clean shot)
                else:
                    h += info['cost_make'] # Need to prepare ingredients and shake
            
            else:
                # Constant fallback
                h += info['cost']
                
        return h
