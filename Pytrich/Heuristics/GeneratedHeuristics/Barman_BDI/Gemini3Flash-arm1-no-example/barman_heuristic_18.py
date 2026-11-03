from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The Barman_BDI domain is constrained by a single shaker and two hands. 
# Every cocktail requires a strict cycle of cleaning the shaker, filling it with ingredients, 
# shaking it, and pouring it into a shot. The shaker is a shared resource across all cocktail goals.
# Admissibility Sketches:
# 1. Task-based Cost: Each task in the network requires a minimum number of primitive operations 
#    to be resolved. For example, 'AchieveContainsShotCocktail' requires at least 4 operations (clean shot, 
#    fill 2 ingredients, shake, pour), which is a lower bound on the remaining work.
# 2. State-aware Goal Detection: If an 'Achieve' task's target state (e.g., contains(shot, cocktail)) 
#    is already satisfied, the cost is 0 (as the Null method will be chosen).

class BarmanBDIGemini3FlashHeuristic18(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_18=
