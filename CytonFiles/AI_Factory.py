from CytonFiles.ExpectimaxAI import ExpectimaxAI
from CytonFiles.MinimaxAI import MinimaxAI
from CytonFiles.RandomAI import RandomAI
from CytonFiles.DqnAI import DqnAI

class AI_Factory:
    def __init__(self, EXPECTIMAX_DEPTH = 5, MINIMAX_DEPTH = 4):
        self.EXPECTIMAX_DEPTH = EXPECTIMAX_DEPTH
        self.MINIMAX_DEPTH = MINIMAX_DEPTH
        self.ai_list = ["Expectimax", "Minimax", "Random", "DQN"]
    
    @staticmethod
    def create_ai(self, strategy, DQN_MODEL_PATH="model.pth"):
        if strategy.lower() == "expectimax":
            return ExpectimaxAI(depth=self.EXPECTIMAX_DEPTH)
        elif strategy.lower() == "minimax":
            return MinimaxAI(depth=self.MINIMAX_DEPTH)
        elif strategy.lower() == "random":
            return RandomAI()
        elif strategy.lower() == "dqn":
            return DqnAI(DQN_MODEL_PATH)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def get_ai_list(self):
        return self.ai_list