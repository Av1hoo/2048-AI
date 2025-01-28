import ExpectimaxAI
import MinimaxAI
import RandomAI
import DqnAI

@staticmethod
class AI_Factory:
    def __init__(self, EXPECTIMAX_DEPTH = 5, MINIMAX_DEPTH = 4):
        self.EXPECTIMAX_DEPTH = EXPECTIMAX_DEPTH
        self.MINIMAX_DEPTH = MINIMAX_DEPTH
        self.ai_list = ["Expectimax", "Minimax", "Random", "DQN"]

    def create_ai(self, strategy):
        if strategy.lower() == "expectimax":
            return ExpectimaxAI(depth=self.EXPECTIMAX_DEPTH)
        elif strategy.lower() == "minimax":
            return MinimaxAI(depth=self.MINIMAX_DEPTH)
        elif strategy.lower() == "random":
            return RandomAI()
        elif strategy.lower() == "dqn":
            return DqnAI()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def get_ai_list(self):
        return self.ai_list