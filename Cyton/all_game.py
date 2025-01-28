import tkinter as tk
import time
from batch_games import Batch_Games
import Visualizer
import Game2048GUI

def main():
    root = tk.Tk()
    app = Game2048GUI(root)
    root.mainloop()

if __name__=="__main__":
    # Global config
    STRATEGY = ["Expectimax", "Minimax", "Random", "DQN"]
    AI_STRATEGY = STRATEGY[0]
    RUN_BATCH = True
    NUM_BATCH_GAMES = 1000
    RUN_MULTIPLE_BATCH = False

    if RUN_BATCH:
        start_time = time.time()
        stats = Batch_Games.run_batch_games(NUM_BATCH_GAMES, AI_STRATEGY)
        print(f"Batch games completed in {time.time()-start_time:.2f} seconds.")
        # If you only run one strategy, pass that stats directly:
        Visualizer.visualize_stats(stats, NUM_BATCH_GAMES)
    elif RUN_MULTIPLE_BATCH:
        stats_all = Batch_Games.run_multiple_batch_strategies(NUM_BATCH_GAMES, STRATEGY)
        Visualizer.visualize_stats(stats_all, NUM_BATCH_GAMES)
    else:
        main()