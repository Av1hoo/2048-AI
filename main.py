# main.py
import tkinter as tk
import time
from CytonFiles import Game2048GUI
from CytonFiles import Batch_Games
from CytonFiles import Visualizer

def main():
    root = tk.Tk()
    app = Game2048GUI.Game2048GUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Global config
    STRATEGY = ["Expectimax", "Minimax", "Random", "DQN"]
    AI_STRATEGY = STRATEGY[0]
    DQN_MODEL_PATH = "model.pth"
    RUN_BATCH = False
    RUN_MULTIPLE_BATCH = True
    NUM_BATCH_GAMES = 1000
    DEPTH_EXPECIMAX = 2
    DEPTH_MINIMAX = 4
    STRATEGY_BATCH = {"Expectimax": 2, "Minimax": 2, "Random": 0, "DQN": 0}

    if RUN_BATCH:
        start_time = time.time()
        stats = Batch_Games.Batch_Games.run_batch_games(
            num_games=NUM_BATCH_GAMES,
            strategy=AI_STRATEGY,
            MODEL_PATH=DQN_MODEL_PATH,
            DEPTH=DEPTH_EXPECIMAX
        )
        print(f"Batch games completed in {time.time()-start_time:.2f} seconds.")
        # If you only run one strategy, pass that stats directly:
        Visualizer.Visualizer.visualize_stats(stats, NUM_BATCH_GAMES, show_plots=True)

    elif RUN_MULTIPLE_BATCH:
        start_time = time.time()
        stats_all = Batch_Games.Batch_Games.run_multiple_batch_strategies(
            num_games=NUM_BATCH_GAMES,
            strategies=STRATEGY_BATCH,
            MODEL_PATH=DQN_MODEL_PATH
        )
        print(f"Multiple batch games completed in {time.time()-start_time:.2f} seconds.")
        Visualizer.Visualizer.visualize_stats(stats_all, NUM_BATCH_GAMES, show_plots=True)

    else:
        main()
