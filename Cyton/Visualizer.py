import matplotlib.pyplot as plt
import time
import os

class Visualizer():
    @staticmethod
    def visualize_stats(stats_all, num_games):
        strategies = list(stats_all.keys())
        # If you run multiple strategies, stats_all is a dict of {strategy: stats_dict}
        if isinstance(stats_all, dict) and all(isinstance(s, dict) for s in stats_all.values()):
            average_scores = [stats_all[s]['average_score'] for s in strategies]
            best_scores = [stats_all[s]['best_score'] for s in strategies]
            plt.figure(figsize=(10, 6))
            plt.bar(strategies, average_scores, color='skyblue')
            plt.xlabel('AI Strategies')
            plt.ylabel('Average Score')
            plt.title(f'Average Scores over {num_games} Games')
            plt.savefig(f'stats/average_scores_{time.strftime("%Y%m%d-%H%M")}.png')
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.bar(strategies, best_scores, color='salmon')
            plt.xlabel('AI Strategies')
            plt.ylabel('Best Score')
            plt.title(f'Best Scores over {num_games} Games')
            plt.savefig(f'stats/best_scores_{time.strftime("%Y%m%d-%H%M")}.png')
            plt.show()

            plt.figure(figsize=(12, 8))
            all_tiles = [2**i for i in range(4, 14)]
            for strategy in strategies:
                histogram = stats_all[strategy]['histogram_highest_tile']
                counts = [histogram.get(tile, 0) for tile in all_tiles]
                plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
            plt.xlabel('Highest Tile')
            plt.ylabel('Frequency')
            plt.title(f'Highest Tile Distribution over {num_games} Games')
            plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
            plt.legend()
            plt.grid(True)
            plt.savefig(f"stats/highest_tile_distribution_{time.strftime('%Y%m%d-%H%M')}.png")
            plt.show()
        else:
            # Single strategy stats
            strategy = stats_all["strategy"]
            avg_score = stats_all["average_score"]
            best_score = stats_all["best_score"]
            histogram = stats_all["histogram_highest_tile"]
            # show plt with the data
            plt.figure(figsize=(12, 8))
            all_tiles = [2**i for i in range(4, 14)]
            counts = [histogram.get(tile, 0) for tile in all_tiles]
            plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
            plt.xlabel('Highest Tile')
            plt.ylabel('Frequency')
            plt.title(f'Highest Tile Distribution over {num_games} Games')
            plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
            plt.legend()
            plt.grid(True)
            # if no folder stats create it
            if not os.path.exists("stats"):
                os.makedirs("stats")
            plt.savefig(f"stats/highest_tile_distribution_{time.strftime('%Y%m%d-%H%M')}.png")
            print(f"Strategy: {strategy} | Average: {avg_score} | Best: {best_score} | Highest Tile Distribution: {histogram}")