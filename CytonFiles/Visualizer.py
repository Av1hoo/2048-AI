# Visualizer.py
import matplotlib.pyplot as plt
import time
import os

class Visualizer:
    @staticmethod
    def visualize_stats(stats_all, num_games, show_plots=False):
        """
        Generates Matplotlib figures based on batch game results.

        Args:
            stats_all (dict): The results from batch games.
            num_games (int): Number of games run in the batch.
            show_plots (bool): If True, display the plots interactively.
        Returns:
            list: A list of Matplotlib figures.
        """
        figures = []
        strategies = list(stats_all.keys())

        # If you run multiple strategies, stats_all is a dict of {strategy: stats_dict}
        if isinstance(stats_all, dict) and all(isinstance(s, dict) for s in stats_all.values()):
            average_scores = [stats_all[s]['average_score'] for s in strategies]
            best_scores = [stats_all[s]['best_score'] for s in strategies]

            # Ensure 'Stats' directory exists
            if not os.path.exists("Stats"):
                os.makedirs("Stats")

            # Average Scores Bar Chart
            fig_avg = plt.figure(figsize=(10, 6))
            plt.bar(strategies, average_scores, color='skyblue')
            plt.xlabel('AI Strategies')
            plt.ylabel('Average Score')
            plt.title(f'Average Scores over {num_games} Games')
            plt.tight_layout()
            figures.append(fig_avg)
            plt.savefig(f'Stats/average_scores_{time.strftime("%Y%m%d-%H%M")}.png')
            if show_plots:
                plt.show()
            plt.close(fig_avg)  # Close the figure to free memory

            # Best Scores Bar Chart
            fig_best = plt.figure(figsize=(10, 6))
            plt.bar(strategies, best_scores, color='salmon')
            plt.xlabel('AI Strategies')
            plt.ylabel('Best Score')
            plt.title(f'Best Scores over {num_games} Games')
            plt.tight_layout()
            figures.append(fig_best)
            plt.savefig(f'Stats/best_scores_{time.strftime("%Y%m%d-%H%M")}.png')
            if show_plots:
                plt.show()
            plt.close(fig_best)

            # Highest Tile Distribution Histogram
            fig_hist = plt.figure(figsize=(12, 8))
            all_tiles = [2**i for i in range(4, 14)]
            bar_width = 0.2
            indices = range(len(all_tiles))

            for idx, strategy in enumerate(strategies):
                histogram = stats_all[strategy]['histogram_highest_tile']
                counts = [histogram.get(tile, 0) for tile in all_tiles]
                plt.bar([i + idx * bar_width for i in indices], counts, alpha=0.7, width=bar_width, label=strategy)

            plt.xticks([i + bar_width * (len(strategies)-1)/2 for i in indices], all_tiles, rotation=45)
            plt.xlabel('Highest Tile')
            plt.ylabel('Frequency')
            plt.title(f'Highest Tile Distribution over {num_games} Games')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            figures.append(fig_hist)
            plt.savefig(f"Stats/highest_tile_distribution_{time.strftime('%Y%m%d-%H%M')}.png")
            if show_plots:
                plt.show()
            plt.close(fig_hist)

        else:
            # Single strategy stats
            strategy = stats_all.get("strategy", "Unknown Strategy")
            avg_score = stats_all.get("average_score", 0)
            best_score = stats_all.get("best_score", 0)
            histogram = stats_all.get("histogram_highest_tile", {})

            # Ensure 'Stats' directory exists
            if not os.path.exists("Stats"):
                os.makedirs("Stats")

            fig_single = plt.figure(figsize=(12, 8))
            all_tiles = [2**i for i in range(4, 14)]
            counts = [histogram.get(tile, 0) for tile in all_tiles]
            plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
            plt.xlabel('Highest Tile')
            plt.ylabel('Frequency')
            plt.title(f'Highest Tile Distribution over {num_games} Games')
            plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            figures.append(fig_single)
            plt.savefig(f"Stats/highest_tile_distribution_{time.strftime('%Y%m%d-%H%M')}.png")
            if show_plots:
                plt.show()
            plt.close(fig_single)

            # Optionally, print the stats
            print(f"Strategy: {strategy} | Average: {avg_score} | Best: {best_score} | Highest Tile Distribution: {histogram}")

        return figures
