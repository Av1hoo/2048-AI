# Visualizer.py
import matplotlib.pyplot as plt
import time
import os

class Visualizer:
    @staticmethod
    def visualize_stats(stats_all, num_games, show_plots=False):
        """
        Generates Matplotlib figures based on batch game results and saves them.
        Returns a dictionary with paths to the saved images.
        """
        plot_paths = {}
        strategies = list(stats_all.keys())
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        stats_dir = "static/Stats"
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        if isinstance(stats_all, dict) and all(isinstance(s, dict) for s in stats_all.values()):
            average_scores = [stats_all[s]['average_score'] for s in strategies]
            best_scores = [stats_all[s]['best_score'] for s in strategies]

            # Average Scores Bar Chart
            fig_avg = plt.figure(figsize=(10, 6))
            plt.bar(strategies, average_scores, color='skyblue')
            plt.xlabel('AI Strategies')
            plt.ylabel('Average Score')
            plt.title(f'Average Scores over {num_games} Games')
            plt.tight_layout()
            avg_score_filename = f"average_scores_{timestamp}.png"
            avg_score_path = os.path.join(stats_dir, avg_score_filename)
            plt.savefig(avg_score_path)
            plot_paths['avg_score_plot'] = f"Stats/{avg_score_filename}"
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
            best_score_filename = f"best_scores_{timestamp}.png"
            best_score_path = os.path.join(stats_dir, best_score_filename)
            plt.savefig(best_score_path)
            plot_paths['best_score_plot'] = f"Stats/{best_score_filename}"
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
            tile_distribution_filename = f"highest_tile_distribution_{timestamp}.png"
            tile_distribution_path = os.path.join(stats_dir, tile_distribution_filename)
            plt.savefig(tile_distribution_path)
            plot_paths['tile_distribution_plot'] = f"Stats/{tile_distribution_filename}"
            if show_plots:
                plt.show()
            plt.close(fig_hist)

        else:
            # Single strategy stats
            strategy = stats_all.get("strategy", "Unknown Strategy")
            avg_score = stats_all.get("average_score", 0)
            best_score = stats_all.get("best_score", 0)
            histogram = stats_all.get("histogram_highest_tile", {})

            fig_single = plt.figure(figsize=(12, 8))
            all_tiles = [2**i for i in range(4, 14)]
            counts = [histogram.get(tile, 0) for tile in all_tiles]
            # Create bars and add count numbers on top
            bars = plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            plt.xlabel('Highest Tile')
            plt.ylabel('Frequency')
            plt.title(f'Highest Tile Distribution over {num_games} Games')
            plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            tile_distribution_filename = f"highest_tile_distribution_{timestamp}.png"
            tile_distribution_path = os.path.join(stats_dir, tile_distribution_filename)
            plt.savefig(tile_distribution_path)
            plot_paths['tile_distribution_plot'] = f"Stats/{tile_distribution_filename}"
            if show_plots:
                plt.show()
            plt.close(fig_single)

            # Optionally, print the stats
            print(f"Strategy: {strategy} | Average: {avg_score} | Best: {best_score} | Highest Tile Distribution: {histogram}")

        return plot_paths
