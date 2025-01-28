import os
import pickle
from collections import defaultdict
from CytonFiles.AI_Factory import AI_Factory
from CytonFiles.C_funcs import init_c_game, bitboard_spawn_func, bitboard_get_max_tile_func, bitboard_move_func, bitboard_is_game_over_func



class Batch_Games():
    @staticmethod
    def run_batch_games(num_games=1000, strategy="expectimax", MODEL_PATH="model.pth", DEPTH=5):
        init_c_game()
        factory = AI_Factory(EXPECTIMAX_DEPTH=DEPTH, MINIMAX_DEPTH=DEPTH)
        ai = factory.create_ai(factory, strategy, MODEL_PATH)
        
        best_score = -1
        best_game = None
        total_score = 0
        histogram_highest_tile = defaultdict(int)

        strategy_safe = strategy.lower()
        best_game_path = f"Best_Games/best_game_{strategy_safe}.pkl"
        
        for game_num in range(1, num_games + 1):
            print(f"Running game {game_num}/{num_games} with strategy {strategy}", end="\r")
            bitboard = 0
            bitboard = bitboard_spawn_func(bitboard)
            bitboard = bitboard_spawn_func(bitboard)
            score = 0
            highest = bitboard_get_max_tile_func(bitboard)
            states = [bitboard]
            moves = []

            while True:
                move = ai.get_move(bitboard)  # same 30-sec limit if needed in AI
                if move is None:
                    break
                newb, sc, moved = bitboard_move_func(bitboard, move)
                if not moved:
                    break
                bitboard = newb
                score += sc
                current_highest = bitboard_get_max_tile_func(bitboard)
                if current_highest > highest:
                    highest = current_highest
                bitboard = bitboard_spawn_func(bitboard)
                moves.append(move)
                states.append(bitboard)

                if bitboard_is_game_over_func(bitboard):
                    break

            total_score += score
            histogram_highest_tile[highest] += 1

            if score > best_score:
                best_score = score
                best_game = {
                    "score": score,
                    "highest": highest,
                    "states": states.copy(),
                    "moves": moves.copy(),
                }
                print(f"New best game #{game_num}: Score={score}, Highest Tile={highest}")
                if os.path.exists(best_game_path):
                    with open(best_game_path, "rb") as f:
                        data = pickle.load(f)
                        if best_score <= data["score"]:
                            pass
                        else:
                            with open(best_game_path, "wb") as f_write:
                                pickle.dump(best_game, f_write)
                            print(f"[INFO] Best game for strategy '{strategy}' saved with Score={best_game['score']}, Highest Tile={best_game['highest']}")
                else:
                    with open(best_game_path, "wb") as f_write:
                        pickle.dump(best_game, f_write)
                    print(f"[INFO] Best game for strategy '{strategy}' saved with Score={best_game['score']}, Highest Tile={best_game['highest']}")

        average_score = total_score / num_games if num_games > 0 else 0
        stats = {
            "strategy": strategy,
            "best_score": best_score,
            "average_score": average_score,
            "histogram_highest_tile": dict(histogram_highest_tile)
        }
        return stats

    @staticmethod
    def run_multiple_batch_strategies(num_games=1000, strategies="expectimax", MODEL_PATH="model.pth"):
        stats_all = {}
        # strategies = {"Expectimax": 2, "Minimax": 4, "Random": 0, "DQN": 0}
        for strategy, depth in strategies.items():
            print(f"\nRunning batch games for strategy: {strategy}")
            stats = Batch_Games.run_batch_games(num_games, strategy, MODEL_PATH, depth)
            stats_all[strategy] = stats
        return stats_all