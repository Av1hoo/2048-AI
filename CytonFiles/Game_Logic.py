# game_logic.py
import os
import pickle
import random
import time 

from CytonFiles.C_funcs import (
    init_c_game, bitboard_spawn, bitboard_move,
    bitboard_is_game_over, bitboard_get_max_tile,
    bitboard_to_board
)
from CytonFiles.AI_Factory import AI_Factory

class Game2048:
    def __init__(self, ai_strategy="Expectimax", best_game_path="Best_Games/best_game.pkl", depth=3):
        # Initialize the C environment just once
        init_c_game()

        # Basic config
        self.ai_strategy = ai_strategy
        self.best_game_path = best_game_path
        
        # AI engine
        self.factory = AI_Factory(EXPECTIMAX_DEPTH = depth, MINIMAX_DEPTH = depth)
        self.ai = self.factory.create_ai(self.ai_strategy)

        # Game state
        self.bitboard = 0
        self.score = 0
        self.highest = 0
        self.game_over = False

        # Tracking
        self.states = []
        self.moves = []
        self.best_score = 0  # Keep track of best score in current session

    def set_depth(self, depth):
        """
        Set the depth for the AI strategy.
        """
        if self.ai_strategy == "Expectimax":
            self.factory.EXPECTIMAX_DEPTH = depth
        elif self.ai_strategy == "Minimax":
            self.factory.MINIMAX_DEPTH = depth

            
    def reset_game(self):
        """
        Start/restart a new game. Clears all previous states and moves.
        """
        init_c_game()
        self.bitboard = 0
        self.bitboard = bitboard_spawn(self.bitboard)
        self.bitboard = bitboard_spawn(self.bitboard)
        self.score = 0
        self.highest = bitboard_get_max_tile(self.bitboard)
        self.game_over = False

        # Clear any previous runs
        self.states.clear()
        self.moves.clear()

        # Record the initial state
        self.states.append(self.bitboard)

    def do_move(self, action):
        """
        Perform a move (0=UP,1=DOWN,2=LEFT,3=RIGHT) if the game is not over.
        Returns True if a move was made, False otherwise.
        """
        if self.game_over:
            return False

        new_board, gained_score, moved = bitboard_move(self.bitboard, action)
        if not moved:
            return False  # no change in board => skip

        self.bitboard = new_board
        self.score += gained_score
        self._update_highest()

        # Spawn a new tile
        self.bitboard = bitboard_spawn(self.bitboard)
        self.moves.append(action)
        self.states.append(self.bitboard)

        # Check if game is over
        if bitboard_is_game_over(self.bitboard):
            self._finish_game()

        return True

    def ai_single_step(self):
        """
        Let the AI choose and perform a single move.
        Returns True if a move was made, False otherwise.
        """
        if self.game_over:
            return False

        move = self.ai.get_move(self.bitboard)
        if move is None:
            self._finish_game()
            return False

        new_board, gained_score, moved = bitboard_move(self.bitboard, move)
        if not moved:
            self._finish_game()
            return False

        self.bitboard = new_board
        self.score += gained_score
        self._update_highest()

        # Spawn a tile
        self.bitboard = bitboard_spawn(self.bitboard)
        self.moves.append(move)
        self.states.append(self.bitboard)

        # Check if game is over
        if bitboard_is_game_over(self.bitboard):
            self._finish_game()

        return True
    
    def play_full_ai(self, max_steps=1000):
        """
        Let the AI continue making moves until the game ends or we hit max_steps.
        """
        steps = 0
        while not self.game_over and steps < max_steps:
            moved = self.ai_single_step()
            if not moved:
                break
            steps += 1

    def _update_highest(self):
        """
        Update the highest tile found so far.
        """
        current_max = bitboard_get_max_tile(self.bitboard)
        if current_max > self.highest:
            self.highest = current_max

    def _finish_game(self):
        """
        Mark game as over and attempt to save if it's the best so far.
        """
        self.game_over = True
        self._save_if_best()

    def _save_if_best(self):
        """
        Compare current game score to the best one stored, and if better, overwrite the best game file.
        """
        best_score_stored = -1
        data_to_store = {
            "score": self.score,
            "highest": self.highest,
            "states": self.states[:],
            "moves": self.moves[:],
        }

        # Make sure directory exists
        os.makedirs(os.path.dirname(self.best_game_path), exist_ok=True)

        # If there's no best game file, store this as the first best.
        if not os.path.exists(self.best_game_path):
            with open(self.best_game_path, "wb") as f:
                pickle.dump(data_to_store, f)
            print(f"[INFO] First best game saved with score={self.score}")
            return

        # Compare with current best game
        with open(self.best_game_path, "rb") as f:
            data = pickle.load(f)

        if self.score <= data["score"]:
            return

        # We have a better score, store new best game
        with open(self.best_game_path, "wb") as f:
            pickle.dump(data_to_store, f)
        print(f"[INFO] New best game with score={self.score}, highest={self.highest}")

    # ---------------------------
    #  Public getters & helpers
    # ---------------------------
    def is_game_over(self):
        return self.game_over

    def get_board_2d(self):
        """
        Return a 2D (4x4) Python list representation for easy UI rendering.
        """
        return bitboard_to_board(self.bitboard)

    def get_score(self):
        return self.score

    def get_highest(self):
        return self.highest

    def get_current_best_score(self):
        return self.best_score  # session best

    def set_current_best_score(self, value):
        self.best_score = value

    def view_best_game(self):
        """
        Return the best game data (states, moves, score, highest) if any, otherwise None.
        """
        if not os.path.exists(self.best_game_path):
            print("No best game found.")
            return None

        with open(self.best_game_path, "rb") as f:
            data = pickle.load(f)
        return data
    
# -- Helpers for storing in session --
    def get_state(self):
        return {
            'bitboard': self.bitboard,
            'score': self.score,
            'highest': self.highest,
            'game_over': self.game_over,
            'ai_strategy': self.ai_strategy,
            # Store current depth (based on whichever strategy is active)
            'depth': (
                self.factory.EXPECTIMAX_DEPTH if self.ai_strategy == "Expectimax"
                else self.factory.MINIMAX_DEPTH if self.ai_strategy == "Minimax"
                else 0
            )
        }

    def set_state(self, state):
        self.bitboard = state['bitboard']
        self.score = state['score']
        self.highest = state['highest']
        self.game_over = state['game_over']
        self.ai_strategy = state.get('ai_strategy', "Expectimax")

        # Reinitialize AI strategy
        self.ai = self.factory.create_ai(self.ai_strategy)

        # Restore depth
        saved_depth = state.get('depth', 4)  # Fallback or choose your own default
        if self.ai_strategy in ["Expectimax", "Minimax"]:
            self.set_depth(saved_depth)
