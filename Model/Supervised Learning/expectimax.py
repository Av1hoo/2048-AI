import numpy as np
from copy import deepcopy
import random
import tkinter as tk

class Game2048:
    def __init__(self, headless=True):
        self.size = 4
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.headless = headless
        if not headless:
            self.setup_ui()

    def setup_ui(self):
        # UI setup code here if you want visual feedback
        pass

    def reset_game(self):
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            # Ensure we're modifying the list, not a tuple
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        moved = False
        for i in range(self.size):
            new_row, did_move, merge_score = self._compress_and_merge(self.board[i])
            if did_move:
                moved = True
                self.score += merge_score
                self.board[i] = new_row  # Assign to list, not tuple
        return moved

    def move_right(self):
        return self._move_helper(lambda row: row[::-1], self.move_left)

    def move_up(self):
        return self._move_helper(lambda board: list(zip(*board)), self.move_left)

    def move_down(self):
        return self._move_helper(lambda board: list(zip(*board[::-1]))[::-1], self.move_left)

    def _move_helper(self, transform, move_func):
        # Convert to list of lists to ensure mutable operations
        board_list = [list(row) for row in transform(self.board)]
        moved = move_func()
        # Convert back to list of lists
        self.board = [list(row) for row in transform(board_list)]
        return moved

    def _compress_and_merge(self, row):
        row = [x for x in row if x != 0]
        new_row = []
        score = 0
        i = 0
        while i < len(row):
            if i + 1 < len(row) and row[i] == row[i + 1]:
                new_row.append(row[i] * 2)
                score += row[i] * 2
                i += 2
            else:
                new_row.append(row[i])
                i += 1
        while len(new_row) < self.size:
            new_row.append(0)
        moved = new_row != row
        return new_row, moved, score

    def execute_move(self, direction):
        if direction == 'Up':
            return self.move_up()
        elif direction == 'Down':
            return self.move_down()
        elif direction == 'Left':
            return self.move_left()
        elif direction == 'Right':
            return self.move_right()
        return False

    def get_available_moves(self):
        return [i for i, move in enumerate(['Up', 'Down', 'Left', 'Right']) if self.execute_move(move)]

    def is_game_over(self):
        return not self.get_available_moves()

    def score_board(self):
        return self.score

    def get_new_state(self, action):
        board_copy = deepcopy(self.board)
        score_copy = self.score
        if self.execute_move(['Up', 'Down', 'Left', 'Right'][action]):
            self.spawn_tile()
            new_state = deepcopy(self.board)
            reward = self.score - score_copy
            done = self.is_game_over()
            self.board = board_copy  # revert changes for next simulation
            self.score = score_copy
            return new_state, reward, done
        return board_copy, 0, False  # No change if the move doesn't alter the board

class Expectimax:
    def __init__(self, depth=4):
        self.depth = depth

    def heuristic(self, game):
        board = game.board
        score = game.score
        monotonicity_score = 0
        empty_cells = 0
        
        for row in board:
            for i in range(3):
                if row[i] == 0:
                    empty_cells += 1
                    continue
                if row[i] == row[i+1]:
                    monotonicity_score += row[i]
                elif row[i] < row[i+1]:
                    monotonicity_score += row[i+1] - row[i]
                else:
                    monotonicity_score -= row[i] - row[i+1]

        # Same for columns
        for col in zip(*board):
            for i in range(3):
                if col[i] == 0:
                    continue
                if col[i] == col[i+1]:
                    monotonicity_score += col[i]
                elif col[i] < col[i+1]:
                    monotonicity_score += col[i+1] - col[i]
                else:
                    monotonicity_score -= col[i] - col[i+1]

        return score + empty_cells * 200 + monotonicity_score

    def expectimax(self, game, depth):
        if depth == 0 or game.is_game_over():
            return self.heuristic(game), None
        
        moves = game.get_available_moves()
        if not moves:
            return self.heuristic(game), None

        best_move = None
        best_score = -float('inf')
        for move in moves:
            new_game = Game2048(headless=True)
            new_game.board = deepcopy(game.board)
            new_game.score = game.score
            
            _, reward, _ = new_game.get_new_state(move)
            # Here, we ensure that expectimax_expect returns a tuple with the score
            score = -self.expectimax_expect(new_game, depth - 1)[0]
            
            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move

    def expectimax_expect(self, game, depth):
        if depth == 0 or game.is_game_over():
            return (self.heuristic(game),)  # Return as a tuple

        empty_cells = [(i, j) for i in range(4) for j in range(4) if game.board[i][j] == 0]
        if not empty_cells:
            return (self.heuristic(game),)

        expected_value = 0
        for i, j in empty_cells:
            game.board[i][j] = 2
            expected_value += 0.9 * self.expectimax(game, depth)[0]
            game.board[i][j] = 4
            expected_value += 0.1 * self.expectimax(game, depth)[0]
            game.board[i][j] = 0  # Reset cell

        return (expected_value / len(empty_cells),)  # Return as a tuple

def play_game_with_expectimax():
    game = Game2048(headless=True)
    expectimax = Expectimax(depth=4)

    while not game.is_game_over():
        _, move = expectimax.expectimax(game, expectimax.depth)
        if move is None:  # This should not happen if there are moves left
            break
        game.execute_move(['Up', 'Down', 'Left', 'Right'][move])
        game.spawn_tile()
        print(f"Move: {['Up', 'Down', 'Left', 'Right'][move]}, Score: {game.score}, Highest Tile: {max([max(row) for row in game.board])}")

    print(f"Game Over. Final Score: {game.score}, Highest Tile: {max([max(row) for row in game.board])}")

if __name__ == "__main__":
    play_game_with_expectimax()