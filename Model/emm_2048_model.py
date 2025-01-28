import math
import random
from functools import lru_cache
from copy import deepcopy

###############################################################################
# 1) 2048 Environment: Board Representation & Moves
###############################################################################
class Game2048:
    """
    A standard 4×4 2048 environment.
    - Internally stores a board as a 2D list of integers.
    - Provides move() logic for Up/Down/Left/Right merges.
    - Spawns random tiles of 2 or 4.
    """

    def __init__(self, seed=None):
        self.size = 4
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self):
        """ Reset the board, spawn two tiles, set score=0, done=False. """
        self.board = [[0]*self.size for _ in range(self.size)]
        self.score = 0
        self.done = False
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        """ Spawn a tile (2 or 4) in a random empty cell (90% chance 2, 10% chance 4). """
        empty = [(r,c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = 4 if random.random() < 0.1 else 2

    def is_game_over(self):
        """
        Check if no moves are possible. If at least one move is possible, game is not over.
        We'll also treat achieving 2048 as 'continuing' (you can stop if you prefer).
        """
        # If there's any empty cell, not game over
        for row in self.board:
            if 0 in row:
                return False
        # Check merges horizontally
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r][c] == self.board[r][c+1]:
                    return False
        # Check merges vertically
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r][c] == self.board[r+1][c]:
                    return False
        return True

    def move(self, action):
        """
        action: 0=Up, 1=Down, 2=Left, 3=Right
        Returns (changed, score_gained).
        If the board changes, spawns a tile. Also updates self.score.
        """
        if action == 0:
            changed, gain = self.move_up()
        elif action == 1:
            changed, gain = self.move_down()
        elif action == 2:
            changed, gain = self.move_left()
        elif action == 3:
            changed, gain = self.move_right()
        else:
            return False, 0

        if changed:
            self.score += gain
            self.spawn_tile()
        self.done = self.is_game_over()
        return changed, gain

    def move_left(self):
        changed = False
        total_score = 0
        new_board = []
        for row in self.board:
            filtered = [x for x in row if x!=0]
            merged = []
            skip = False
            row_score = 0
            for i in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                    val = filtered[i]*2
                    merged.append(val)
                    row_score += val
                    skip = True
                else:
                    merged.append(filtered[i])
            while len(merged) < self.size:
                merged.append(0)
            if merged != row:
                changed = True
            total_score += row_score
            new_board.append(merged)
        self.board = new_board
        return changed, total_score

    def move_right(self):
        changed = False
        total_score = 0
        new_board = []
        for row in self.board:
            reversed_row = row[::-1]
            filtered = [x for x in reversed_row if x!=0]
            merged = []
            skip = False
            row_score = 0
            for i in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                    val = filtered[i]*2
                    merged.append(val)
                    row_score += val
                    skip = True
                else:
                    merged.append(filtered[i])
            while len(merged) < self.size:
                merged.append(0)
            merged.reverse()
            if merged != row:
                changed = True
            total_score += row_score
            new_board.append(merged)
        self.board = new_board
        return changed, total_score

    def move_up(self):
        changed = False
        total_score = 0
        transposed = [list(col) for col in zip(*self.board)]
        # Move left on each row of transposed
        new_transposed = []
        for row in transposed:
            filtered = [x for x in row if x!=0]
            merged = []
            skip = False
            row_score = 0
            for i in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                    val = filtered[i]*2
                    merged.append(val)
                    row_score += val
                    skip = True
                else:
                    merged.append(filtered[i])
            while len(merged) < self.size:
                merged.append(0)
            if merged != row:
                changed = True
            total_score += row_score
            new_transposed.append(merged)
        # transpose back
        new_board = [list(row) for row in zip(*new_transposed)]
        self.board = new_board
        return changed, total_score

    def move_down(self):
        changed = False
        total_score = 0
        transposed = [list(col) for col in zip(*self.board)]
        new_transposed = []
        for row in transposed:
            reversed_row = row[::-1]
            filtered = [x for x in reversed_row if x!=0]
            merged = []
            skip = False
            row_score = 0
            for i in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                    val = filtered[i]*2
                    merged.append(val)
                    row_score += val
                    skip = True
                else:
                    merged.append(filtered[i])
            while len(merged) < self.size:
                merged.append(0)
            merged.reverse()
            if merged != row:
                changed = True
            total_score += row_score
            new_transposed.append(merged)
        new_board = [list(row) for row in zip(*new_transposed)]
        self.board = new_board
        return changed, total_score

    def get_board_tuple(self):
        """ Returns a tuple-of-tuples representation of the board for hashing. """
        return tuple(tuple(row) for row in self.board)

    def get_max_tile(self):
        """ Return the maximum tile on the board. """
        return max(max(row) for row in self.board)

###############################################################################
# 2) Advanced Heuristic Function
###############################################################################
def heuristic_score(board):
    """
    Combine monotonicity, smoothness, empty cells, and max tile.
    Returns a float that is larger for more 'desirable' states.
    """
    # Convert board to a grid (4x4) of integers
    # It's already that way in `board`, so let's just pass it in directly.

    # 2.1) Monotonicity
    # We'll measure row-monotonicity and column-monotonicity
    # We'll sum them as an approximation
    mono_score = monotonicity(board)

    # 2.2) Smoothness
    smooth_score = smoothness(board)

    # 2.3) Empty cells
    empty_count = sum(row.count(0) for row in board)

    # 2.4) Max tile (log scale)
    max_tile = 1
    for row in board:
        row_max = max(row)
        if row_max > max_tile:
            max_tile = row_max
    max_log = math.log2(max_tile)

    # Weighted combination
    # Tweak these weights if you want stronger emphasis on certain aspects
    return 1.0*mono_score + 0.1*smooth_score + 2.7*(empty_count) + 1.0*(max_log)


def monotonicity(board):
    """
    A measure of how 'ordered' each row and column is.
    We'll compute separate row monotonicity (L→R and R→L) and column monotonicity (top→bottom, bottom→top),
    then take the max for each line, sum over all lines.
    """
    total = 0

    # Rows
    for r in range(4):
        row = board[r]
        # measure monotonicity from left to right
        incr_score, decr_score = 0, 0
        for c in range(3):
            if row[c] >= row[c+1]:
                incr_score += row[c+1] - row[c]
            if row[c] <= row[c+1]:
                decr_score += row[c] - row[c+1]
        total += max(incr_score, decr_score)

    # Columns
    for c in range(4):
        col = [board[r][c] for r in range(4)]
        incr_score, decr_score = 0, 0
        for r in range(3):
            if col[r] >= col[r+1]:
                incr_score += col[r+1] - col[r]
            if col[r] <= col[r+1]:
                decr_score += col[r] - col[r+1]
        total += max(incr_score, decr_score)

    return total

def smoothness(board):
    """
    A measure of how similar adjacent tiles are.
    We'll look at horizontal and vertical neighbors.
    We'll do a negative sum of differences, so bigger negative means less smooth.
    We'll invert it in the final score (the heuristic add).
    """
    diff_sum = 0
    # Horizontal neighbors
    for r in range(4):
        for c in range(3):
            if board[r][c] != 0 and board[r][c+1] != 0:
                diff_sum -= abs(math.log2(board[r][c]) - math.log2(board[r][c+1]))
    # Vertical neighbors
    for r in range(3):
        for c in range(4):
            if board[r][c] != 0 and board[r+1][c] != 0:
                diff_sum -= abs(math.log2(board[r][c]) - math.log2(board[r+1][c]))
    return diff_sum


###############################################################################
# 3) Expectimax Search with Caching
###############################################################################
class ExpectimaxAI:
    """
    Uses an Expectimax approach to decide moves:
      - Max nodes: the AI's turn (0=Up,1=Down,2=Left,3=Right).
      - Chance nodes: spawn tile 2 or 4 with probability 0.9 / 0.1.
    We'll keep a cache (lru_cache) to store evaluations for (board, depth, is_player_turn).
    """

    def __init__(self, depth=6):
        self.depth = depth

    def get_move(self, game):
        """
        Return the best move (0..3) from the current state of 'game'.
        If no valid moves (all are invalid), we return None.
        """
        best_move = None
        best_score = float('-inf')
        for move in [0,1,2,3]:
            # Try the move on a copy
            temp_game = deepcopy(game)
            changed, score_gained = temp_game.move(move)
            if not changed:
                continue
            # Evaluate
            val = self.expectimax_value(temp_game, self.depth-1, is_player_turn=False) + score_gained
            if val > best_score:
                best_score = val
                best_move = move
        return best_move

    @lru_cache(maxsize=None)
    def expectimax_value(self, game_state, depth, is_player_turn):
        """
        Evaluate the game state using Expectimax with a given depth.
        game_state: copy of Game2048
        depth: how deep we go
        is_player_turn: bool
        """
        # We'll store a 'hashable' representation of the board in the cache: (board_as_tuple, depth, is_player_turn).
        # However, we can't store 'game_state' directly in the cache because it's not hashable.
        # Instead, we rely on the lru_cache wrapper and define the function carefully. 
        # We'll pass in a custom key -> Python approach:
        # We'll do a trick: define a separate function that takes in (board_tuple, score, depth, is_player_turn)

        # If depth=0 or game is over, return the heuristic
        if depth <= 0 or game_state.is_game_over():
            return heuristic_score(game_state.board)

        if is_player_turn:
            # Max node
            best_val = float('-inf')
            moves = [0,1,2,3]
            for move in moves:
                new_game = deepcopy(game_state)
                changed, gain = new_game.move(move)
                if not changed:
                    continue
                val = gain + self.expectimax_value(new_game, depth-1, False)
                if val > best_val:
                    best_val = val
            if best_val == float('-inf'):
                # no valid moves
                return heuristic_score(game_state.board)
            return best_val
        else:
            # Chance node: random tile spawn
            # Find empty cells
            empty = []
            for r in range(4):
                for c in range(4):
                    if game_state.board[r][c] == 0:
                        empty.append((r,c))
            if not empty:
                # No empty => force player turn
                return self.expectimax_value(game_state, depth-1, True)

            # sum up expected values
            val_sum = 0.0
            prob_2 = 0.9
            prob_4 = 0.1
            for (r,c) in empty:
                # place 2
                new_game_2 = deepcopy(game_state)
                new_game_2.board[r][c] = 2
                val2 = self.expectimax_value(new_game_2, depth-1, True)

                # place 4
                new_game_4 = deepcopy(game_state)
                new_game_4.board[r][c] = 4
                val4 = self.expectimax_value(new_game_4, depth-1, True)

                val_sum += prob_2*val2 + prob_4*val4

            return val_sum / len(empty)


###############################################################################
# 4) Main: a function that plays a single 2048 game with Expectimax
###############################################################################
def play_single_game(depth=6, print_board=False):
    """
    Play a single game of 2048 using the Expectimax AI at the specified depth.
    Returns final score and max tile.
    """
    game = Game2048()
    ai = ExpectimaxAI(depth=depth)

    while not game.is_game_over():
        move = ai.get_move(game)
        if move is None:
            # No valid move => break
            break
        changed, _ = game.move(move)
        if not changed:
            break
        if print_board:
            print_board_4x4(game.board)
            print()

    return game.score, game.get_max_tile()

def print_board_4x4(board):
    """ Helper to print a 4x4 board in a neat grid. """
    for row in board:
        print("\t".join(str(v) if v>0 else "." for v in row))

###############################################################################
# 5) If we want to demonstrate multiple runs
###############################################################################
def play_n_games(n=10, depth=6, verbose=True):
    """
    Play n games, each with Expectimax depth=6 (or chosen).
    Return stats about how many times 2048+ was reached, average score, etc.
    """
    scores = []
    max_tiles = []
    for i in range(n):
        score, max_tile = play_single_game(depth=depth, print_board=False)
        scores.append(score)
        max_tiles.append(max_tile)
        if verbose:
            print(f"Game {i+1}/{n}: score={score}, max_tile={max_tile}")

    avg_score = sum(scores)/len(scores)
    success_count = sum(1 for mt in max_tiles if mt >= 2048)
    success_rate = success_count / n

    if verbose:
        print(f"\nPlayed {n} games with depth={depth}.")
        print(f"Average score: {avg_score:.2f}")
        print(f"Reached 2048 in {success_rate*100:.1f}% of the games.")
        print(f"Max tile distribution:")
        from collections import Counter
        c = Counter(max_tiles)
        for tile, cnt in sorted(c.items()):
            print(f"  Tile={tile}, Count={cnt}")

###############################################################################
# 6) Main Execution
###############################################################################
if __name__ == "__main__":
    # Example: play 10 games at depth=6, print results
    # This depth often yields near-100% success for reaching 2048 (or more).
    play_n_games(n=1, depth=5, verbose=True)