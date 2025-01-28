import tkinter as tk
import random
import math
import torch
import torch.nn as nn
from copy import deepcopy
import pickle
import os
from functools import lru_cache

###############################################################################
# 1) Expectimax Snippet (Renamed Game2048 -> GameState2048)
###############################################################################
class GameState2048:
    """
    A standard 4×4 2048 environment for Expectimax search (no GUI).
    This is separate from the Tkinter-based Game2048 to avoid naming conflicts.
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
        Check if no moves are possible. 
        We'll consider a state 'done' if no merges and no empty spaces exist.
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
        If the board changes, spawn a tile. Also updates self.score.
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

    def get_max_tile(self):
        """ Return the maximum tile on the board. """
        return max(max(row) for row in self.board)

###############################################################################
# 2) Heuristic & Expectimax
###############################################################################
def monotonicity(board):
    """
    A measure of how 'ordered' each row and column is.
    We compute row-monotonicity and column-monotonicity,
    then sum them.
    """
    total = 0

    # Rows
    for r in range(4):
        row = board[r]
        incr_score, decr_score = 0, 0
        for c in range(3):
            # difference = next - current
            diff = row[c+1] - row[c]
            if diff > 0:
                incr_score += diff
            else:
                decr_score -= diff
        # we take the max of "strictly increasing" vs "strictly decreasing"
        # (some implementations do negative differences; can be tweaked)
        total += max(incr_score, decr_score)

    # Columns
    for c in range(4):
        col = [board[r][c] for r in range(4)]
        incr_score, decr_score = 0, 0
        for r in range(3):
            diff = col[r+1] - col[r]
            if diff > 0:
                incr_score += diff
            else:
                decr_score -= diff
        total += max(incr_score, decr_score)

    return total

def smoothness(board):
    """
    A measure of how similar adjacent tiles are.
    We'll look at horizontal and vertical neighbors,
    summing negative differences in log space.
    """
    diff_sum = 0
    for r in range(4):
        for c in range(3):
            if board[r][c] != 0 and board[r][c+1] != 0:
                diff_sum -= abs(math.log2(board[r][c]) - math.log2(board[r][c+1]))
    for r in range(3):
        for c in range(4):
            if board[r][c] != 0 and board[r+1][c] != 0:
                diff_sum -= abs(math.log2(board[r][c]) - math.log2(board[r+1][c]))
    return diff_sum

def heuristic_score(board):
    """
    Weighted combination of monotonicity, smoothness, empty cells, and max tile.
    """
    mono = monotonicity(board)
    smooth = smoothness(board)
    empty_count = sum(row.count(0) for row in board)
    max_tile = max(max(row) for row in board)
    max_log = math.log2(max_tile) if max_tile>0 else 0

    # You can tweak these weights
    return 1.0*mono + 0.1*smooth + 2.7*(empty_count) + 1.0*(max_log)

class ExpectimaxAI:
    """
    Standard Expectimax with chance nodes (2 or 4).
    """

    def __init__(self, depth=6):
        self.depth = depth

    def get_move(self, game):
        """
        Return the best move (0..3) for the current state in 'game' (a GameState2048).
        """
        best_move = None
        best_score = float('-inf')
        for move in [0,1,2,3]:
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
        Evaluate the game state with depth-limited expectimax.
        """
        if depth <= 0 or game_state.is_game_over():
            return heuristic_score(game_state.board)

        if is_player_turn:
            # Max node
            best_val = float('-inf')
            for move in [0,1,2,3]:
                new_game = deepcopy(game_state)
                changed, gained = new_game.move(move)
                if not changed:
                    continue
                val = gained + self.expectimax_value(new_game, depth-1, False)
                if val > best_val:
                    best_val = val
            if best_val == float('-inf'):  # no valid moves
                return heuristic_score(game_state.board)
            return best_val
        else:
            # Chance node
            empty = []
            for r in range(4):
                for c in range(4):
                    if game_state.board[r][c] == 0:
                        empty.append((r,c))
            if not empty:
                # no empty => treat as player node
                return self.expectimax_value(game_state, depth-1, True)

            prob_2 = 0.9
            prob_4 = 0.1
            val_sum = 0.0
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
# 3) Helper: get_best_move_expectimax()
###############################################################################
def get_best_move_expectimax(board, depth=6):
    """
    Given a 4×4 board from the Tkinter GUI game,
    use the ExpectimaxAI to pick the best move (0=Up,1=Down,2=Left,3=Right).
    Returns None if no valid moves.
    """
    # Clone the board into a GameState2048
    game = GameState2048()
    # Start empty, then copy the board exactly.
    game.board = [row[:] for row in board]
    game.done = game.is_game_over()

    ai = ExpectimaxAI(depth=depth)
    return ai.get_move(game)


###############################################################################
# 4) Your Existing Tkinter Game (Unmodified logic for DQN + UI),
#    with an added key 'e' for Expectimax single-step.
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNModel(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=256, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

model = DQNModel()  
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Action index to direction
action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

def board_to_exponents(board):
    """
    Convert a 4×4 board of tile values to exponents [0..15].
    """
    exps = []
    for row in board:
        for val in row:
            if val == 0:
                exps.append(0)
            else:
                exp = int(math.log2(val))
                if exp > 15:  # clamp
                    exp = 15
                exps.append(exp)
    return exps

def predict_moves_sorted_dqn(board):
    """
    Given a 4x4 board, return a list of directions sorted by descending Q-value.
    """
    exps = board_to_exponents(board)
    state_t = torch.tensor(exps, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t)  # shape [1,4]
    sorted_indices = torch.argsort(q_values, dim=1, descending=True).squeeze(0).tolist()
    return [action_map[i] for i in sorted_indices]


class Game2048:
    def __init__(self, master, headless=False):
        self.size = 4
        self.board = [[0]*4 for _ in range(4)]
        self.score = 0
        self.highest_tile = 0
        self.game_over = False
        self.history = []
        self.moves_history = []
        self.best_score = 0
        self.game_states = []
        self.best_game_path = "best_game.pkl"

        self.bg_color = "#BBADA0"
        self.cell_colors = {
            0:    "#CDC1B4",
            2:    "#EEE4DA",
            4:    "#EDE0C8",
            8:    "#F2B179",
            16:   "#F59563",
            32:   "#F67C5F",
            64:   "#F65E3B",
            128:  "#EDCF72",
            256:  "#EDCC61",
            512:  "#EDC850",
            1024: "#EDC53F",
            2048: "#EDC22E",
        }

        self.headless = headless
        self.master = master

        if not self.headless:
            self._build_ui()
            self.reset_game()

    def _build_ui(self):
        self.master.title("2048 w/ DQN + Expectimax")
        self.master.geometry("600x500")
        self.master.resizable(False, False)

        self.top_frame = tk.Frame(self.master, bg=self.bg_color)
        self.top_frame.pack(pady=5)

        self.score_label = tk.Label(self.top_frame, text="Score: 0",
                                    font=("Helvetica", 14, "bold"), bg=self.bg_color)
        self.score_label.pack(side="left", padx=10)

        self.best_label = tk.Label(self.top_frame, text="Best: 0",
                                   font=("Helvetica", 14, "bold"), bg=self.bg_color)
        self.best_label.pack(side="left", padx=10)

        self.highest_label = tk.Label(self.top_frame, text="Highest: 0",
                                      font=("Helvetica", 14, "bold"), bg=self.bg_color)
        self.highest_label.pack(side="left", padx=10)

        self.game_over_label = tk.Label(self.top_frame, text="",
                                        font=("Helvetica", 16, "bold"),
                                        fg="red", bg=self.bg_color)
        self.game_over_label.pack(side="left", padx=10)

        self.main_frame = tk.Frame(self.master, bg=self.bg_color)
        self.main_frame.pack(padx=10, pady=10)
        self.tiles = []
        for r in range(4):
            row_tiles = []
            for c in range(4):
                lbl = tk.Label(self.main_frame, text="", bg=self.cell_colors[0],
                               font=("Helvetica", 20, "bold"), width=4, height=2)
                lbl.grid(row=r, column=c, padx=5, pady=5)
                row_tiles.append(lbl)
            self.tiles.append(row_tiles)

        self.bottom_frame = tk.Frame(self.master, bg=self.bg_color)
        self.bottom_frame.pack()

        btn_restart = tk.Button(self.bottom_frame, text="Restart (R)",
                                command=self.key_restart, bg="#8f7a66",
                                fg="white", font=("Helvetica", 12, "bold"))
        btn_restart.pack(side="left", padx=5)

        btn_viewer = tk.Button(self.bottom_frame, text="View Best (V)",
                               command=self.view_best_game, bg="#8f7a66",
                               fg="white", font=("Helvetica", 12, "bold"))
        btn_viewer.pack(side="left", padx=5)

        # Key bindings
        self.master.bind("r", self.key_restart)
        self.master.bind("R", self.key_restart)

        self.master.bind("<Up>", self.key_up)
        self.master.bind("<Down>", self.key_down)
        self.master.bind("<Left>", self.key_left)
        self.master.bind("<Right>", self.key_right)

        # Single-step DQN (i) or continuous DQN (c)
        self.master.bind("i", self.key_ml_single)
        self.master.bind("c", self.key_ml_continuous)

        # Single-step Expectimax (e)
        self.master.bind("e", self.key_em_single)

        # Open best viewer
        self.master.bind("v", lambda e: self.view_best_game())

    def reset_game(self):
        self.board = [[0]*4 for _ in range(4)]
        self.score = 0
        self.game_over = False
        self.highest_tile = 0
        self.history.clear()
        self.moves_history.clear()
        self.game_states.clear()

        self.spawn_tile()
        self.spawn_tile()
        self.game_states.append(deepcopy(self.board))

        self.update_ui()

    # Basic arrow-key moves
    def key_left(self, event=None):
        self.apply_move("Left")

    def key_right(self, event=None):
        self.apply_move("Right")

    def key_up(self, event=None):
        self.apply_move("Up")

    def key_down(self, event=None):
        self.apply_move("Down")

    # Single-step DQN
    def key_ml_single(self, event=None):
        if self.game_over:
            return
        sorted_dirs = predict_moves_sorted_dqn(self.board)
        for d in sorted_dirs:
            moved = self._execute(d)
            if moved:
                self.history.append((None, None))
                self.spawn_tile()
                self.game_states.append(deepcopy(self.board))
                self.moves_history.append(d)
                self.update_ui()
                if self.is_game_over():
                    self.finish_game()
                return
        self.finish_game()

    # Continuous DQN
    def key_ml_continuous(self, event=None):
        def step():
            if self.game_over:
                return
            sorted_dirs = predict_moves_sorted_dqn(self.board)
            moved_any = False
            for d in sorted_dirs:
                m = self._execute(d)
                if m:
                    self.spawn_tile()
                    self.game_states.append(deepcopy(self.board))
                    self.moves_history.append(d)
                    moved_any = True
                    break
            self.update_ui()
            if self.is_game_over() or not moved_any:
                self.finish_game()
            else:
                self.master.after(50, step)

        step()

    # NEW: Single-step Expectimax
    def key_em_single(self, event=None):
        if self.game_over:
            return
        best_move = get_best_move_expectimax(self.board, depth=6)
        # If no valid move, game over
        if best_move is None:
            self.finish_game()
            return
        # Convert best_move (0..3) to text direction
        move_map = {0:'Up', 1:'Down', 2:'Left', 3:'Right'}
        direction = move_map[best_move]
        moved = self._execute(direction)
        if moved:
            self.history.append((None, None))
            self.spawn_tile()
            self.game_states.append(deepcopy(self.board))
            self.moves_history.append(direction)
            self.update_ui()
            if self.is_game_over():
                self.finish_game()
        else:
            self.finish_game()

    # Common move apply
    def apply_move(self, direction):
        if self.game_over:
            return
        pre_board = deepcopy(self.board)
        pre_score = self.score
        moved = self._execute(direction)
        if moved:
            self.history.append((pre_board, pre_score))
            self.spawn_tile()
            self.game_states.append(deepcopy(self.board))
            self.moves_history.append(direction)
            self.update_ui()
            if self.is_game_over():
                self.finish_game()
        return moved

    def _execute(self, direction):
        if direction == "Up":
            return self.move_up()
        elif direction == "Down":
            return self.move_down()
        elif direction == "Left":
            return self.move_left()
        elif direction == "Right":
            return self.move_right()
        return False

    def spawn_tile(self):
        empty = [(r,c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if not empty:
            return
        r,c = random.choice(empty)
        self.board[r][c] = 4 if random.random()<0.1 else 2

    def move_left(self):
        changed = False
        total_score = 0
        for r in range(4):
            row = self.board[r]
            new_row, sc, did_move = self._compress_merge(row)
            self.board[r] = new_row
            if did_move:
                changed = True
                total_score += sc
        if changed:
            self.score += total_score
        return changed

    def move_right(self):
        changed = False
        total_score = 0
        for r in range(4):
            row = self.board[r][::-1]
            new_row, sc, did_move = self._compress_merge(row)
            new_row.reverse()
            if did_move:
                changed = True
                total_score += sc
            self.board[r] = new_row
        if changed:
            self.score += total_score
        return changed

    def move_up(self):
        changed = False
        total_score = 0
        for c in range(4):
            col = [self.board[r][c] for r in range(4)]
            new_col, sc, did_move = self._compress_merge(col)
            if did_move:
                changed = True
                total_score += sc
            for r in range(4):
                self.board[r][c] = new_col[r]
        if changed:
            self.score += total_score
        return changed

    def move_down(self):
        changed = False
        total_score = 0
        for c in range(4):
            col = [self.board[r][c] for r in range(4)][::-1]
            new_col, sc, did_move = self._compress_merge(col)
            new_col.reverse()
            if new_col != [self.board[r][c] for r in range(4)]:
                changed = True
                total_score += sc
            for r in range(4):
                self.board[r][c] = new_col[r]
        if changed:
            self.score += total_score
        return changed

    def _compress_merge(self, arr):
        original = list(arr)
        filtered = [x for x in arr if x!=0]
        merged = []
        sc = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip=False
                continue
            if i<len(filtered)-1 and filtered[i]==filtered[i+1]:
                val = filtered[i]*2
                merged.append(val)
                sc+=val
                skip=True
            else:
                merged.append(filtered[i])
        while len(merged)<4:
            merged.append(0)
        did_move = (merged!=original)
        hi = max(merged)
        if hi>self.highest_tile:
            self.highest_tile=hi
        return merged, sc, did_move

    def is_game_over(self):
        for row in self.board:
            if 0 in row:
                return False
        for r in range(4):
            for c in range(3):
                if self.board[r][c] == self.board[r][c+1]:
                    return False
        for c in range(4):
            for r in range(3):
                if self.board[r][c] == self.board[r+1][c]:
                    return False
        return True

    def finish_game(self):
        self.game_over = True
        self.save_if_best()
        if not self.headless:
            self.game_over_label.config(text="Game Over!")

    def save_if_best(self):
        current_score = self.score
        best_score_stored = -1
        if os.path.exists(self.best_game_path):
            try:
                with open(self.best_game_path, "rb") as f:
                    data = pickle.load(f)
                best_score_stored = data.get("score", -1)
            except:
                pass
        if current_score > best_score_stored:
            data_to_store = {
                "score": self.score,
                "highest_tile": self.highest_tile,
                "states": self.game_states,
                "moves": self.moves_history
            }
            with open(self.best_game_path, "wb") as f:
                pickle.dump(data_to_store, f)
            print(f"New best game saved with score={self.score}, highest tile={self.highest_tile}")

    def update_ui(self):
        if self.headless:
            return
        for r in range(4):
            for c in range(4):
                val = self.board[r][c]
                color = self.cell_colors.get(val, "#3C3A32")
                txt = str(val) if val>0 else ""
                self.tiles[r][c].config(text=txt, bg=color)
        if self.score> self.best_score:
            self.best_score = self.score
        self.score_label.config(text=f"Score: {self.score}")
        self.best_label.config(text=f"Best: {self.best_score}")
        self.highest_label.config(text=f"Highest: {self.highest_tile}")
        self.master.update_idletasks()

    def key_restart(self, event=None):
        self.reset_game()
        if not self.headless:
            self.game_over_label.config(text="")

    def view_best_game(self):
        if not os.path.exists(self.best_game_path):
            print("No best_game.pkl found.")
            return
        with open(self.best_game_path, "rb") as f:
            data = pickle.load(f)
        states = data["states"]
        moves = data["moves"]
        score = data["score"]
        hi_tile = data["highest_tile"]

        viewer = BestGameViewer(self.master, states, moves, score, hi_tile, self.cell_colors)
        viewer.show_frame(0)

class BestGameViewer(tk.Toplevel):
    def __init__(self, master, states, moves, final_score, final_highest, cell_colors):
        super().__init__(master)
        self.title("Best Game Viewer")
        self.geometry("400x400")
        self.resizable(False, False)

        self.states = states
        self.moves = moves
        self.final_score = final_score
        self.final_highest = final_highest
        self.cell_colors = cell_colors
        self.idx = 0

        self.label_info = tk.Label(self, text="", font=("Helvetica", 12, "bold"))
        self.label_info.pack()

        self.grid_frame = tk.Frame(self, bg="#BBADA0")
        self.grid_frame.pack(pady=5)

        self.tiles = []
        for r in range(4):
            row_tiles = []
            for c in range(4):
                lbl = tk.Label(self.grid_frame, text="", bg=cell_colors[0],
                               font=("Helvetica", 16, "bold"), width=4, height=2)
                lbl.grid(row=r, column=c, padx=3, pady=3)
                row_tiles.append(lbl)
            self.tiles.append(row_tiles)

        nav_frame = tk.Frame(self)
        nav_frame.pack()

        btn_prev = tk.Button(nav_frame, text="Prev", command=self.prev_state)
        btn_prev.pack(side="left", padx=5)

        btn_next = tk.Button(nav_frame, text="Next", command=self.next_state)
        btn_next.pack(side="left", padx=5)

        btn_auto = tk.Button(nav_frame, text="Play", command=self.auto_play)
        btn_auto.pack(side="left", padx=5)

    def show_frame(self, idx):
        self.idx = idx
        if self.idx<0:
            self.idx=0
        if self.idx>= len(self.states):
            self.idx = len(self.states)-1

        board = self.states[self.idx]
        for r in range(4):
            for c in range(4):
                val = board[r][c]
                color = self.cell_colors.get(val, "#3C3A32")
                txt = str(val) if val>0 else ""
                self.tiles[r][c].config(text=txt, bg=color)

        msg = f"Move {self.idx}/{len(self.states)-1}"
        if self.idx>0 and self.idx-1< len(self.moves):
            msg += f" | Action: {self.moves[self.idx-1]}"
        msg += f" | Score={self.final_score}, Highest={self.final_highest}"
        self.label_info.config(text=msg)

    def next_state(self):
        self.show_frame(self.idx+1)

    def prev_state(self):
        self.show_frame(self.idx-1)

    def auto_play(self):
        if self.idx >= len(self.states)-1:
            return
        self.next_state()
        self.after(50, self.auto_play)


def run_headless_until_2048():
    try:
        game = Game2048(None, headless=True)
        game.reset_game()
        print("Running headless until 2048...")

        games = 0
        while True:
            while not game.is_game_over():
                # Using the DQN approach in a loop:
                sorted_dirs = predict_moves_sorted_dqn(game.board)
                moved = False
                for d in sorted_dirs:
                    moved = game.apply_move(d)
                    if moved:
                        if game.highest_tile >= 2048:
                            print("Reached 2048!")
                            for row in game.board:
                                print(row)
                            return True
                        break
                if not moved:
                    game.reset_game()
            games += 1
            if games % 50 == 0:
                print("Games played:", games, end="\r")
            game.reset_game()
    except Exception as e:
        print(f"Error in headless mode: {e}")
        return False
    
def run_expectimax_until_2048():
    try:
        game = GameState2048()
        print("Running Expectimax until 2048...")
        games = 0
        while True:
            while not game.is_game_over():
                
                ai = ExpectimaxAI(depth=6)
                move = ai.get_move(game)
                changed, score_gained = game.move(move)
                if not changed:
                    break
                if game.get_max_tile() >= 2048:
                    print("Reached 2048!")
                    for row in game.board:
                        print(row)
                    return True
            # save the game if better
            print("Score:", game.score, "Max tile:", game.get_max_tile())
            print(game.board[0], game.board[1], game.board[2], game.board[3], sep="\n")
            games += 1
            if games % 50 == 0:
                print("Games played:", games, end="\r")
            game.reset()
    except Exception as e:
        print(f"Error in Expectimax mode: {e}")
        return False


import math, random, pickle
from functools import lru_cache

###############################################################################
# 1) Build LUTs for Row Merges (Move-Left)
###############################################################################
# row_left_table[row16] = new_row16 after move-left
# row_score_table[row16] = points gained during that row merge

row_left_table = [0]*65536
row_score_table = [0]*65536

def build_row_tables():
    """
    For each possible 16-bit row, interpret it as 4 nibbles (each 0..15),
    simulate a 'move-left' merge, then store:
      - row_left_table[row] = new 16-bit row
      - row_score_table[row] = points gained from merges
    """
    for row_val in range(65536):
        # Extract 4 nibbles
        # nibble 0 = (row_val >>  0) & 0xF  => leftmost in textual reading,
        # but we'll treat nibble 0 as the left cell in the code for convenience.
        # (You can swap interpretation if you prefer.)
        tiles = [
            (row_val >>  0) & 0xF,
            (row_val >>  4) & 0xF,
            (row_val >>  8) & 0xF,
            (row_val >> 12) & 0xF
        ]
        # Convert these tile-exponents into a list we can "compress+merge"
        # Example: [2,2,1,0] means [4,4,2,_] in actual 2048 values if we consider 2^2=4, 2^1=2, etc.

        # 1) Filter out zero
        filtered = [t for t in tiles if t != 0]
        merged = []
        score_gain = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                # merge
                merged_val = filtered[i] + 1  # exponent +1 => e.g. 2^2 + 2^2 => 2^3
                # the actual numeric tile is 2^(merged_val), so add that to score
                score_gain += (1 << merged_val)  # 2^(merged_val)
                merged.append(merged_val)
                skip = True
            else:
                merged.append(filtered[i])

        # fill with zeros up to length 4
        merged.extend([0]*(4 - len(merged)))

        # Rebuild a 16-bit row
        # merged[0] => lowest nibble, merged[1] => next nibble, ...
        new_val = ((merged[0] & 0xF)
                   | ((merged[1] & 0xF) << 4)
                   | ((merged[2] & 0xF) << 8)
                   | ((merged[3] & 0xF) << 12))

        row_left_table[row_val] = new_val
        row_score_table[row_val] = score_gain

def reverse_row(row16):
    """
    Reverse the 4 nibbles of a 16-bit row.
    Example: row16 = ABCD (in hex nibbles) => DCBA
    """
    # Extract nibbles
    n0 = (row16 >>  0) & 0xF
    n1 = (row16 >>  4) & 0xF
    n2 = (row16 >>  8) & 0xF
    n3 = (row16 >> 12) & 0xF
    return (n3) | (n2 << 4) | (n1 << 8) | (n0 << 12)

###############################################################################
# 2) Bitboard Representation
###############################################################################
# The 64-bit "bitboard" is conceptually 4 rows (row0..row3), each row is 16 bits.
# row0 is the lowest 16 bits, row1 next, row2 next, row3 highest 16 bits:
#
#     bits [ 0..15] = row0
#     bits [16..31] = row1
#     bits [32..47] = row2
#     bits [48..63] = row3
#
# In normal 2048 reading, row0 might be the "top row" or the "bottom row",
# you can choose. Just be consistent in get/set functions.

def get_row(bitboard, row_idx):
    """
    Return the 16-bit representation of row 'row_idx' from the bitboard.
    row_idx=0 => lowest 16 bits
    """
    return (bitboard >> (16 * row_idx)) & 0xFFFF

def set_row(bitboard, row_idx, newrow):
    """
    Return new bitboard with row 'row_idx' replaced by 'newrow' (16 bits).
    """
    shift = 16 * row_idx
    # Clear old row
    mask = 0xFFFF << shift
    bitboard &= ~mask
    # Insert new row
    bitboard |= (newrow & 0xFFFF) << shift
    return bitboard

def shift_left(bitboard):
    """
    Move-left operation on the entire 4×4 board (bitboard).
    Returns (new_bitboard, total_score, changed).
    We do each row's LUT, sum scores, track if changed.
    """
    moved = False
    total_score = 0
    new_b = bitboard
    for r in range(4):
        row = get_row(new_b, r)
        new_row = row_left_table[row]
        if new_row != row:
            moved = True
        sc = row_score_table[row]
        total_score += sc
        new_b = set_row(new_b, r, new_row)
    return new_b, total_score, moved

def shift_right(bitboard):
    """
    Move-right by reversing each row, using the LUT for move-left,
    then reversing back.
    """
    moved = False
    total_score = 0
    new_b = bitboard
    for r in range(4):
        row = get_row(new_b, r)
        rev = reverse_row(row)
        new_rev = row_left_table[rev]
        if new_rev != rev:
            moved = True
        sc = row_score_table[rev]
        total_score += sc
        new_row = reverse_row(new_rev)
        new_b = set_row(new_b, r, new_row)
    return new_b, total_score, moved

def transpose(bitboard):
    """
    Transpose the 4×4 board: rows become columns, columns become rows.
    This is a standard trick to handle up/down with the same row LUT logic.
    """
    # We can do a bit-swap approach, or decode each cell. For clarity:
    # decode each row, build 4 arrays, transpose, re-encode.
    # But let's do a slightly optimized approach.

    # We'll extract each nibble (16 total), reorder them into row/col transpose.

    # We'll do a simpler decode approach:
    cells = [0]*16
    for r in range(4):
        row = get_row(bitboard, r)
        for c in range(4):
            cells[r*4 + c] = (row >> (4*c)) & 0xF

    # Now transpose so cell(r,c) -> cell(c,r)
    newcells = [0]*16
    for r in range(4):
        for c in range(4):
            newcells[c*4 + r] = cells[r*4 + c]

    # re-encode into bitboard
    new_b = 0
    for r in range(4):
        rowval = 0
        for c in range(4):
            rowval |= (newcells[r*4 + c] << (4*c))
        new_b = set_row(new_b, r, rowval)
    return new_b

def shift_up(bitboard):
    """
    Move-up is transpose -> shift-left -> transpose.
    """
    b2 = transpose(bitboard)
    b2, sc, moved = shift_left(b2)
    b2 = transpose(b2)
    return b2, sc, moved

def shift_down(bitboard):
    """
    Move-down is transpose -> shift-right -> transpose.
    """
    b2 = transpose(bitboard)
    b2, sc, moved = shift_right(b2)
    b2 = transpose(b2)
    return b2, sc, moved

def bitboard_move(bitboard, action):
    """
    action: 0=Up, 1=Down, 2=Left, 3=Right
    Returns (new_bitboard, score_gained, changed).
    """
    if action == 0:
        return shift_up(bitboard)
    elif action == 1:
        return shift_down(bitboard)
    elif action == 2:
        return shift_left(bitboard)
    elif action == 3:
        return shift_right(bitboard)
    else:
        return bitboard, 0, False

###############################################################################
# 3) Spawning Tiles & Checking Game Over
###############################################################################
def bitboard_count_empty(bitboard):
    """ Count how many of the 16 cells (nibbles) are zero. """
    count = 0
    for _ in range(16):
        if (bitboard & 0xF) == 0:
            count += 1
        bitboard >>= 4
    return count

def bitboard_spawn(bitboard):
    """
    Spawn a tile (2 or 4) into a random empty nibble (90% chance 2).
    Returns new_bitboard.
    If no empty, returns the same bitboard.
    """
    empty_count = bitboard_count_empty(bitboard)
    if empty_count == 0:
        return bitboard

    # pick a random empty index
    r = random.randrange(empty_count)
    val = 1 if random.random() < 0.9 else 2  # exponent 1 => tile=2, exponent=2 => tile=4

    # find the r-th empty nibble
    new_b = bitboard
    shift = 0
    empty_seen = 0
    while True:
        nib = (new_b >> shift) & 0xF
        if nib == 0:
            if empty_seen == r:
                # place val
                new_b &= ~(0xF << shift)  # clear
                new_b |= (val << shift)   # set
                return new_b
            empty_seen += 1
        shift += 4

def bitboard_is_game_over(bitboard):
    """
    Check if no moves are possible (and no empty). If there's any empty => not over.
    Or if any merge is possible => not over.
    """
    # Quick check for any empty
    if bitboard_count_empty(bitboard) > 0:
        return False

    # Check merges horizontally
    tmp = bitboard
    for r in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        # if row or merges, we can do a quick check of nibble pairs
        for c in range(3):
            left_nib = (row >> (4*c)) & 0xF
            right_nib = (row >> (4*(c+1))) & 0xF
            if left_nib == right_nib:
                return False

    # Check merges vertically
    # easiest is to transpose and do the same
    t = transpose(bitboard)
    tmp = t
    for r in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(3):
            up_nib = (row >> (4*c)) & 0xF
            down_nib = (row >> (4*(c+1))) & 0xF
            if up_nib == down_nib:
                return False
    return True

def bitboard_get_max_tile(bitboard):
    """ Return the actual numeric value of the largest tile. """
    max_exponent = 0
    tmp = bitboard
    for _ in range(16):
        nib = tmp & 0xF
        if nib > max_exponent:
            max_exponent = nib
        tmp >>= 4
    return (1 << max_exponent)  # 2^exponent

###############################################################################
# 4) Heuristic & Expectimax (with LRU Cache)
###############################################################################

def bitboard_heuristic(bitboard):
    """
    Combine monotonicity, smoothness, empty cells, max tile, etc.
    We decode the 16 nibbles for convenience.
    """
    # decode
    tiles = []
    tmp = bitboard
    for _ in range(16):
        tiles.append(tmp & 0xF)
        tmp >>= 4

    # Rebuild as a 4×4 in exponent form
    # row0: tiles[0..3], row1: tiles[4..7], ...
    board_exp = []
    idx = 0
    for r in range(4):
        row = tiles[idx:idx+4]
        idx+=4
        board_exp.append(row)
    # We'll rely on the same logic as normal heuristics, but treat them as exponents.

    # monotonicity
    mono = monotonicity_exponents(board_exp)
    # smoothness
    smooth = smoothness_exponents(board_exp)
    # empty
    empty_count = sum(row.count(0) for row in board_exp)
    # max tile
    max_e = max(max(row) for row in board_exp)
    max_log = max_e  # already exponent
    # Weighted
    return 1.0*mono + 0.1*smooth + 2.7*(empty_count) + 1.0*max_log

def monotonicity_exponents(board_exp):
    """
    Similar to standard monotonicity, but the "value" is the exponent.
    """
    total = 0
    # Rows
    for r in range(4):
        row = board_exp[r]
        incr, decr = 0, 0
        for c in range(3):
            if row[c+1] > row[c]:
                incr += (row[c+1] - row[c])
            else:
                decr += (row[c] - row[c+1])
        total += max(incr, decr)
    # Cols
    for c in range(4):
        col = [board_exp[r][c] for r in range(4)]
        incr, decr = 0, 0
        for r in range(3):
            if col[r+1] > col[r]:
                incr += (col[r+1] - col[r])
            else:
                decr += (col[r] - col[r+1])
        total += max(incr, decr)
    return total

def smoothness_exponents(board_exp):
    """
    Negative sum of difference in adjacent exponents. 
    We'll invert it so that bigger negative => less smooth => smaller final value.
    We'll just accumulate a negative sum here.
    """
    diff_sum = 0
    for r in range(4):
        for c in range(3):
            if board_exp[r][c] != 0 and board_exp[r][c+1] != 0:
                diff_sum -= abs(board_exp[r][c+1] - board_exp[r][c])
    for c in range(4):
        for r in range(3):
            if board_exp[r][c] != 0 and board_exp[r+1][c] != 0:
                diff_sum -= abs(board_exp[r+1][c] - board_exp[r][c])
    return diff_sum

class ExpectimaxBitboard:
    def __init__(self, depth=6):
        self.depth = depth

    def get_best_move(self, bitboard):
        """
        Returns best move (0..3) from the current bitboard state,
        or None if no valid moves.
        """
        best_move = None
        best_val = float("-inf")
        for action in [0,1,2,3]:
            new_b, score_gained, changed = bitboard_move(bitboard, action)
            if not changed:
                continue
            val = score_gained + self.expectimax_value(new_b, self.depth-1, False)
            if val > best_val:
                best_val = val
                best_move = action
        return best_move

    @lru_cache(None)
    def expectimax_value(self, bitboard, depth, is_player_turn):
        if depth <= 0 or bitboard_is_game_over(bitboard):
            return bitboard_heuristic(bitboard)
        if is_player_turn:
            # max node
            best_val = float("-inf")
            found_move = False
            for action in [0,1,2,3]:
                new_b, sc, changed = bitboard_move(bitboard, action)
                if changed:
                    found_move = True
                    val = sc + self.expectimax_value(new_b, depth-1, False)
                    if val > best_val:
                        best_val = val
            if not found_move:
                return bitboard_heuristic(bitboard)
            return best_val
        else:
            # chance node: spawn 2 or 4 in all empty cells
            empty_count = bitboard_count_empty(bitboard)
            if empty_count == 0:
                # no empty => treat as player node
                return self.expectimax_value(bitboard, depth-1, True)
            prob2 = 0.9
            prob4 = 0.1
            val_sum = 0.0

            # We'll need to iterate over all empty cells
            # Instead of physically placing, we can do a loop. 
            # For speed, though, we need to do a small iteration. We'll decode nibbles quickly:
            shift = 0
            tmp_b = bitboard
            for _ in range(16):
                nib = (tmp_b & 0xF)
                if nib == 0:
                    # place 2
                    b2 = (bitboard & ~(0xF << shift)) | (1 << shift)  # nibble=1 => tile=2
                    v2 = self.expectimax_value(b2, depth-1, True)
                    # place 4
                    b4 = (bitboard & ~(0xF << shift)) | (2 << shift)  # nibble=2 => tile=4
                    v4 = self.expectimax_value(b4, depth-1, True)
                    val_sum += prob2*v2 + prob4*v4
                tmp_b >>= 4
                shift += 4

            return val_sum / empty_count

###############################################################################
# 5) Example: play a single game headless, save best game
###############################################################################

def play_single_game_bitboard(depth=6, print_steps=False):
    """
    Play a single 2048 game using the bitboard + expectimax until game over.
    Returns final_score, max_tile, plus the move history & board states.
    """
    # Initialize random board
    b = 0
    b = bitboard_spawn(b)
    b = bitboard_spawn(b)

    ai = ExpectimaxBitboard(depth=depth)

    states = []
    moves = []
    scores = []
    current_score = 0

    while not bitboard_is_game_over(b):
        states.append(b)
        scores.append(current_score)

        best_move = ai.get_best_move(b)
        if best_move is None:
            # no valid move => game over
            break
        b2, sc, changed = bitboard_move(b, best_move)
        if not changed:
            # no moves => game over
            break
        current_score += sc
        moves.append(best_move)
        b = bitboard_spawn(b2)

        if print_steps:
            print(f"Move={best_move}, Gained={sc}, Score={current_score}")

    # final
    states.append(b)
    scores.append(current_score)
    max_tile = bitboard_get_max_tile(b)
    if print_steps:
        print(f"Final Score={current_score}, Max Tile={max_tile}")
    return current_score, max_tile, states, moves, scores

def save_if_best(score, max_tile, states, moves, scores, filename="expectimax_best_game.pkl"):
    """
    Compare with existing best file, and if our `score` is higher, overwrite.
    We'll store raw bitboards for states, plus scores, moves, etc.
    """
    best_stored = -1
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        best_stored = data.get("score", -1)
    except:
        pass

    if score > best_stored:
        newdata = {
            "score": score,
            "max_tile": max_tile,
            "states": states,
            "moves": moves,
            "scores": scores
        }
        with open(filename, "wb") as f:
            pickle.dump(newdata, f)
        print(f"New best game saved with score={score}, max_tile={max_tile}")

###############################################################################
# 6) Utility: Convert your 4×4 (values) board <-> bitboard
###############################################################################
def board_to_bitboard(board_4x4):
    """
    Given a 4x4 matrix of tile values (2,4,8,16,...,0 for empty),
    convert to a single 64-bit int with nibble exponents.
    """
    b = 0
    shift = 0
    for r in range(4):
        for c in range(4):
            val = board_4x4[r][c]
            if val > 0:
                exponent = int(math.log2(val))
            else:
                exponent = 0
            b |= (exponent & 0xF) << shift
            shift += 4
    return b

def bitboard_to_board(bitboard):
    """
    Decode the 64-bit int into a 4x4 with tile values.
    """
    board_4x4 = [[0]*4 for _ in range(4)]
    shift = 0
    for r in range(4):
        for c in range(4):
            nib = (bitboard >> shift) & 0xF
            if nib > 0:
                board_4x4[r][c] = (1 << nib)  # 2^nib
            else:
                board_4x4[r][c] = 0
            shift += 4
    return board_4x4

###############################################################################
# 7) Demo main
###############################################################################

def main_demo():
    # Build the row LUTs once
    build_row_tables()

    # Play a single game with depth=6, see final
    final_score, max_tile, states, moves, scores = play_single_game_bitboard(
        depth=6, print_steps=True
    )
    print(f"Final Score: {final_score}, Max Tile: {max_tile}")
    # Save if best
    save_if_best(final_score, max_tile, states, moves, scores)



def main():
    root = tk.Tk()
    game = Game2048(root, headless=False)
    root.mainloop()

if __name__ == "__main__":
    # Example headless run:
    # run_headless_until_2048()
    # run_expectimax_until_2048()
    build_row_tables()
    main_demo()
    # Or launch the GUI:
    main()
