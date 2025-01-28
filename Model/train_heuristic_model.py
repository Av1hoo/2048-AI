"""
train_heuristic_model.py

Trains a neural network to imitate a Corner+Monotonicity heuristic for 2048.
Saves the trained model to "model.pth".

Usage:
  python train_heuristic_model.py
"""

import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


###############################################################################
# Simple 2048 Environment (No GUI)
###############################################################################
class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.reset()

    def reset(self):
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()

    def get_empty_cells(self):
        empties = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empties.append((r, c))
        return empties

    def spawn_tile(self):
        empties = self.get_empty_cells()
        if not empties:
            return
        r, c = random.choice(empties)
        # 90% chance of 2, 10% chance of 4
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    def move_up(self):
        moved = False
        for col in range(self.size):
            column_data = [self.board[row][col] for row in range(self.size)]
            new_col, col_moved, score_gain = self._compress_merge(column_data)
            if col_moved:
                moved = True
            for row in range(self.size):
                self.board[row][col] = new_col[row]
            self.score += score_gain
        return moved

    def move_down(self):
        moved = False
        for col in range(self.size):
            column_data = [self.board[row][col] for row in range(self.size)][::-1]
            new_col, col_moved, score_gain = self._compress_merge(column_data)
            new_col.reverse()
            if col_moved:
                moved = True
            for row in range(self.size):
                self.board[row][col] = new_col[row]
            self.score += score_gain
        return moved

    def move_left(self):
        moved = False
        for row in range(self.size):
            row_data = self.board[row]
            new_row, row_moved, score_gain = self._compress_merge(row_data)
            if row_moved:
                moved = True
            self.board[row] = new_row
            self.score += score_gain
        return moved

    def move_right(self):
        moved = False
        for row in range(self.size):
            row_data = self.board[row][::-1]
            new_row, row_moved, score_gain = self._compress_merge(row_data)
            new_row.reverse()
            if row_moved:
                moved = True
            self.board[row] = new_row
            self.score += score_gain
        return moved

    def _compress_merge(self, tiles):
        """
        1) Slide non-zero tiles up (remove zeros in between).
        2) Merge adjacent equal tiles from left to right.
        3) Slide again to fill any new gaps.
        """
        filtered = [t for t in tiles if t != 0]
        merged = []
        moved = False
        score_gain = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i < len(filtered) - 1 and filtered[i] == filtered[i+1]:
                new_val = filtered[i]*2
                merged.append(new_val)
                score_gain += new_val
                skip = True
                moved = True
            else:
                merged.append(filtered[i])

        # Pad with zeros
        while len(merged) < len(tiles):
            merged.append(0)

        # Check if effectively changed from original
        if merged != tiles:
            moved = True

        return merged, moved, score_gain

    def is_game_over(self):
        # 1) If any empty cell => not over
        for row in self.board:
            if 0 in row:
                return False

        # 2) If any merge possibility horizontally
        for row in range(self.size):
            for col in range(self.size - 1):
                if self.board[row][col] == self.board[row][col + 1]:
                    return False

        # 3) If any merge possibility vertically
        for col in range(self.size):
            for row in range(self.size - 1):
                if self.board[row][col] == self.board[row + 1][col]:
                    return False

        # If we found no empty cells and no merges => game over
        return True

    def clone(self):
        g = Game2048(self.size)
        g.board = deepcopy(self.board)
        g.score = self.score
        return g

    def step(self, action_str):
        """
        Execute the action in { 'Up', 'Down', 'Left', 'Right' }.
        Return True if the board changed, False otherwise.
        """
        if action_str == 'Up':
            return self.move_up()
        elif action_str == 'Down':
            return self.move_down()
        elif action_str == 'Left':
            return self.move_left()
        elif action_str == 'Right':
            return self.move_right()
        else:
            return False


###############################################################################
# 3) Corner + Monotonicity Heuristic
###############################################################################
def evaluate_heuristic(board):
    """
    Compute a single numeric score that tries to measure:
    - Monotonicity (rows and columns should be strictly increasing or decreasing)
    - Corner building (big tiles in the corners)

    The weighting is somewhat arbitrary.
    """
    size = len(board)
    
    # 1) Monotonicity
    # We measure row-wise monotonicity and column-wise monotonicity.
    # A row that is strictly increasing or decreasing should get a higher score.
    # We'll do a simple approach: sum(|tiles[i] - tiles[i+1]|) inversely.
    # The lower these differences, the more monotonic. We'll store a negative penalty.
    # Then we invert sign so "more monotonic => higher score".
    monotonic_score = 0

    # Row monotonicity
    for r in range(size):
        row_vals = board[r]
        # measure difference
        for c in range(size - 1):
            monotonic_score -= abs(row_vals[c] - row_vals[c+1])

    # Column monotonicity
    for c in range(size):
        col_vals = [board[r][c] for r in range(size)]
        for r in range(size - 1):
            monotonic_score -= abs(col_vals[r] - col_vals[r+1])

    # 2) Corner preference
    # Check corners. Let's do a simple approach:
    #   big_tile_in_corner = (board[0][0] + board[0][size-1] + board[size-1][0] + board[size-1][size-1]) * 0.5
    # So that if big tiles accumulate in corners, we get a bonus.
    corners = [board[0][0], board[0][size-1], board[size-1][0], board[size-1][size-1]]
    corner_score = 0.5 * sum(corners)

    # Combine
    # Weighted sum with some scale factors
    # (You can tune these to see which works better)
    return monotonic_score + corner_score


def choose_heuristic_move(game: Game2048):
    """
    For each possible move in {Up, Down, Left, Right}:
      1) Clone game
      2) Execute move
      3) If valid, compute heuristic
    Pick the move with the highest heuristic.
    
    Return (best_move_str, best_score).
    If no move is valid, return (None, -inf).
    """
    moves = ['Up', 'Down', 'Left', 'Right']
    best_move = None
    best_value = -float('inf')

    for m in moves:
        g_clone = game.clone()
        changed = g_clone.step(m)
        if not changed:
            continue  # invalid move

        val = evaluate_heuristic(g_clone.board)
        if val > best_value:
            best_value = val
            best_move = m

    return best_move, best_value


###############################################################################
# 4) Data Collection: Generate (state, action) from the Heuristic
###############################################################################
def board_to_exponents(board):
    """
    Convert a 4×4 board of tile values (0,2,4,8,...) 
    to a list of 16 exponents [0..15].
    Clamps any tile > 2^15 to exponent 15.
    """
    exps = []
    for row in board:
        for val in row:
            if val == 0:
                exps.append(0)
            else:
                exp = int(math.log2(val))
                if exp > 15:
                    exp = 15
                exps.append(exp)
    return exps


def collect_heuristic_data(num_episodes=1000, max_moves=1000):
    """
    Let the heuristic policy play 'num_episodes' games.
    Collect all (board_exponents, action_idx) pairs into a list.
    Return the dataset as a list of (input, label).
      input = [16 floats] board exponents
      label = int in {0,1,2,3} representing (Up,Down,Left,Right)
    """
    move_map = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
    data = []
    start_time = time.time()

    for ep in range(num_episodes):
        env = Game2048(size=4)
        for _ in range(max_moves):
            if env.is_game_over():
                break

            # State (board exps)
            state_exps = board_to_exponents(env.board)

            # Action
            best_move, _ = choose_heuristic_move(env)
            if best_move is None:
                # no valid moves => game over
                break

            # record
            data.append((state_exps, move_map[best_move]))
            # print each 1000 games
            if ep % 7 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / ep if ep else 0
                eta = (num_episodes - ep) * avg_time
                print(f"Games - {ep} / {num_episodes} - {len(data)} data points -Time passed: {time.time() - start_time:.2f}s ETA: {eta:.2f}s", end='\r')

            # step
            env.step(best_move)
            env.spawn_tile()

    return data


###############################################################################
# 5) Define a Model, Train It, Save model.pth
###############################################################################
class ImitationModel(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128, action_dim=4):
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


def train_model():
    # 1) Collect data
    print("Collecting data from heuristic policy...")
    data = collect_heuristic_data(num_episodes=50000, max_moves=1000)
    print(f"Data size: {len(data)} (state-action pairs)")

    # 2) Create train & val splits
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data   = data[split_idx:]

    print(f"Train data: {len(train_data)}, Val data: {len(val_data)}")

    # 3) Create model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImitationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4) Training loop
    epochs = 10
    batch_size = 256

    def iterate_minibatches(dataset, batch_size):
        # a simple generator
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            xs, ys = [], []
            for (state_exps, action_idx) in batch:
                xs.append(state_exps)
                ys.append(action_idx)
            yield (torch.tensor(xs, dtype=torch.float32),
                   torch.tensor(ys, dtype=torch.long))

    for ep in range(epochs):
        # ---- Train pass ----
        model.train()
        total_loss = 0.0
        for xb, yb in iterate_minibatches(train_data, batch_size):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_data)

        # ---- Validation pass ----
        model.eval()
        total_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in iterate_minibatches(val_data, batch_size):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                total_val_loss += loss.item() * len(xb)

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()

        val_loss = total_val_loss / len(val_data)
        val_acc  = correct / len(val_data)

        print(f"Epoch {ep+1}/{epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # 5) Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train_model()
