"""
train_expectimax_model.py

Trains a neural network to imitate an Expectimax search policy for 2048.
Saves the trained model to "model.pth".

Usage:
  python train_expectimax_model.py
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

    def spawn_tile(self, force_tile=None):
        """
        Spawn a tile (2 or 4) in a random empty cell.
        If 'force_tile' is specified, use that instead of random choice of 2/4.
        """
        empties = self.get_empty_cells()
        if not empties:
            return
        r, c = random.choice(empties)
        if force_tile is not None:
            self.board[r][c] = force_tile
        else:
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
        1) Slide non-zero tiles (remove zeros in between).
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
        if len(self.get_empty_cells()) > 0:
            return False

        # Check merges horizontally
        for row in range(self.size):
            for col in range(self.size - 1):
                if self.board[row][col] == self.board[row][col + 1]:
                    return False

        # Check merges vertically
        for col in range(self.size):
            for row in range(self.size - 1):
                if self.board[row][col] == self.board[row + 1][col]:
                    return False

        # No empties, no merges => game over
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
# 2) Evaluation / Heuristic
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


def evaluate_board(board):
    """
    More sophisticated evaluation:
    1) Monotonicity (rows/cols that are strictly or nearly strictly increasing/decreasing)
    2) Smoothness (penalty if adjacent tiles differ by large amounts)
    3) Empty cell bonus
    4) Largest tile in corner bonus

    Weighted sum of these factors.
    """
    size = len(board)

    # 1) Monotonicity
    # We'll measure how "ordered" each row/col is. 
    # We do a sign check: are we consistently increasing or decreasing?
    # We'll accumulate a positive value if consistent, negative otherwise.
    mono_score = 0

    # row monotonicity
    for r in range(size):
        row_vals = board[r]
        for c in range(size - 1):
            diff = row_vals[c+1] - row_vals[c]
            if diff > 0:
                mono_score += 1
            elif diff < 0:
                mono_score += 1

    # col monotonicity
    for c in range(size):
        col_vals = [board[r][c] for r in range(size)]
        for r in range(size - 1):
            diff = col_vals[r+1] - col_vals[r]
            if diff > 0:
                mono_score += 1
            elif diff < 0:
                mono_score += 1

    # 2) Smoothness (we want adjacent tiles to be close in value)
    smooth_score = 0
    for r in range(size):
        for c in range(size - 1):
            smooth_score -= abs(board[r][c] - board[r][c+1]) / 4.0
    for c in range(size):
        for r in range(size - 1):
            smooth_score -= abs(board[r][c] - board[r+1][c]) / 4.0

    # 3) Empty cell bonus
    empties = sum(cell == 0 for row in board for cell in row)
    empty_score = 50 * empties  # each empty cell is +50

    # 4) Largest tile in corner
    corners = [
        board[0][0],
        board[0][size-1],
        board[size-1][0],
        board[size-1][size-1]
    ]
    max_tile = max(cell for row in board for cell in row)
    corner_bonus = 0
    if max_tile in corners:
        corner_bonus = max_tile * 0.5

    # Weighted sum
    return 0.3 * mono_score + 1.0 * smooth_score + 1.0 * empty_score + 1.0 * corner_bonus


###############################################################################
# 3) Expectimax Search
###############################################################################
def expectimax(game: Game2048, depth: int, is_chance_turn: bool):
    """
    Depth-limited Expectimax search.
    Returns a float "value" of the board at best play from current state.
    """
    if depth == 0 or game.is_game_over():
        return evaluate_board(game.board)

    if not is_chance_turn:
        # Player's turn -> max over actions
        best_value = -float('inf')
        for action in ['Up', 'Down', 'Left', 'Right']:
            g_clone = game.clone()
            changed = g_clone.step(action)
            if not changed:
                continue
            val = expectimax(g_clone, depth - 1, True)
            if val > best_value:
                best_value = val
        return best_value if best_value != -float('inf') else evaluate_board(game.board)
    else:
        # Chance turn -> average over possible spawns
        empties = game.get_empty_cells()
        if not empties:
            return evaluate_board(game.board)

        total_value = 0.0
        for (r, c) in empties:
            # spawn a 2
            g_clone2 = game.clone()
            g_clone2.board[r][c] = 2
            val2 = expectimax(g_clone2, depth - 1, False)

            # spawn a 4
            g_clone4 = game.clone()
            g_clone4.board[r][c] = 4
            val4 = expectimax(g_clone4, depth - 1, False)

            # Weighted average for tile spawn probabilities: 90% 2, 10% 4
            total_value += 0.9 * val2 + 0.1 * val4

        return total_value / len(empties)


def choose_best_move_expectimax(game: Game2048, search_depth=3):
    """
    Finds the best move (Up/Down/Left/Right) by running expectimax for each action.
    Returns (best_action, best_value).
    If no move is valid, returns (None, -inf).
    """
    best_move = None
    best_value = -float('inf')

    for action in ['Up', 'Down', 'Left', 'Right']:
        g_clone = game.clone()
        changed = g_clone.step(action)
        if not changed:
            continue
        # Now it's the "chance" turn
        val = expectimax(g_clone, search_depth - 1, True)
        if val > best_value:
            best_value = val
            best_move = action

    return best_move, best_value


###############################################################################
# 4) Data Collection: Generate (state, action) via Expectimax
###############################################################################
def collect_data_expectimax(num_episodes=2000, max_moves=1000, search_depth=3):
    """
    Let the expectimax policy play 'num_episodes' games.
    Collect all (board_exponents, action_idx) pairs into a list.
    """
    move_map = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
    dataset = []
    start_time = time.time()

    for ep in range(num_episodes):
        env = Game2048()
        moves_done = 0
        while not env.is_game_over() and moves_done < max_moves:
            # Record current state
            state_exps = board_to_exponents(env.board)

            # Decide best move
            best_move, _ = choose_best_move_expectimax(env, search_depth=search_depth)
            if best_move is None:
                break  # no valid moves => game over

            dataset.append((state_exps, move_map[best_move]))

            # Apply the move + spawn tile
            env.step(best_move)
            env.spawn_tile()
            moves_done += 1

        # simple progress print
        if (ep+1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {ep+1}/{num_episodes} - data size so far={len(dataset)}, time={elapsed:.2f}s")

    return dataset


###############################################################################
# 5) Define a Model, Train It, Save model.pth
###############################################################################
class ImitationModel(nn.Module):
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


def train_model():
    # 1) Collect data from an Expectimax policy
    print("Collecting data with Expectimax policy (depth=4)...")
    data = collect_data_expectimax(
        num_episodes=100,    # Adjust for your runtime constraints
        max_moves=1000,
        search_depth=7
    )
    print(f"Data size: {len(data)} (state-action pairs)")

    # 2) Shuffle & split into train/val
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data   = data[split_idx:]
    print(f"Train data: {len(train_data)}, Val data: {len(val_data)}")

    # 3) Create model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImitationModel().to(device)

    # You can experiment with a different LR or optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4) Training loop
    epochs = 10
    batch_size = 256

    def iterate_minibatches(dataset, batch_size):
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
            logits = model(xb)  # shape [B,4]
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

        print(f"Epoch {ep+1}/{epochs} | train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # 5) Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train_model()