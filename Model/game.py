import statistics
import tkinter as tk
import random
from copy import deepcopy
import math
import torch
import torch.nn as nn
import numpy as np

###############################################################################
# 1) Load the Trained DQN Model Once (Global or in Game Init)
###############################################################################
class DQNModel(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128, action_dim=4):
        """
        For a 4x4 2048 board, state_dim=16 (the exponents).
        We'll produce 4 Q-values (Up, Down, Left, Right).
        """
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)  # shape [batch_size, 4 Q-values]

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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQNModel()  # Ensure this matches the training architecture
checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
model_dict = model.state_dict()
filtered_dict = {}
for k, v in checkpoint.items():
    if k in model_dict and v.shape == model_dict[k].shape:
        filtered_dict[k] = v

model_dict.update(filtered_dict)
model.load_state_dict(model_dict, strict=False)
# model.load_state_dict(
#     torch.load(
#         "model.pth",
#         map_location=device,
#         weights_only=True
#     )
# )


model.to(device)
model.eval()

# A helper to map action indices to directions
action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

###############################################################################
# 2) Convert Board Tiles -> Exponents [0..15], Clamping if > 2^15
###############################################################################
def board_to_exponents(board):
    """
    Convert a 4×4 board of tile values (0,2,4,8,...) 
    to a list of 16 exponents [0..15].
    """
    exps = []
    for row in board:
        for val in row:
            if val == 0:
                exps.append(0)
            else:
                exp = int(math.log2(val))
                # clamp any exponent above 15
                if exp > 15:
                    exp = 15
                exps.append(exp)
    return exps

###############################################################################
# 3) Predict the Next Move Using DQN
###############################################################################
def predict_moves_sorted_dqn(board):
    """
    Predicts and returns a list of moves sorted by descending Q-values.

    Args:
        board (list of lists): The current 4x4 game board.

    Returns:
        list of str: List of directions ('Up', 'Down', 'Left', 'Right') sorted by descending Q-value.
    """
    # Convert the board to exponents
    exponents = board_to_exponents(board)
    
    # Convert to tensor and send to the appropriate device
    state = torch.tensor(exponents, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, 16]

    with torch.no_grad():
        q_values = model(state)  # Shape: [1, 4]

    # Get actions sorted by descending Q-value
    sorted_action_indices = torch.argsort(q_values, dim=1, descending=True).squeeze(0).tolist()

    # Handle the case when there's only one action
    if isinstance(sorted_action_indices, int):
        sorted_action_indices = [sorted_action_indices]

    # Map action indices to directions
    sorted_directions = [action_map[action] for action in sorted_action_indices]

    return sorted_directions



###############################################################################
# Main 2048 Game Class
###############################################################################
class Game2048:
    def __init__(self, master=None, headless=False):
        """
        If headless=True, we skip the UI setup entirely.
        """
        self.size = 4
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.highest_tile = 0
        self.best_score = 0
        self.history = []

        # Colors (still can store if you want, or skip in headless mode)
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

        # If not headless, create the Tkinter UI
        self.headless = headless
        self.master = master
        if not self.headless:
            self.master = master
            self.master.title("2048")
            self.master.geometry("600x450")
            self.master.resizable(False, False)

            ############################################################################
            # Top frame for Score, Best, Game-Over label, Restart & Undo Buttons
            ############################################################################
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

            

            ############################################################################
            # Main grid for 4x4 tiles
            ############################################################################
            self.main_frame = tk.Frame(self.master, bg=self.bg_color)
            self.main_frame.pack(padx=10, pady=10)
            self.tiles = []

            for i in range(self.size):
                row_tiles = []
                for j in range(self.size):
                    label = tk.Label(self.main_frame, text="", bg=self.cell_colors[0],
                                    font=("Helvetica", 20, "bold"), width=4, height=2)
                    label.grid(row=i, column=j, padx=5, pady=5)
                    row_tiles.append(label)
                self.tiles.append(row_tiles)

            # Put the buttons in a separate frame for better alignment, on the right bottom
            self.bottom_frame = tk.Frame(self.master, bg=self.bg_color)
            self.bottom_frame.pack(pady=15)
            self.restart_button = tk.Button(self.bottom_frame, text="Restart (R)",
                                            command=self.key_restart, bg="#8f7a66",
                                            fg="white", font=("Helvetica", 12, "bold"))
            self.restart_button.pack(side="right", padx=10)

            self.undo_button = tk.Button(self.bottom_frame, text="Undo (Z)",
                                        command=self.key_undo, bg="#8f7a66",
                                        fg="white", font=("Helvetica", 12, "bold"))
            self.undo_button.pack(side="left", padx=10)

            

            ############################################################################
            # Key bindings for manual moves
            ############################################################################
            self.master.bind("<Up>", self.key_up)
            self.master.bind("<Down>", self.key_down)
            self.master.bind("<Left>", self.key_left)
            self.master.bind("<Right>", self.key_right)

            ############################################################################
            # Key bindings for ML moves
            ############################################################################
            self.master.bind("i", self.key_ml_single)  # single ML move
            self.master.bind("I", self.key_ml_single)
            self.master.bind("c", self.key_ml_continuous)  # continuous ML moves
            self.master.bind("C", self.key_ml_continuous)

            ############################################################################
            # Key bindings for Restart & Undo
            ############################################################################
            self.master.bind("r", self.key_restart)
            self.master.bind("R", self.key_restart)
            self.master.bind("z", self.key_undo)
            self.master.bind("Z", self.key_undo)

            # Start a new game
            self.reset_game()
        else:
            # Headless: just reset the board but no UI
            self.reset_game()
        

    ###########################################################################
    # Game Initialization / Reset
    ###########################################################################
    def reset_game(self):
        """Clears board, resets score, spawns initial tiles, hides 'Game Over'."""
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.history.clear()
        if not self.headless:
            self.game_over_label.config(text="")  # Hide "Game Over" message

        # Spawn initial tiles
        self.spawn_new_tile()
        self.spawn_new_tile()

        self.update_ui()

    ###########################################################################
    # Key Move Handlers (Up, Down, Left, Right)
    ###########################################################################
    def key_left(self, event=None):
        # Save board & score before move
        pre_board = deepcopy(self.board)
        pre_score = self.score

        moved = self.move_left()

        # Only if moved => store state for undo, spawn tile, update UI
        if moved:
            self.history.append((pre_board, pre_score))
            self.spawn_new_tile()
            self.update_ui()
            if self.is_game_over():
                self.show_game_over()

    def key_right(self, event=None):
        pre_board = deepcopy(self.board)
        pre_score = self.score

        moved = self.move_right()
        if moved:
            self.history.append((pre_board, pre_score))
            self.spawn_new_tile()
            self.update_ui()
            if self.is_game_over():
                self.show_game_over()

    def key_up(self, event=None):
        pre_board = deepcopy(self.board)
        pre_score = self.score

        moved = self.move_up()
        if moved:
            self.history.append((pre_board, pre_score))
            self.spawn_new_tile()
            self.update_ui()
            if self.is_game_over():
                self.show_game_over()

    def key_down(self, event=None):
        pre_board = deepcopy(self.board)
        pre_score = self.score

        moved = self.move_down()
        if moved:
            self.history.append((pre_board, pre_score))
            self.spawn_new_tile()
            self.update_ui()
            if self.is_game_over():
                self.show_game_over()

    ###########################################################################
    # ML Move Handlers
    ###########################################################################
    def key_ml_single(self, event=None):
        """Perform a single ML-driven move with fallback."""
        if self.is_game_over():
            return

        # 1) Get moves sorted by descending Q-value
        sorted_directions = predict_moves_sorted_dqn(self.board)

        # 2) Save current state (for undo)
        pre_board = deepcopy(self.board)
        pre_score = self.score

        # 3) Attempt each move in sorted order until a valid move is found
        for direction in sorted_directions:
            moved = self.execute_move(direction)
            if moved:
                self.history.append((pre_board, pre_score))
                self.spawn_new_tile()
                self.update_ui()
                print(f"ML single-move picked: {direction}" , end="")
                if self.is_game_over():
                    self.show_game_over()
                return  # Exit after a successful move

        # 4) If no valid moves found, declare game over
        self.show_game_over()


    
    def execute_move(self, direction):
        """Call move_up/move_down/move_left/move_right depending on the direction.
        Return True if the board changed, else False.
        """
        if direction == 'Up':
            return self.move_up()
        elif direction == 'Down':
            return self.move_down()
        elif direction == 'Left':
            return self.move_left()
        elif direction == 'Right':
            return self.move_right()
        else:
            return False

    def key_ml_continuous(self, event=None):
        """Continuously perform ML-driven moves until no moves are possible or game over."""
        if self.is_game_over():
            return

        def step():
            if self.is_game_over():
                self.show_game_over()
                return

            # 1) Get moves sorted by descending Q-value
            sorted_directions = predict_moves_sorted_dqn(self.board)

            # 2) Save board/score for undo
            pre_board = deepcopy(self.board)
            pre_score = self.score

            # 3) Attempt each move in sorted order until a valid move is found
            moved = False
            chosen_direction = None
            for direction in sorted_directions:
                if self.execute_move(direction):
                    moved = True
                    chosen_direction = direction
                    break

            if moved:
                self.history.append((pre_board, pre_score))
                self.spawn_new_tile()
                self.update_ui()
                # print all in the same line
                # print(f"ML continuous move picked: {chosen_direction} ", end="")
                if not self.is_game_over():
                    # Schedule the next step (adjust delay as needed)
                    self.master.after(50, step)
                else:
                    self.show_game_over()
            else:
                # No valid move found => game over
                # print("ML continuous move: No valid moves found. Game Over!")
                self.show_game_over()

        step()

    ###########################################################################
    # Restart & Undo
    ###########################################################################
    def key_restart(self, event=None):
        """Restart game (clear board, new tiles, reset score)."""
        self.reset_game()

    def key_undo(self, event=None):
        """Undo the last move (restore board & score)."""
        if not self.history:
            return
        last_board, last_score = self.history.pop()
        self.board = last_board
        self.score = last_score
        self.highest_tile = max([max(row) for row in self.board])
        self.update_ui()

    ###########################################################################
    # 2048 Core Logic
    ###########################################################################
    def spawn_new_tile(self):
        """Spawns a new tile (2 or 4) in a random empty spot."""
        empty_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board[r][c] == 0
        ]
        if not empty_cells:
            return
        r, c = random.choice(empty_cells)
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    def move_left(self):
        moved = False
        total_merge_score = 0
        for row in range(self.size):
            row_data = self.board[row]
            new_row, did_move, merge_score = self._compress_and_merge(row_data)
            if did_move:
                moved = True
                total_merge_score += merge_score
            self.board[row] = new_row
        if moved:
            self.score += total_merge_score
        return moved

    def move_right(self):
        moved = False
        total_merge_score = 0
        for row in range(self.size):
            row_data = self.board[row][::-1]
            new_row, did_move, merge_score = self._compress_and_merge(row_data)
            new_row.reverse()
            if did_move:
                moved = True
                total_merge_score += merge_score
            self.board[row] = new_row
        if moved:
            self.score += total_merge_score
        return moved

    def move_up(self):
        moved = False
        total_merge_score = 0
        for col in range(self.size):
            col_data = [self.board[row][col] for row in range(self.size)]
            new_col, did_move, merge_score = self._compress_and_merge(col_data)
            if did_move:
                moved = True
                total_merge_score += merge_score
            for row in range(self.size):
                self.board[row][col] = new_col[row]
        if moved:
            self.score += total_merge_score
        return moved

    def move_down(self):
        moved = False
        total_merge_score = 0
        for col in range(self.size):
            col_data = [self.board[row][col] for row in range(self.size)][::-1]
            new_col, did_move, merge_score = self._compress_and_merge(col_data)
            new_col.reverse()
            if did_move:
                moved = True
                total_merge_score += merge_score
            for row in range(self.size):
                self.board[row][col] = new_col[row]
        if moved:
            self.score += total_merge_score
        return moved

    def _compress_and_merge(self, tiles):
        """
        1. Remove zeros
        2. Merge adjacent equal tiles
        3. Pad with zeros
        4. Return (new_tiles, did_move, merge_score)
        """
        original = list(tiles)
        filtered = [t for t in tiles if t != 0]

        merged = []
        merge_score = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i < len(filtered) - 1 and filtered[i] == filtered[i+1]:
                merged_val = filtered[i]*2
                merged.append(merged_val)
                merge_score += merged_val
                skip = True
            else:
                merged.append(filtered[i])

        # Pad with zeros
        while len(merged) < len(tiles):
            merged.append(0)

        did_move = (merged != original)
        self.highest_tile = max(self.highest_tile, max(merged))
        return merged, did_move, merge_score

    def is_game_over(self):
        """Check if no more moves are possible or 2048 is present."""
        # Check for 2048 => define as game over (or you can define as "Win" scenario)
        for row in self.board:
            if 2048 in row:
                return True

        # Check any empty cell => not game over
        for row in self.board:
            if 0 in row:
                return False

        # Check merges horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False

        # Check merges vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i][j] == self.board[i + 1][j]:
                    return False

        return True

    ###########################################################################
    # UI Updates & Game Over
    ###########################################################################
    def update_ui(self):
        """Refresh the tile labels and scores."""
        if self.headless:
            return
        # Update tiles
        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i][j]
                color = self.cell_colors.get(val, "#3C3A32")  # default for big values
                text = str(val) if val != 0 else ""
                self.tiles[i][j].config(text=text, bg=color)

        # Update score & best score
        if self.score > self.best_score:
            self.best_score = self.score
        self.score_label.config(text=f"Score: {self.score}")
        self.best_label.config(text=f"Best: {self.best_score}")
        self.highest_label.config(text=f"Highest: {self.highest_tile}")

        self.master.update_idletasks()

    def show_game_over(self):
        if self.headless:
            return
        """Display 'Game Over!' in the top frame."""
        self.game_over_label.config(text="Game Over!")

###############################################################################
# Main Function
###############################################################################

def run_single_game_random():
    """
    Creates a headless Game2048 instance and keeps applying random moves 
    until game is over. Returns (final_score, highest_tile, move_count).
    """
    game = Game2048(headless=True)
    move_count = 0

    # While not game over, pick a random direction among ['Up','Down','Left','Right'].
    # Only apply it if it changes the board. If not, pick another random direction.
    directions = ['Up','Down','Left','Right']
    while not game.is_game_over():
        direction = random.choice(directions)
        changed = game.execute_move(direction)
        if changed:
            game.spawn_new_tile()
            move_count += 1

    return (game.score, game.highest_tile, move_count)

def predict_moves_descending(board):
    """
    1) Convert board -> exponents -> tensor(1,16)
    2) Get logits from model
    3) Sort moves by descending logit
    4) Return a list of moves in descending confidence, e.g. ['Right','Down','Left','Up']
    """
    # Convert 4x4 board to exponents
    exps = board_to_exponents(board)
    x = torch.tensor(exps, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)  # shape [1,4]

    # We have model outputs for moves in this order: 0=Right,1=Left,2=Down,3=Up
    move_idx_sorted = torch.argsort(logits, dim=1, descending=True).squeeze(0)

    # Corrected mapping
    idx_to_str = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}


    # Return list of directions in descending confidence
    moves_in_desc_order = [idx_to_str[idx.item()] for idx in move_idx_sorted]
    return moves_in_desc_order

def run_single_game_model():
    game = Game2048(headless=True)
    game.reset_game()

    move_count = 0
    while not game.is_game_over():
        moves_sorted = predict_moves_sorted_dqn(game.board)
        moved = False
        for move in moves_sorted:
            if game.execute_move(move):
                move_count += 1      # count the move
                game.spawn_new_tile()
                # no need for UI
                moved = True
                break
        if not moved:
            # No valid move found
            break

    return game.score, game.highest_tile, move_count


def run_1000_games_random_vs_model(num_games=1000):
    random_results = []
    model_results = []

    # ----- 1) Run random approach -----
    print(f"Running {num_games} random games...")
    for _ in range(num_games):
        score, highest_tile, moves_count = run_single_game_random()
        random_results.append((score, highest_tile, moves_count))

    # ----- 2) Run ML approach -----
    print(f"Running {num_games} model-based games...")
    for _ in range(num_games):
        score, highest_tile, moves_count = run_single_game_model()
        model_results.append((score, highest_tile, moves_count))

    # Unpack results
    rand_scores     = [r[0] for r in random_results]
    rand_high_tiles = [r[1] for r in random_results]
    rand_moves      = [r[2] for r in random_results]

    model_scores     = [r[0] for r in model_results]
    model_high_tiles = [r[1] for r in model_results]
    model_moves      = [r[2] for r in model_results]

    # Compute simple stats: mean, max, min
    # (You can compute median, standard deviation, etc. if you want.)
    def stats_str(values):
        return (f"avg={statistics.mean(values):.1f}, "
                f"min={min(values)}, "
                f"max={max(values)}")

    print("\n---------- Results Comparison ----------")
    print("Random Approach:")
    print(f"  Score:      {stats_str(rand_scores)}")
    print(f"  HighestTile:{stats_str(rand_high_tiles)}")
    print(f"  Moves:      {stats_str(rand_moves)}")
    rand_count_tiles = [rand_high_tiles.count(2**i) for i in range(16)]
    filtered_tiles = [(2**i, c) for i, c in enumerate(rand_count_tiles) if c > 0]
    print("  Tiles:" , end=" ")
    for tile, count in filtered_tiles:
        # all in the same line
        print(f"{tile}:{count}", end=", ")
    

    print("\nModel Approach:")
    print(f"  Score:      {stats_str(model_scores)}")
    print(f"  HighestTile:{stats_str(model_high_tiles)}")
    print(f"  Moves:      {stats_str(model_moves)}")
    # print how many for each tile, for example 256 -> 82 times
    model_count_tiles = [model_high_tiles.count(2**i) for i in range(16)]
    filtered_tiles = [(2**i, c) for i, c in enumerate(model_count_tiles) if c > 0]
    print("  Tiles:" , end=" ")
    for tile, count in filtered_tiles:
        # all in the same line
        print(f"{tile}:{count}", end=", ")
    

    # You could do a direct comparison, e.g. which has higher average score
    avg_random_score = statistics.mean(rand_scores)
    avg_model_score  = statistics.mean(model_scores)

    winner = "Model" if avg_model_score > avg_random_score else "Random"
    print(f"\nOverall winner by average score: {winner}!")

def main():
    root = tk.Tk()
    game = Game2048(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    # Note: The following line should be removed or placed outside the __main__ block
    run_1000_games_random_vs_model(num_games=1000)