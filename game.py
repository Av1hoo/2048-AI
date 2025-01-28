import tkinter as tk
import random
import math
import torch
import torch.nn as nn
import os
import pickle
from functools import lru_cache

################################################################################
# 1) Global Configuration
################################################################################

# Adjust as needed:
EXPECTIMAX_DEPTH = 5   # smaller depth => faster
DQN_MODEL_PATH   = "model.pth"  # your trained DQN model
BEST_GAME_PATH   = "best_game.pkl"

# If you want, you can reduce delays (in ms) when running continuous DQN
CONTINUOUS_STEP_DELAY = 50


################################################################################
# 2) Build the Row-LUT for Move-Left (Speeding up merges)
################################################################################
# We'll store two tables of size 65536 (2^16):
#   row_left_table[row16]  => new 16-bit after move-left
#   row_score_table[row16] => points gained from merges
#
# Each 'row16' is a 4-nibble representation of the row (lowest nibble = leftmost tile).
# Example nibble range: 0..15, where 0 means empty, 1 means tile=2, 2 means tile=4, etc.

row_left_table = [0]*65536
row_score_table = [0]*65536

def build_row_tables():
    for row_val in range(65536):
        # Extract the 4 nibbles:
        # nibble0 is row_val & 0xF, nibble1 is (row_val>>4) & 0xF, etc.
        tiles = [
            (row_val >> 0) & 0xF,
            (row_val >> 4) & 0xF,
            (row_val >> 8) & 0xF,
            (row_val >> 12) & 0xF
        ]
        filtered = [t for t in tiles if t != 0]
        merged = []
        score = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                merged_val = filtered[i] + 1  # exponent + 1
                # actual numeric tile = 2^(merged_val)
                score += (1 << merged_val)
                merged.append(merged_val)
                skip = True
            else:
                merged.append(filtered[i])
        # pad with zeros
        merged.extend([0]*(4-len(merged)))
        # new 16-bit row
        new_row_val = (
            (merged[0] & 0xF)
            | ((merged[1] & 0xF) << 4)
            | ((merged[2] & 0xF) << 8)
            | ((merged[3] & 0xF) << 12)
        )
        row_left_table[row_val] = new_row_val
        row_score_table[row_val] = score

def reverse_row(row16):
    # Reverse the 4 nibbles of row16: (n0 n1 n2 n3) => (n3 n2 n1 n0)
    n0 = (row16 >>  0) & 0xF
    n1 = (row16 >>  4) & 0xF
    n2 = (row16 >>  8) & 0xF
    n3 = (row16 >> 12) & 0xF
    return (n3) | (n2 << 4) | (n1 << 8) | (n0 << 12)


################################################################################
# 3) Bitboard Utility: get_row, set_row, transpose, moves
################################################################################
# Layout in a 64-bit: row0 is bits [0..15], row1 is [16..31], row2 [32..47], row3 [48..63].

def get_row(bitboard, row_idx):
    return (bitboard >> (16 * row_idx)) & 0xFFFF

def set_row(bitboard, row_idx, row16):
    shift = 16 * row_idx
    # clear that row, then insert:
    mask = 0xFFFF << shift
    bitboard &= ~mask
    bitboard |= (row16 & 0xFFFF) << shift
    return bitboard

def transpose(bitboard):
    """
    Transpose 4x4 in nibble-based form.
    We'll decode the 16 nibbles, reorder, re-encode.
    """
    cells = [0]*16
    tmp = bitboard
    for r in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(4):
            cells[r*4 + c] = (row >> (4*c)) & 0xF

    # Transpose => cell(r,c) -> cell(c,r)
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

def shift_left(bitboard):
    """ Move-left on all 4 rows. """
    moved = False
    total_score = 0
    new_b = bitboard
    for r in range(4):
        row = get_row(new_b, r)
        newrow = row_left_table[row]
        if newrow != row:
            moved = True
        total_score += row_score_table[row]
        new_b = set_row(new_b, r, newrow)
    return new_b, total_score, moved

def shift_right(bitboard):
    """ Move-right by reversing each row, LUT for left, then reversing back. """
    moved = False
    total_score = 0
    new_b = bitboard
    for r in range(4):
        row = get_row(new_b, r)
        rev_row = reverse_row(row)
        new_rev = row_left_table[rev_row]
        if new_rev != rev_row:
            moved = True
        total_score += row_score_table[rev_row]
        final_row = reverse_row(new_rev)
        new_b = set_row(new_b, r, final_row)
    return new_b, total_score, moved

def shift_up(bitboard):
    """ Move-up = transpose -> shift-left -> transpose back. """
    b2 = transpose(bitboard)
    b2, sc, m = shift_left(b2)
    b2 = transpose(b2)
    return b2, sc, m

def shift_down(bitboard):
    """ Move-down = transpose -> shift-right -> transpose back. """
    b2 = transpose(bitboard)
    b2, sc, m = shift_right(b2)
    b2 = transpose(b2)
    return b2, sc, m

def bitboard_move(bitboard, action):
    """
    action: 0=Up, 1=Down, 2=Left, 3=Right
    Return (new_bitboard, score_gained, changed).
    """
    if action == 0:
        return shift_up(bitboard)
    elif action == 1:
        return shift_down(bitboard)
    elif action == 2:
        return shift_left(bitboard)
    elif action == 3:
        return shift_right(bitboard)
    # fallback
    return bitboard, 0, False


################################################################################
# 4) Spawning, Checking Game Over, etc.
################################################################################
def bitboard_count_empty(bitboard):
    count = 0
    tmp = bitboard
    for _ in range(16):
        if (tmp & 0xF) == 0:
            count += 1
        tmp >>= 4
    return count

def bitboard_spawn(bitboard):
    """ Spawn tile 2 or 4 (exponent=1 or 2) into a random empty nibble. """
    empty_count = bitboard_count_empty(bitboard)
    if empty_count == 0:
        return bitboard
    r = random.randrange(empty_count)
    val = 1 if random.random()<0.9 else 2  # exponent=1 => tile=2, exponent=2 => tile=4
    shift = 0
    seen = 0
    tmp_b = bitboard
    while True:
        nib = (tmp_b & 0xF)
        if nib == 0:
            if seen == r:
                # place val here
                bitboard &= ~(0xF << shift)
                bitboard |= (val << shift)
                return bitboard
            seen += 1
        shift += 4
        tmp_b >>= 4

def bitboard_is_game_over(bitboard):
    """ True if no moves possible. """
    # check any empty
    if bitboard_count_empty(bitboard) > 0:
        return False

    # check horizontal merges
    tmp = bitboard
    for _ in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(3):
            nib1 = (row >> (4*c)) & 0xF
            nib2 = (row >> (4*(c+1))) & 0xF
            if nib1 == nib2:
                return False

    # check vertical merges => transpose + same check
    t = transpose(bitboard)
    tmp = t
    for _ in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(3):
            nib1 = (row >> (4*c)) & 0xF
            nib2 = (row >> (4*(c+1))) & 0xF
            if nib1 == nib2:
                return False

    return True

def bitboard_get_max_tile(bitboard):
    """ Return numeric value of largest tile. """
    max_exp = 0
    tmp = bitboard
    for _ in range(16):
        nib = tmp & 0xF
        if nib > max_exp:
            max_exp = nib
        tmp >>= 4
    return (1 << max_exp)  # 2^max_exp


################################################################################
# 5) Utility: Convert bitboard <-> 4×4 matrix of integers
################################################################################
def bitboard_to_board(bitboard):
    board = [[0]*4 for _ in range(4)]
    shift = 0
    for r in range(4):
        for c in range(4):
            nib = (bitboard >> shift) & 0xF
            board[r][c] = (1 << nib) if nib > 0 else 0
            shift += 4
    return board

def board_to_bitboard(board_4x4):
    b = 0
    shift = 0
    for r in range(4):
        for c in range(4):
            val = board_4x4[r][c]
            if val > 0:
                exponent = int(math.log2(val))
                b |= (exponent << shift)
            shift += 4
    return b


################################################################################
# 6) Heuristic + Expectimax w/ Bitboard
################################################################################
def monotonicity_exponents(board_exp):
    total = 0
    # rows
    for r in range(4):
        incr = 0
        decr = 0
        for c in range(3):
            if board_exp[r][c+1] > board_exp[r][c]:
                incr += (board_exp[r][c+1] - board_exp[r][c])
            else:
                decr += (board_exp[r][c] - board_exp[r][c+1])
        total += max(incr, decr)
    # cols
    for c in range(4):
        incr = 0
        decr = 0
        for r in range(3):
            if board_exp[r+1][c] > board_exp[r][c]:
                incr += (board_exp[r+1][c] - board_exp[r][c])
            else:
                decr += (board_exp[r][c] - board_exp[r+1][c])
        total += max(incr, decr)
    return total

def smoothness_exponents(board_exp):
    diff = 0
    for r in range(4):
        for c in range(3):
            if board_exp[r][c] != 0 and board_exp[r][c+1] != 0:
                diff -= abs(board_exp[r][c+1] - board_exp[r][c])
    for c in range(4):
        for r in range(3):
            if board_exp[r][c] != 0 and board_exp[r+1][c] != 0:
                diff -= abs(board_exp[r+1][c] - board_exp[r][c])
    return diff

def bitboard_heuristic(bitboard):
    # decode as exponents
    exps = []
    tmp = bitboard
    for _ in range(16):
        exps.append(tmp & 0xF)
        tmp >>= 4
    # reshape
    board_exp = [exps[i*4:(i+1)*4] for i in range(4)]
    mono  = monotonicity_exponents(board_exp)
    sm    = smoothness_exponents(board_exp)
    empty = exps.count(0)
    max_e = max(exps)
    # Weighted
    return 1.0*mono + 0.1*sm + 2.7*empty + 1.0*max_e

class ExpectimaxBitboard:
    def __init__(self, depth=EXPECTIMAX_DEPTH):
        self.depth = depth

    def get_best_move(self, bitboard):
        best_move = None
        best_val = float("-inf")
        for action in [0,1,2,3]:
            new_b, sc, moved = bitboard_move(bitboard, action)
            if not moved:
                continue
            val = sc + self.expectimax_value(new_b, self.depth-1, False)
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
            any_move = False
            for action in [0,1,2,3]:
                new_b, sc, moved = bitboard_move(bitboard, action)
                if moved:
                    any_move = True
                    val = sc + self.expectimax_value(new_b, depth-1, False)
                    if val > best_val:
                        best_val = val
            if not any_move:
                return bitboard_heuristic(bitboard)
            return best_val
        else:
            # chance node
            empty_count = bitboard_count_empty(bitboard)
            if empty_count == 0:
                return self.expectimax_value(bitboard, depth-1, True)
            prob2 = 0.9
            prob4 = 0.1
            val_sum = 0.0

            tmp_b = bitboard
            shift = 0
            for _ in range(16):
                nib = (tmp_b & 0xF)
                if nib == 0:
                    # place 2 => exponent=1
                    b2 = (bitboard & ~(0xF << shift)) | (1 << shift)
                    v2 = self.expectimax_value(b2, depth-1, True)
                    # place 4 => exponent=2
                    b4 = (bitboard & ~(0xF << shift)) | (2 << shift)
                    v4 = self.expectimax_value(b4, depth-1, True)
                    val_sum += prob2*v2 + prob4*v4
                tmp_b >>= 4
                shift += 4

            return val_sum / empty_count


################################################################################
# 7) DQN Model + Predict
################################################################################
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

# Attempt to load model if available
dqn_model = None
if os.path.exists(DQN_MODEL_PATH):
    dqn_model = DQNModel()
    # Some torch.load's require removing extra keys:
    checkpoint = torch.load(DQN_MODEL_PATH, map_location=device, weights_only=True)
    dqn_model.load_state_dict(checkpoint, strict=False)
    dqn_model.to(device)
    dqn_model.eval()

def bitboard_to_exponents_list(bitboard):
    arr = []
    tmp = bitboard
    for _ in range(16):
        arr.append(tmp & 0xF)
        tmp >>= 4
    return arr

def predict_moves_sorted_dqn(bitboard):
    if dqn_model is None:
        return []  # no model loaded
    exps = bitboard_to_exponents_list(bitboard)
    state_t = torch.tensor(exps, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = dqn_model(state_t).cpu().numpy()[0]  # shape [4]
    actions_sorted = sorted(range(4), key=lambda i: q_values[i], reverse=True)
    # map numeric to strings for display
    action_map = {0:'Up',1:'Down',2:'Left',3:'Right'}
    return [action_map[a] for a in actions_sorted]


################################################################################
# 8) GUI + Game Logic (Tkinter)
################################################################################
class Game2048GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("2048 - Bitboard Version")
        self.master.resizable(False, False)

        # Build row LUT once
        build_row_tables()

        # Initialize bitboard
        self.bitboard = 0
        self.score = 0
        self.highest_tile = 0
        self.game_over = False
        self.history = []
        self.moves_history = []
        self.game_states = []

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
            2048: "#EDC22E"
        }
        self.best_score = 0
        self.create_ui()
        self.reset_game()

    def create_ui(self):
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
                                command=self.restart_game, bg="#8f7a66",
                                fg="white", font=("Helvetica", 12, "bold"))
        btn_restart.pack(side="left", padx=5)

        btn_viewer = tk.Button(self.bottom_frame, text="View Best (V)",
                               command=self.view_best_game, bg="#8f7a66",
                               fg="white", font=("Helvetica", 12, "bold"))
        btn_viewer.pack(side="left", padx=5)

        self.master.bind("r", lambda e: self.restart_game())
        self.master.bind("R", lambda e: self.restart_game())

        self.master.bind("<Up>", lambda e: self.key_move(0))    # Up
        self.master.bind("<Down>", lambda e: self.key_move(1))  # Down
        self.master.bind("<Left>", lambda e: self.key_move(2))  # Left
        self.master.bind("<Right>", lambda e: self.key_move(3)) # Right

        # Single-step DQN (i) or continuous DQN (c)
        self.master.bind("i", lambda e: self.move_dqn_single())
        self.master.bind("c", lambda e: self.move_dqn_continuous())

        # Single-step Expectimax (e)
        self.master.bind("e", lambda e: self.move_em_single())

        # View best
        self.master.bind("v", lambda e: self.view_best_game())

    def reset_game(self):
        self.bitboard = 0
        self.score = 0
        self.highest_tile = 0
        self.game_over = False
        self.history.clear()
        self.moves_history.clear()
        self.game_states.clear()
        # spawn 2
        self.bitboard = bitboard_spawn(self.bitboard)
        # spawn 2
        self.bitboard = bitboard_spawn(self.bitboard)
        self.game_states.append(self.bitboard)
        self.update_ui()

    def restart_game(self):
        self.reset_game()
        self.game_over_label.config(text="")

    def key_move(self, action):
        if self.game_over:
            return
        pre_bb = self.bitboard
        new_b, sc, moved = bitboard_move(pre_bb, action)
        if not moved:
            return
        self.bitboard = new_b
        self.score += sc
        self.update_highest_tile()
        self.bitboard = bitboard_spawn(self.bitboard)
        self.game_states.append(self.bitboard)
        self.moves_history.append(action)
        self.update_ui()
        if bitboard_is_game_over(self.bitboard):
            self.finish_game()

    def move_dqn_single(self):
        if self.game_over or not dqn_model:
            return
        dirs = predict_moves_sorted_dqn(self.bitboard)  # e.g. ["Down", "Left", ...]
        # map str->num
        action_map = {"Up":0,"Down":1,"Left":2,"Right":3}
        for d in dirs:
            a = action_map[d]
            new_b, sc, moved = bitboard_move(self.bitboard, a)
            if moved:
                self.bitboard = new_b
                self.score += sc
                self.update_highest_tile()
                self.bitboard = bitboard_spawn(self.bitboard)
                self.moves_history.append(a)
                self.game_states.append(self.bitboard)
                self.update_ui()
                if bitboard_is_game_over(self.bitboard):
                    self.finish_game()
                return
        self.finish_game()

    def move_dqn_continuous(self):
        if self.game_over or not dqn_model:
            return

        def step():
            if self.game_over:
                return
            dirs = predict_moves_sorted_dqn(self.bitboard)
            action_map = {"Up":0,"Down":1,"Left":2,"Right":3}
            moved_any = False
            for d in dirs:
                a = action_map[d]
                new_b, sc, moved = bitboard_move(self.bitboard, a)
                if moved:
                    self.bitboard = new_b
                    self.score += sc
                    self.update_highest_tile()
                    self.bitboard = bitboard_spawn(self.bitboard)
                    self.moves_history.append(a)
                    self.game_states.append(self.bitboard)
                    moved_any = True
                    break
            self.update_ui()
            if bitboard_is_game_over(self.bitboard) or not moved_any:
                self.finish_game()
            else:
                self.master.after(CONTINUOUS_STEP_DELAY, step)

        step()

    def move_em_single(self):
        if self.game_over:
            return
        ai = ExpectimaxBitboard(depth=EXPECTIMAX_DEPTH)
        move = ai.get_best_move(self.bitboard)
        if move is None:
            self.finish_game()
            return
        new_b, sc, moved = bitboard_move(self.bitboard, move)
        if not moved:
            self.finish_game()
            return
        self.bitboard = new_b
        self.score += sc
        self.update_highest_tile()
        self.bitboard = bitboard_spawn(self.bitboard)
        self.moves_history.append(move)
        self.game_states.append(self.bitboard)
        self.update_ui()
        if bitboard_is_game_over(self.bitboard):
            self.finish_game()

    def update_ui(self):
        board = bitboard_to_board(self.bitboard)
        for r in range(4):
            for c in range(4):
                val = board[r][c]
                color = self.cell_colors.get(val, "#3C3A32")
                txt = str(val) if val>0 else ""
                self.tiles[r][c].config(text=txt, bg=color)
        if self.score> self.best_score:
            self.best_score = self.score
        self.score_label.config(text=f"Score: {self.score}")
        self.best_label.config(text=f"Best: {self.best_score}")
        self.highest_label.config(text=f"Highest: {self.highest_tile}")
        self.master.update_idletasks()

    def update_highest_tile(self):
        mt = bitboard_get_max_tile(self.bitboard)
        if mt > self.highest_tile:
            self.highest_tile = mt

    def finish_game(self):
        self.game_over = True
        self.game_over_label.config(text="Game Over!")
        self.save_if_best()

    def save_if_best(self):
        current_score = self.score
        best_score_stored = -1
        if os.path.exists(BEST_GAME_PATH):
            try:
                with open(BEST_GAME_PATH, "rb") as f:
                    data = pickle.load(f)
                best_score_stored = data.get("score", -1)
            except:
                pass
        if current_score > best_score_stored:
            data_to_store = {
                "score": self.score,
                "highest_tile": self.highest_tile,
                "states": self.game_states[:],    # store bitboards
                "moves": self.moves_history[:],  # numeric actions 0..3
            }
            with open(BEST_GAME_PATH, "wb") as f:
                pickle.dump(data_to_store, f)
            print(f"New best game saved with score={self.score}, highest tile={self.highest_tile}")

    def view_best_game(self):
        if not os.path.exists(BEST_GAME_PATH):
            print("No best_game.pkl found.")
            return
        with open(BEST_GAME_PATH, "rb") as f:
            data = pickle.load(f)
        states = data["states"]
        moves = data["moves"]
        score = data["score"]
        hi_tile = data["highest_tile"]
        viewer = BestGameViewer(self.master, states, moves, score, hi_tile)
        viewer.show_frame(0)


################################################################################
# 9) Best Game Viewer
################################################################################
class BestGameViewer(tk.Toplevel):
    def __init__(self, master, states, moves, final_score, final_highest):
        super().__init__(master)
        self.title("Best Game Viewer")
        self.geometry("400x400")
        self.resizable(False, False)

        self.states = states
        self.moves = moves
        self.final_score = final_score
        self.final_highest = final_highest
        self.idx = 0

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
            2048: "#EDC22E"
        }

        self.label_info = tk.Label(self, text="", font=("Helvetica", 12, "bold"))
        self.label_info.pack()

        self.grid_frame = tk.Frame(self, bg=self.bg_color)
        self.grid_frame.pack(pady=5)

        self.tiles = []
        for r in range(4):
            row_tiles = []
            for c in range(4):
                lbl = tk.Label(self.grid_frame, text="", bg=self.cell_colors[0],
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
        self.idx = max(0, min(idx, len(self.states)-1))
        bitb = self.states[self.idx]
        board = bitboard_to_board(bitb)
        for r in range(4):
            for c in range(4):
                val = board[r][c]
                color = self.cell_colors.get(val, "#3C3A32")
                txt = str(val) if val>0 else ""
                self.tiles[r][c].config(text=txt, bg=color)

        msg = f"Move {self.idx}/{len(self.states)-1}"
        if self.idx>0 and self.idx-1< len(self.moves):
            direction_map = {0:'Up',1:'Down',2:'Left',3:'Right'}
            msg += f" | Action: {direction_map[self.moves[self.idx-1]]}"
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
        self.after(100, self.auto_play)


################################################################################
# 10) Main entry
################################################################################
def main():
    root = tk.Tk()
    app = Game2048GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
