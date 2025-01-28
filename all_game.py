import tkinter as tk
import math
import random
import os
import pickle
from functools import lru_cache
from tkinter import filedialog
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

###############################################################################
# 1) Global Settings
###############################################################################
# Adjust as needed.
# Strategy: "Expectimax", "Minimax", "Random", "Rotating", "MCTS", "DQN"
STRATEGY = ["Expectimax", "Minimax", "Random", "Rotating", "MCTS", "DQN"]
AI_STRATEGY = STRATEGY[0] # or "Minimax", "Random", "Rotating", etc.
EXPECTIMAX_DEPTH = 3        # search depth
MINIMAX_DEPTH = 4
MCTS_ROLLOUTS = 200          # for a future MCTS (placeholder)
BEST_GAME_PATH = "best_game.pkl"

# For continuous AI steps, how many ms delay
CONTINUOUS_STEP_DELAY = 10

# If you have a DQN model:
DQN_MODEL_PATH = "model.pth"
USE_DQN = True  # set to False if you don't have a model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# 2) Row-Merge LUTs for Speed (each row is 16 bits)
###############################################################################
row_left_table = [0]*65536
row_score_table = [0]*65536

def build_row_tables():
    """
    Precompute how each 16-bit row transforms when moved left,
    plus the immediate score gained from merges.
    """
    for row_val in range(65536):
        # extract 4 nibbles
        t0 = (row_val >>  0) & 0xF
        t1 = (row_val >>  4) & 0xF
        t2 = (row_val >>  8) & 0xF
        t3 = (row_val >> 12) & 0xF
        row = [t0, t1, t2, t3]

        filtered = [x for x in row if x != 0]
        merged = []
        score = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                val = filtered[i] + 1
                score += (1 << val)  # 2^(val)
                merged.append(val)
                skip = True
            else:
                merged.append(filtered[i])
        merged.extend([0]*(4-len(merged)))

        new_val = (
            (merged[0] & 0xF)
            | ((merged[1] & 0xF) << 4)
            | ((merged[2] & 0xF) << 8)
            | ((merged[3] & 0xF) << 12)
        )
        row_left_table[row_val] = new_val
        row_score_table[row_val] = score

def reverse_row(row16):
    n0 = (row16 >>  0) & 0xF
    n1 = (row16 >>  4) & 0xF
    n2 = (row16 >>  8) & 0xF
    n3 = (row16 >> 12) & 0xF
    return (n3) | (n2 << 4) | (n1 << 8) | (n0 << 12)

###############################################################################
# 3) Bitboard Representation + Moves
###############################################################################
# 64-bit integer, 4 rows × 16 bits each (lowest 16 bits = row0, then row1, etc.)

def get_row(bitboard, row_idx):
    return (bitboard >> (16*row_idx)) & 0xFFFF

def set_row(bitboard, row_idx, row16):
    shift = 16*row_idx
    mask = 0xFFFF << shift
    bitboard &= ~mask
    bitboard |= (row16 << shift)
    return bitboard

def transpose(bitboard):
    """
    Transpose 4×4 nibble matrix. (Decode → reorder → re-encode.)
    """
    cells = [0]*16
    tmp = bitboard
    for r in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(4):
            cells[r*4 + c] = (row >> (4*c)) & 0xF

    newcells = [0]*16
    for r in range(4):
        for c in range(4):
            newcells[c*4 + r] = cells[r*4 + c]

    b2 = 0
    for r in range(4):
        rowval = 0
        for c in range(4):
            rowval |= (newcells[r*4 + c] << (4*c))
        b2 = set_row(b2, r, rowval)
    return b2

def shift_left(bitboard):
    moved = False
    score = 0
    b2 = bitboard
    for r in range(4):
        row = get_row(b2, r)
        new_row = row_left_table[row]
        if new_row != row:
            moved = True
        score += row_score_table[row]
        b2 = set_row(b2, r, new_row)
    return b2, score, moved

def shift_right(bitboard):
    moved = False
    score = 0
    b2 = bitboard
    for r in range(4):
        row = get_row(b2, r)
        rev = reverse_row(row)
        new_rev = row_left_table[rev]
        if new_rev != rev:
            moved = True
        score += row_score_table[rev]
        fin = reverse_row(new_rev)
        b2 = set_row(b2, r, fin)
    return b2, score, moved

def shift_up(bitboard):
    t = transpose(bitboard)
    t2, sc, mv = shift_left(t)
    return transpose(t2), sc, mv

def shift_down(bitboard):
    t = transpose(bitboard)
    t2, sc, mv = shift_right(t)
    return transpose(t2), sc, mv

def bitboard_move(bitboard, action):
    """
    action: 0=Up,1=Down,2=Left,3=Right
    Returns (new_bitboard, score_gained, moved).
    """
    if action == 0:
        return shift_up(bitboard)
    elif action == 1:
        return shift_down(bitboard)
    elif action == 2:
        return shift_left(bitboard)
    elif action == 3:
        return shift_right(bitboard)
    return bitboard, 0, False

###############################################################################
# 4) Spawn, Check Game Over, Convert
###############################################################################
def bitboard_count_empty(bitboard):
    cnt = 0
    tmp = bitboard
    for _ in range(16):
        if (tmp & 0xF) == 0:
            cnt += 1
        tmp >>= 4
    return cnt

def bitboard_spawn(bitboard):
    """
    Spawn tile (2 or 4) in a random empty nibble.
    """
    empties = bitboard_count_empty(bitboard)
    if empties == 0:
        return bitboard

    r = random.randrange(empties)
    val = 1 if random.random() < 0.9 else 2  # exponent=1 => tile=2, exponent=2 => tile=4

    tmp_b = bitboard
    shift = 0
    seen = 0
    while True:
        nib = (tmp_b & 0xF)
        if nib == 0:
            if seen == r:
                # place val
                bitboard &= ~(0xF << shift)
                bitboard |= (val << shift)
                return bitboard
            seen+=1
        shift += 4
        tmp_b >>= 4

def bitboard_is_game_over(bitboard):
    """
    True if no moves possible (no empty & no merges).
    """
    if bitboard_count_empty(bitboard) > 0:
        return False
    # check merges horizontally
    tmp = bitboard
    for _ in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(3):
            n1 = (row >> (4*c)) & 0xF
            n2 = (row >> (4*(c+1))) & 0xF
            if n1 == n2:
                return False
    # check merges vertically
    t = transpose(bitboard)
    tmp = t
    for _ in range(4):
        row = tmp & 0xFFFF
        tmp >>= 16
        for c in range(3):
            n1 = (row >> (4*c)) & 0xF
            n2 = (row >> (4*(c+1))) & 0xF
            if n1 == n2:
                return False
    return True

def bitboard_get_max_tile(bitboard):
    mx = 0
    tmp = bitboard
    for _ in range(16):
        nib = tmp & 0xF
        if nib>mx:
            mx=nib
        tmp >>= 4
    return (1<<mx)

def bitboard_to_board(bitboard):
    board = [[0]*4 for _ in range(4)]
    shift=0
    for r in range(4):
        for c in range(4):
            nib = (bitboard >> shift) & 0xF
            board[r][c] = (1<<nib) if nib>0 else 0
            shift+=4
    return board

def board_to_bitboard(mat):
    b=0
    shift=0
    for r in range(4):
        for c in range(4):
            val = mat[r][c]
            if val>0:
                exp = int(math.log2(val))
                b |= (exp<<shift)
            shift+=4
    return b

###############################################################################
# 5) Advanced Heuristics: Corner-Building, Monotonicity, Empties, Score, etc.
###############################################################################
def corner_weighting(bitboard):
    """
    Encourage largest tile in corner by applying a positional weighting.
    For example, a classic approach is to treat the board as:
      [  4,  3,  2,  1]
      [  5,  6,  7,  8]
      [ 12, 11, 10,  9]
      [ 13, 14, 15, 16]
    or so, then multiply exponents. We'll do something simpler: we sum
    row*(some factor)+col*(some factor).
    """
    # decode exps
    exps = []
    tmp = bitboard
    for _ in range(16):
        exps.append(tmp & 0xF)
        tmp >>= 4
    # We'll do a snaking weighting:
    # top-left corner is highest weight => to encourage big tiles there
    # Just an example "gradient"
    # or you can define a 16-element table with custom values.
    weights = [
       50,  4,  3,  2,
       10,  5,  1,  1,
        5,  2,  1,  1,
        2,  1,  1,  1
    ]
    value = 0
    for i in range(16):
        value += exps[i] * weights[i]
    return value

def monotonicity(bitboard):
    """
    Row+column monotonicity in exponents. Similar to standard 2048 approach.
    """
    # decode
    bmat = []
    tmp = bitboard
    for r in range(4):
        row = []
        rr = tmp & 0xFFFF
        tmp >>=16
        for c in range(4):
            row.append((rr>>(4*c)) & 0xF)
        bmat.append(row)

    total=0
    # row monotonic
    for r in range(4):
        incr,decr=0,0
        for c in range(3):
            if bmat[r][c+1]>bmat[r][c]:
                incr+=(bmat[r][c+1]-bmat[r][c])
            else:
                decr+=(bmat[r][c]-bmat[r][c+1])
        total+=max(incr,decr)
    # col
    for c in range(4):
        incr,decr=0,0
        for r in range(3):
            if bmat[r+1][c]>bmat[r][c]:
                incr+=(bmat[r+1][c]-bmat[r][c])
            else:
                decr+=(bmat[r][c]-bmat[r+1][c])
        total+=max(incr,decr)
    return total

def count_empty(bitboard):
    return bitboard_count_empty(bitboard)

def advanced_heuristic(bitboard):
    """
    Combine corner weighting, monotonicity, empties, and
    a small bonus for largest exponent.
    """
    empties = count_empty(bitboard)
    corner_val = corner_weighting(bitboard)
    mono_val = monotonicity(bitboard)
    max_tile = bitboard_get_max_tile(bitboard)
    max_exp = int(math.log2(max_tile)) if max_tile>0 else 0
    # Weighted sum
    return (3.0 * corner_val
            + 1.0 * mono_val
            + 2.0 * empties
            + 2.0 * max_exp
    )

###############################################################################
# 6) Multiple AI Strategies
###############################################################################

########################################
# 6.1) Expectimax (search by depth)
########################################
class ExpectimaxAI:
    def __init__(self, depth=EXPECTIMAX_DEPTH):
        self.depth = depth

    def get_move(self, bitboard):
        best_val = float("-inf")
        best_move = None
        for action in [0,1,2,3]:
            nb, sc, moved = bitboard_move(bitboard, action)
            if not moved:
                continue
            val = sc + self.expectimax_value(nb, self.depth-1, is_player=False)
            if val>best_val:
                best_val = val
                best_move = action
        return best_move

    @lru_cache(maxsize=8000)
    def expectimax_value(self, bitboard, depth, is_player):
        if depth<=0 or bitboard_is_game_over(bitboard):
            return advanced_heuristic(bitboard)

        if is_player:
            # max node
            best_val = float("-inf")
            anymove=False
            for a in [0,1,2,3]:
                b2, sc, moved = bitboard_move(bitboard, a)
                if moved:
                    anymove=True
                    val = sc + self.expectimax_value(b2, depth-1, False)
                    if val>best_val:
                        best_val = val
            if not anymove:
                return advanced_heuristic(bitboard)
            return best_val
        else:
            # chance node
            empty = bitboard_count_empty(bitboard)
            if empty==0:
                return self.expectimax_value(bitboard, depth-1, True)
            val_sum = 0
            prob2, prob4 = 0.9, 0.1
            shift=0
            tmp=bitboard
            for _ in range(16):
                nib=tmp & 0xF
                if nib==0:
                    # place 2 => exponent=1
                    b2 = (bitboard & ~(0xF << shift)) | (1<<shift)
                    v2 = self.expectimax_value(b2, depth-1, True)
                    # place 4 => exponent=2
                    b4 = (bitboard & ~(0xF << shift)) | (2<<shift)
                    v4 = self.expectimax_value(b4, depth-1, True)
                    val_sum += prob2*v2 + prob4*v4
                tmp >>=4
                shift+=4
            return val_sum/empty

########################################
# 6.2) Minimax (similar to Expectimax, but chance node replaced by adversarial)
########################################
class MinimaxAI:
    def __init__(self, depth=MINIMAX_DEPTH):
        self.depth=depth

    def get_move(self, bitboard):
        best_val = float("-inf")
        best_move=None
        for a in [0,1,2,3]:
            nb, sc, moved = bitboard_move(bitboard, a)
            if moved:
                val = sc + self.minimax_value(nb, self.depth-1, is_player=False)
                if val>best_val:
                    best_val=val
                    best_move=a
        return best_move

    @lru_cache(maxsize=8000)
    def minimax_value(self, bitboard, depth, is_player):
        if depth<=0 or bitboard_is_game_over(bitboard):
            return advanced_heuristic(bitboard)
        if is_player:
            # maximize
            best_val = float("-inf")
            anymove=False
            for a in [0,1,2,3]:
                nb, sc, moved = bitboard_move(bitboard, a)
                if moved:
                    anymove=True
                    v = sc + self.minimax_value(nb, depth-1, False)
                    if v>best_val:
                        best_val=v
            if not anymove:
                return advanced_heuristic(bitboard)
            return best_val
        else:
            # "opponent" tries to minimize. We'll pick the worst spawn location for the player
            empty = bitboard_count_empty(bitboard)
            if empty==0:
                return self.minimax_value(bitboard, depth-1, True)
            best_val = float("inf")
            shift=0
            tmp=bitboard
            # We'll assume the "worst" tile is always a 4 if that is more punishing.
            # Or we try both 2/4 to see which is worst.
            for _ in range(16):
                nib = tmp & 0xF
                if nib==0:
                    # place a 2
                    b2 = (bitboard & ~(0xF<<shift)) | (1<<shift)
                    v2 = self.minimax_value(b2, depth-1, True)
                    # place a 4
                    b4 = (bitboard & ~(0xF<<shift)) | (2<<shift)
                    v4 = self.minimax_value(b4, depth-1, True)
                    # the "opponent" picks the min (lowest) outcome
                    # i.e. tries to sabotage the player
                    worst = min(v2, v4)
                    if worst<best_val:
                        best_val=worst
                tmp >>=4
                shift+=4
            return best_val

########################################
# 6.3) Random Trials (just picks random valid move)
########################################
class RandomAI:
    def get_move(self, bitboard):
        random.shuffle(ACTIONS)
        for a in ACTIONS:
            nb, sc, moved = bitboard_move(bitboard, a)
            if moved:
                return a
        return None

########################################
# 6.4) Rotating Moves (Up,Left,Down,Right,...)
########################################
class RotatingAI:
    def __init__(self):
        self.idx=0
    def get_move(self, bitboard):
        a = ACTIONS[self.idx % len(ACTIONS)]
        self.idx+=1
        # check if valid
        nb,sc,moved = bitboard_move(bitboard, a)
        if moved:
            return a
        else:
            # if invalid, try next
            self.idx+=1
            return ACTIONS[(self.idx-1) % len(ACTIONS)]

# We might add "Ordered Moves" or "Corner Spam" by always returning [Up,Left] if valid, etc.
ACTIONS = [0,1,2,3]  # Up,Down,Left,Right (our standard order)

########################################
# 6.5) Monte Carlo Tree Search (Stub)
########################################
class MctsAI:
    """
    Placeholder. The real MCTS would expand states, do rollouts, backprop.
    For brevity, we just randomize or do a partial approach.
    """
    def __init__(self, rollouts=MCTS_ROLLOUTS):
        self.rollouts = rollouts

    def get_move(self, bitboard):
        # simple approach: for each possible move, do random rollouts, pick best average
        best_move = None
        best_score = -999999
        for a in ACTIONS:
            nb, sc, moved = bitboard_move(bitboard, a)
            if not moved:
                continue
            # do random rollouts from nb
            # accumulate average
            sum_ = 0
            for _ in range(self.rollouts//4):
                sum_ += self.random_rollout(nb)
            avg_ = sum_ / (self.rollouts//4)
            if avg_+sc > best_score:
                best_score = avg_+sc
                best_move = a
        return best_move

    def random_rollout(self, bitboard):
        # random steps until gameover, then return advanced_heuristic
        tmp = bitboard
        for _ in range(50):  # cap
            if bitboard_is_game_over(tmp):
                break
            moves = []
            for a in ACTIONS:
                _,_,mov = bitboard_move(tmp, a)
                if mov:
                    moves.append(a)
            if not moves:
                break
            a = random.choice(moves)
            nb, sc, _ = bitboard_move(tmp, a)
            tmp = bitboard_spawn(nb)
        return advanced_heuristic(tmp)


###############################################################################
# 7) (Optional) DQN Model
###############################################################################
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

dqn_model = None
if USE_DQN and os.path.exists(DQN_MODEL_PATH):
    dqn_model = DQNModel()
    ckpt = torch.load(DQN_MODEL_PATH, map_location=device, weights_only=True)
    # If mismatch keys, remove strict=False:
    dqn_model.load_state_dict(ckpt, strict=False)
    dqn_model.to(device)
    dqn_model.eval()

def bitboard_to_explist(bb):
    arr=[]
    for _ in range(16):
        arr.append(bb & 0xF)
        bb>>=4
    return arr

class DqnAI:
    def get_move(self, bitboard):
        if not dqn_model:
            # fallback if no model
            return None
        exps = bitboard_to_explist(bitboard)
        st = torch.tensor(exps, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q = dqn_model(st).cpu().numpy()[0]
        # pick best Q
        best_a = max(range(4), key=lambda i: q[i])
        # if that doesn't move, try next best
        order = sorted(range(4), key=lambda i: q[i], reverse=True)
        for a in order:
            nb,_,m = bitboard_move(bitboard, a)
            if m:
                return a
        return None
    
def create_ai(strategy):
        if strategy=="Expectimax":
            return ExpectimaxAI(EXPECTIMAX_DEPTH)
        elif strategy=="Minimax":
            return MinimaxAI(MINIMAX_DEPTH)
        elif strategy=="Random":
            return RandomAI()
        elif strategy=="Rotating":
            return RotatingAI()
        elif strategy=="MCTS":
            return MctsAI(MCTS_ROLLOUTS)
        elif strategy=="DQN":
            return DqnAI()
        # fallback
        return RandomAI()


###############################################################################
# 8) Tkinter GUI & Controller
###############################################################################
class Game2048GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("2048 Advanced - Bitboard")
        self.master.resizable(False, False)

        # Build row LUT
        build_row_tables()

        self.bitboard = 0
        self.score = 0
        self.highest = 0
        self.game_over = False

        self.history=[]
        self.states=[]
        self.moves=[]
        self.best_score=0

        # Decide which AI to use
        # You can easily switch this line:
        self.ai = create_ai(AI_STRATEGY)

        self.bg_color="#BBADA0"
        self.cell_colors = {
            0:"#CDC1B4", 2:"#EEE4DA", 4:"#EDE0C8", 8:"#F2B179",16:"#F59563",32:"#F67C5F",64:"#F65E3B",
            128:"#EDCF72",256:"#EDCC61",512:"#EDC850",1024:"#EDC53F",2048:"#EDC22E"
        }
        self.make_ui()
        self.reset_game()

    def make_ui(self):
        self.top_frame = tk.Frame(self.master, bg=self.bg_color)
        self.top_frame.pack(pady=5)

        self.lbl_score = tk.Label(self.top_frame, text="Score:0",
                                  font=("Helvetica",14,"bold"), bg=self.bg_color)
        self.lbl_score.pack(side="left", padx=10)
        self.lbl_best = tk.Label(self.top_frame, text="Best:0",
                                 font=("Helvetica",14,"bold"), bg=self.bg_color)
        self.lbl_best.pack(side="left", padx=10)
        self.lbl_highest = tk.Label(self.top_frame, text="Highest:0",
                                    font=("Helvetica",14,"bold"), bg=self.bg_color)
        self.lbl_highest.pack(side="left",padx=10)
        self.lbl_over = tk.Label(self.top_frame, text="",
                                 font=("Helvetica",16,"bold"),fg="red",bg=self.bg_color)
        self.lbl_over.pack(side="left",padx=10)

        self.main_frame = tk.Frame(self.master,bg=self.bg_color)
        self.main_frame.pack(padx=10,pady=10)
        self.tiles = []
        for r in range(4):
            row_tiles=[]
            for c in range(4):
                lbl = tk.Label(self.main_frame, text="", bg=self.cell_colors[0],
                               font=("Helvetica",20,"bold"), width=4, height=2)
                lbl.grid(row=r, column=c, padx=5, pady=5)
                row_tiles.append(lbl)
            self.tiles.append(row_tiles)

        self.bottom_frame = tk.Frame(self.master,bg=self.bg_color)
        self.bottom_frame.pack()

        btn_restart = tk.Button(self.bottom_frame, text="Restart (R)",
                                command=self.restart_game, bg="#8f7a66",
                                fg="white", font=("Helvetica",12,"bold"))
        btn_restart.pack(side="left", padx=5)

        btn_view = tk.Button(self.bottom_frame, text="View Best (V)",
                             command=self.view_best_game, bg="#8f7a66",
                             fg="white", font=("Helvetica",12,"bold"))
        btn_view.pack(side="left", padx=5)

        # Binds
        self.master.bind("r", lambda e: self.restart_game())
        self.master.bind("R", lambda e: self.restart_game())
        self.master.bind("<Up>", lambda e: self.do_move(0))
        self.master.bind("<Down>", lambda e: self.do_move(1))
        self.master.bind("<Left>", lambda e: self.do_move(2))
        self.master.bind("<Right>", lambda e: self.do_move(3))
        # Single-step AI: 'i'
        self.master.bind("i", lambda e: self.ai_single_step())
        # Continuous AI: 'c'
        self.master.bind("c", lambda e: self.ai_continuous())
        # View best: 'v'
        self.master.bind("v", lambda e: self.view_best_game())

    def reset_game(self):
        self.bitboard=0
        self.bitboard = bitboard_spawn(self.bitboard)
        self.bitboard = bitboard_spawn(self.bitboard)
        self.score=0
        self.highest= bitboard_get_max_tile(self.bitboard)
        self.game_over=False
        self.history.clear()
        self.states.clear()
        self.moves.clear()
        self.states.append(self.bitboard)
        self.update_ui()
        self.lbl_over.config(text="")

    def restart_game(self):
        self.reset_game()

    def do_move(self, action):
        if self.game_over:
            return
        newb, sc, moved = bitboard_move(self.bitboard, action)
        if not moved:
            return
        self.bitboard = newb
        self.score+=sc
        self.update_highest()
        self.bitboard = bitboard_spawn(self.bitboard)
        self.moves.append(action)
        self.states.append(self.bitboard)
        self.update_ui()
        if bitboard_is_game_over(self.bitboard):
            self.finish_game()

    def ai_single_step(self):
        if self.game_over:
            return
        move = self.ai.get_move(self.bitboard)
        if move is None:
            self.finish_game()
            return
        newb, sc, moved = bitboard_move(self.bitboard, move)
        if not moved:
            self.finish_game()
            return
        self.bitboard = newb
        self.score+=sc
        self.update_highest()
        self.bitboard = bitboard_spawn(self.bitboard)
        self.moves.append(move)
        self.states.append(self.bitboard)
        self.update_ui()
        if bitboard_is_game_over(self.bitboard):
            self.finish_game()

    def ai_continuous(self):
        def step():
            if self.game_over:
                return
            move = self.ai.get_move(self.bitboard)
            if move is None:
                self.finish_game()
                return
            nb, sc, mv = bitboard_move(self.bitboard, move)
            if not mv:
                self.finish_game()
                return
            self.bitboard = nb
            self.score+=sc
            self.update_highest()
            self.bitboard = bitboard_spawn(self.bitboard)
            self.moves.append(move)
            self.states.append(self.bitboard)
            self.update_ui()
            if bitboard_is_game_over(self.bitboard):
                self.finish_game()
            else:
                self.master.after(CONTINUOUS_STEP_DELAY, step)
        step()

    def update_highest(self):
        mt = bitboard_get_max_tile(self.bitboard)
        if mt>self.highest:
            self.highest=mt

    def update_ui(self):
        b = bitboard_to_board(self.bitboard)
        for r in range(4):
            for c in range(4):
                val = b[r][c]
                clr = self.cell_colors.get(val, "#3C3A32")
                txt = str(val) if val>0 else ""
                self.tiles[r][c].config(text=txt, bg=clr)
        if self.score> self.best_score:
            self.best_score=self.score
        self.lbl_score.config(text=f"Score:{self.score}")
        self.lbl_best.config(text=f"Best:{self.best_score}")
        self.lbl_highest.config(text=f"Highest:{self.highest}")
        self.master.update_idletasks()

    def finish_game(self):
        self.game_over=True
        self.lbl_over.config(text="Game Over!")
        self.save_if_best()

    def save_if_best(self):
        best_score_stored=-1
        if os.path.exists(BEST_GAME_PATH):
            try:
                with open(BEST_GAME_PATH,"rb") as f:
                    data = pickle.load(f)
                best_score_stored = data.get("score",-1)
            except:
                pass
        if self.score>best_score_stored:
            data_to_store = {
                "score":self.score,
                "highest":self.highest,
                "states":self.states[:],
                "moves":self.moves[:],
            }
            with open(BEST_GAME_PATH,"wb") as f:
                pickle.dump(data_to_store,f)
            print(f"[INFO] New best game with score={self.score}, highest={self.highest}")

    def view_best_game(self):
        if not os.path.exists(BEST_GAME_PATH):
            print("No best game found.")
            return
        with open(BEST_GAME_PATH,"rb") as f:
            data = pickle.load(f)
        states = data["states"]
        moves = data["moves"]
        sc = data["score"]
        hi = data["highest"]

        viewer = BestGameViewer(self.master, states, moves, sc, hi)
        viewer.show_frame(0)

###############################################################################
# 9) BestGameViewer
###############################################################################
class BestGameViewer(tk.Toplevel):
    def __init__(self, master, states, moves, final_score, final_highest):
        super().__init__(master)
        self.title("Best Game Viewer")
        self.resizable(False,False)
        self.geometry("470x330")
        self.states=states
        self.moves=moves
        self.final_score=final_score
        self.final_highest=final_highest
        self.slider_val = 30
        self.idx=0

        self.bg_color="#BBADA0"
        self.cell_colors={
            0:"#CDC1B4", 2:"#EEE4DA", 4:"#EDE0C8", 8:"#F2B179",16:"#F59563",32:"#F67C5F",64:"#F65E3B",
            128:"#EDCF72",256:"#EDCC61",512:"#EDC850",1024:"#EDC53F",2048:"#EDC22E"
        }

        self.lbl_info = tk.Label(self, text="", font=("Helvetica",12,"bold"))
        self.lbl_info.pack()

        self.grid_frame = tk.Frame(self, bg=self.bg_color)
        self.grid_frame.pack()

        self.tiles=[]
        for r in range(4):
            row_tiles=[]
            for c in range(4):
                lb = tk.Label(self.grid_frame, text="", bg=self.cell_colors[0],
                              font=("Helvetica",16,"bold"), width=4, height=2)
                lb.grid(row=r, column=c, padx=3, pady=3)
                row_tiles.append(lb)
            self.tiles.append(row_tiles)

        nav = tk.Frame(self)
        nav.pack()

        tk.Button(nav, text="Prev", command=self.prev_state).pack(side="left", padx=5)
        tk.Button(nav, text="Next", command=self.next_state).pack(side="left", padx=5)
        tk.Button(nav, text="Play", command=self.auto_play).pack(side="left", padx=5)
        # add slider that chagne up the speed and update slider_val
        self.slider = tk.Scale(nav, from_=10, to=1000, orient="horizontal", length=200)
        self.slider.set(self.slider_val)
        self.slider.pack(side="left", padx=5)
        self.slider.bind("<ButtonRelease-1>", self.update_speed)
        # add option to load another game
        tk.Button(nav, text="Load Another", command=self.load_another_game).pack(side="left", padx=5)
    
    def load_another_game(self):
        path = filedialog.askopenfilename(title="Select Best Game File",
                                          filetypes=(("Pickle files","*.pkl"),("All files","*.*")))
        if not path:
            return
        with open(path,"rb") as f:
            data = pickle.load(f)
        self.states = data["states"]
        self.moves = data["moves"]
        self.final_score = data["score"]
        self.final_highest = data["highest"]
        self.idx=0
        self.show_frame(0)
    
    def update_speed(self, e):
        self.slider_val = self.slider.get()


    def show_frame(self, i):
        if i<0: i=0
        if i>= len(self.states): i=len(self.states)-1
        self.idx=i
        b = self.states[i]
        mat = bitboard_to_board(b)
        for r in range(4):
            for c in range(4):
                val = mat[r][c]
                clr = self.cell_colors.get(val,"#3C3A32")
                txt = str(val) if val>0 else ""
                self.tiles[r][c].config(text=txt, bg=clr)

        msg = f"Move {i}/{len(self.states)-1}"
        if i>0 and i-1<len(self.moves):
            direction_map = {0:"Up",1:"Down",2:"Left",3:"Right"}
            mv = self.moves[i-1]
            msg+=f" | Action: {direction_map[mv]}"
        msg+=f" | Score={self.final_score}, Highest={self.final_highest}"
        self.lbl_info.config(text=msg)

    def next_state(self):
        self.show_frame(self.idx+1)

    def prev_state(self):
        self.show_frame(self.idx-1)

    def auto_play(self):
        if self.idx>=len(self.states)-1:
            return
        self.next_state()
        self.after(self.slider_val, self.auto_play)

###############################################################################
# 10) Batch Game Runner
###############################################################################
def run_batch_games(num_games=1000, strategy=AI_STRATEGY):
    build_row_tables()
    ai = create_ai(strategy)

    best_score = -1
    best_game = None
    total_score = 0
    histogram_highest_tile = defaultdict(int)

    for game_num in range(1, num_games + 1):
        print(f"Running game {game_num}/{num_games} with strategy {strategy}", end="\r")
        bitboard = 0
        bitboard = bitboard_spawn(bitboard)
        bitboard = bitboard_spawn(bitboard)
        score = 0
        highest = bitboard_get_max_tile(bitboard)
        states = [bitboard]
        moves = []

        while True:
            move = ai.get_move(bitboard)
            if move is None:
                break
            newb, sc, moved = bitboard_move(bitboard, move)
            if not moved:
                break
            bitboard = newb
            score += sc
            current_highest = bitboard_get_max_tile(bitboard)
            if current_highest > highest:
                highest = current_highest
            bitboard = bitboard_spawn(bitboard)
            moves.append(move)
            states.append(bitboard)

            if bitboard_is_game_over(bitboard):
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

    # Calculate average score
    average_score = total_score / num_games if num_games > 0 else 0

    # Compile stats
    stats = {
        "strategy": strategy,
        "best_score": best_score,
        "average_score": average_score,
        "histogram_highest_tile": dict(histogram_highest_tile)
    }

    # Save best game if applicable
    if best_game:
        # Save best game per strategy with unique filename
        strategy_safe = strategy.lower()
        best_game_path = f"best_game_{strategy_safe}.pkl"
        # check if the saved file is better than the current best game
        if os.path.exists(best_game_path):
            with open(best_game_path, "rb") as f:
                data = pickle.load(f)
                if best_score <= data["score"]:
                    pass
                else:
                    with open(best_game_path, "wb") as f:
                        pickle.dump(best_game, f)
                    print(f"[INFO] Best game for strategy '{strategy}' saved with Score={best_game['score']}, Highest Tile={best_game['highest']}")
        else:
            with open(best_game_path, "wb") as f:
                pickle.dump(best_game, f)
            print(f"[INFO] Best game for strategy '{strategy}' saved with Score={best_game['score']}, Highest Tile={best_game['highest']}")
    return stats

def run_multiple_batch_strategies(num_games=1000, strategies=STRATEGY):
    stats_all = {}
    for strategy in strategies:
        print(f"\nRunning batch games for strategy: {strategy}")
        stats = run_batch_games(num_games, strategy)
        stats_all[strategy] = stats
    return stats_all

def visualize_stats(stats_all, num_games):
    strategies = list(stats_all.keys())
    average_scores = [stats_all[s]['average_score'] for s in strategies]
    best_scores = [stats_all[s]['best_score'] for s in strategies]

    # Plot Average Scores
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, average_scores, color='skyblue')
    plt.xlabel('AI Strategies')
    plt.ylabel('Average Score')
    plt.title(f'Average Scores over {num_games} Games')
    plt.savefig('average_scores.png')
    plt.show()

    # Plot Best Scores
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, best_scores, color='salmon')
    plt.xlabel('AI Strategies')
    plt.ylabel('Best Score')
    plt.title(f'Best Scores over {num_games} Games')
    plt.savefig('best_scores.png')
    plt.show()

    # Plot Highest Tile Distribution
    plt.figure(figsize=(12, 8))
    
    # Create list of all possible tile values
    all_tiles = [2**i for i in range(4, 14)]  # 2 to 2048
    
    for strategy in strategies:
        histogram = stats_all[strategy]['histogram_highest_tile']
        
        # Fill in counts, use 0 for missing values
        counts = [histogram.get(tile, 0) for tile in all_tiles]
        
        # Create bar plot with slight offset for each strategy
        plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
    plt.xlabel('Highest Tile')
    plt.ylabel('Frequency')
    plt.title(f'Highest Tile Distribution over {num_games} Games')
    plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig('highest_tile_distribution.png')
    plt.show()



###############################################################################
# 11) Main
###############################################################################
def main():
    root = tk.Tk()
    app = Game2048GUI(root)
    root.mainloop()

RUN_BATCH = True
NUM_BATCH_GAMES = 1000
RUN_MULTIPLE_BATCH = False

import time
if __name__=="__main__":
    if RUN_BATCH:
        start_time = time.time()
        run_batch_games(NUM_BATCH_GAMES, AI_STRATEGY)
        print(f"Batch games completed in {time.time()-start_time:.2f} seconds.")
    elif RUN_MULTIPLE_BATCH:
        stats_all = run_multiple_batch_strategies(NUM_BATCH_GAMES, STRATEGY)
        visualize_stats(stats_all, NUM_BATCH_GAMES)
    else:
        main()