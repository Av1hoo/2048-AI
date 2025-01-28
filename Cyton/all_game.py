import tkinter as tk
import math
import random
import os
import pickle
from functools import lru_cache
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from tkinter import filedialog
import ctypes
import sys
import time
from game_viewer import BestGameViewer

###############################################################################
# 1) Load the C Library via ctypes
###############################################################################
libname = "C/c_game2048.dll"
c2048 = ctypes.CDLL(libname)

c2048.init_c_game_2048.argtypes = []
c2048.init_c_game_2048.restype = None

c2048.bitboard_move.argtypes = [ctypes.c_uint64, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_bool)]
c2048.bitboard_move.restype = ctypes.c_uint64

c2048.bitboard_count_empty.argtypes = [ctypes.c_uint64]
c2048.bitboard_count_empty.restype = ctypes.c_int

c2048.bitboard_spawn.argtypes = [ctypes.c_uint64]
c2048.bitboard_spawn.restype = ctypes.c_uint64

c2048.bitboard_is_game_over.argtypes = [ctypes.c_uint64]
c2048.bitboard_is_game_over.restype = ctypes.c_bool

c2048.bitboard_get_max_tile.argtypes = [ctypes.c_uint64]
c2048.bitboard_get_max_tile.restype = ctypes.c_int

Int4Array = ctypes.c_int * 4
Board4x4 = Int4Array * 4
c2048.bitboard_to_board_c.argtypes = [ctypes.c_uint64, Board4x4]
c2048.bitboard_to_board_c.restype = None

c2048.board_to_bitboard_c.argtypes = [Board4x4]
c2048.board_to_bitboard_c.restype = ctypes.c_uint64

c2048.advanced_heuristic_c.argtypes = [ctypes.c_uint64]
c2048.advanced_heuristic_c.restype = ctypes.c_double

def init_c_game():
    c2048.init_c_game_2048()

def bitboard_move(bitboard, action):
    out_score = ctypes.c_int(0)
    out_moved = ctypes.c_bool(False)
    nb = c2048.bitboard_move(ctypes.c_uint64(bitboard),
                             ctypes.c_int(action),
                             ctypes.byref(out_score),
                             ctypes.byref(out_moved))
    return nb, out_score.value, out_moved.value

def bitboard_count_empty(bitboard):
    return c2048.bitboard_count_empty(ctypes.c_uint64(bitboard))

def bitboard_spawn(bitboard):
    return c2048.bitboard_spawn(ctypes.c_uint64(bitboard))

def bitboard_is_game_over(bitboard):
    return c2048.bitboard_is_game_over(ctypes.c_uint64(bitboard))

def bitboard_get_max_tile(bitboard):
    return c2048.bitboard_get_max_tile(ctypes.c_uint64(bitboard))

def bitboard_to_board(bitboard):
    b = Board4x4()
    c2048.bitboard_to_board_c(ctypes.c_uint64(bitboard), b)
    pyboard = []
    for r in range(4):
        row = []
        for c in range(4):
            row.append(b[r][c])
        pyboard.append(row)
    return pyboard

def board_to_bitboard(mat):
    b = Board4x4()
    for r in range(4):
        for c in range(4):
            b[r][c] = mat[r][c]
    return c2048.board_to_bitboard_c(b)

def advanced_heuristic(bitboard):
    return c2048.advanced_heuristic_c(ctypes.c_uint64(bitboard))

###############################################################################
# 3) Some global config
###############################################################################
STRATEGY = ["Expectimax", "Minimax", "Random", "Rotating", "MCTS", "DQN"]
AI_STRATEGY = STRATEGY[0]
EXPECTIMAX_DEPTH = 3
TIME_LIMIT = 30.0
MINIMAX_DEPTH = 4
MCTS_ROLLOUTS = 200
BEST_GAME_PATH = "Best_Games/best_game.pkl"
CONTINUOUS_STEP_DELAY = 1

DQN_MODEL_PATH = "model.pth"
USE_DQN = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = [0,1,2,3]  # Up,Down,Left,Right

###############################################################################
# 4) Example DQN Model (Optional)
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
    try:
        ckpt = torch.load(DQN_MODEL_PATH, map_location=device)
        dqn_model.load_state_dict(ckpt, strict=False)
        dqn_model.to(device)
        dqn_model.eval()
    except:
        dqn_model = None

def bitboard_to_explist(bb):
    arr=[]
    for _ in range(16):
        arr.append(bb & 0xF)
        bb >>= 4
    return arr

###############################################################################
# 5) AI classes with a 30-second time limit
###############################################################################
@lru_cache(maxsize=6000)
class ExpectimaxAI:
    def __init__(self, depth=EXPECTIMAX_DEPTH):
        self.depth = depth
        self.cache = {}
        # For timing:
        self.start_time = 0
        self.timed_out = False
        self.time_limit = TIME_LIMIT

    def get_move(self, bitboard):
        """
        We start timing here. If we exceed 30 seconds in recursion,
        we stop searching and return whatever best we found so far.
        """
        self.start_time = time.time()
        self.timed_out = False
        best_val = float("-inf")
        best_move = None

        for action in ACTIONS:
            # Check time limit:
            if time.time() - self.start_time > self.time_limit:
                # Timed out; return best so far
                print("Timed out getting move")
                return best_move

            nb, sc, moved = bitboard_move(bitboard, action)
            if not moved:
                continue

            val = sc + self.expectimax_value(nb, self.depth - 1, False)

            if val > best_val:
                best_val = val
                best_move = action

            if self.timed_out:
                # If any deeper call timed out, stop immediately
                break

        return best_move

    def expectimax_value(self, bitboard, depth, is_player):
        # Immediately return if we are over time:
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return 0  # fallback value

        key = (bitboard, depth, is_player)
        if key in self.cache:
            return self.cache[key]

        if depth <= 0 or bitboard_is_game_over(bitboard):
            val = advanced_heuristic(bitboard)
            self.cache[key] = val
            return val

        if is_player:
            best_val = float("-inf")
            anymove = False
            for a in ACTIONS:
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nb, sc, moved = bitboard_move(bitboard, a)
                if moved:
                    anymove = True
                    val = sc + self.expectimax_value(nb, depth - 1, False)
                    if val > best_val:
                        best_val = val
                if self.timed_out:
                    break
            if not anymove:
                val = advanced_heuristic(bitboard)
                self.cache[key] = val
                return val

            self.cache[key] = best_val
            return best_val
        else:
            empty = bitboard_count_empty(bitboard)
            if empty == 0:
                val = self.expectimax_value(bitboard, depth - 1, True)
                self.cache[key] = val
                return val

            val_sum = 0.0
            prob2, prob4 = 0.9, 0.1
            shift = 0
            tmp = bitboard
            for _ in range(16):
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nib = (tmp & 0xF)
                if nib == 0:
                    b2 = (bitboard & ~(0xF << shift)) | (1 << shift)
                    v2 = self.expectimax_value(b2, depth - 1, True)
                    b4 = (bitboard & ~(0xF << shift)) | (2 << shift)
                    v4 = self.expectimax_value(b4, depth - 1, True)
                    val_sum += (prob2 * v2 + prob4 * v4)
                tmp >>= 4
                shift += 4

            # If we timed out halfway, no sense dividing partial sums by empty,
            # but let's still do it so we can return a partial expectation:
            val = val_sum / empty if empty > 0 else 0
            self.cache[key] = val
            return val

class MinimaxAI:
    def __init__(self, depth=MINIMAX_DEPTH):
        self.depth = depth
        self.cache = {}
        # For timing:
        self.start_time = 0
        self.timed_out = False
        self.time_limit = TIME_LIMIT

    def get_move(self, bitboard):
        self.start_time = time.time()
        self.timed_out = False
        best_val = float("-inf")
        best_move = None

        for a in ACTIONS:
            if time.time() - self.start_time > self.time_limit:
                return best_move

            nb, sc, moved = bitboard_move(bitboard, a)
            if moved:
                val = sc + self.minimax_value(nb, self.depth - 1, False)
                if val > best_val:
                    best_val = val
                    best_move = a
            if self.timed_out:
                break

        return best_move

    def minimax_value(self, bitboard, depth, is_player):
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return 0

        key = (bitboard, depth, is_player)
        if key in self.cache:
            return self.cache[key]

        if depth <= 0 or bitboard_is_game_over(bitboard):
            val = advanced_heuristic(bitboard)
            self.cache[key] = val
            return val

        if is_player:
            best_val = float("-inf")
            anymove = False
            for a in ACTIONS:
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nb, sc, moved = bitboard_move(bitboard, a)
                if moved:
                    anymove = True
                    v = sc + self.minimax_value(nb, depth - 1, False)
                    if v > best_val:
                        best_val = v
                if self.timed_out:
                    break
            if not anymove:
                val = advanced_heuristic(bitboard)
                self.cache[key] = val
                return val
            self.cache[key] = best_val
            return best_val
        else:
            empty = bitboard_count_empty(bitboard)
            if empty == 0:
                val = self.minimax_value(bitboard, depth - 1, True)
                self.cache[key] = val
                return val
            best_val = float("inf")
            shift = 0
            tmp = bitboard
            for _ in range(16):
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nib = tmp & 0xF
                if nib == 0:
                    b2 = (bitboard & ~(0xF << shift)) | (1 << shift)
                    v2 = self.minimax_value(b2, depth - 1, True)
                    b4 = (bitboard & ~(0xF << shift)) | (2 << shift)
                    v4 = self.minimax_value(b4, depth - 1, True)
                    worst = v2 if v2 < v4 else v4
                    if worst < best_val:
                        best_val = worst
                tmp >>= 4
                shift += 4
            self.cache[key] = best_val
            return best_val

class RandomAI:
    def get_move(self, bitboard):
        random.shuffle(ACTIONS)
        for a in ACTIONS:
            nb, sc, moved = bitboard_move(bitboard, a)
            if moved:
                return a
        return None

class RotatingAI:
    def __init__(self):
        self.idx=0
    def get_move(self, bitboard):
        a = ACTIONS[self.idx % len(ACTIONS)]
        self.idx += 1
        nb, sc, moved = bitboard_move(bitboard, a)
        if moved:
            return a
        else:
            self.idx += 1
            return ACTIONS[(self.idx-1) % len(ACTIONS)]

class MctsAI:
    def __init__(self, rollouts=MCTS_ROLLOUTS):
        self.rollouts = rollouts
        # For timing:
        self.start_time = 0
        self.timed_out = False
        self.time_limit = TIME_LIMIT

    def get_move(self, bitboard):
        self.start_time = time.time()
        self.timed_out = False

        best_move = None
        best_score = float("-inf")

        # We'll just do a fixed number of trials per action,
        # but also check time before each rollout.
        for a in ACTIONS:
            nb, sc, moved = bitboard_move(bitboard, a)
            if not moved:
                continue

            sum_ = 0
            trials = max(1, self.rollouts // 4)
            for _ in range(trials):
                # Check time
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                sum_ += self.random_rollout(nb)

            avg_ = sum_ / trials if trials > 0 else 0
            total_score = sc + avg_
            if total_score > best_score:
                best_score = total_score
                best_move = a

            if self.timed_out:
                break

        return best_move

    def random_rollout(self, bitboard):
        tmp = bitboard
        for _ in range(50):
            if bitboard_is_game_over(tmp):
                break
            moves = []
            for a in ACTIONS:
                _,_,m = bitboard_move(tmp, a)
                if m:
                    moves.append(a)
            if not moves:
                break
            chosen = random.choice(moves)
            nb, sc, _ = bitboard_move(tmp, chosen)
            tmp = bitboard_spawn(nb)
            # Also check time in each step of rollout
            if time.time() - self.start_time > self.time_limit:
                self.timed_out = True
                break
        return advanced_heuristic(tmp)

class DqnAI:
    def get_move(self, bitboard):
        if not dqn_model:
            return None
        exps = bitboard_to_explist(bitboard)
        st = torch.tensor(exps, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q = dqn_model(st).cpu().numpy()[0]
        order = sorted(range(4), key=lambda i: q[i], reverse=True)
        for a in order:
            nb, _, moved = bitboard_move(bitboard, a)
            if moved:
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
    return RandomAI()

###############################################################################
# 6) The Tkinter GUI and controller (unchanged except where noted)
###############################################################################
class Game2048GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("2048 Advanced - C Accelerated")
        self.master.resizable(False, False)

        init_c_game()

        self.bitboard = 0
        self.score = 0
        self.highest = 0
        self.game_over = False

        self.history=[]
        self.states=[]
        self.moves=[]
        self.best_score=0

        self.ai = create_ai(AI_STRATEGY)

        self.bg_color="#BBADA0"
        self.cell_colors = {
            0:"#CDC1B4", 2:"#EEE4DA", 4:"#EDE0C8", 8:"#F2B179",16:"#F59563",32:"#F67C5F",64:"#F65E3B",
            128:"#EDCF72",256:"#EDCC61",512:"#EDC850",1024:"#EDC53F",2048:"#EDC22E", 4096:"#6BC910", 8192:"#63BE07"
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

        self.master.bind("r", lambda e: self.restart_game())
        self.master.bind("R", lambda e: self.restart_game())
        self.master.bind("<Up>", lambda e: self.do_move(0))
        self.master.bind("<Down>", lambda e: self.do_move(1))
        self.master.bind("<Left>", lambda e: self.do_move(2))
        self.master.bind("<Right>", lambda e: self.do_move(3))
        self.master.bind("i", lambda e: self.ai_single_step())
        self.master.bind("c", lambda e: self.ai_continuous())
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
        self.score += sc
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
        self.score += sc
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
            self.score += sc
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
# 7) BestGameViewer (unchanged)
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

        self.slider = tk.Scale(nav, from_=10, to=1000, orient="horizontal", length=200)
        self.slider.set(self.slider_val)
        self.slider.pack(side="left", padx=5)
        self.slider.bind("<ButtonRelease-1>", self.update_speed)

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
        b = bitboard_to_board(self.states[i])
        for r in range(4):
            for c in range(4):
                val = b[r][c]
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
# 10) Batch Game Runner (unchanged, or you can add the same time checks if needed)
###############################################################################
def create_ai(strategy):
    if strategy.lower() == "expectimax":
        return ExpectimaxAI(depth=EXPECTIMAX_DEPTH)
    elif strategy.lower() == "minimax":
        return MinimaxAI(depth=MINIMAX_DEPTH)
    elif strategy.lower() == "random":
        return RandomAI()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def bitboard_spawn_func(bitboard):
    return c2048.bitboard_spawn(ctypes.c_uint64(bitboard))

def bitboard_get_max_tile_func(bitboard):
    return c2048.bitboard_get_max_tile(ctypes.c_uint64(bitboard))

def bitboard_move_func(bitboard, action):
    out_score = ctypes.c_int(0)
    out_moved = ctypes.c_bool(False)
    new_bitboard = c2048.bitboard_move(ctypes.c_uint64(bitboard),
                                      ctypes.c_int(action),
                                      ctypes.byref(out_score),
                                      ctypes.byref(out_moved))
    return new_bitboard, out_score.value, out_moved.value

def bitboard_is_game_over_func(bitboard):
    return c2048.bitboard_is_game_over(ctypes.c_uint64(bitboard))

def run_batch_games(num_games=1000, strategy=AI_STRATEGY):
    init_c_game()
    ai = create_ai(strategy)
    
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

def run_multiple_batch_strategies(num_games=1000, strategies=STRATEGY):
    stats_all = {}
    for strategy in strategies:
        print(f"\nRunning batch games for strategy: {strategy}")
        stats = run_batch_games(num_games, strategy)
        stats_all[strategy] = stats
    return stats_all

def visualize_stats(stats_all, num_games):
    strategies = list(stats_all.keys())
    # If you run multiple strategies, stats_all is a dict of {strategy: stats_dict}
    if isinstance(stats_all, dict) and all(isinstance(s, dict) for s in stats_all.values()):
        average_scores = [stats_all[s]['average_score'] for s in strategies]
        best_scores = [stats_all[s]['best_score'] for s in strategies]
        plt.figure(figsize=(10, 6))
        plt.bar(strategies, average_scores, color='skyblue')
        plt.xlabel('AI Strategies')
        plt.ylabel('Average Score')
        plt.title(f'Average Scores over {num_games} Games')
        plt.savefig(f'stats/average_scores_{time.strftime("%Y%m%d-%H%M")}.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(strategies, best_scores, color='salmon')
        plt.xlabel('AI Strategies')
        plt.ylabel('Best Score')
        plt.title(f'Best Scores over {num_games} Games')
        plt.savefig(f'stats/best_scores_{time.strftime("%Y%m%d-%H%M")}.png')
        plt.show()

        plt.figure(figsize=(12, 8))
        all_tiles = [2**i for i in range(4, 14)]
        for strategy in strategies:
            histogram = stats_all[strategy]['histogram_highest_tile']
            counts = [histogram.get(tile, 0) for tile in all_tiles]
            plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
        plt.xlabel('Highest Tile')
        plt.ylabel('Frequency')
        plt.title(f'Highest Tile Distribution over {num_games} Games')
        plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"stats/highest_tile_distribution_{time.strftime('%Y%m%d-%H%M')}.png")
        plt.show()
    else:
        # Single strategy stats
        strategy = stats_all["strategy"]
        avg_score = stats_all["average_score"]
        best_score = stats_all["best_score"]
        histogram = stats_all["histogram_highest_tile"]
        # show plt with the data
        plt.figure(figsize=(12, 8))
        all_tiles = [2**i for i in range(4, 14)]
        counts = [histogram.get(tile, 0) for tile in all_tiles]
        plt.bar(range(len(all_tiles)), counts, alpha=0.7, label=strategy)
        plt.xlabel('Highest Tile')
        plt.ylabel('Frequency')
        plt.title(f'Highest Tile Distribution over {num_games} Games')
        plt.xticks(range(len(all_tiles)), all_tiles, rotation=45)
        plt.legend()
        plt.grid(True)
        # if no folder stats create it
        if not os.path.exists("stats"):
            os.makedirs("stats")
        plt.savefig(f"stats/highest_tile_distribution_{time.strftime('%Y%m%d-%H%M')}.png")
        print(f"Strategy: {strategy} | Average: {avg_score} | Best: {best_score} | Highest Tile Distribution: {histogram}")

###############################################################################
# 8) Main
###############################################################################
def main():
    root = tk.Tk()
    app = Game2048GUI(root)
    root.mainloop()

RUN_BATCH = True
NUM_BATCH_GAMES = 1000
RUN_MULTIPLE_BATCH = False

if __name__=="__main__":
    if RUN_BATCH:
        start_time = time.time()
        stats = run_batch_games(NUM_BATCH_GAMES, AI_STRATEGY)
        print(f"Batch games completed in {time.time()-start_time:.2f} seconds.")
        # If you only run one strategy, pass that stats directly:
        visualize_stats(stats, NUM_BATCH_GAMES)
    elif RUN_MULTIPLE_BATCH:
        stats_all = run_multiple_batch_strategies(NUM_BATCH_GAMES, STRATEGY)
        visualize_stats(stats_all, NUM_BATCH_GAMES)
    else:
        main()