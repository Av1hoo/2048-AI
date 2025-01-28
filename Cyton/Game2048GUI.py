import os
import pickle
import tkinter as tk
from game_viewer import BestGameViewer
from C_funcs import *
import ai_factory

# Global config
STRATEGY = ["Expectimax", "Minimax", "Random", "DQN"]
AI_STRATEGY = STRATEGY[0]
BEST_GAME_PATH = "Best_Games/best_game.pkl"
CONTINUOUS_STEP_DELAY = 1

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

        self.ai = ai_factory.AI_Factory.create_ai(AI_STRATEGY)

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
