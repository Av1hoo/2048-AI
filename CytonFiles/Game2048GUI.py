# Game2048GUI.py
import os
import pickle
import tkinter as tk

from CytonFiles.Game_viewer import BestGameViewer
from CytonFiles.Game_Logic import Game2048 

# Configure how fast continuous AI steps happen
CONTINUOUS_STEP_DELAY = 1

class Game2048GUI:
    def __init__(self, master, ai_strategy="Expectimax", DEPTH=3):
        self.master = master
        self.master.title("2048 Advanced - C Accelerated")
        self.master.resizable(False, False)

        # Create the game logic object
        self.game = Game2048(ai_strategy="Expectimax", best_game_path="Best_Games/best_game.pkl", depth=DEPTH)
        self.game.reset_game()

        # Track best score in this session
        self.session_best_score = 0

        # GUI colors & styling
        self.bg_color = "#BBADA0"
        self.cell_colors = {
            0:"#CDC1B4",     2:"#EEE4DA",    4:"#EDE0C8",    8:"#F2B179",
            16:"#F59563",    32:"#F67C5F",   64:"#F65E3B",   128:"#EDCF72",
            256:"#EDCC61",   512:"#EDC850", 1024:"#EDC53F", 2048:"#EDC22E",
            4096:"#6BC910", 8192:"#63BE07"
        }

        # Build UI
        self.make_ui()
        self.update_ui()

    def make_ui(self):
        """
        Build the TK widgets: score labels, board, buttons, etc.
        """
        self.top_frame = tk.Frame(self.master, bg=self.bg_color)
        self.top_frame.pack(pady=5)

        self.lbl_score = tk.Label(self.top_frame, text="Score: 0",
                                  font=("Helvetica",14,"bold"), bg=self.bg_color)
        self.lbl_score.pack(side="left", padx=10)

        self.lbl_best = tk.Label(self.top_frame, text="Best: 0",
                                 font=("Helvetica",14,"bold"), bg=self.bg_color)
        self.lbl_best.pack(side="left", padx=10)

        self.lbl_highest = tk.Label(self.top_frame, text="Highest: 0",
                                    font=("Helvetica",14,"bold"), bg=self.bg_color)
        self.lbl_highest.pack(side="left", padx=10)

        self.lbl_over = tk.Label(self.top_frame, text="",
                                 font=("Helvetica",16,"bold"), fg="red", bg=self.bg_color)
        self.lbl_over.pack(side="left", padx=10)

        # Main board
        self.main_frame = tk.Frame(self.master,bg=self.bg_color)
        self.main_frame.pack(padx=10,pady=10)

        self.tiles = []
        for r in range(4):
            row_tiles = []
            for c in range(4):
                lbl = tk.Label(
                    self.main_frame,
                    text="",
                    bg=self.cell_colors[0],
                    font=("Helvetica",20,"bold"),
                    width=4, height=2
                )
                lbl.grid(row=r, column=c, padx=5, pady=5)
                row_tiles.append(lbl)
            self.tiles.append(row_tiles)

        # Bottom Frame with Buttons
        self.bottom_frame = tk.Frame(self.master,bg=self.bg_color)
        self.bottom_frame.pack()

        btn_restart = tk.Button(
            self.bottom_frame, text="Restart (R)",
            command=self.restart_game, bg="#8f7a66",
            fg="white", font=("Helvetica",12,"bold")
        )
        btn_restart.pack(side="left", padx=5)

        btn_view = tk.Button(
            self.bottom_frame, text="View Best (V)",
            command=self.view_best_game, bg="#8f7a66",
            fg="white", font=("Helvetica",12,"bold")
        )
        btn_view.pack(side="left", padx=5)

        # Bind keyboard inputs
        self.master.bind("r", lambda e: self.restart_game())
        self.master.bind("R", lambda e: self.restart_game())
        self.master.bind("<Up>",    lambda e: self.do_move(0))
        self.master.bind("<Down>",  lambda e: self.do_move(1))
        self.master.bind("<Left>",  lambda e: self.do_move(2))
        self.master.bind("<Right>", lambda e: self.do_move(3))
        self.master.bind("i", lambda e: self.ai_single_step())
        self.master.bind("c", lambda e: self.ai_continuous())
        self.master.bind("v", lambda e: self.view_best_game())

    def update_ui(self):
        """
        Refresh the board tiles, score labels, etc. to reflect the current game state.
        """
        board_data = self.game.get_board_2d()
        for r in range(4):
            for c in range(4):
                val = board_data[r][c]
                color = self.cell_colors.get(val, "#3C3A32")
                text_val = str(val) if val > 0 else ""
                self.tiles[r][c].config(text=text_val, bg=color)

        # Update best score if needed
        current_score = self.game.get_score()
        if current_score > self.session_best_score:
            self.session_best_score = current_score

        self.lbl_score.config(text=f"Score: {current_score}")
        self.lbl_best.config(text=f"Best: {self.session_best_score}")
        self.lbl_highest.config(text=f"Highest: {self.game.get_highest()}")
        self.master.update_idletasks()

    def restart_game(self):
        self.game.reset_game()
        self.lbl_over.config(text="")
        self.update_ui()

    def do_move(self, action):
        if self.game.is_game_over():
            return
        moved = self.game.do_move(action)
        if moved:
            self.update_ui()
        if self.game.is_game_over():
            self.finish_game()

    def ai_single_step(self):
        if self.game.is_game_over():
            return
        moved = self.game.ai_single_step()
        if moved:
            self.update_ui()
        if self.game.is_game_over():
            self.finish_game()

    def ai_continuous(self):
        def step():
            if self.game.is_game_over():
                return
            moved = self.game.ai_single_step()
            if not moved:
                self.finish_game()
                return

            self.update_ui()
            if self.game.is_game_over():
                self.finish_game()
            else:
                # Schedule next AI move
                self.master.after(CONTINUOUS_STEP_DELAY, step)

        step()

    def finish_game(self):
        self.lbl_over.config(text="Game Over!")

    def view_best_game(self):
        data = self.game.view_best_game()
        if data is None:
            print("No best game found.")
            return

        states  = data["states"]
        moves   = data["moves"]
        sc      = data["score"]
        hi      = data["highest"]

        viewer = BestGameViewer(self.master, states, moves, sc, hi)
        viewer.show_frame(0)
