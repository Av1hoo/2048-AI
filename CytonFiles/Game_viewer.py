import tkinter as tk
import pickle
from tkinter import filedialog

class BestGameViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Best Game Viewer")
        self.resizable(False, False)
        self.geometry("470x350")
        self.stop = True
        
        self.bg_color = "#BBADA0"
        self.cell_colors = {
            0: "#CDC1B4", 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179",
            16: "#F59563", 32: "#F67C5F", 64: "#F65E3B",
            128: "#EDCF72", 256: "#EDCC61", 512: "#EDC850",
            1024: "#EDC53F", 2048: "#EDC22E"
        }
        
        # Initial state: show Load Game button
        self.init_frame = tk.Frame(self, bg=self.bg_color)
        self.init_frame.pack(expand=True, fill='both')
        
        load_btn = tk.Button(
            self.init_frame, 
            text="Load Game", 
            command=self.load_game, 
            font=("Helvetica", 14)
        )
        load_btn.pack(pady=130)
    
    def bitboard_to_board(self, bitboard):
        b = [[0]*4 for _ in range(4)]
        # the bitboard is given as powers of 2, convert to actual values
        for i in range(16):
            r = i // 4
            c = i % 4
            val = bitboard & 0xf
            b[r][c] = 0 if val == 0 else 2**val
            bitboard >>= 4
        pyboard = []
        for r in range(4):
            row = []
            for c in range(4):
                row.append(b[r][c])
            pyboard.append(row)
        return pyboard
        
    def load_game(self):
        path = filedialog.askopenfilename(
            title="Select Best Game File",
            filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
        )
        if not path:
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.states = data["states"]
        self.moves = data["moves"]
        self.final_score = data["score"]
        self.final_highest = data["highest"]
        self.idx = 0
        self.slider_val = 30
        
        # Remove initial frame
        self.init_frame.destroy()
        
        # Set up viewer
        self.setup_viewer()
        self.show_frame(0)
    
    def setup_viewer(self):
        # Info label
        self.lbl_info = tk.Label(self, text="", font=("Helvetica", 12, "bold"))
        self.lbl_info.pack()
        
        # Grid frame
        self.grid_frame = tk.Frame(self, bg=self.bg_color)
        self.grid_frame.pack()
        
        self.tiles = []
        for r in range(4):
            row_tiles = []
            for c in range(4):
                lb = tk.Label(
                    self.grid_frame, 
                    text="", 
                    bg=self.cell_colors[0],
                    font=("Helvetica", 16, "bold"), 
                    width=4, 
                    height=2
                )
                lb.grid(row=r, column=c, padx=3, pady=3)
                row_tiles.append(lb)
            self.tiles.append(row_tiles)
        
        # Navigation frame
        nav = tk.Frame(self)
        nav.pack(pady=10)
        
        tk.Button(nav, text="Prev", command=self.prev_state).pack(side="left", padx=5)
        tk.Button(nav, text="Next", command=self.next_state).pack(side="left", padx=5)
        tk.Button(nav, text="Play", command=self.click_play).pack(side="left", padx=5)
        
        self.slider = tk.Scale(
            nav, 
            from_=10, 
            to=1000, 
            orient="horizontal", 
            length=100,
            label="Speed (ms):"
        )
        self.slider.set(self.slider_val)
        self.slider.pack(side="left", padx=5)
        self.slider.bind("<ButtonRelease-1>", self.update_speed)

        self.slider_frames = tk.Scale(
            nav, 
            from_=0, 
            to=len(self.states)-1, 
            orient="horizontal", 
            length=100,
            label="Frame:"
        )
        self.slider_frames.pack(side="left", padx=5)
        self.slider_frames.set(self.idx)
        self.slider_frames.bind("<ButtonRelease-1>", self.update_frame)
        
        tk.Button(nav, text="Load Another", command=self.load_another_game).pack(side="left", padx=5)
    
    def load_another_game(self):
        path = filedialog.askopenfilename(
            title="Select Best Game File",
            filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
        )
        if not path:
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.states = data["states"]
        self.moves = data["moves"]
        self.final_score = data["score"]
        self.final_highest = data["highest"]
        self.idx = 0
        self.show_frame(0)
    
    def update_speed(self, event):
        self.slider_val = self.slider.get()

    def update_frame(self, event):
        self.idx = self.slider_frames.get()
        self.show_frame(self.idx)

    def on_state_slider_move(self, val):
        # Convert the slider value to integer index
        try:
            i = int(float(val))
            self.show_frame(i)
        except ValueError:
            pass  # Ignore invalid values
    
    def show_frame(self, i):
        if i < 0:
            i = 0
        if i >= len(self.states):
            i = len(self.states) - 1
        self.idx = i
        b = self.bitboard_to_board(self.states[i])
        for r in range(4):
            for c in range(4):
                val = b[r][c]
                clr = self.cell_colors.get(val, "#3C3A32")
                txt = str(val) if val > 0 else ""
                self.tiles[r][c].config(text=txt, bg=clr)
        
        msg = f"Move {i}/{len(self.states)-1}"
        if i > 0 and i-1 < len(self.moves):
            direction_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
            mv = self.moves[i-1]
            msg += f" | Action: {direction_map.get(mv, 'Unknown')}"
        msg += f" | Score={self.final_score}, Highest={self.final_highest}"
        self.lbl_info.config(text=msg)
    
    def next_state(self):
        self.show_frame(self.idx + 1)
    
    def prev_state(self):
        self.show_frame(self.idx - 1)

    def click_play(self):
        if self.stop:
            self.stop = False
            self.auto_play()
        else:
            self.stop = True
    
    def auto_play(self):
        if self.idx >= len(self.states)-1 or self.stop:
            return
        self.next_state()
        self.after(self.slider_val, self.auto_play)

def main():
    app = BestGameViewer()
    app.mainloop()

if __name__ == "__main__":
    main()
