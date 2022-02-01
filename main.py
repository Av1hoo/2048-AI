import tkinter as tk
import random
import colors as c


class Game(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("2048")

        self.main_grid = tk.Frame(
            self, bg=c.GRID_COLOR, bd=3, width=600, height=600
        )
        self.main_grid.grid(pady=(100, 0))
        self.make_GUI()
        self.start_game()
        
        # controls
        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<Up>", self.up)
        self.master.bind("<Down>", self.down)
        self.master.bind("<z>", self.undo)
        
        # use matrix to keep track of the board
        self.last_matrix = [[0] * 4 for _ in range(4)]
        self.last_last_matrix = [[0] * 4 for _ in range(4)]
        self.last_score = 0

        self.mainloop()

    def make_GUI(self):
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.main_grid,
                    width=150,
                    height=150,
                    bg=c.EMPTY_CELL_COLOR
                )
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg=c.EMPTY_CELL_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)

        score_frame = tk.Frame(self)
        score_frame.place(relx=0.5, y=45, anchor="center")
        tk.Label(
            score_frame,
            text="Score",
            font=c.SCORE_LABEL_FONT
        ).grid(row=0)

        undo_frame = tk.Frame(self)
        undo_frame.place(relx=0.5, x=230, y=55)
        tk.Label(undo_frame,
                 text="",
                 font=c.UNDO_FONT
                 ).grid(row=1)

        self.undo_label = tk.Label(undo_frame, text="Undo", font=c.UNDO_FONT)
        self.undo_label.grid(row=1)

        self.score_label = tk.Label(score_frame, text="0", font=c.SCORE_LABEL_FONT)
        self.score_label.grid(row=1)



    def start_game(self):
        self.matrix = [[0] * 4 for _ in range(4)]
        
        # start with random tiles
        row = random.randint(0, 3)
        column = random.randint(0, 3)
        self.matrix[row][column] = 2
        self.cells[row][column]["frame"].configure(bg=c.CELL_COLORS[2])
        self.cells[row][column]["number"].configure(
            bg=c.CELL_COLORS[2],
            fg=c.CELL_NUMBER_COLORS[2],
            font=c.CELL_NUMBER_FONTS[2],
            text=str(2)
        )
        while self.matrix[row][column] != 0:
            row = random.randint(0, 3)
            column = random.randint(0, 3)
        self.matrix[row][column] = 2
        self.cells[row][column]["frame"].configure(bg=c.CELL_COLORS[2])
        self.cells[row][column]["number"].configure(
            bg=c.CELL_COLORS[2],
            fg=c.CELL_NUMBER_COLORS[2],
            font=c.CELL_NUMBER_FONTS[2],
            text=str(2)
        )
        self.score = 0

    def stack(self):
        new_matrix = [[0] * 4 for _ in range(4)]
        for i in range(4):
            fill_position = 0
            for j in range(4):
                if self.matrix[i][j] != 0:
                    new_matrix[i][fill_position] = self.matrix[i][j]
                    fill_position += 1
        self.matrix = new_matrix

    def combine(self):
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] != 0 and self.matrix[i][j] == self.matrix[i][j + 1]:
                    self.matrix[i][j] *= 2
                    self.matrix[i][j + 1] = 0
                    self.score += self.matrix[i][j]

    def reverse(self):
        new_matrix = []
        for i in range(4):
            new_matrix.append([])
            for j in range(4):
                new_matrix[i].append(self.matrix[i][3 - j])
        self.matrix = new_matrix

    def transpose(self):
        new_matrix = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_matrix[i][j] = self.matrix[j][i]
        self.matrix = new_matrix

    def add_new_tile(self):
        self.undo_label.configure(text="Undo")
        row = random.randint(0, 3)
        column = random.randint(0, 3)
        while self.matrix[row][column] != 0:
            row = random.randint(0, 3)
            column = random.randint(0, 3)
        self.matrix[row][column] = random.choice([2, 2, 2, 2, 4])


    def undo(self, event):
        self.undo_label.configure(text="")
        print(self.game_over_label.pack_info())
        try:
            self.game_over_label.destroy()
            #self.game_over_frame.grid_forget()
        except AttributeError:
            pass
        # undo the score
        if self.last_matrix != [[0] * 4 for _ in range(4)]:
            self.matrix = self.last_matrix
            self.score = self.last_score
        self.update_GUI()

    def update_GUI(self):
        self.check_lose()
        for i in range(4):
            for j in range(4):
                cell_value = self.matrix[i][j]
                if cell_value == 0:
                    self.cells[i][j]["frame"].configure(bg=c.EMPTY_CELL_COLOR)
                    self.cells[i][j]["number"].configure(bg=c.EMPTY_CELL_COLOR, text="")
                else:
                    self.cells[i][j]["frame"].configure(bg=c.CELL_COLORS[cell_value])
                    self.cells[i][j]["number"].configure(
                        bg=c.CELL_COLORS[cell_value],
                        fg=c.CELL_NUMBER_COLORS[cell_value],
                        font=c.CELL_NUMBER_FONTS[cell_value],
                        text=str(cell_value)
                    )
        self.score_label.configure(text=self.score)
        self.update_idletasks()

    def left(self, event):
        if self.last_matrix != self.matrix:
            self.last_matrix = self.matrix
        self.last_score = self.score
        self.stack()
        self.combine()
        self.stack()
        if self.matrix != self.last_matrix:
            self.last_last_matrix = self.last_matrix
            self.add_new_tile()
        else:
            self.last_matrix = self.last_last_matrix
        self.update_GUI()
        self.check_lose()

    def right(self, event):
        if self.last_matrix != self.matrix:
            self.last_matrix = self.matrix
        self.last_score = self.score
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        if self.matrix != self.last_matrix:
            self.last_last_matrix = self.last_matrix
            self.add_new_tile()
        else:
            self.last_matrix = self.last_last_matrix
        self.update_GUI()
        self.check_lose()

    def up(self, event):
        if self.last_matrix != self.matrix:
            self.last_matrix = self.matrix
        self.last_score = self.score
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        if self.matrix != self.last_matrix:
            self.last_last_matrix = self.last_matrix
            self.add_new_tile()
        else:
            self.last_matrix = self.last_last_matrix
        self.update_GUI()
        self.check_lose()

    def down(self, event):
        if self.last_matrix != self.matrix:
            self.last_matrix = self.matrix
        self.last_score = self.score
        self.transpose()
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.transpose()
        if self.matrix != self.last_matrix:
            self.last_last_matrix = self.last_matrix
            self.add_new_tile()
        else:
            self.last_matrix = self.last_last_matrix
        self.update_GUI()
        self.check_lose()

    def horz(self):
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] == self.matrix[i][j + 1]:
                    return True
        return False

    def vert(self):
        for i in range(3):
            for j in range(4):
                if self.matrix[i][j] == self.matrix[i + 1][j]:
                    return True
        return False


    def check_lose(self):
        if not any(0 in row for row in self.matrix) and not self.horz() and not self.vert():
            self.game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            self.game_over_frame.grid()
            self.game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
            self.game_over_label = tk.Label(
                self.game_over_frame,
                text="Game Over !",
                font=c.GAME_OVER_FONT,
                bg=c.LOSER_BG,
                fg=c.GAME_OVER_FONT_COLOR
            )
            self.game_over_label.pack()




def main():
    Game()


if __name__ == "__main__":
    main()
