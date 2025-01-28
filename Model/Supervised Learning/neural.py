import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F

class Game2048:
    def __init__(self, headless=True):
        self.size = 4
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.headless = headless
        if not headless:
            self.setup_ui()

    def reset_game(self):
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        moved = False
        for i in range(self.size):
            new_row, did_move, merge_score = self._compress_and_merge(self.board[i])
            if did_move:
                moved = True
                self.score += merge_score
                self.board[i] = new_row
        return moved

    def move_right(self):
        return self._move_helper(lambda row: row[::-1], self.move_left)

    def move_up(self):
        return self._move_helper(lambda board: list(zip(*board)), self.move_left)

    def move_down(self):
        return self._move_helper(lambda board: list(zip(*board[::-1]))[::-1], self.move_left)

    def _move_helper(self, transform, move_func):
        board_list = [list(row) for row in transform(self.board)]
        moved = move_func()
        self.board = [list(row) for row in transform(board_list)]
        return moved

    def _compress_and_merge(self, row):
        row = [x for x in row if x != 0]
        new_row = []
        score = 0
        i = 0
        while i < len(row):
            if i + 1 < len(row) and row[i] == row[i + 1]:
                new_row.append(row[i] * 2)
                score += row[i] * 2
                i += 2
            else:
                new_row.append(row[i])
                i += 1
        while len(new_row) < self.size:
            new_row.append(0)
        moved = new_row != row
        return new_row, moved, score

    def execute_move(self, direction):
        if direction == 'Up':
            return self.move_up()
        elif direction == 'Down':
            return self.move_down()
        elif direction == 'Left':
            return self.move_left()
        elif direction == 'Right':
            return self.move_right()
        return False

    def is_game_over(self):
        return not any(self.execute_move(m) for m in ['Up', 'Down', 'Left', 'Right'])

    def get_board_state(self):
        return np.array(self.board).flatten()

    def calculate_corner_building_score(self):
        """Encourage tiles to be in the corners (top-left, top-right, bottom-left, bottom-right)."""
        score = 0
        # Check for corner tiles (top-left, top-right, bottom-left, bottom-right)
        corner_tiles = [
            self.board[0][0],  # Top-left corner
            self.board[0][3],  # Top-right corner
            self.board[3][0],  # Bottom-left corner
            self.board[3][3],  # Bottom-right corner
        ]
        for tile in corner_tiles:
            if tile > 0:
                score += tile  # Reward higher tiles in the corners
        return score

    def calculate_monotonicity_score(self):
        """Encourage a monotonic arrangement (increasing or decreasing)."""
        score = 0

        # Row-wise monotonicity (increasing or decreasing)
        for row in self.board:
            for i in range(3):
                if row[i] > 0 and row[i] == row[i + 1]:
                    score += row[i]  # Reward matching adjacent tiles
                elif row[i] < row[i + 1]:
                    score += row[i + 1] - row[i]  # Reward increasing sequence
                elif row[i] > row[i + 1]:
                    score -= row[i] - row[i + 1]  # Penalize decreasing sequence

        # Column-wise monotonicity (increasing or decreasing)
        for col in zip(*self.board):
            for i in range(3):
                if col[i] > 0 and col[i] == col[i + 1]:
                    score += col[i]
                elif col[i] < col[i + 1]:
                    score += col[i + 1] - col[i]
                elif col[i] > col[i + 1]:
                    score -= col[i] - col[i + 1]

        return score

    def get_heuristics(self):
        corner_building_score = self.calculate_corner_building_score()
        monotonicity_score = self.calculate_monotonicity_score()
        return corner_building_score + monotonicity_score


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(16 + 1, 128)  # Adjusted input size to 17
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 4 possible moves (Up, Down, Left, Right)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



def train_game():
    model = QNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    game = Game2048(headless=True)
    epochs = 50000

    for epoch in range(epochs):
        game.reset_game()
        state = game.get_board_state()
        heuristics = game.get_heuristics()

        # Combine the board state with the heuristic feature
        state_with_heuristics = np.concatenate([state, [heuristics]], axis=0)  # Size: 17
        state_with_heuristics = torch.tensor(state_with_heuristics, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Choose random action (consider implementing epsilon-greedy)
        action = random.choice([0, 1, 2, 3])  # 0=Up, 1=Down, 2=Left, 3=Right

        # Perform action and get new state and reward
        moved = game.execute_move(['Up', 'Down', 'Left', 'Right'][action])
        reward = game.score
        next_state = game.get_board_state()
        next_heuristics = game.get_heuristics()

        # Combine the next board state with the next heuristic feature
        next_state_with_heuristics = np.concatenate([next_state, [next_heuristics]], axis=0)  # Size: 17
        next_state_with_heuristics = torch.tensor(next_state_with_heuristics, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Compute Q-values
        q_values = model(state_with_heuristics)
        next_q_values = model(next_state_with_heuristics)
        target = q_values.clone()
        target[0, action] = reward + 0.9 * next_q_values.max().item()

        # Loss and backpropagation
        loss = criterion(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')



if __name__ == "__main__":
    train_game()
