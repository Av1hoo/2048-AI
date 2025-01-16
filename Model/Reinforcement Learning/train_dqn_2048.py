import math
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

##############################################################################
# 1) Define the 2048 Game Environment (No GUI)
##############################################################################
class GameEnv2048:
    """
    A headless 2048 environment that supports:
      - reset()
      - step(action)
      - get_state()
      - is_game_over()

    Action mapping:
      0 = Up
      1 = Down
      2 = Left
      3 = Right
    """
    def __init__(self, size=4):
        self.size = size
        self.board = None
        self.score = 0
        self.highest_tile = 0
        self.reset()

    def reset(self):
        self.board = [[0]*self.size for _ in range(self.size)]
        self.score = 0
        self.highest_tile = 0
        self._spawn_new_tile()
        self._spawn_new_tile()
        return self.get_state()

    def get_state(self):
        """
        Return the current board as a 16-dim vector (exponents in [0..15]).
        (You can return other representations if you prefer.)
        """
        exps = []
        for row in self.board:
            for val in row:
                if val == 0:
                    exps.append(0)
                else:
                    exp = int(math.log2(val))
                    exps.append(min(exp, 15))  # clamp to 15
        return np.array(exps, dtype=np.float32)

    def step(self, action):
        """
        action: 0=Up, 1=Down, 2=Left, 3=Right
        Returns: (next_state, reward, done, info)
        """
        prev_score = self.score

        moved = False
        if action == 0:
            moved = self._move_up()
        elif action == 1:
            moved = self._move_down()
        elif action == 2:
            moved = self._move_left()
        elif action == 3:
            moved = self._move_right()

        reward = 0
        done = False

        # If the board changed, we gained merges => score might have increased
        if moved:
            # Reward is the score increment
            reward = self.score - prev_score
            self._spawn_new_tile()

        # Check if game is over
        if self._is_game_over():
            done = True

        next_state = self.get_state()
        return next_state, reward, done, {}

    def _spawn_new_tile(self):
        """Spawn a 2 or 4 tile randomly in an empty cell."""
        empty_cells = [(r, c)
                       for r in range(self.size)
                       for c in range(self.size)
                       if self.board[r][c] == 0]
        if not empty_cells:
            return
        r, c = random.choice(empty_cells)
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    def _is_game_over(self):
        """Check if no more moves are possible or 2048 tile is reached."""
        # If 2048 is reached, consider it 'done' (you can define differently if you like)
        for row in self.board:
            if 2048 in row:
                return True

        # If any cell is empty => not game over
        for row in self.board:
            if 0 in row:
                return False

        # Check horizontal merges
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r][c] == self.board[r][c+1]:
                    return False

        # Check vertical merges
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r][c] == self.board[r+1][c]:
                    return False

        return True

    def _move_left(self):
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

    def _move_right(self):
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

    def _move_up(self):
        moved = False
        total_merge_score = 0
        for col in range(self.size):
            col_data = [self.board[r][col] for r in range(self.size)]
            new_col, did_move, merge_score = self._compress_and_merge(col_data)
            if did_move:
                moved = True
                total_merge_score += merge_score
            for r in range(self.size):
                self.board[r][col] = new_col[r]
        if moved:
            self.score += total_merge_score
        return moved

    def _move_down(self):
        moved = False
        total_merge_score = 0
        for col in range(self.size):
            col_data = [self.board[r][col] for r in range(self.size)][::-1]
            new_col, did_move, merge_score = self._compress_and_merge(col_data)
            new_col.reverse()
            if did_move:
                moved = True
                total_merge_score += merge_score
            for r in range(self.size):
                self.board[r][col] = new_col[r]
        if moved:
            self.score += total_merge_score
        return moved

    def _compress_and_merge(self, tiles):
        """
        1) Remove zeros
        2) Merge adjacent equal tiles
        3) Pad with zeros
        4) Return (new_tiles, did_move, merge_score)
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

        # pad with zeros
        while len(merged) < len(tiles):
            merged.append(0)

        did_move = (merged != original)
        self.highest_tile = max(self.highest_tile, max(merged))
        return merged, did_move, merge_score


##############################################################################
# 2) Define the DQN Model & Replay Buffer
##############################################################################
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


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.memory)


##############################################################################
# 3) Training Loop for DQN
##############################################################################
def train_dqn(env,
              num_episodes=5000,
              gamma=0.99,
              batch_size=64,
              lr=1e-3,
              epsilon_start=1.0,
              epsilon_end=0.02,
              epsilon_decay=0.9995,
              target_update_freq=1000):
    """
    Trains a DQN on the 2048 environment.
    Returns the trained policy_net.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = 16
    action_dim = 4

    policy_net = DQNModel(state_dim=state_dim, action_dim=action_dim).to(device)
    target_net = DQNModel(state_dim=state_dim, action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    epsilon = epsilon_start
    steps_done = 0

    for episode in range(num_episodes):
        state = env.reset()  # shape: [16]
        done = False
        episode_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)  # shape: [1,4]
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            # Decrease epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            steps_done += 1

            # If we have enough samples, train
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)

                states_t = torch.FloatTensor(states_b).to(device)       # [batch,16]
                actions_t = torch.LongTensor(actions_b).to(device)      # [batch]
                rewards_t = torch.FloatTensor(rewards_b).to(device)     # [batch]
                next_states_t = torch.FloatTensor(next_states_b).to(device)
                dones_t = torch.FloatTensor(dones_b).to(device)

                # Current Q
                q_values = policy_net(states_t)                  # [batch,4]
                # Gather the Q-values for the chosen actions
                q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [batch]

                # Next Q (using target_net)
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0]  # [batch]

                # Q-target
                target_q = rewards_t + gamma * max_next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target_net periodically
                if steps_done % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.1f}, Epsilon: {epsilon:.3f}")

    return policy_net


##############################################################################
# 4) Main: Train & Save the Model to model.pth
##############################################################################
def main():
    env = GameEnv2048(size=4)
    trained_model = train_dqn(env, num_episodes=200)

    # Save the trained model
    torch.save(trained_model.state_dict(), "model.pth")
    print("Training finished, model saved to model.pth")


if __name__ == "__main__":
    main()
