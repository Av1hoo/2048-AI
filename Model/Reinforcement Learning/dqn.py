import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import math
from copy import deepcopy
from game import Game2048

# Assuming you've implemented a Game2048Env class or similar for 2048 game environment
class Game2048Env:
    def __init__(self, headless=True):
        self.game = Game2048(headless=headless)
        self.steps = 0

    def reset(self):
        self.game.reset_game()
        self.steps = 0
        return self._get_state()

    def step(self, action):
        pre_board = deepcopy(self.game.board)
        pre_score = self.game.score

        if action == 0:
            moved = self.game.move_up()
        elif action == 1:
            moved = self.game.move_down()
        elif action == 2:
            moved = self.game.move_left()
        else:  # action == 3
            moved = self.game.move_right()

        self.steps += 1

        if moved:
            self.game.history.append((pre_board, pre_score))
            self.game.spawn_new_tile()
            new_state = self._get_state()
            reward = self.game.score - pre_score
            done = self.game.is_game_over()
            return new_state, reward, done, {}
        else:
            # If move didn't change the board, it's like a no-op in terms of state change
            return self._get_state(), 0, False, {}

    def _get_state(self):
        return np.array([tile if tile != 0 else 0 for row in self.game.board for tile in row])

    def render(self):
        if not self.game.headless:
            self.game.update_ui()

# Define DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128, action_dim=4):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Game2048Env(headless=True)
memory = ReplayMemory(100000)
model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(batch.state, device=device, dtype=torch.float)
    action_batch = torch.tensor(batch.action, device=device, dtype=torch.long)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], device=device, dtype=torch.float)
    
    state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

num_episodes = 1000
steps_done = 0

for i_episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

        next_state, reward, done, _ = env.step(action.item())
        memory.push(state, action, next_state, reward, done)
        state = next_state
        total_reward += reward

        loss = optimize_model()
        
        if done:
            if max(env.game.board) == 2048:
                print(f"Episode {i_episode} reached 2048!")
            print(f"Episode {i_episode} finished after {env.steps} steps with score {total_reward}")
            break

    if i_episode % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())

torch.save(model.state_dict(), 'model.pth')