import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

###############################################################################
# 1) Minimal 2048 Environment
###############################################################################
class Game2048Env:
    """
    A stripped-down 2048 environment without a GUI.
    - Observation: a 16-element array of exponents [0..15], with 0 for empty.
    - Action: 0=Up, 1=Down, 2=Left, 3=Right.
    - step(action) -> (next_state, reward, done).
    """
    def __init__(self, size=4, seed=None):
        self.size = size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.board = [[0]*self.size for _ in range(self.size)]
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()
        return self.get_observation()

    def get_observation(self):
        """
        Convert the board’s tile values (0,2,4,8,...) into exponents with max=15.
        E.g., tile=0 -> exponent=0, tile=2 -> exponent=1, tile=4->2, 8->3, ...
        Larger than 2^15 is clamped to exponent=15.
        Returns a list of length 16.
        """
        obs = []
        for row in self.board:
            for val in row:
                if val == 0:
                    obs.append(0)
                else:
                    exp = int(math.log2(val))
                    exp = min(exp, 15)  # clamp
                    obs.append(exp)
        return obs

    def spawn_tile(self):
        """Spawn a 2 or 4 in an empty cell."""
        empty = [(r,c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
        if not empty:
            return
        r,c = random.choice(empty)
        self.board[r][c] = 4 if random.random()<0.1 else 2

    def is_game_over(self):
        if any(2048 in row for row in self.board):
            return True
        # If there's any empty cell, not over
        if any(0 in row for row in self.board):
            return False
        # Check merges horizontally
        for i in range(self.size):
            for j in range(self.size-1):
                if self.board[i][j] == self.board[i][j+1]:
                    return False
        # Check merges vertically
        for j in range(self.size):
            for i in range(self.size-1):
                if self.board[i][j] == self.board[i+1][j]:
                    return False
        return True

    def step(self, action):
        """
        action: 0=Up, 1=Down, 2=Left, 3=Right
        Returns: (next_observation, reward, done)
        """
        prev_board = deepcopy(self.board)
        prev_score = self.score

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        reward = self.score - prev_score
        done = False
        if moved:
            self.spawn_tile()
            done = self.is_game_over()

        return self.get_observation(), reward, done

    def move_left(self):
        moved = False
        total_merge_score = 0
        for row_idx in range(self.size):
            row = self.board[row_idx]
            compressed, changed, merge_score = self.compress_and_merge(row)
            if changed:
                moved = True
                total_merge_score += merge_score
            self.board[row_idx] = compressed
        self.score += total_merge_score
        return moved

    def move_right(self):
        moved = False
        total_merge_score = 0
        for row_idx in range(self.size):
            row = self.board[row_idx][::-1]
            compressed, changed, merge_score = self.compress_and_merge(row)
            compressed.reverse()
            if changed:
                moved = True
                total_merge_score += merge_score
            self.board[row_idx] = compressed
        self.score += total_merge_score
        return moved

    def move_up(self):
        moved = False
        total_merge_score = 0
        for col in range(self.size):
            col_vals = [self.board[r][col] for r in range(self.size)]
            compressed, changed, merge_score = self.compress_and_merge(col_vals)
            if changed:
                moved = True
                total_merge_score += merge_score
            for r in range(self.size):
                self.board[r][col] = compressed[r]
        self.score += total_merge_score
        return moved

    def move_down(self):
        moved = False
        total_merge_score = 0
        for col in range(self.size):
            col_vals = [self.board[r][col] for r in range(self.size)][::-1]
            compressed, changed, merge_score = self.compress_and_merge(col_vals)
            compressed.reverse()
            if changed:
                moved = True
                total_merge_score += merge_score
            for r in range(self.size):
                self.board[r][col] = compressed[r]
        self.score += total_merge_score
        return moved

    @staticmethod
    def compress_and_merge(tiles):
        """
        1) Remove zeros
        2) Merge adjacent equals
        3) Pad with zeros
        Return (new_tiles, changed, merge_score)
        """
        original = list(tiles)
        filtered = [t for t in tiles if t!=0]
        merged = []
        merge_score = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip=False
                continue
            if i<len(filtered)-1 and filtered[i] == filtered[i+1]:
                val = filtered[i]*2
                merged.append(val)
                merge_score += val
                skip = True
            else:
                merged.append(filtered[i])
        while len(merged)<len(tiles):
            merged.append(0)

        changed = (merged != original)
        return merged, changed, merge_score


###############################################################################
# 2) Simple MCTS Implementation for 2048
###############################################################################
class MCTSNode:
    """
    Each node holds:
     - state: the environment observation (16 exponents)
     - parent: parent node
     - visits: how many times node was visited
     - q_value: total value from rollouts
     - children: dict(action -> MCTSNode)
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.q_value = 0.0
        self.children = {}  # action -> child MCTSNode
        self.untried_actions = [0,1,2,3]  # up, down, left, right

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.41):
        """
        Select the child with the highest UCB (Upper Confidence Bound).
        UCB = Q / visits + c_param * sqrt( 2 * ln(parent_visits) / visits ).
        """
        best = None
        best_ucb = -1e9
        for action, child in self.children.items():
            ucb = (child.q_value / (child.visits + 1e-7)) + \
                  c_param * math.sqrt(2*math.log(self.visits+1) / (child.visits+1e-7))
            if ucb > best_ucb:
                best_ucb = ucb
                best = child
        return best

    def expand(self, env):
        """
        Take one untried action, apply it to the current state, create a new child node.
        Remove that action from untried_actions.
        """
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)

        # Clone env to apply the step
        cloned_env = clone_env(env)
        cloned_env.board = to_board(self.state)
        cloned_env.score = 0  # we only track incremental changes
        # Step
        obs_next, reward, done = cloned_env.step(action)

        child_node = MCTSNode(state=obs_next, parent=self)
        self.children[action] = child_node
        return child_node, reward, done

    def update(self, reward):
        """
        Update this node's Q-value and visit count.
        """
        self.q_value += reward
        self.visits += 1


def clone_env(env):
    """Create a new Game2048Env with the same board setup (no random state duplication)."""
    cloned = Game2048Env(size=env.size)
    cloned.board = deepcopy(env.board)
    cloned.score = env.score
    return cloned

def to_board(state):
    """
    Convert a state array of exponents [0..15] -> 4x4 board.
    """
    board = []
    idx = 0
    for _ in range(4):
        row = []
        for _ in range(4):
            exp = state[idx]
            row.append(0 if exp==0 else (2**exp))
            idx += 1
        board.append(row)
    return board


def default_policy_rollout(env, depth_limit=20):
    """
    Rollout from the current env using a random policy up to depth_limit or game-over.
    Return total reward from this rollout.
    """
    cloned = clone_env(env)
    total_reward = 0
    for _ in range(depth_limit):
        if cloned.is_game_over():
            break
        action = random.choice([0,1,2,3])  # random action
        _, reward, done = cloned.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def tree_policy(node, env):
    """
    Selection + Expansion:
    - While the node is fully expanded, descend to best_child.
    - If not fully expanded, expand one child.
    """
    current = node
    current_env = clone_env(env)
    current_env.board = to_board(current.state)

    while True:
        if not current.is_fully_expanded():
            child, reward, done = current.expand(current_env)
            return child, reward, done, current_env
        else:
            # descend
            current = current.best_child()
            # step environment to match that child's state (we need the reward for rollout)
            # to do that, find which action leads to that child's state
            action_found = None
            for act, child_node in current.parent.children.items():
                if child_node is current:
                    action_found = act
                    break
            # step environment
            _, reward, done = current_env.step(action_found)
            if done:
                return current, reward, done, current_env

def mcts_search(root_state, env, n_iterations=100):
    """
    Perform n_iterations of MCTS from the root_state in the given env.
    Return the best action found from root.
    """
    root_node = MCTSNode(root_state)

    for _ in range(n_iterations):
        # 1) Clone environment for the search
        search_env = clone_env(env)
        search_env.board = to_board(root_state)

        # 2) Selection + Expansion
        node, immediate_reward, done, search_env = tree_policy(root_node, search_env)

        # 3) If not done, rollout from here
        if not done:
            rollout_reward = default_policy_rollout(search_env)
        else:
            rollout_reward = 0

        # total reward from node perspective
        total_reward = immediate_reward + rollout_reward

        # 4) Backpropagate
        backprop_node = node
        while backprop_node is not None:
            backprop_node.update(total_reward)
            backprop_node = backprop_node.parent

    # Once done, pick the child with highest average Q-value
    # We interpret that child’s action as the best move.
    best_action, best_avg_q = None, -1e9
    for action, child in root_node.children.items():
        avg_q = child.q_value / (child.visits + 1e-7)
        if avg_q > best_avg_q:
            best_avg_q = avg_q
            best_action = action

    return best_action


###############################################################################
# 3) Simple Neural Network Model
###############################################################################
class Net(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)  # shape [batch_size, 4]


###############################################################################
# 4) Training Loop: Use MCTS to pick moves, store (state, action) pairs, train
###############################################################################
def train_mcts(num_episodes=50, mcts_iters=100):
    """
    1. Create environment
    2. For each episode, run MCTS at each step to pick best_action
    3. Store (state, action) into a buffer
    4. Periodically train on the buffer (supervised => classify action from state)
    5. Save model.pth
    """
    env = Game2048Env()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # We'll keep a small replay buffer of (state, action)
    buffer_states = []
    buffer_actions = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            # 1) Use MCTS to pick an action
            action = mcts_search(root_state=obs, env=env, n_iterations=mcts_iters)

            # 2) Step env
            next_obs, reward, done = env.step(action)

            # 3) Store in buffer
            buffer_states.append(obs)
            buffer_actions.append(action)

            # Move on
            obs = next_obs

        # Simple printing
        print(f"Episode {episode+1}/{num_episodes}, Score={env.score}")

        # 4) Periodically train (e.g. every episode)
        if len(buffer_states) > 32:  # small check
            # Sample from buffer
            batch_size = 32
            idxs = np.random.choice(len(buffer_states), batch_size, replace=False)
            batch_states = [buffer_states[i] for i in idxs]
            batch_actions = [buffer_actions[i] for i in idxs]

            # Convert to tensors
            s_t = torch.tensor(batch_states, dtype=torch.float32)
            a_t = torch.tensor(batch_actions, dtype=torch.long)

            # Forward
            logits = model(s_t)
            loss = loss_fn(logits, a_t)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f" Train loss={loss.item():.4f}")

    # Done training, save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved model to model.pth")


###############################################################################
# 5) Run if main
###############################################################################
if __name__ == "__main__":
    train_mcts(num_episodes=20, mcts_iters=50)
