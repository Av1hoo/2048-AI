import math
import random
import gym
import numpy as np
from gym import spaces

class Game2048Env(gym.Env):
    """
    A custom 2048 environment for Gym.
    Observations: 16-dim array of tile exponents, each in [0..15].
    Actions: 4 discrete moves (0=Up, 1=Down, 2=Left, 3=Right).
    Reward: the *score increment* obtained each step (you can modify).
    Episode ends when no more moves are possible or we reach some max-steps.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, size=4, max_steps=2000):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # Up,Down,Left,Right
        # Each cell is an exponent in [0..15], so observation is shape (16,)
        # We'll store them as int, but gym requires a float or Box, so we use Box [0..15].
        self.observation_space = spaces.Box(
            low=0, high=15, shape=(size*size,), dtype=np.int32
        )
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.step_count = 0
        # spawn 2 tiles
        self._spawn_tile()
        self._spawn_tile()
        return self._get_obs()

    def step(self, action):
        """
        action: 0=Up, 1=Down, 2=Left, 3=Right
        returns: (obs, reward, done, info)
        """
        old_score = self.score
        moved = self._move(action)
        reward = 0
        if moved:
            # Score increment from merges
            reward = self.score - old_score
            self._spawn_tile()

        self.step_count += 1
        done = (not self._can_move()) or (self.step_count >= self.max_steps)
        obs = self._get_obs()
        info = {}
        return obs, float(reward), done, info

    def render(self, mode="human"):
        # Simple text render of the board
        exp_board = (2**self.board) * (self.board > 0)  # convert exponents to actual tile values
        print(exp_board)
        print(f"Score: {self.score}")

    def _get_obs(self):
        # Flatten the board to shape (16,)
        return self.board.flatten()

    def _spawn_tile(self):
        empty = [(r, c) for r in range(self.size)
                         for c in range(self.size)
                         if self.board[r, c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        # 90% chance of 2 => exponent=1, 10% => exponent=2
        # or you can treat exponent=0 for empty, exponent=1 => value=2, etc.
        # We'll store exponent=1 => tile=2, exponent=2 => tile=4
        self.board[r, c] = 2 if random.random() < 0.1 else 1

    def _can_move(self):
        # If any cell is empty => can move
        if np.any(self.board == 0):
            return True
        # Check merges horizontally
        for r in range(self.size):
            for c in range(self.size-1):
                if self.board[r,c] == self.board[r,c+1]:
                    return True
        # Check merges vertically
        for c in range(self.size):
            for r in range(self.size-1):
                if self.board[r,c] == self.board[r+1,c]:
                    return True
        return False

    def _move(self, action):
        """
        action: 0=Up,1=Down,2=Left,3=Right
        returns True if board changed
        """
        moved = False
        if action == 0:
            # Up
            for col in range(self.size):
                col_vals = self.board[:,col]
                merged, gained = self._merge_line(col_vals)
                if not np.array_equal(col_vals, merged):
                    moved = True
                self.board[:,col] = merged
                self.score += gained
        elif action == 1:
            # Down
            for col in range(self.size):
                col_vals = self.board[:,col][::-1]
                merged, gained = self._merge_line(col_vals)
                merged = merged[::-1]
                original = self.board[:,col]
                if not np.array_equal(original, merged):
                    moved = True
                self.board[:,col] = merged
                self.score += gained
        elif action == 2:
            # Left
            for row in range(self.size):
                row_vals = self.board[row,:]
                merged, gained = self._merge_line(row_vals)
                if not np.array_equal(row_vals, merged):
                    moved = True
                self.board[row,:] = merged
                self.score += gained
        else:
            # Right
            for row in range(self.size):
                row_vals = self.board[row,:][::-1]
                merged, gained = self._merge_line(row_vals)
                merged = merged[::-1]
                original = self.board[row,:]
                if not np.array_equal(original, merged):
                    moved = True
                self.board[row,:] = merged
                self.score += gained

        return moved

    def _merge_line(self, line):
        """
        Given a 1D array of exponents, merges them 2048-style.
        Returns (merged_line, score_gained).
        """
        # Filter out zeros
        filtered = [x for x in line if x != 0]
        merged = []
        score_gained = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i < len(filtered)-1 and filtered[i] == filtered[i+1]:
                val = filtered[i] + 1  # merging exponents => exponent+1
                merged.append(val)
                # Score gained = 2^val
                score_gained += (2 ** val)
                skip = True
            else:
                merged.append(filtered[i])
        # Pad with zeros
        while len(merged) < len(line):
            merged.append(0)
        return np.array(merged), score_gained
