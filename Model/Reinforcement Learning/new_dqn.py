import numpy as np
import torch
import random
import os
from collections import deque
import pickle
import time
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Import your Agent and QNetwork ---
from agent import Agent
from model import QNetwork

class Improved2048Env:
    def __init__(self):
        self.action_size = 4
        self.done = False
        self.score = 0
        self.state_size = 4 * 4  
        self.board = np.zeros((4,4), dtype=int) 

    def reset(self):
        self.board = np.zeros((4,4), dtype=int)
        self.done = False
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self._get_state()

    def _get_state(self):
        return self.board.flatten().astype(np.float32)

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

    def step(self, action):
        old_board = self.board.copy()
        if action == 0:  # Up
            self.board = self._move_up(self.board)
        elif action == 1:  # Down
            self.board = np.flipud(self._move_up(np.flipud(self.board)))
        elif action == 2:  # Left
            self.board = self._move_left(self.board)
        else:  # Right
            self.board = np.fliplr(self._move_left(np.fliplr(self.board)))
        
        reward = self._calculate_reward(old_board)
        self.score += reward
        self.done = not np.any(self.board == 0)  # Game over if no more moves possible
        if not self.done:
            self.add_random_tile()
        next_state = self._get_state()
        return next_state, reward, self.done

    def _move_up(self, board):
        for col in range(4):
            board[:, col] = self._merge_line(board[:, col])
        return board

    def _move_left(self, board):
        for row in range(4):
            board[row, :] = self._merge_line(board[row, :])
        return board

    def _merge_line(self, line):
        line = line[line != 0]
        i = 0
        while i < len(line) - 1:
            if line[i] == line[i + 1]:
                line[i] *= 2
                line = np.delete(line, i + 1)
                i += 1  # Skip next tile as it's merged
            i += 1
        return np.pad(line, (0, 4 - len(line)), 'constant')

    def _calculate_reward(self, old_board):
        new_board = self.board
        score_change = np.sum(new_board) - np.sum(old_board)
        max_tile = np.max(new_board)
        if max_tile >= 2048:  # Bonus for achieving high tiles
            score_change += 1000
        return score_change

    def current_state(self):
        return self._get_state()

def dqn_train(env, agent, n_episodes=50000, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    scores = []
    eps = eps_start
    optimizer = torch.optim.Adam(agent.qnetwork_local.parameters(), lr=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1000, verbose=True)

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            if random.random() < eps:
                action = random.randint(0, env.action_size - 1)
                action_values = np.zeros((1, env.action_size))  # Define action_values for random actions
            else:
                action_values = agent.act(state)
                action = np.argmax(action_values)

            next_state, reward, done = env.step(action)
            # Now action_values is defined in both cases
            agent.step(state, action, reward, next_state, done, abs(reward - action_values[0, action]), 1.0)

            state = next_state
            score += reward

            if len(agent.memory) > agent.batch_size:
                agent.learn(learn_iterations=1, mode='weighted_error', gamma=0.99, weight=None)

            if i_episode % 10 == 0:
                agent.soft_update(agent.qnetwork_local, agent.qnetwork_target)

        eps = max(eps_end, eps * eps_decay)
        scores.append(score)
        
        if i_episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {i_episode}/{n_episodes}, Avg Score: {avg_score:.2f}, Eps: {eps:.3f}")
            scheduler.step(avg_score)  # Adjust learning rate based on performance

        if i_episode % 1000 == 0:
            eval_score = evaluate_agent(env, agent, num_games=100)
            print(f"Evaluation Score at Episode {i_episode}: {eval_score:.2f}")

    return scores

def evaluate_agent(env, agent, num_games=100):
    scores = []
    for _ in range(num_games):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = np.argmax(agent.act(state, eps=0))
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state
        scores.append(score)
    return np.mean(scores)

def main():
    env = Improved2048Env()
    agent = Agent(
        state_size=16,
        action_size=4,
        seed=42,
        fc1_units=512,
        fc2_units=512,
        fc3_units=512,
        buffer_size=100000,
        batch_size=1024,
        lr=5e-4,
        use_expected_rewards=True,
        predict_steps=3,
        gamma=0.99,
        tau=0.001
    )

    scores = dqn_train(env, agent, n_episodes=50000)

    # Save the model after training
    torch.save(agent.qnetwork_local.state_dict(), "model.pth")
    print("Training complete. Model weights saved to model.pth.")

if __name__ == "__main__":
    main()