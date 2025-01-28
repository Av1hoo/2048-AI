import gym
import numpy as np
import time
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from gym_2048 import Game2048Env

def main():
    # Create environment
    env = Game2048Env(size=4, max_steps=2000)
    
    # # SB3 expects either a single env or a vec-env.
    # # We'll wrap it in a DummyVecEnv for convenience:
    # vec_env = DummyVecEnv([lambda: env])

    def make_env():
        return Game2048Env(size=4, max_steps=2000)

    num_envs = 8  # Number of parallel environments
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Choose an algorithm: PPO or DQN. Let's pick PPO first:
    # model = PPO("MlpPolicy", vec_env, verbose=1)

    # Or choose DQN:
    # model = DQN("MlpPolicy", vec_env, verbose=1,
    #             learning_rate=1e-4,
    #             buffer_size=5_000_000,
    #             learning_starts=10_000,
    #             batch_size=64,
    #             tau=0.99,
    #             target_update_interval=1_000,
    #             train_freq=4,
    #             gamma=0.99)
    
    # Instead of DQN, use PPO
    model = PPO("MlpPolicy", vec_env, verbose=1,
                learning_rate=1e-4,
                n_steps=2048,  # Number of steps to run for each environment per update
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device="cuda")


    # Train for some timesteps. 
    total_timesteps = 1_000_000  # You may need 1M+ steps for consistent 2048 tile.

    start_time = time.time()  # Record start time

    # Train the model with callback to display ETA
    for steps_done in range(0, total_timesteps, 10_000):  # Update every 10k steps
        model.learn(total_timesteps=10_000, reset_num_timesteps=False)
        
        # Time elapsed
        elapsed_time = time.time() - start_time
        avg_time_per_step = elapsed_time / (steps_done + 10_000)  # Avoid div by zero
        remaining_time = (total_timesteps - (steps_done + 10_000)) * avg_time_per_step

        # Convert time format
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

        print(f"Progress: {steps_done + 10_000}/{total_timesteps} | "
              f"Elapsed: {elapsed_time_str} | ETA: {remaining_time_str}")

    # Save the trained model
    model.save("model_2048_rl")
    torch.save(model.policy.state_dict(), "model.pth")

    # Test the model after training:
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Use the trained model to predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    env.render()
    print(f"Test game finished, total_reward={total_reward}")

if __name__ == "__main__":
    main()
