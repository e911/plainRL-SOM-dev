# To see the list of available arguments, run: python generate_sample_rb.py --help
# python generate_sample_rb.py --n_timesteps 60000 --game_nm 'CartPole-v1'

import os
import numpy as np
import tyro
import torch
import gymnasium as gym
from args import Args
from datetime import datetime
from tqdm import tqdm
from log_utils import run_time

args = tyro.cli(Args)

env = gym.make(args.game_nm)
print(f"Env: {env.unwrapped.spec.id}")
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

folder_dir = "/home/sr8685/GitHub/RL-SOM/generate_sample_rb/rb_samples"
file_name = "rb_samples_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(args.game_nm) + "_"

total_run_time = run_time()
n = 70 # len(agent.logger.log_folder) + 15
dashes = "-" * n

if args.timesteps_episodes == 1:
    # Sample runs for CartPole-v1 with random play for a fixed number of episodes
    file = file_name + str(args.n_episodes) + "_eps.txt"
    with open(f"{folder_dir}/{file}", "w") as f:
        for e in tqdm(range(args.n_episodes), desc="Sampling episodes"):
            obs, _ = env.reset()
            state = torch.as_tensor(np.array(obs))
            done = False

            while not done:
                # action = np.random.randint(2)
                action = env.action_space.sample()

                state_next, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                action = torch.as_tensor([action], dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                state_next = torch.as_tensor(np.array(state_next))
                done = torch.as_tensor([done], dtype=torch.float)
                
                transition = (state, action, reward, state_next, done)
                f.write(str(transition) + "\n")
                
                state = state_next
else:
    # Sample runs for CartPole-v1 with random play for a fixed number of timesteps
    file = file_name + str(args.n_timesteps) + "_steps.txt"
    with open(f"{folder_dir}/{file}", "w") as f:
        obs, _ = env.reset()
        state = torch.as_tensor(np.array(obs))
        done = False
        for t in tqdm(range(args.n_timesteps), desc="Sampling timesteps"):
            # action = np.random.randint(2)
            action = env.action_space.sample()

            state_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(np.array(state_next))
            done = torch.as_tensor([done], dtype=torch.float)
            
            transition = (state, action, reward, state_next, done)
            f.write(str(transition) + "\n")
            
            state = state_next

            if done:
                obs, _ = env.reset()
                state = torch.as_tensor(np.array(obs))
                done = False

print("\nSamples saved to folder:", folder_dir, "with file name:", file)
total_run_time.block_end()
total_run_time.print_time(n)
print(f"\nExecuted file: {os.path.abspath(__file__)}\n")