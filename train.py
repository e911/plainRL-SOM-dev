import os
import gymnasium as gym
from args import Args
import tyro
from agent import DQNAgent
from log_utils import run_time

args = tyro.cli(Args)

env = gym.make(args.game_nm)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

total_run_time = run_time()
n = 70 # len(agent.logger.log_folder) + 15
dashes = "-" * n

print(f"\n{dashes}\nEnv: {args.game_nm}\nTotal Episodes: {args.n_episodes}\n{dashes}\n")
agent = DQNAgent(state_dim, action_dim, args=args)

max_score, max_ep = agent.train(env, args)

print(f"\n{dashes}\n\nMax Score: {max_score:.1f} | Episode: {max_ep}")
total_run_time.block_end()
total_run_time.print_time(n)
print(f"\nExecuted file: {os.path.abspath(__file__)}\n");