import os
from typing import Tuple
from dataclasses import dataclass, field

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    notes: str = "RL-SOM"
    """notes regarding the run"""
    wandb_project_name: str = "RL-SOM"
    """the wandb's project name"""
    wandb_entity: str = "sr8685-rit"
    """the entity (team) of wandb's project"""
    
    update_target_freq: int = 500
    """the frequency to update the target network"""
    game_nm: str = 'CartPole-v1'
    """the name of the game/environment"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    n_episodes: int = 500
    """total episodes to run the experiments"""
    n_timesteps: int = 10000
    """total timesteps to run the experiments"""
    timesteps_episodes: int = 0
    """whether to use timesteps or episodes for running the experiments; 0 for timesteps, 1 for episodes"""
    max_episode_steps: int = 500
    """the maximum episode steps for the environment"""
    grid_size: Tuple[int, int] = (10, 10)
    """the size of the SOM grid"""
    lr: float = 1e-3
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    eps_start: float = 1.0
    """the initial epsilon for epsilon-greedy action selection"""
    eps_end: float = 0.01
    """the final epsilon for epsilon-greedy action selection"""
    eps_decay: float = 0.995
    """the decay rate for epsilon-greedy action selection"""
    
