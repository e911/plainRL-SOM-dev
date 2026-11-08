from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks import DQN
from SOMBuffer import SOMBuffer, WeightedSOMBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, outs=64, args=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.eps = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.outs = outs
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.loss_fn = nn.MSELoss()

    def train(self, env: Any, args: Any, rb: Any | None = None) -> tuple[float, int]:
        """Train for ``args.n_episodes`` episodes with one replay buffer.

        Args:
            env: Gymnasium environment.
            args: Experiment arguments (``n_episodes``, ``batch_size``, ``update_target_freq``).
            rb: Replay buffer with ``add``/``sample``; ``None`` uses ``WeightedSOMBuffer()``.

        Returns:
            ``(max_score, max_ep)``: the best training-episode return and its episode.
        """
        if rb is None:
            rb = WeightedSOMBuffer()
        # rb = SOMBuffer()

        is_train = False
        total_steps = 0
        max_score = 0
        max_ep = 0

        for e in range(args.n_episodes):
            is_train = total_steps > 150
            state, _ = env.reset()
            ep_reward = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                total_steps += 1
                
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Only a real ending is a terminal state; the time limit (truncated) is not.
                done = terminated
                ep_reward += reward

                # rb.add(state, action, reward, next_state, done, total_steps)
                rb.add(state, action, reward, next_state, done, e)
                state = next_state

                if is_train:
                    states, actions, rewards, next_states, dones = rb.sample(args.batch_size)
                    loss = self.update(states, actions, rewards, next_states, dones)

                if total_steps % args.update_target_freq == 0:
                    self.update_target_network()

            self.decay_epsilon()

            if e % 10 == 0:
                if is_train:
                    print(f"Episode: {e} | Score: {ep_reward:.1f} | Total Steps: {total_steps} | Epsilon: {self.eps:.4f} | Loss: {loss:.4f}")
                else:
                    print(f"Episode: {e} | Score: {ep_reward:.1f} | Total Steps: {total_steps} | Epsilon: {self.eps:.4f}")

            if ep_reward > max_score:
                max_score = ep_reward
                max_ep = e

        env.close()
        return max_score, max_ep

    def train_steps(
        self,
        env: Any,
        eval_env: Any,
        rb: Any,
        n_steps: int,
        checkpoints: Sequence[int],
        batch_size: int,
        update_target_freq: int,
        eval_episodes: int = 10,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Train for a fixed number of environment steps and evaluate greedily at checkpoints.

        The loop is :meth:`train`'s (learning starts after step 150, target
        sync every ``update_target_freq`` steps, epsilon decays once per
        episode), run for ``n_steps`` steps instead of a number of episodes.
        Only real endings are stored as terminal.

        Args:
            env: Training environment.
            eval_env: Separate environment for the greedy evaluations.
            rb: Replay buffer with ``add``/``sample``.
            n_steps: Environment steps to train for.
            checkpoints: Steps after which to evaluate.
            batch_size: Minibatch size.
            update_target_freq: Steps between target-network syncs.
            eval_episodes: Greedy episodes per evaluation.
            seed: Seed of the first training reset and of the evaluation episodes.

        Returns:
            Dict with ``eval`` (checkpoint -> mean greedy return),
            ``train_returns`` (every finished training episode's return) and
            ``episodes``.
        """
        wanted = set(checkpoints)
        evals: dict[int, float] = {}
        train_returns: list[float] = []
        e, total_steps = 0, 0
        state, _ = env.reset(seed=seed)
        ep_reward, is_train = 0.0, False
        while total_steps < n_steps:
            total_steps += 1
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            rb.add(state, action, reward, next_state, terminated, e)
            state = next_state
            if is_train:
                self.update(*rb.sample(batch_size))
            if total_steps % update_target_freq == 0:
                self.update_target_network()
            if terminated or truncated:
                train_returns.append(ep_reward)
                self.decay_epsilon()
                e += 1
                is_train = total_steps > 150
                state, _ = env.reset()
                ep_reward = 0.0
            if total_steps in wanted:
                evals[total_steps] = self.evaluate(eval_env, eval_episodes, seed=seed * 1000 + total_steps)
        return {"eval": evals, "train_returns": train_returns, "episodes": e}

    def evaluate(self, env: Any, n_episodes: int, seed: int) -> float:
        """Mean return of the greedy policy.

        Args:
            env: Evaluation environment.
            n_episodes: Episodes to run.
            seed: Seed of the first reset (later resets continue its stream).

        Returns:
            Mean undiscounted return over the episodes.
        """
        total = 0.0
        for i in range(n_episodes):
            state, _ = env.reset(seed=seed if i == 0 else None)
            done = False
            while not done:
                with torch.no_grad():
                    q = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                state, reward, terminated, truncated, _ = env.step(int(torch.argmax(q).item()))
                total += reward
                done = terminated or truncated
        return total / n_episodes

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, states, actions, rewards, next_states, dones):
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q values for current states
        q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        # Compute target Q values for next states
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + (self.gamma * max_next_q_values * (1 - dones_tensor))

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)