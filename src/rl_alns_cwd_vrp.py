#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL-guided ALNS for the Quzhou CWD benchmark."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import BenchmarkInstance, clone_solution, evaluate_solution, seed_everything
from alns_cwd_vrp import (
    ACTION_PAIRS,
    build_initial_solution,
    perturb,
    small_instance_exact_polish,
    state_features,
)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def train_step(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay_buffer: deque,
    batch_size: int,
    gamma: float,
) -> float:
    batch = random.sample(replay_buffer, batch_size)
    states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
    rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    next_states = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32)
    dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

    current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        target = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(current_q, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def solve_with_rl_alns(
    instance: BenchmarkInstance,
    iterations: int = 550,
    seed: int = 0,
    exact_polish_threshold: int = 12,
    exact_polish_time_limit: int = 180,
) -> Dict:
    seed_everything(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    start_time = time.time()
    current = build_initial_solution(instance, seed=seed)
    current_metrics = evaluate_solution(instance, current)
    best = clone_solution(current)
    best_metrics = dict(current_metrics)

    q_net = QNetwork(state_dim=8, action_dim=len(ACTION_PAIRS))
    target_net = QNetwork(state_dim=8, action_dim=len(ACTION_PAIRS))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    replay_buffer: deque = deque(maxlen=2500)
    epsilon = 0.45
    epsilon_floor = 0.05
    gamma = 0.96
    batch_size = 48
    target_sync = 40
    accepted = 0
    temperature = max(current_metrics["objective"] * 0.04, 10.0)
    history: List[Dict] = []
    losses: List[float] = []
    action_counts = np.zeros(len(ACTION_PAIRS), dtype=int)

    for iteration in range(iterations):
        state = state_features(instance, current_metrics, best_metrics, iteration, iterations)
        with torch.no_grad():
            q_values = q_net(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy()
        if random.random() < epsilon:
            candidate_actions = random.sample(range(len(ACTION_PAIRS)), k=min(3, len(ACTION_PAIRS)))
        else:
            candidate_actions = list(np.argsort(q_values)[-3:][::-1])

        trial_records = []
        for action_index in candidate_actions:
            candidate, action_info = perturb(instance, current, action_index, random)
            candidate_metrics = evaluate_solution(instance, candidate)
            trial_records.append((candidate_metrics["objective"], action_index, candidate, candidate_metrics, action_info))
        trial_records.sort(key=lambda item: item[0])
        _, action_index, candidate, candidate_metrics, action_info = trial_records[0]
        delta = candidate_metrics["objective"] - current_metrics["objective"]
        accept = False

        normalized_gain = (current_metrics["objective"] - candidate_metrics["objective"]) / max(
            best_metrics["objective"], 1.0
        )
        reward = normalized_gain
        if candidate_metrics["objective"] + 1e-9 < best_metrics["objective"]:
            reward += 0.12
        if not candidate_metrics["feasible"]:
            reward -= 0.08

        if delta <= 0:
            accept = True
        else:
            threshold = math.exp(-delta / max(temperature, 1e-6))
            if random.random() < threshold:
                accept = True
                reward -= 0.02

        next_metrics = current_metrics
        if accept:
            current = candidate
            current_metrics = candidate_metrics
            next_metrics = current_metrics
            accepted += 1
            if candidate_metrics["objective"] + 1e-9 < best_metrics["objective"]:
                best = clone_solution(candidate)
                best_metrics = dict(candidate_metrics)
        else:
            reward -= 0.01

        next_state = state_features(instance, next_metrics, best_metrics, iteration + 1, iterations)
        replay_buffer.append((state, action_index, reward, next_state, 0.0))
        action_counts[action_index] += 1

        if len(replay_buffer) >= batch_size:
            losses.append(train_step(q_net, target_net, optimizer, replay_buffer, batch_size, gamma))

        if (iteration + 1) % target_sync == 0:
            target_net.load_state_dict(q_net.state_dict())

        history.append(
            {
                "iteration": iteration,
                "current_objective": current_metrics["objective"],
                "best_objective": best_metrics["objective"],
                "temperature": temperature,
                "epsilon": epsilon,
                "accepted": accept,
                "action": ACTION_PAIRS[action_index],
                "reward": reward,
                "loss": losses[-1] if losses else None,
            }
        )
        epsilon = max(epsilon_floor, epsilon * 0.9945)
        temperature *= 0.995

    best, best_metrics, polish_info = small_instance_exact_polish(
        instance,
        best,
        best_metrics,
        customer_threshold=exact_polish_threshold,
        time_limit=exact_polish_time_limit,
    )

    elapsed = time.time() - start_time
    return {
        "instance": instance.name,
        "algorithm": "RL-ALNS",
        "seed": seed,
        "elapsed_seconds": elapsed,
        "accepted_moves": accepted,
        "acceptance_rate": accepted / max(iterations, 1),
        "best_metrics": best_metrics,
        "best_solution": best,
        "history": history,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "action_counts": {
            f"{destroy}+{repair}": int(count)
            for (destroy, repair), count in zip(ACTION_PAIRS, action_counts)
        },
        "postprocess": polish_info,
    }


def main() -> None:
    from pathlib import Path

    instance = BenchmarkInstance.load(Path(__file__).resolve().parents[1] / "instances" / "QZ-real-1.json")
    result = solve_with_rl_alns(instance, iterations=350, seed=20260311)
    metrics = result["best_metrics"]
    print(f"RL-ALNS finished for {instance.name}")
    print(f"  Objective: {metrics['objective']:.2f}")
    print(f"  Open facilities: {metrics['open_facilities']}")
    print(f"  Routes: {metrics['routes']}")
    print(f"  Distance: {metrics['travel_distance']:.2f}")
    print(f"  Feasible: {metrics['feasible']}")


if __name__ == "__main__":
    main()
