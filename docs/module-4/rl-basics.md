---
title: "Module 4: RL Basics"
description: "Fundamentals of Reinforcement Learning and Markov Decision Processes"
sidebar_position: 2
slug: /module-4/rl-basics
keywords: [reinforcement learning, MDP, value functions, robotics, AI]
---

# RL Basics: Foundations of Robot Learning

## Introduction

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. In robotics, this approach is particularly powerful as robots can learn complex behaviors through trial and error, adapting to their environment without requiring explicit programming for every situation.

## Markov Decision Processes (MDP)

### Definition and Components

A Markov Decision Process is a mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision maker. An MDP is defined by the tuple (S, A, P, R, γ):

- **S**: Set of states (the possible configurations of the robot/environment)
- **A**: Set of actions (the possible actions the robot can take)
- **P**: Transition probabilities P(s'|s, a) - probability of moving to state s' from state s with action a
- **R**: Reward function R(s, a) or R(s, a, s') - immediate reward for taking action a in state s
- **γ**: Discount factor (0 ≤ γ < 1) - how much future rewards are valued

### The Markov Property

The Markov property states that the future state depends only on the current state and action, not on the sequence of events that preceded it:

```
P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, s_1, a_1, ..., s_t, a_t)
```

### Value Functions

Value functions estimate how good it is for an agent to be in a given state. There are two main types:

#### State-Value Function
```
V^π(s) = E_π[G_t | S_t = s]
```
Where G_t is the total discounted reward from time t onwards.

#### Action-Value Function (Q-Function)
```
Q^π(s, a) = E_π[G_t | S_t = s, A_t = a]
```

## Basic RL Algorithms

### Value Iteration

```python
import numpy as np

def value_iteration(P, R, gamma=0.9, theta=1e-6):
    """
    Value iteration algorithm for solving MDPs

    Args:
        P: Transition probability matrix P[s][a] = [(prob, next_state, reward)]
        R: Reward matrix
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        Optimal value function and policy
    """
    n_states = len(P)
    n_actions = len(P[0])

    # Initialize value function
    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            Q_s = np.zeros(n_actions)

            # Calculate Q-value for each action
            for a in range(n_actions):
                q = 0
                for prob, next_state, reward in P[s][a]:
                    q += prob * (reward + gamma * V[next_state])
                Q_s[a] = q

            # Update value function with best action
            V[s] = np.max(Q_s)
            delta = max(delta, abs(v - V[s]))

        # Check for convergence
        if delta < theta:
            break

    # Extract optimal policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        Q_s = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward in P[s][a]:
                Q_s[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(Q_s)

    return V, policy
```

### Q-Learning

Q-Learning is a model-free RL algorithm that can learn optimal policies without knowing the environment dynamics:

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Q-Learning agent for discrete state-action spaces
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))

    def act(self, state):
        """
        Choose action using epsilon-greedy policy
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(range(self.n_actions))
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example: Simple grid world environment
class GridWorld:
    def __init__(self, width=5, height=5, goal=(4, 4)):
        self.width = width
        self.height = height
        self.goal = goal
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self._get_state()

    def _get_state(self):
        return self.agent_pos[0] * self.width + self.agent_pos[1]

    def _is_valid_pos(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def step(self, action):
        """
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        x, y = self.agent_pos
        if action == 0:  # up
            new_pos = (x-1, y)
        elif action == 1:  # right
            new_pos = (x, y+1)
        elif action == 2:  # down
            new_pos = (x+1, y)
        elif action == 3:  # left
            new_pos = (x, y-1)
        else:
            new_pos = (x, y)  # invalid action

        # Check if new position is valid
        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos

        # Calculate reward
        if self.agent_pos == self.goal:
            reward = 100  # Goal reached
            done = True
        else:
            reward = -1  # Time penalty
            done = False

        return self._get_state(), reward, done, {}

    def render(self):
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        for row in grid:
            print(' '.join(row))
        print()

# Training loop example
def train_grid_world():
    env = GridWorld()
    agent = QLearningAgent(n_states=25, n_actions=4)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while steps < 100:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    print("Training completed!")

    # Test the trained agent
    print("\nTesting trained agent:")
    state = env.reset()
    env.render()

    for step in range(20):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        env.render()

        if done:
            print("Goal reached!")
            break

# Uncomment to run training
# train_grid_world()
```

## Policy vs Value-Based Methods

### Value-Based Methods
- Learn an optimal value function (like Q-learning)
- Derive policy from value function (e.g., greedy with respect to value)
- Examples: Q-learning, SARSA, Deep Q-Networks (DQN)

### Policy-Based Methods
- Directly learn the optimal policy
- Parameterize policy as π(a|s, θ) and optimize parameters θ
- Better for continuous action spaces

## Robot-Specific Considerations

### Continuous Action Spaces

Many robotics applications involve continuous action spaces (e.g., joint velocities, motor torques). For these, we use:

- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
```

## Exploration vs Exploitation

A key challenge in RL is balancing exploration (trying new actions) and exploitation (using known good actions). Common strategies include:

- **ε-greedy**: With probability ε, take random action; otherwise, take best-known action
- **Upper Confidence Bound (UCB)**: Choose actions based on uncertainty
- **Thompson Sampling**: Sample from action-value distribution

## Summary

This section covered the fundamental concepts of Reinforcement Learning, including Markov Decision Processes, value functions, and basic algorithms like Q-learning. These form the foundation for applying RL to robotics, where agents can learn complex behaviors through interaction with their environment.

Continue with [Advanced Techniques](./advanced-techniques) to explore Deep RL methods.