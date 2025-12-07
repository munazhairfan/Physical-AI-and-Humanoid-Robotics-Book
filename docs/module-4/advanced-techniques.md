---
title: "Module 4: Advanced RL Techniques"
description: "Deep Reinforcement Learning, policy gradients, and actor-critic methods for robotics"
sidebar_position: 3
slug: /module-4/advanced-techniques
keywords: [deep RL, policy gradients, actor-critic, robotics, neural networks]
---

# Advanced RL Techniques for Robotics

## Introduction

While basic RL algorithms provide a solid foundation, real-world robotics applications often require more sophisticated approaches. Advanced RL techniques leverage deep neural networks to handle high-dimensional state and action spaces, enabling robots to learn complex behaviors in continuous environments.

## Deep Q-Networks (DQN)

Deep Q-Networks extend Q-learning to handle high-dimensional state spaces using neural networks as function approximators.

### Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main network
        self.q_network = DQN(state_size, action_size).to(self.device)
        # Target network for stable training
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Copy weights to target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example training loop
def train_dqn():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.n)

    episodes = 1000
    target_update_freq = 100

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle newer gym versions
        total_reward = 0

        for t in range(500):
            action = agent.act(state)
            result = env.step(action)

            # Handle different gym versions
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

            if done:
                break

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
```

### Key DQN Innovations

1. **Experience Replay**: Store experiences in a replay buffer and sample randomly to break correlation between consecutive samples
2. **Target Network**: Use a separate target network that is updated periodically to provide stable targets
3. **ε-greedy Exploration**: Balance exploration vs exploitation

## Policy Gradient Methods

Policy gradient methods directly optimize the policy parameters to maximize expected return.

### REINFORCE Algorithm

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # Store log probability for gradient computation
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        # Compute discounted rewards
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Compute loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Reset storage
        self.log_probs = []
        self.rewards = []

# Example usage for CartPole
def train_reinforce():
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(state_dim=env.observation_space.shape[0],
                          action_dim=env.action_space.n)

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0

        while True:
            action = agent.act(state)
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            agent.store_reward(reward)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_policy()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Actor-Critic Methods

Actor-critic methods combine value-based and policy-based approaches, using an actor to select actions and a critic to evaluate them.

### Advantage Actor-Critic (A2C)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)

        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, state_value = self.model(state)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action).item(), state_value.item()

    def update(self, state, action, reward, next_state, done, gamma=0.99):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)

        action_probs, state_value = self.model(state)
        _, next_state_value = self.model(next_state)

        # Compute advantage
        target = reward + gamma * next_state_value * (1 - done)
        advantage = target - state_value

        # Compute losses
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        total_loss = actor_loss + 0.5 * critic_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

# Example training
def train_a2c():
    env = gym.make('CartPole-v1')
    agent = A2CAgent(state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.n)

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0

        while True:
            action, log_prob, state_value = agent.act(state)
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Deep Deterministic Policy Gradient (DDPG)

DDPG is designed for continuous action spaces, which are common in robotics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.memory = deque(maxlen=100000)

    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        noise = np.random.normal(0, noise, size=action.shape)
        action = action + noise
        return np.clip(action, -self.max_action, self.max_action)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=100, gamma=0.99, tau=0.005):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        state = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        action = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        reward = torch.FloatTensor([e[2] for e in batch]).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        done = torch.BoolTensor([e[4] for e in batch]).to(self.device).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (gamma * target_q * (1 - done))

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Example for continuous control
def train_ddpg():
    # Example using a simple environment
    class SimpleEnv:
        def __init__(self):
            self.state = np.random.random(4)
            self.action_space = type('ActionSpace', (), {'shape': [1]})()

        def reset(self):
            self.state = np.random.random(4)
            return self.state

        def step(self, action):
            reward = -np.sum((self.state - action)**2)  # Simple quadratic reward
            self.state = self.state + 0.1 * (np.random.random(4) - 0.5)  # Add noise
            done = False
            return self.state, reward, done, {}

    env = SimpleEnv()
    agent = DDPGAgent(state_dim=4, action_dim=1, max_action=1.0)

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(100):
            action = agent.act(state)
            if isinstance(action, np.ndarray) and action.shape == (1,):
                action = action[0]
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm that maximizes both expected return and entropy for better exploration.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(SACActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))

        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()

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

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SACActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.alpha = alpha
        self.memory = deque(maxlen=100000)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=256, gamma=0.99, tau=0.005):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        state = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        action = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        reward = torch.FloatTensor([e[2] for e in batch]).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        done = torch.BoolTensor([e[4] for e in batch]).to(self.device).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (gamma * target_q * (1 - done))

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_action, log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Example usage
def train_sac():
    # Simple example with placeholder environment
    class SimpleEnv:
        def __init__(self):
            self.state = np.random.random(3)
            self.action_space = type('ActionSpace', (), {'shape': [2]})()

        def reset(self):
            self.state = np.random.random(3)
            return self.state

        def step(self, action):
            reward = -np.sum((self.state - action[:len(self.state)])**2)
            self.state = self.state + 0.1 * (np.random.random(3) - 0.5)
            done = False
            return self.state, reward, done, {}

    env = SimpleEnv()
    agent = SACAgent(state_dim=3, action_dim=2, max_action=1.0)

    episodes = 500

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Robotics-Specific Applications

### Robot Control with RL

```python
import numpy as np
import torch
import torch.nn as nn

class RobotControlEnvironment:
    def __init__(self):
        # Robot state: [x, y, theta, x_dot, y_dot, theta_dot]
        self.state = np.zeros(6)
        self.target = np.array([1.0, 1.0, 0.0])  # [x, y, theta]
        self.dt = 0.01
        self.max_steps = 1000
        self.steps = 0

    def reset(self):
        self.state = np.random.uniform(-0.5, 0.5, 6)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        # Action: [v_x, v_y, omega] - linear velocities and angular velocity
        v_x, v_y, omega = action

        # Update state using simple kinematic model
        x, y, theta, x_dot, y_dot, theta_dot = self.state

        # Update velocities
        self.state[3] = v_x  # x_dot
        self.state[4] = v_y  # y_dot
        self.state[5] = omega  # theta_dot

        # Update positions
        self.state[0] += self.state[3] * self.dt  # x
        self.state[1] += self.state[4] * self.dt  # y
        self.state[2] += self.state[5] * self.dt  # theta

        # Compute reward
        pos_error = np.linalg.norm(self.state[:2] - self.target[:2])
        orient_error = abs(self.state[2] - self.target[2])

        # Reward based on proximity to target
        reward = -pos_error - 0.1 * orient_error - 0.01 * np.sum(np.abs(action))

        self.steps += 1
        done = self.steps >= self.max_steps or pos_error < 0.1

        return self.state.copy(), reward, done, {}

class RobotControlNN(nn.Module):
    def __init__(self, state_dim=6, action_dim=3):
        super(RobotControlNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Actions in [-1, 1] range
        )

        # Scale actions to appropriate range
        self.action_scale = torch.tensor([1.0, 1.0, 1.0])  # [v_x_max, v_y_max, omega_max]

    def forward(self, state):
        action = self.network(state)
        return action * self.action_scale

# Training function for robot control
def train_robot_control():
    env = RobotControlEnvironment()
    agent = RobotControlNN()
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(100):  # Max steps per episode
            state_tensor = torch.FloatTensor(state)
            action_tensor = agent(state_tensor)
            action = action_tensor.detach().numpy()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
```

## Stability and Safety Considerations

When applying RL to real robots, safety is paramount. Key considerations include:

1. **Action Bounds**: Always constrain actions to safe limits
2. **Reward Shaping**: Design rewards that encourage safe behavior
3. **Simulation-to-Real Transfer**: Use domain randomization to improve robustness
4. **Safety Filters**: Implement hard constraints that override learned policies

## Summary

This section covered advanced RL techniques including Deep Q-Networks, policy gradients, actor-critic methods, and algorithms for continuous control. These techniques enable robots to learn complex behaviors in high-dimensional state and action spaces, making them suitable for real-world applications.

Continue with [Control Integration](./control-integration) to learn how to combine RL with traditional control systems.