---
title: "Module 4: Assignments"
description: "Reinforcement Learning and control systems assignments for robotics"
sidebar_position: 6
slug: /module-4/assignments
keywords: [reinforcement learning, assignments, robotics, control systems, exercises]
---

# Module 4 Assignments: Reinforcement Learning & Control

## Overview

This section contains assignments designed to reinforce your understanding of Reinforcement Learning and control systems in robotics. Each assignment builds upon the concepts covered in this module, progressing from basic implementation to complex applications.

## Beginner Exercises

### Exercise 1: Q-Learning Implementation
**Objective**: Implement a Q-Learning agent to solve a simple grid world navigation problem.

**Requirements**:
1. Create a 10x10 grid world with obstacles
2. Implement the Q-learning algorithm with ε-greedy exploration
3. Train the agent to navigate from start to goal position
4. Visualize the learned Q-table as a heatmap

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width-1, height-1)
        self.obstacles = [(2, 2), (3, 3), (4, 4), (5, 5)]  # Define obstacles
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self._get_state()

    def _get_state(self):
        return self.agent_pos[0] * self.width + self.agent_pos[1]

    def _is_valid_pos(self, pos):
        x, y = pos
        return (0 <= x < self.width and 0 <= y < self.height and
                pos not in self.obstacles)

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

        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos

        # Calculate reward
        if self.agent_pos == self.goal:
            reward = 100  # Goal reached
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -10  # Hit obstacle
            done = False
        else:
            reward = -1  # Time penalty
            done = False

        return self._get_state(), reward, done, {}

# TODO: Implement Q-learning agent and training loop
# Expected output: Agent should learn to navigate to goal avoiding obstacles
# Validation checkpoint: Success rate > 80% after 1000 episodes
```

**Validation Checkpoints**:
- Agent learns to avoid obstacles
- Success rate > 80% after 1000 episodes
- Q-table converges to reasonable values

### Exercise 2: PID Controller Design
**Objective**: Design and tune a PID controller for a simple robotic system.

**Requirements**:
1. Implement a PID controller class with standard parameters
2. Apply it to control a simulated motor system
3. Tune parameters using Ziegler-Nichols method
4. Analyze performance metrics (rise time, settling time, overshoot)

```python
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        # TODO: Implement PID control law
        pass

class MotorSystem:
    def __init__(self, inertia=0.1, damping=0.05):
        self.inertia = inertia
        self.damping = damping
        self.position = 0
        self.velocity = 0

    def update(self, torque, dt=0.01):
        # TODO: Implement motor dynamics
        pass

# TODO: Design control system and tune PID parameters
# Expected output: Stable control with minimal overshoot and fast settling
# Validation checkpoint: Settling time < 2 seconds, overshoot < 10%
```

**Validation Checkpoints**:
- Controller stabilizes the system
- Settling time < 2 seconds
- Overshoot < 10%
- Zero steady-state error

### Exercise 3: Value Iteration Algorithm
**Objective**: Implement the value iteration algorithm to solve a simple MDP.

**Requirements**:
1. Create a simple MDP environment (e.g., inventory management)
2. Implement value iteration algorithm
3. Extract optimal policy from value function
4. Compare with random policy performance

```python
def value_iteration(transitions, rewards, gamma=0.9, theta=1e-6):
    """
    transitions: dict[s][a] = [(prob, next_state, reward)]
    rewards: reward function
    gamma: discount factor
    theta: convergence threshold
    """
    # TODO: Implement value iteration algorithm
    pass

# TODO: Create simple MDP and solve with value iteration
# Expected output: Optimal value function and policy
# Validation checkpoint: Algorithm converges within 1000 iterations
```

**Validation Checkpoints**:
- Algorithm converges within 1000 iterations
- Optimal policy outperforms random policy
- Value function is monotonically improving

## Intermediate Exercises

### Exercise 4: Deep Q-Network for Continuous Control
**Objective**: Implement a DQN agent for a continuous control problem using action discretization.

**Requirements**:
1. Create a DQN agent with experience replay
2. Apply to a continuous control environment (e.g., cart-pole)
3. Implement ε-greedy exploration with decay
4. Visualize training progress and learned policy

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        # TODO: Implement DQN architecture
        pass

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # TODO: Implement DQN agent with experience replay
        pass

    def act(self, state, training=True):
        # TODO: Implement ε-greedy action selection
        pass

    def remember(self, state, action, reward, next_state, done):
        # TODO: Implement experience storage
        pass

    def replay(self, batch_size=32):
        # TODO: Implement experience replay training
        pass

# TODO: Train DQN agent on cart-pole environment
# Expected output: Agent learns to balance pole
# Validation checkpoint: Average reward > 200 over 100 episodes
```

**Validation Checkpoints**:
- Agent learns to balance pole consistently
- Average reward > 200 over 100 episodes
- Experience replay improves learning stability

### Exercise 5: Policy Gradient for Robot Arm Control
**Objective**: Implement a policy gradient method to control a simple robot arm.

**Requirements**:
1. Create a neural network policy for continuous action space
2. Implement REINFORCE algorithm with baseline
3. Apply to 2-DOF planar robot arm reaching task
4. Analyze learning curves and final performance

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        # TODO: Implement policy network with Gaussian output
        pass

    def forward(self, state):
        # TODO: Return action distribution parameters
        pass

class RobotArmEnv:
    def __init__(self):
        # 2-DOF planar arm with state [joint1, joint2, vel1, vel2, target_x, target_y]
        self.state_dim = 6
        self.action_dim = 2  # Joint torques
        self.reset()

    def reset(self):
        # TODO: Reset arm to random configuration
        pass

    def step(self, action):
        # TODO: Implement arm dynamics and reward function
        pass

# TODO: Implement policy gradient training
# Expected output: Robot arm learns to reach target positions
# Validation checkpoint: Success rate > 70% on reaching task
```

**Validation Checkpoints**:
- Robot arm learns to reach target positions
- Success rate > 70% on reaching task
- Policy converges to stable behavior

### Exercise 6: Actor-Critic for Navigation
**Objective**: Implement an Actor-Critic method for mobile robot navigation.

**Requirements**:
1. Create Actor-Critic network architecture
2. Apply to navigation task with obstacles
3. Implement advantage estimation
4. Compare performance with policy gradient methods

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        # TODO: Implement shared feature extractor
        # TODO: Implement separate actor and critic heads
        pass

    def forward(self, state):
        # TODO: Return action probabilities and state value
        pass

class NavigationEnv:
    def __init__(self):
        # State: [robot_x, robot_y, robot_theta, goal_x, goal_y, obstacle_x, obstacle_y]
        self.state_dim = 7
        self.action_dim = 2  # [linear_vel, angular_vel]
        self.reset()

    def reset(self):
        # TODO: Initialize navigation environment
        pass

    def step(self, action):
        # TODO: Implement navigation dynamics and reward
        pass

# TODO: Train Actor-Critic agent for navigation
# Expected output: Agent learns efficient navigation policy
# Validation checkpoint: Navigation success rate > 85%
```

**Validation Checkpoints**:
- Agent learns efficient navigation policy
- Navigation success rate > 85%
- Actor-Critic outperforms policy gradients

## Advanced Projects

### Project 1: Deep Reinforcement Learning for Robotic Manipulation
**Objective**: Develop a complete DRL system for a robotic manipulation task.

**Requirements**:
1. Choose a manipulation environment (e.g., reaching, grasping, pushing)
2. Implement a state-of-the-art DRL algorithm (SAC, TD3, or PPO)
3. Include proper reward shaping and exploration strategies
4. Evaluate on multiple test scenarios
5. Analyze sample efficiency and generalization

**Implementation Hints**:
- Use appropriate neural network architecture for continuous control
- Implement proper action bounds and state normalization
- Include safety mechanisms to prevent damage to robot
- Consider simulation-to-real transfer techniques

**Validation Checkpoints**:
- Task success rate > 90% in simulation
- Algorithm demonstrates stable learning curves
- Generalizes to new goal positions
- Safe execution with bounded control actions

### Project 2: Learning-Based Adaptive Control System
**Objective**: Design a hybrid control system that combines traditional control with learning-based adaptation.

**Requirements**:
1. Implement a traditional controller (e.g., PID, MPC) as baseline
2. Add a learning component that adapts controller parameters
3. Ensure stability and safety constraints are maintained
4. Evaluate performance improvement over baseline controller
5. Test robustness to disturbances and model uncertainties

**Implementation Hints**:
- Use safe exploration techniques during learning
- Implement parameter constraints to maintain stability
- Consider multi-objective optimization (performance + safety)
- Include monitoring systems to detect unsafe conditions

**Validation Checkpoints**:
- Hybrid system outperforms baseline controller
- Stability guaranteed throughout learning process
- Adapts effectively to disturbances
- Maintains safety constraints during operation

## Submission Guidelines

### Code Requirements
- All code must be well-documented with comments
- Include proper error handling and validation
- Use appropriate data structures and algorithms
- Follow clean code principles and best practices

### Report Requirements
- Describe your approach and implementation details
- Include analysis of results and performance metrics
- Discuss challenges encountered and how you addressed them
- Compare your results with baseline methods where applicable
- Include visualizations of learning curves and final performance

### Evaluation Criteria
- Correctness of implementation (40%)
- Performance and efficiency (30%)
- Code quality and documentation (20%)
- Analysis and insights (10%)

## Resources and References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Levine, S. (2018). Reinforcement Learning for Robotics
- Proportional-Integral-Derivative (PID) Controller Design and Implementation
- OpenAI Spinning Up: Deep RL Learning Resources