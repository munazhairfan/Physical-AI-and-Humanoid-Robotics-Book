---
title: "Glossary"
description: "Comprehensive glossary of terms used in Physical AI & Humanoid Robotics"
sidebar_position: 1
slug: /appendix/glossary
keywords: [glossary, robotics, AI, terminology, definitions]
---

# Glossary

## A

**Action Space**: The set of all possible actions that an agent can take in a reinforcement learning environment.

**Actor-Critic**: A reinforcement learning algorithm that combines value-based and policy-based methods, using an actor to select actions and a critic to evaluate them.

**Adaptive Control**: A control system that adjusts its parameters in real-time based on changes in the system or environment.

**Affine Transformation**: A geometric transformation that preserves points, straight lines, and planes, including translation, rotation, scaling, and shearing.

**Artificial Neural Network (ANN)**: A computational model inspired by biological neural networks, used for machine learning tasks.

**Asynchronous Advantage Actor-Critic (A3C)**: A distributed reinforcement learning algorithm that uses multiple agents to learn in parallel.

## B

**Bayesian Filter**: A mathematical framework for estimating the state of a system using probabilistic models and sensor measurements.

**Behavior Tree**: A hierarchical structure used in robotics and AI for organizing and executing complex behaviors.

**Bezier Curve**: A parametric curve used in computer graphics and motion planning, defined by control points.

**Breadth-First Search (BFS)**: A graph traversal algorithm that explores all neighbors at the present depth before moving to nodes at the next depth level.

## C

**Cartesian Space**: The 3D space defined by X, Y, Z coordinates used to describe positions and orientations in robotics.

**Centroidal Dynamics**: The dynamics of a robot's center of mass and angular momentum, important for humanoid locomotion.

**Configuration Space (C-Space)**: The space of all possible configurations of a robot, used in motion planning to represent obstacles and free space.

**Control Barrier Function (CBF)**: A mathematical function used to ensure safety constraints in control systems.

**Control Lyapunov Function (CLF)**: A function used to prove stability of control systems and design stabilizing controllers.

**Convolutional Neural Network (CNN)**: A type of neural network particularly effective for image processing and computer vision tasks.

**Coordinate System**: A system for defining positions and orientations in space, including world, base, and tool coordinate systems.

## D

**Deep Deterministic Policy Gradient (DDPG)**: A model-free, off-policy reinforcement learning algorithm for continuous action spaces.

**Deep Q-Network (DQN)**: A reinforcement learning algorithm that combines Q-learning with deep neural networks.

**Denavit-Hartenberg (DH) Parameters**: A convention for defining coordinate frames in robotic manipulator kinematics.

**Differential Drive**: A type of wheeled robot locomotion using two independently controlled wheels on either side of the robot.

**Dijkstra's Algorithm**: A graph search algorithm that finds the shortest paths between nodes with non-negative edge weights.

**Dynamic Movement Primitives (DMP)**: A method for learning and reproducing complex movements in robotics.

## E

**End-Effector**: The tool or device at the end of a robotic arm that interacts with the environment.

**Episodic Memory**: In reinforcement learning, the memory of sequences of states, actions, and rewards from complete episodes.

**Euclidean Distance**: The straight-line distance between two points in Euclidean space.

**Extrinsic Parameters**: Camera parameters that describe the position and orientation of the camera in the world coordinate system.

## F

**Forward Kinematics**: The process of calculating the position and orientation of the end-effector based on joint angles.

**Forward Propagation**: In neural networks, the process of computing outputs from inputs through the network layers.

**Fused Multiply-Add (FMA)**: A hardware operation that performs multiplication and addition in a single step, important for AI computations.

## G

**Gaussian Process**: A probabilistic model used for regression and classification tasks, particularly useful for uncertainty quantification.

**Gaussian Mixture Model (GMM)**: A probabilistic model that represents a distribution as a mixture of multiple Gaussian distributions.

**Generalized Coordinates**: A set of parameters that define the configuration of a mechanical system relative to a reference configuration.

**Gradient Descent**: An optimization algorithm that minimizes a function by iteratively moving in the direction of steepest descent.

**Graph-based Path Planning**: Path planning algorithms that represent the environment as a graph of connected nodes.

## H

**Hamiltonian Mechanics**: A reformulation of classical mechanics that describes the evolution of a physical system using Hamilton's equations.

**Heuristic Function**: In search algorithms, a function that estimates the cost to reach the goal from a given state.

**Holonomic System**: A mechanical system where the constraints on motion are integrable, allowing for direct control of all degrees of freedom.

**Homogeneous Transformation**: A 4x4 matrix used to represent rotation and translation in 3D space.

## I

**Inverse Kinematics**: The process of determining joint angles required to achieve a desired end-effector position and orientation.

**Inverse Dynamics**: The calculation of forces and torques required to produce a given motion.

**Intrinsic Parameters**: Camera parameters that describe the internal characteristics of the camera, such as focal length and optical center.

**Imitation Learning**: A machine learning approach where an agent learns to perform tasks by mimicking expert demonstrations.

## J

**Jacobian Matrix**: A matrix of partial derivatives that describes the relationship between joint velocities and end-effector velocities in robotics.

**Jerk**: The rate of change of acceleration; important in trajectory planning for smooth motion.

**Joint Space**: The space defined by the joint angles of a robotic manipulator.

## K

**Kalman Filter**: An algorithm that uses a series of measurements observed over time to estimate unknown variables, particularly in the presence of noise.

**Kinematic Chain**: A system of rigid bodies connected by joints, forming a mechanical linkage.

**Kinematics**: The study of motion without considering the forces that cause it.

## L

**Lagrangian Mechanics**: A reformulation of classical mechanics that uses the Lagrangian function to describe the dynamics of a system.

**Lateral Movement**: Side-to-side motion, particularly relevant for wheeled robots with specific drive mechanisms.

**Learning Rate**: A hyperparameter in machine learning that controls how much to change the model in response to the estimated error each time the model weights are updated.

**Linear Quadratic Regulator (LQR)**: An optimal control technique that minimizes a quadratic cost function for linear systems.

**Logistic Regression**: A statistical model used for binary classification tasks.

## M

**Machine Learning**: A field of artificial intelligence that focuses on building systems that can learn from data without being explicitly programmed.

**Manipulability**: A measure of how well a robotic manipulator can move in different directions at a given configuration.

**Markov Decision Process (MDP)**: A mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker.

**Minimum Jerk Trajectory**: A trajectory that minimizes the jerk (third derivative of position) for smooth motion.

**Model Predictive Control (MPC)**: An advanced control method that uses a model of the system to predict future behavior and optimize control actions.

**Monte Carlo Method**: A computational technique that uses random sampling to solve problems that might be deterministic in principle.

## N

**Neural Network**: A computational model inspired by biological neural networks, used for machine learning tasks.

**Node**: In ROS (Robot Operating System), an executable process that works as part of a ROS system.

**Non-holonomic System**: A mechanical system with constraints that are not integrable, limiting the possible motions.

## O

**Occupancy Grid**: A probabilistic representation of space used in robotics for mapping and navigation.

**Odometry**: The use of data from motion sensors to estimate change in position over time.

**Optimization-based Planning**: Motion planning approaches that formulate path planning as an optimization problem.

## P

**Path Planning**: The process of finding a collision-free path from a start to goal configuration.

**Perception Pipeline**: A sequence of processing steps that transform sensor data into meaningful information about the environment.

**PID Controller**: A control loop feedback mechanism that calculates an error value as the difference between desired and measured values.

**Point Cloud**: A set of data points in space, typically representing the external surface of an object captured by 3D sensors.

**Policy Gradient**: A class of reinforcement learning algorithms that optimize the policy directly by gradient ascent.

**Proportional-Derivative (PD) Controller**: A control system that uses proportional and derivative terms to control a process.

**Proportional-Integral-Derivative (PID) Controller**: A control system that uses proportional, integral, and derivative terms to control a process.

## Q

**Q-Learning**: A model-free reinforcement learning algorithm that learns a policy telling an agent what action to take under what circumstances.

**Quaternion**: A mathematical construct used to represent rotations and orientations in 3D space, avoiding gimbal lock issues.

**Q-Value**: In reinforcement learning, the expected utility of taking a given action in a given state and following an optimal policy thereafter.

## R

**Rapidly-exploring Random Tree (RRT)**: A path planning algorithm that builds a tree of possible configurations by randomly exploring the configuration space.

**RRT***: An asymptotically optimal variant of RRT that provides better path quality.

**Reactive System**: A system that responds to changes in its environment in real-time.

**Reinforcement Learning**: A type of machine learning where agents learn to make decisions by interacting with an environment.

**Representational State Transfer (REST)**: An architectural style for designing networked applications, often used in robotics APIs.

**Robot Operating System (ROS)**: A flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster.

**Rolling Shutter**: A method of image capture in which the image is captured line by line across the sensor, rather than all at once.

## S

**Sampling-based Planning**: Motion planning algorithms that explore the configuration space by randomly sampling configurations.

**Sensor Fusion**: The process of combining data from multiple sensors to achieve better accuracy and reliability than could be achieved by using a single sensor.

**Simultaneous Localization and Mapping (SLAM)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

**Soft Actor-Critic (SAC)**: An off-policy actor-critic algorithm that maximizes both expected return and entropy for better exploration.

**State-Action-Reward-State-Action (SARSA)**: An on-policy temporal difference learning method in reinforcement learning.

**State Estimation**: The process of estimating the state of a system from noisy and incomplete measurements.

**State Space**: The set of all possible states of a system, used in control theory and robotics.

**Stereovision**: The process of extracting 3D information from 2D images captured from different viewpoints.

**Spline**: A piecewise polynomial function used for smooth curve fitting and trajectory generation.

## T

**Temporal Difference (TD) Learning**: A prediction-based reinforcement learning method that learns to predict values of states based on other predictions.

**Trajectory Optimization**: The process of finding the optimal path through space and time for a robot to follow.

**Transform**: In robotics, a mathematical operation that converts coordinates from one frame to another.

**Twist**: In robotics, a 6D vector representing linear and angular velocities.

## V

**Value Iteration**: An algorithm for finding optimal policies in Markov Decision Processes by iteratively updating value functions.

**Variational Autoencoder (VAE)**: A generative neural network that learns to encode data into a latent space and decode it back.

**Vector Database**: A specialized database designed to store and search vector embeddings efficiently.

## W

**Wheel Odometry**: The use of sensors to measure the rotation of wheels to estimate the distance traveled by a wheeled robot.

**Workspace**: The volume of space that a robot manipulator can reach with its end-effector.

## X, Y, Z

**Zero Moment Point (ZMP)**: A concept used in bipedal locomotion to describe the point on the ground where the moment of the ground reaction force is zero.