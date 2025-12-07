---
title: Kinematics
description: Understanding forward and inverse kinematics, Jacobians, and kinematic chains for robot motion
sidebar_position: 2
---

# Kinematics

## Introduction

Kinematics is the study of motion without considering the forces that cause it. In robotics, kinematics deals with the relationship between the joint variables of a robot and the position and orientation of its end-effector. Understanding kinematics is fundamental to controlling robot motion, as it allows us to determine where a robot's end-effector will be based on its joint angles, and conversely, what joint angles are needed to achieve a desired end-effector position.

## Forward Kinematics

Forward kinematics (FK) is the process of calculating the position and orientation of a robot's end-effector given the joint angles and the geometric parameters of the robot. This is a straightforward calculation that involves multiplying transformation matrices corresponding to each joint.

### Mathematical Foundation

For a robot with n joints, the forward kinematics can be expressed as:

$$ T = A_1(\theta_1) \cdot A_2(\theta_2) \cdot A_3(\theta_3) \cdot \ldots \cdot A_n(\theta_n) $$

Where:
- $T$ is the transformation matrix representing the end-effector pose
- $A_i(\theta_i)$ is the transformation matrix for joint $i$ as a function of its joint angle $\theta_i$

### Denavit-Hartenberg Convention

The Denavit-Hartenberg (DH) convention provides a systematic way to define coordinate frames on robotic links and joints. Each joint is described by four parameters:
- $a_i$: link length
- $\alpha_i$: link twist
- $d_i$: link offset
- $\theta_i$: joint angle

The transformation matrix for a single joint using DH parameters is:

$$
A_i =
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### Example: 2-DOF Planar Robot

Let's consider a simple 2-DOF planar robot with two revolute joints and link lengths $L_1$ and $L_2$:

```python
import numpy as np

def forward_kinematics_2dof(theta1, theta2, L1, L2):
    """
    Calculate forward kinematics for a 2-DOF planar robot

    Args:
        theta1: Joint angle 1 (radians)
        theta2: Joint angle 2 (radians)
        L1: Length of link 1
        L2: Length of link 2

    Returns:
        x, y: End-effector position
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

    return x, y

# Example usage
theta1 = np.pi/4  # 45 degrees
theta2 = np.pi/6  # 30 degrees
L1 = 1.0
L2 = 0.8

x, y = forward_kinematics_2dof(theta1, theta2, L1, L2)
print(f"End-effector position: ({x:.3f}, {y:.3f})")
```

## Inverse Kinematics

Inverse kinematics (IK) is the reverse process of forward kinematics - determining the joint angles required to achieve a desired end-effector position and orientation. This is generally more complex than forward kinematics because:

1. Multiple solutions may exist for the same end-effector pose
2. No solution may exist if the desired pose is outside the robot's workspace
3. The solution may require numerical methods for complex robots

### Analytical Solutions

For simple robots with specific geometries, analytical solutions to inverse kinematics can be derived. Let's consider the 2-DOF planar robot again:

```python
import numpy as np

def inverse_kinematics_2dof(x, y, L1, L2):
    """
    Calculate inverse kinematics for a 2-DOF planar robot

    Args:
        x, y: Desired end-effector position
        L1: Length of link 1
        L2: Length of link 2

    Returns:
        theta1, theta2: Joint angles (radians)
    """
    # Calculate distance from origin to end-effector
    r = np.sqrt(x**2 + y**2)

    # Check if the position is reachable
    if r > L1 + L2:
        raise ValueError("Position is outside the workspace")
    if r < abs(L1 - L2):
        raise ValueError("Position is inside the workspace but unreachable")

    # Calculate theta2 using law of cosines
    cos_theta2 = (L1**2 + L2**2 - r**2) / (2 * L1 * L2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    # Calculate theta1
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2

# Example usage
x = 1.2
y = 0.8
L1 = 1.0
L2 = 0.8

theta1, theta2 = inverse_kinematics_2dof(x, y, L1, L2)
print(f"Joint angles: theta1={theta1:.3f} rad, theta2={theta2:.3f} rad")

# Verify with forward kinematics
x_fk, y_fk = forward_kinematics_2dof(theta1, theta2, L1, L2)
print(f"Verification: ({x_fk:.3f}, {y_fk:.3f}) vs desired ({x}, {y})")
```

### Numerical Methods

For more complex robots, numerical methods such as the Jacobian-based approach are used. The basic idea is to iteratively adjust joint angles to minimize the error between the current and desired end-effector poses.

## Jacobians

The Jacobian matrix relates the joint velocities to the end-effector velocities. It's crucial for understanding robot dynamics, force control, and singularity analysis.

### Mathematical Definition

The Jacobian $J$ is defined as:

$$
\begin{bmatrix}
\dot{x} \\
\dot{y} \\
\dot{z} \\
\dot{\alpha} \\
\dot{\beta} \\
\dot{\gamma}
\end{bmatrix}
= J(q)
\begin{bmatrix}
\dot{\theta_1} \\
\dot{\theta_2} \\
\vdots \\
\dot{\theta_n}
\end{bmatrix}
$$

Where $q$ represents the joint angles and the left side represents the end-effector velocities.

### Geometric Jacobian

For a robot with revolute joints, the geometric Jacobian can be computed as:

$$
J =
\begin{bmatrix}
J_v \\
J_\omega
\end{bmatrix}
$$

Where:
- $J_v$ relates joint velocities to linear end-effector velocity
- $J_\omega$ relates joint velocities to angular end-effector velocity

For the $i$-th joint:
- Linear velocity contribution: $J_{v,i} = z_{i-1} \times (p_n - p_{i-1})$
- Angular velocity contribution: $J_{\omega,i} = z_{i-1}$

Where $z_{i-1}$ is the axis of actuation of joint $i$, $p_n$ is the end-effector position, and $p_{i-1}$ is the position of joint $i-1$.

### Singularity Analysis

Singularities occur when the Jacobian matrix loses rank, meaning the robot loses one or more degrees of freedom in Cartesian space. At singular configurations:
- The robot cannot move in certain directions
- Joint velocities can become very large
- The inverse Jacobian does not exist

```python
import numpy as np

def jacobian_2dof(theta1, theta2, L1, L2):
    """
    Calculate the Jacobian for a 2-DOF planar robot

    Args:
        theta1, theta2: Joint angles
        L1, L2: Link lengths

    Returns:
        J: 2x2 Jacobian matrix
    """
    J = np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2), -L2*np.sin(theta1 + theta2)],
        [L1*np.cos(theta1) + L2*np.cos(theta1 + theta2), L2*np.cos(theta1 + theta2)]
    ])

    return J

def is_singular(J):
    """
    Check if the Jacobian is singular by computing its determinant
    """
    det = np.linalg.det(J)
    return abs(det) < 1e-6

# Example: Check for singularities
theta1 = 0
theta2 = 0
L1 = 1.0
L2 = 0.8

J = jacobian_2dof(theta1, theta2, L1, L2)
print(f"Jacobian:\n{J}")
print(f"Is singular: {is_singular(J)}")
print(f"Determinant: {np.linalg.det(J):.6f}")
```

## Kinematic Chains

Robots can be classified based on their kinematic chain structure:

### Serial Robots
- Joints connected in sequence from base to end-effector
- Most common type of industrial robot
- Advantages: large workspace, simple control
- Disadvantages: lower stiffness, less payload capacity

### Parallel Robots
- Multiple kinematic chains connect base to end-effector
- Examples: Stewart platform, delta robots
- Advantages: high stiffness, high payload capacity
- Disadvantages: limited workspace, complex control

### Hybrid Robots
- Combination of serial and parallel structures
- Used in specialized applications

## Humanoid & Mobile Robot Examples

### Humanoid Robot Legs

For a humanoid robot leg with hip, knee, and ankle joints, the kinematic model becomes more complex:

```python
def humanoid_leg_ik(x, y, z, thigh_length, shank_length):
    """
    Simplified inverse kinematics for a humanoid leg
    (2D sagittal plane)
    """
    # Calculate knee angle using law of cosines
    leg_length = np.sqrt(x**2 + (y**2 + z**2))

    # Check reachability
    if leg_length > thigh_length + shank_length:
        raise ValueError("Leg position unreachable")

    # Knee angle
    cos_knee = (thigh_length**2 + shank_length**2 - leg_length**2) / (2 * thigh_length * shank_length)
    knee_angle = np.pi - np.arccos(np.clip(cos_knee, -1, 1))

    # Hip angles
    hip_yaw = np.arctan2(y, z)
    hip_pitch = np.arctan2(-x, np.sqrt(y**2 + z**2))

    # Additional calculation for hip roll to align foot
    hip_roll = 0  # Simplified

    return hip_pitch, hip_roll, knee_angle
```

### Mobile Robot Kinematics

For mobile robots, kinematics describes the relationship between wheel velocities and robot motion:

#### Differential Drive

```python
def differential_drive_kinematics(v_left, v_right, wheel_base):
    """
    Forward kinematics for differential drive robot

    Args:
        v_left, v_right: Left and right wheel velocities
        wheel_base: Distance between wheels

    Returns:
        v_x, v_y, omega: Linear and angular velocities of robot
    """
    v_x = (v_right + v_left) / 2
    omega = (v_right - v_left) / wheel_base

    # For pure 2D motion, v_y = 0
    return v_x, 0, omega

def differential_drive_inverse(v_x, omega, wheel_base):
    """
    Inverse kinematics for differential drive robot
    """
    v_left = v_x - omega * wheel_base / 2
    v_right = v_x + omega * wheel_base / 2

    return v_left, v_right
```

## Summary

Kinematics forms the foundation for understanding and controlling robot motion. Forward kinematics allows us to predict where the robot's end-effector will be given joint angles, while inverse kinematics enables us to determine the required joint angles to achieve a desired end-effector pose. The Jacobian matrix provides the relationship between joint and Cartesian velocities, which is crucial for advanced control strategies.

Understanding these concepts is essential for implementing motion planning and control algorithms, as they provide the mathematical framework for describing robot capabilities and limitations.