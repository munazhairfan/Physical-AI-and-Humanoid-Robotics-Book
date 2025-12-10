# Quickstart Guide: Module 5: Robotics Simulation Frameworks (Gazebo, Webots, PyBullet)

**Feature**: `006-simulation-framework` | **Date**: 2025-12-11

## Getting Started with Robotics Simulation

This quickstart guide will help you set up the basic environment for robotics simulation frameworks.

### Prerequisites
- Ubuntu 18.04/20.04 or equivalent Linux distribution (for full Gazebo support)
- Python 3.6+
- C++ development environment
- Basic understanding of robot modeling (URDF/SDF)

### Setup Instructions for Gazebo

1. **Install Gazebo**
   ```bash
   sudo apt-get install gazebo libgazebo-dev
   # For ROS Noetic: sudo apt-get install ros-noetic-gazebo-*
   # For ROS 2: sudo apt-get install ros-<ros2-distro>-gazebo-*
   ```

2. **Basic Gazebo Launch**
   ```bash
   # Launch Gazebo with empty world
   gazebo
   
   # Launch with a specific world file
   gazebo my_world.world
   ```

### Setup Instructions for Webots

1. **Install Webots**
   ```bash
   # Download from https://www.cyberbotics.com/
   # Or install via snap: sudo snap install webots
   ```

2. **Basic Webots Launch**
   ```bash
   # Launch Webots
   webots
   
   # Launch with a specific world file
   webots my_world.wbt
   ```

### Setup Instructions for PyBullet

1. **Install PyBullet**
   ```bash
   pip install pybullet
   ```

2. **Basic PyBullet Example**
   ```python
   import pybullet as p
   import pybullet_data

   # Connect to physics server
   physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

   # Set gravity
   p.setGravity(0, 0, -9.8)

   # Load plane and robot
   p.setAdditionalSearchPath(pybullet_data.getDataPath())
   planeId = p.loadURDF("plane.urdf")
   robotStartPos = [0, 0, 1]
   robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
   robotId = p.loadURDF("r2d2.urdf", robotStartPos, robotStartOrientation)

   # Run simulation
   for i in range(10000):
       p.stepSimulation()
       # Add your control code here

   # Disconnect
   p.disconnect()
   ```

### Key Concepts to Master
1. Physics engine fundamentals and differences between frameworks
2. Robot modeling and importing (URDF, SDF, PROTO)
3. Sensor simulation and data processing
4. Control system integration
5. Multi-robot simulation techniques

### Next Steps
- Complete the full module documentation in `/docs/module-5/`
- Try the hands-on examples in `/docs/module-5/examples/`
- Work through the assignments in `/docs/module-5/assignments.md`

### Troubleshooting
- Ensure your graphics drivers support OpenGL for visualization
- Check that robot models are properly formatted
- Verify that all dependencies are installed correctly
- For Gazebo: make sure ROS environment is properly sourced if using ROS integration