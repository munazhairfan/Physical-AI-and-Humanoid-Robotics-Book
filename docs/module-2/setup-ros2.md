# ROS2 Setup for Perception & Computer Vision Module

## Installing ROS2

This module uses ROS2 for sensor examples and integration. Follow these steps to set up ROS2:

### Ubuntu/Debian Installation
```bash
# Setup locale
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2
sudo apt install python3-colcon-common-extensions
sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv ros-humble-image-transport ros-humble-compressed-image-transport
```

### Windows Installation (Alternative approach)
For Windows users, consider using ROS2 in WSL2 or Docker:

```bash
# Using Docker
docker run -it --rm ros:humble
```

### Environment Setup
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# For Windows with WSL2
source /opt/ros/humble/setup.bash
```

### Python Integration
```bash
pip install rospkg catkin-pkg
```