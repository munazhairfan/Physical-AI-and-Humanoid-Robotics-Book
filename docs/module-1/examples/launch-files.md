---
title: Launch Files Example
sidebar_label: Launch Files
---

# Launch Files Example

This example demonstrates how to create and use launch files in ROS2 to manage complex robot systems with multiple nodes.

## Overview

Launch files in ROS2 allow you to start multiple nodes with specific configurations simultaneously. They're essential for managing complex robotic systems with multiple components like sensors, controllers, and AI nodes.

## Python Launch File Example

### Basic Launch File

Create a Python launch file `my_robot_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Launch the talker node
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker_node',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[
                ('chatter', 'my_chatter')
            ],
            output='screen'
        ),

        # Launch the listener node
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='listener_node',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[
                ('chatter', 'my_chatter')
            ],
            output='screen'
        )
    ])
```

### Advanced Launch File with Namespacing

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.actions import SetParameter

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='robot1')
    namespace = LaunchConfiguration('namespace', default='')

    ld = LaunchDescription()

    # Set environment variables if needed
    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'))

    # Create a group with namespace if specified
    group_action = GroupAction(
        condition=IfCondition(namespace),
        actions=[
            PushRosNamespace(namespace),
            # Robot-specific nodes
            Node(
                package='my_robot_package',
                executable='robot_controller',
                name='controller',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': robot_name}
                ],
                output='screen'
            ),
            Node(
                package='my_robot_package',
                executable='sensor_processor',
                name='sensor_processor',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': robot_name}
                ],
                output='screen'
            )
        ]
    )
    ld.add_action(group_action)

    # Nodes without namespace (or in default namespace)
    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', '/path/to/my/rviz/config.rviz'],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    ))

    # Parameter server node
    ld.add_action(SetParameter(name='use_sim_time', value=use_sim_time))

    return ld
```

### Launch File with Conditional Launching

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    use_gazebo = LaunchConfiguration('use_gazebo', default='false')

    # Get package share directory
    pkg_share = get_package_share_directory('my_robot_package')

    ld = LaunchDescription()

    # Include another launch file conditionally
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'gazebo.launch.py')
        ),
        condition=IfCondition(use_gazebo)
    )
    ld.add_action(gazebo_launch)

    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description':
                open(os.path.join(pkg_share, 'urdf', 'my_robot.urdf')).read()}
        ],
        output='screen'
    )
    ld.add_action(robot_state_publisher)

    # Launch joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    ld.add_action(joint_state_publisher)

    # Launch RViz conditionally
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'config.rviz')],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(use_rviz),
        output='screen'
    )
    ld.add_action(rviz_node)

    # Robot controller node
    controller_node = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    ld.add_action(controller_node)

    return ld
```

## YAML Launch File Example

ROS2 also supports YAML-based launch files:

```yaml
launch:

  - set_parameter:
      name: use_sim_time
      value: false

  - node:
      pkg: demo_nodes_cpp
      exec: talker
      name: talker_node
      params:
        - {use_sim_time: false}
      remap:
        - {from: chatter, to: my_chatter}
      output: screen

  - node:
      pkg: demo_nodes_cpp
      exec: listener
      name: listener_node
      params:
        - {use_sim_time: false}
      remap:
        - {from: chatter, to: my_chatter}
      output: screen

  - log:
      msg: "Launch file loaded successfully"
```

## Namespacing and Remapping Example

### Namespacing

Namespacing allows you to organize nodes and topics into separate namespaces:

```python
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    # Launch nodes in a specific namespace
    return LaunchDescription([
        PushRosNamespace('robot1'),
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace='robot1'
        ),
        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor',
            namespace='robot1'
        )
    ])
```

### Remapping

Remapping allows you to change topic names at launch time:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='sensor_node',
            name='lidar_sensor',
            remappings=[
                ('/scan', '/robot1/laser_scan'),  # Remap scan topic
                ('/tf', '/robot1/tf'),           # Remap tf topic
            ]
        )
    ])
```

## Running Launch Files

1. **Save the launch file** in your package's `launch/` directory

2. **Run the launch file**:
   ```bash
   ros2 launch my_robot_package my_robot_launch.py
   ```

3. **With arguments**:
   ```bash
   ros2 launch my_robot_package my_robot_launch.py use_sim_time:=true robot_name:=my_robot
   ```

## Multi-Robot Launch Example

For managing multiple robots in the same system:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments for robot configuration
    robot1_namespace = LaunchConfiguration('robot1_namespace', default='robot1')
    robot2_namespace = LaunchConfiguration('robot2_namespace', default='robot2')

    return LaunchDescription([
        # Robot 1 nodes
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace=robot1_namespace,
            parameters=[{'robot_id': 1}],
            remappings=[
                ('/cmd_vel', f'/{robot1_namespace}/cmd_vel'),
                ('/odom', f'/{robot1_namespace}/odom'),
            ]
        ),

        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor',
            namespace=robot1_namespace,
            remappings=[
                ('/scan', f'/{robot1_namespace}/scan'),
            ]
        ),

        # Robot 2 nodes
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace=robot2_namespace,
            parameters=[{'robot_id': 2}],
            remappings=[
                ('/cmd_vel', f'/{robot2_namespace}/cmd_vel'),
                ('/odom', f'/{robot2_namespace}/odom'),
            ]
        ),

        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor',
            namespace=robot2_namespace,
            remappings=[
                ('/scan', f'/{robot2_namespace}/scan'),
            ]
        ),

        # Visualization node that can see both robots
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/path/to/multi_robot_config.rviz']
        )
    ])
```

## Key Concepts Demonstrated

1. **Launch File Creation**: How to create Python and YAML launch files
2. **Node Launching**: Starting multiple nodes with specific configurations
3. **Launch Arguments**: Parameterizing launch files for flexibility
4. **Namespacing**: Organizing nodes and topics into separate namespaces
5. **Remapping**: Changing topic names at launch time
6. **Conditional Launching**: Including/excluding nodes based on conditions
7. **Multi-Robot Systems**: Managing multiple robots in a single launch file
8. **Parameter Management**: Setting parameters for nodes at launch time

## Best Practices

- Use launch files to manage complex robot systems
- Parameterize launch files for flexibility
- Use namespaces to organize multi-robot systems
- Include error handling and logging
- Document your launch files with comments
- Use consistent naming conventions

This example shows how to create and use launch files in ROS2 to manage complex robotic systems with multiple nodes, parameters, and configurations.