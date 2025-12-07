---
title: "Module 2: Robotic Nervous System (ROS2) - Core Concepts"
description: "Core concepts of ROS2 framework and robotic communication"
sidebar_position: 2
slug: /module-2/core-concepts
keywords: [robotics, ros2, middleware, communication, nodes, topics, services]
---

# Module 2: Robotic Nervous System (ROS2) - Core Concepts

## Introduction

This section covers the fundamental concepts that form the basis of the ROS2 framework. Understanding these concepts is essential for developing effective robotic communication systems using ROS2.

## The ROS2 Architecture

### Client Library Abstraction

ROS2 uses client libraries (rclpy for Python, rclcpp for C++) that provide a consistent API across programming languages:

```
Application Code
       ↓
ROS Client Library (rclpy/rclcpp)
       ↓
ROS Middleware (RMW)
       ↓
DDS Implementation
```

### Communication Patterns

ROS2 supports multiple communication patterns:

1. **Publish-Subscribe (Topics)**: Asynchronous, one-to-many communication
2. **Request-Response (Services)**: Synchronous, one-to-one communication
3. **Action-Based (Actions)**: Goal-oriented communication with feedback

## Nodes

### Definition and Purpose

A node is an executable process that works as part of a ROS2 system. Nodes are the fundamental building blocks of a ROS2 application:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

Nodes follow a specific lifecycle:
- **Unconfigured**: Node created but not configured
- **Inactive**: Configured but not active
- **Active**: Fully operational
- **Finalized**: Node is shutting down

## Topics and Messages

### Topic Communication

Topics enable publish-subscribe communication between nodes:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

### Message Types

Messages define the data structure for communication:

```python
# Custom message definition (in msg/MyMessage.msg)
string name
int32 id
float64[] values
bool is_valid
```

## Services

### Service Communication

Services enable request-response communication:

```python
# Service definition (in srv/AddTwoInts.srv)
int64 a
int64 b
---
int64 sum

# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {request.a} + {request.b} = {response.sum}')
        return response

# Service client
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
import sys

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        future = self.cli.call_async(self.req)
        return future
```

## Actions

### Action Communication

Actions provide goal-oriented communication with feedback:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

## Parameters

### Node Configuration

Parameters allow runtime configuration of nodes:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('topics_to_subscribe', ['topic1', 'topic2'])

        # Get parameter values
        self.param_value = self.get_parameter('my_parameter').value
        self.threshold = self.get_parameter('threshold').value

        # Set callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'threshold' and param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                if param.value < 0.0 or param.value > 1.0:
                    return SetParametersResult(successful=False, reason='Threshold must be between 0 and 1')
        return SetParametersResult(successful=True)
```

## Launch Files

### Composable Nodes and Launch Systems

Launch files allow you to start multiple nodes together:

```xml
<!-- launch/example.launch.xml -->
<launch>
  <node pkg="demo_nodes_cpp" exec="talker" name="talker">
    <param name="message" value="Hello World"/>
    <param name="count" value="10"/>
  </node>

  <node pkg="demo_nodes_cpp" exec="listener" name="listener"/>

  <node pkg="demo_nodes_cpp" exec="add_two_ints_server" name="server"/>
</launch>
```

Or using Python launch files:

```python
# launch/example_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker',
            parameters=[
                {'message': 'Hello World'},
                {'count': 10}
            ]
        ),
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='listener'
        ),
        Node(
            package='demo_nodes_cpp',
            executable='add_two_ints_server',
            name='server'
        )
    ])
```

## Quality of Service (QoS)

### QoS Profiles

QoS settings control the behavior of publishers and subscribers:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Define QoS profile for real-time communication
realtime_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

# Use in publisher/subscriber
publisher = self.create_publisher(String, 'topic', realtime_qos)
subscription = self.create_subscription(String, 'topic', callback, realtime_qos)
```

## Services vs Topics vs Actions

### When to Use Each

| Communication Type | Use Case | Characteristics |
|-------------------|----------|-----------------|
| Topics | Sensor data, continuous streams | Asynchronous, many-to-many |
| Services | One-time requests, queries | Synchronous, request-response |
| Actions | Long-running tasks with feedback | Goal-oriented, cancellable |

## Package Structure

### ROS2 Package Organization

A standard ROS2 package follows this structure:

```
my_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── setup.py                # Build configuration for Python
├── setup.cfg               # Installation configuration
├── my_package/             # Python modules
│   ├── __init__.py
│   └── my_module.py
├── src/                    # C++ source files
├── include/my_package/     # C++ headers
├── msg/                    # Custom message definitions
├── srv/                    # Custom service definitions
├── action/                 # Custom action definitions
├── launch/                 # Launch files
├── config/                 # Configuration files
└── test/                   # Test files
```

## Communication Patterns in Robotics

### Sensor Integration

ROS2 enables integration of multiple sensors:

```
Camera Node → Image Topics → Perception Nodes
Lidar Node → Point Cloud Topics → Mapping Nodes
IMU Node → Sensor Topics → State Estimation Nodes
```

### Control Systems

Communication between perception and control:

```
Perception → Topics → Decision Making → Actions → Control Systems
```

## Summary

This section covered the core concepts of ROS2 including nodes, topics, services, actions, parameters, and launch systems. These concepts form the foundation for building distributed robotic applications that can communicate effectively and coordinate their activities.

Continue with [Architecture](./architecture) to learn about ROS2 system design patterns.