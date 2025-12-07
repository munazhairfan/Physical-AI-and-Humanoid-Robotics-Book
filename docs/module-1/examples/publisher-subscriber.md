---
title: Publisher/Subscriber Example
sidebar_label: Publisher/Subscriber
---

# Publisher/Subscriber Example

This example demonstrates the fundamental communication pattern in ROS2 using publishers and subscribers. This is the backbone of ROS2's distributed communication system.

## Overview

The publisher/subscriber pattern (also known as topics) enables asynchronous communication between nodes. Publishers send messages to topics, and subscribers receive messages from topics. This creates a decoupled communication system where publishers and subscribers don't need to know about each other directly.

## Prerequisites

- ROS2 installed (Humble Hawksbill or later recommended)
- Python 3.8+ environment
- Basic understanding of ROS2 concepts

## Simple Publisher Example

Let's create a simple publisher that sends "Hello World" messages:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Simple Subscriber Example

Now let's create a subscriber that receives and logs the messages:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


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


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Running the Examples

1. **Save the publisher code** as `talker.py` and the subscriber code as `listener.py` in your ROS2 package.

2. **Make the files executable**:
   ```bash
   chmod +x talker.py
   chmod +x listener.py
   ```

3. **Source your ROS2 environment**:
   ```bash
   source /opt/ros/humble/setup.bash  # or your ROS2 distribution
   ```

4. **Run the publisher in one terminal**:
   ```bash
   ros2 run your_package_name talker
   ```

5. **Run the subscriber in another terminal**:
   ```bash
   ros2 run your_package_name listener
   ```

You should see the publisher sending messages and the subscriber receiving them.

## Custom Message Creation

ROS2 allows you to create custom message types for more complex data structures. Here's how to create a custom message:

### 1. Create the message definition

Create a file `msg/Num.msg` in your package with the following content:

```
int64 num
```

### 2. Update package.xml

Add the following dependencies to your `package.xml`:

```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

### 3. Update CMakeLists.txt

Add the following to your `CMakeLists.txt`:

```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Num.msg"
)
```

### 4. Use the custom message in code

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from your_package_name.msg import Num  # Replace with your package name


class CustomPublisher(Node):

    def __init__(self):
        super().__init__('custom_publisher')
        self.publisher_ = self.create_publisher(Num, 'custom_topic', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = Num()
        msg.num = self.counter
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.num}')
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    custom_publisher = CustomPublisher()
    rclpy.spin(custom_publisher)
    custom_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Demonstration

Quality of Service profiles allow you to control the communication behavior between publishers and subscribers. Here's an example showing different QoS profiles:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String


class QoSPublisher(Node):

    def __init__(self):
        super().__init__('qos_publisher')

        # Create different QoS profiles
        # Reliable communication (default)
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Best effort communication (faster, but messages may be lost)
        best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.reliable_publisher = self.create_publisher(String, 'reliable_topic', reliable_qos)
        self.best_effort_publisher = self.create_publisher(String, 'best_effort_topic', best_effort_qos)

        self.counter = 0

        # Create timer for reliable publisher
        self.reliable_timer = self.create_timer(1.0, self.reliable_timer_callback)
        # Create timer for best effort publisher
        self.best_effort_timer = self.create_timer(0.5, self.best_effort_timer_callback)

    def reliable_timer_callback(self):
        msg = String()
        msg.data = f'Reliable message {self.counter}'
        self.reliable_publisher.publish(msg)
        self.get_logger().info(f'Published reliably: {msg.data}')
        self.counter += 1

    def best_effort_timer_callback(self):
        msg = String()
        msg.data = f'Best effort message {self.counter}'
        self.best_effort_publisher.publish(msg)
        self.get_logger().info(f'Published with best effort: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    qos_publisher = QoSPublisher()
    rclpy.spin(qos_publisher)
    qos_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

The subscriber for QoS demonstration:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String


class QoSSubscriber(Node):

    def __init__(self):
        super().__init__('qos_subscriber')

        # Use matching QoS profiles
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.reliable_subscription = self.create_subscription(
            String,
            'reliable_topic',
            self.reliable_listener_callback,
            reliable_qos)

        self.best_effort_subscription = self.create_subscription(
            String,
            'best_effort_topic',
            self.best_effort_listener_callback,
            best_effort_qos)

    def reliable_listener_callback(self, msg):
        self.get_logger().info(f'Received reliable: {msg.data}')

    def best_effort_listener_callback(self, msg):
        self.get_logger().info(f'Received best effort: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    qos_subscriber = QoSSubscriber()
    rclpy.spin(qos_subscriber)
    qos_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Key Concepts Demonstrated

1. **Node Creation**: How to create ROS2 nodes using the rclpy library
2. **Publishers**: Creating publishers to send messages to topics
3. **Subscribers**: Creating subscribers to receive messages from topics
4. **Timers**: Using timers to periodically publish messages
5. **Custom Messages**: How to define and use custom message types
6. **QoS Profiles**: Different communication reliability options
7. **Logging**: Using ROS2's built-in logging system

## QoS Profile Comparison

| Profile | Reliability | Durability | Use Case |
|---------|-------------|------------|----------|
| Reliable | All messages delivered | Only new messages | Control systems, critical data |
| Best Effort | Messages may be lost | Only new messages | Sensor data, streaming |
| Transient Local | All messages delivered | All messages stored | Configuration, initial data |

This example demonstrates the fundamental communication pattern in ROS2 and shows how to implement it with both standard and custom message types, as well as different Quality of Service configurations.