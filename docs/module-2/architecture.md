---
title: "Module 2: Robotic Nervous System (ROS2) - Architecture"
description: "System architecture for ROS2-based robotic communication"
sidebar_position: 3
slug: /module-2/architecture
keywords: [robotics, ros2, middleware, architecture, design, system, communication]
---

# Module 2: Robotic Nervous System (ROS2) - Architecture

## Introduction

This section explores the architectural patterns and system design principles for implementing robotic communication systems using ROS2. Understanding ROS2 architecture is crucial for building scalable, maintainable, and robust robotic applications.

## ROS2 System Architecture

### Three-Layer Architecture

ROS2 follows a three-layer architecture:

```
Application Layer
       ↓
ROS Client Libraries (rclpy, rclcpp)
       ↓
ROS Middleware (RMW)
       ↓
DDS Implementation (Fast DDS, Cyclone DDS, RTI Connext)
```

### Node-Based Architecture

ROS2 systems are organized as a collection of interconnected nodes:

```
Sensor Nodes → Processing Nodes → Control Nodes → Actuator Nodes
       ↓           ↓              ↓              ↓
   Camera,     Perception,    Decision,     Motors,
   IMU,       Mapping,       Planning      Servos
   LiDAR      Localization   Control
```

## Design Patterns

### Publisher-Subscriber Pattern

The most common pattern for continuous data streams:

```python
class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.publisher = self.create_publisher(SensorData, 'sensor_data', 10)
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

    def publish_sensor_data(self):
        msg = SensorData()
        # Populate message with sensor readings
        self.publisher.publish(msg)

class ProcessingNode(Node):
    def __init__(self):
        super().__init__('processing_node')
        self.subscription = self.create_subscription(
            SensorData, 'sensor_data', self.process_data, 10)

    def process_data(self, msg):
        # Process the received sensor data
        pass
```

### Client-Server Pattern

For request-response interactions:

```python
class NavigationServer(Node):
    def __init__(self):
        super().__init__('navigation_server')
        self.srv = self.create_service(NavigateToPose, 'navigate_to_pose',
                                      self.navigate_to_pose)

    def navigate_to_pose(self, request, response):
        # Process navigation request
        response.success = True
        return response

class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')
        self.cli = self.create_client(NavigateToPose, 'navigate_to_pose')

    async def send_goal(self, pose):
        request = NavigateToPose.Request()
        request.pose = pose
        future = await self.cli.call_async(request)
        return future.result()
```

### Component-Based Architecture

For complex nodes that need to be decomposed:

```python
from rclpy_components.component_manager import ComponentManager

class ModularNode(Node):
    def __init__(self):
        super().__init__('modular_node')

        # Load components dynamically
        self.components = {}
        self.load_component('perception', PerceptionComponent)
        self.load_component('planning', PlanningComponent)
        self.load_component('control', ControlComponent)

    def load_component(self, name, component_class):
        self.components[name] = component_class(self)

class PerceptionComponent:
    def __init__(self, node):
        self.node = node
        self.subscriber = node.create_subscription(
            SensorMsg, 'sensors', self.process_sensors, 10)
        self.publisher = node.create_publisher(
            PerceptionMsg, 'perception_output', 10)

    def process_sensors(self, msg):
        # Process sensor data
        pass
```

## Communication Architectures

### Data-Centric Architecture

DDS enables data-centric communication where data is the primary focus:

```python
# Publisher defines data schema
from sensor_msgs.msg import LaserScan

class LaserPublisher(Node):
    def __init__(self):
        super().__init__('laser_publisher')
        # Define QoS based on data requirements
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(LaserScan, 'laser_scan', sensor_qos)
```

### Service-Oriented Architecture

For command and control interactions:

```python
# Service interface defines contract
class RobotControlInterface:
    def __init__(self, node):
        self.node = node
        self.services = {
            'move_base': node.create_service(MoveBase, 'move_base', self.handle_move_base),
            'gripper_control': node.create_service(GripperCommand, 'gripper', self.handle_gripper),
            'navigation': node.create_service(Navigate, 'navigate', self.handle_navigate)
        }

    def handle_move_base(self, request, response):
        # Handle move base command
        pass

    def handle_gripper(self, request, response):
        # Handle gripper command
        pass

    def handle_navigate(self, request, response):
        # Handle navigation command
        pass
```

## Distributed Architecture Patterns

### Masterless Architecture

ROS2's DDS-based architecture eliminates the need for a central master:

```
Node A ←→ DDS Middleware ←→ Node B
   ↓                          ↓
DDS Implementation      DDS Implementation
```

### Multi-Robot Communication

For coordinating multiple robots:

```python
class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Robot-specific topics
        self.robot_publishers = {}
        self.robot_subscribers = {}

        for robot_id in range(self.num_robots):
            topic_name = f'robot_{robot_id}/command'
            self.robot_publishers[robot_id] = self.create_publisher(
                RobotCommand, topic_name, 10)

            feedback_topic = f'robot_{robot_id}/feedback'
            self.robot_subscribers[robot_id] = self.create_subscription(
                RobotFeedback, feedback_topic,
                lambda msg, rid=robot_id: self.handle_robot_feedback(msg, rid), 10)
```

## Performance Architecture Considerations

### Real-Time Architecture

For time-critical applications:

```python
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor

class RealTimeNode(Node):
    def __init__(self):
        # Initialize with real-time context if needed
        super().__init__('real_time_node', context=Context())

        # Use timer with specific period for real-time constraints
        self.timer = self.create_timer(
            0.01,  # 10ms period
            self.real_time_callback,
            clock=Self.CLOCK_UNSTABLE
        )

    def real_time_callback(self):
        # Time-critical processing
        pass

# Executor with real-time considerations
def main():
    rclpy.init()
    node = RealTimeNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
```

### Resource Management Architecture

Managing computational and communication resources:

```python
class ResourceAwareNode(Node):
    def __init__(self):
        super().__init__('resource_aware_node')

        # Monitor resource usage
        self.resource_monitor = ResourceMonitor()

        # Adaptive QoS based on resource availability
        self.base_qos = QoSProfile(depth=10)
        self.adaptive_qos = self.adjust_qos_for_resources()

        self.publisher = self.create_publisher(
            AdaptiveMsg, 'adaptive_topic', self.adaptive_qos)

    def adjust_qos_for_resources(self):
        resources = self.resource_monitor.get_resources()

        if resources['bandwidth'] < 0.5:
            # Reduce message frequency
            return QoSProfile(depth=1)
        elif resources['cpu'] > 0.8:
            # Reduce processing
            return QoSProfile(depth=5)
        else:
            return self.base_qos
```

## Error Handling Architecture

### Fault-Tolerant Design

Designing systems that can handle failures gracefully:

```python
import rclpy.time
from rclpy.duration import Duration

class FaultTolerantNode(Node):
    def __init__(self):
        super().__init__('fault_tolerant_node')

        # Health monitoring
        self.heartbeat_publisher = self.create_publisher(Heartbeat, 'heartbeat', 10)
        self.heartbeat_timer = self.create_timer(1.0, self.publish_heartbeat)

        # Timeout mechanisms
        self.last_sensor_time = self.get_clock().now()
        self.sensor_timeout = Duration(seconds=5.0)

        # Fallback strategies
        self.fallback_mode = False

    def sensor_callback(self, msg):
        self.last_sensor_time = self.get_clock().now()
        if self.fallback_mode:
            self.exit_fallback_mode()

        # Process sensor data
        self.process_sensor_data(msg)

    def check_sensor_health(self):
        current_time = self.get_clock().now()
        if current_time - self.last_sensor_time > self.sensor_timeout:
            if not self.fallback_mode:
                self.enter_fallback_mode()

    def enter_fallback_mode(self):
        self.get_logger().warn('Entering fallback mode due to sensor timeout')
        self.fallback_mode = True
        # Implement safe behavior

    def exit_fallback_mode(self):
        self.get_logger().info('Exiting fallback mode')
        self.fallback_mode = False
```

## Security Architecture

### Secure Communication Patterns

Implementing security in ROS2 systems:

```python
class SecureNode(Node):
    def __init__(self):
        # Initialize with security context
        super().__init__('secure_node')

        # Encrypted communication
        self.secure_publisher = self.create_publisher(
            SecureMsg, 'secure_topic',
            qos_profile=QoSProfile(depth=10)
        )

        # Authentication mechanisms
        self.authenticate_peers()

    def authenticate_peers(self):
        # Implement peer authentication
        pass
```

## Scalability Architecture

### Microservices Architecture

Decomposing complex systems into smaller services:

```
Perception Service → Planning Service → Control Service
       ↓                   ↓               ↓
Object Detection    Path Planning    Motor Control
Depth Estimation    Trajectory Gen   Safety Monitor
```

### Load Balancing

Distributing work across multiple nodes:

```python
class LoadBalancerNode(Node):
    def __init__(self):
        super().__init__('load_balancer')

        # Multiple worker nodes
        self.workers = []
        for i in range(4):  # 4 worker nodes
            worker_cli = self.create_client(WorkRequest, f'worker_{i}/process')
            self.workers.append(worker_cli)

        self.current_worker = 0

        # Work distribution
        self.work_sub = self.create_subscription(
            WorkItem, 'incoming_work', self.distribute_work, 10)

    def distribute_work(self, work_item):
        # Round-robin distribution
        worker_cli = self.workers[self.current_worker]
        self.current_worker = (self.current_worker + 1) % len(self.workers)

        # Send work to worker
        request = WorkRequest.Request()
        request.work = work_item
        future = worker_cli.call_async(request)
```

## Integration Architecture

### Legacy System Integration

Connecting ROS2 with existing systems:

```python
class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')

        # ROS2 interface
        self.ros_publisher = self.create_publisher(ROSMsg, 'ros_topic', 10)
        self.ros_subscriber = self.create_subscription(
            ROSMsg, 'ros_topic', self.ros_callback, 10)

        # Legacy system interface
        self.legacy_interface = LegacySystemInterface()

        # Bridge timer
        self.bridge_timer = self.create_timer(0.05, self.bridge_callback)

    def ros_callback(self, msg):
        # Forward ROS2 message to legacy system
        self.legacy_interface.send_data(msg)

    def bridge_callback(self):
        # Poll legacy system for data
        legacy_data = self.legacy_interface.receive_data()
        if legacy_data:
            ros_msg = self.convert_to_ros(legacy_data)
            self.ros_publisher.publish(ros_msg)
```

## Summary

This section covered the architectural patterns and design principles for ROS2-based robotic systems, including node organization, communication patterns, distributed architectures, performance considerations, error handling, security, and integration approaches. These architectural concepts are essential for building robust, scalable, and maintainable robotic applications.

Continue with [Nodes, Topics, Services](./nodes-topics-services) to explore core communication mechanisms in detail.