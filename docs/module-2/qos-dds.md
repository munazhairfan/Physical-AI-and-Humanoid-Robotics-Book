---
title: "Module 2: Robotic Nervous System (ROS2) - QoS and DDS"
description: "Quality of Service and Data Distribution Service in ROS2"
sidebar_position: 5
slug: /module-2/qos-dds
keywords: [robotics, ros2, qos, dds, quality of service, middleware, communication]
---

# Module 2: Robotic Nervous System (ROS2) - QoS and DDS

## Introduction

Quality of Service (QoS) and Data Distribution Service (DDS) are fundamental to ROS2's communication architecture. DDS provides the underlying communication infrastructure, while QoS policies allow fine-tuning of communication behavior to meet specific application requirements.

## Data Distribution Service (DDS) Overview

### DDS Architecture

DDS (Data Distribution Service) is an OMG (Object Management Group) standard for real-time, distributed communication. In ROS2, DDS serves as the middleware layer:

```
ROS2 Application Layer
       ↓
ROS Client Libraries (rclpy/rclcpp)
       ↓
ROS Middleware (RMW - ROS Middleware)
       ↓
DDS Implementation (Fast DDS, Cyclone DDS, RTI Connext)
```

### DDS Entities

DDS defines several key entities that enable communication:

1. **Domain**: A communication plane that isolates DDS applications
2. **DomainParticipant**: An application's connection to a domain
3. **Topic**: A named data channel
4. **Publisher**: Entity that sends data
5. **Subscriber**: Entity that receives data
6. **DataWriter**: Specific endpoint for writing data
7. **DataReader**: Specific endpoint for reading data

### DDS Communication Model

DDS uses a publish-subscribe model with data-centricity:

```python
# Underlying DDS concepts in ROS2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# In ROS2, these DDS concepts are abstracted but still relevant:
# - Topic corresponds to DDS Topic
# - Publisher corresponds to DDS DataWriter
# - Subscriber corresponds to DDS DataReader
# - QoS settings map to DDS QoS policies
```

## Quality of Service (QoS) Policies

### QoS Policy Types

ROS2 provides several QoS policies to control communication behavior:

#### 1. Reliability Policy

Controls whether messages are delivered reliably:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Reliable: Ensure all messages are delivered (like TCP)
reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE
)

# Best Effort: Send messages without guarantee (like UDP)
best_effort_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

# Use cases:
# - Reliable: Command messages, critical sensor data, state updates
# - Best Effort: Camera images, LiDAR scans, telemetry data
```

#### 2. Durability Policy

Controls how messages are retained for late-joining subscribers:

```python
from rclpy.qos import DurabilityPolicy

# Volatile: Don't store messages for late joiners
volatile_qos = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

# Transient Local: Store messages for late joiners (local writer only)
transient_local_qos = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)

# Use cases:
# - Volatile: Real-time sensor data, streaming data
# - Transient Local: Configuration parameters, initial state, maps
```

#### 3. History Policy

Controls how many samples are stored:

```python
from rclpy.qos import HistoryPolicy

# Keep Last: Store only the most recent samples
keep_last_qos = QoSProfile(
    depth=5,  # Keep 5 most recent samples
    history=HistoryPolicy.KEEP_LAST
)

# Keep All: Store all samples (limited by resource limits)
keep_all_qos = QoSProfile(
    history=HistoryPolicy.KEEP_ALL
)

# Use cases:
# - Keep Last: Sensor data, control commands
# - Keep All: Log data, historical information
```

#### 4. Deadline Policy

Specifies the maximum time between sample publications:

```python
from rclpy.duration import Duration

deadline_qos = QoSProfile(
    depth=10,
    deadline=Duration(seconds=1)  # Samples must be published within 1 second
)

# Use case: Time-critical systems where data has expiration
```

#### 5. Lifespan Policy

Specifies how long samples are valid:

```python
lifespan_qos = QoSProfile(
    depth=10,
    lifespan=Duration(seconds=30)  # Samples expire after 30 seconds
)

# Use case: Data that becomes stale after some time
```

#### 6. Liveliness Policy

Controls how liveliness is monitored:

```python
from rclpy.qos import LivelinessPolicy

liveliness_qos = QoSProfile(
    depth=10,
    liveliness=LivelinessPolicy.AUTOMATIC,
    liveliness_lease_duration=Duration(seconds=10)
)

# Use case: Monitoring if publishers/subscribers are alive
```

### Complete QoS Profile Examples

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

# Sensor data: High frequency, can lose some data
SENSOR_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

# Command data: Must be reliable, critical
COMMAND_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

# Configuration data: Must be available to late joiners
CONFIG_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST
)

# Log data: Keep all historical information
LOG_QOS = QoSProfile(
    depth=1000,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_ALL
)
```

## QoS Matching and Compatibility

### QoS Compatibility Rules

For communication to occur, publisher and subscriber QoS must be compatible:

```python
# Compatible QoS combinations:
# Publisher: RELIABLE -> Subscriber: RELIABLE (✓)
# Publisher: RELIABLE -> Subscriber: BEST_EFFORT (✓)
# Publisher: BEST_EFFORT -> Subscriber: RELIABLE (✗)
# Publisher: BEST_EFFORT -> Subscriber: BEST_EFFORT (✓)

class QoSMismatchExample(Node):
    def __init__(self):
        super().__init__('qos_mismatch_example')

        # This will work - subscriber accepts RELIABLE or BEST_EFFORT
        self.subscriber = self.create_subscription(
            String, 'topic', self.callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE  # More restrictive
            )
        )

        # This publisher is compatible - it's RELIABLE
        self.publisher = self.create_publisher(
            String, 'topic',
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE  # Compatible
            )
        )

    def callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

### QoS Event Handling

Monitor QoS-related events:

```python
from rclpy.qos_event import SubscriptionEventCallbacks, PublisherEventCallbacks
from rclpy.qos import Lifespan, Deadline, LivelinessPolicy

class QoSEventNode(Node):
    def __init__(self):
        super().__init__('qos_event_node')

        # Set up event callbacks for publisher
        publisher_event_callbacks = PublisherEventCallbacks(
            deadline_callback=self.deadline_callback,
            liveliness_callback=self.liveliness_callback
        )

        self.publisher = self.create_publisher(
            String, 'qos_events',
            qos_profile=QoSProfile(
                depth=10,
                deadline=Duration(seconds=1),
                liveliness=LivelinessPolicy.MANUAL_BY_TOPIC,
                liveliness_lease_duration=Duration(seconds=2)
            ),
            event_callbacks=publisher_event_callbacks
        )

        # Set up event callbacks for subscriber
        subscription_event_callbacks = SubscriptionEventCallbacks(
            deadline_callback=self.sub_deadline_callback,
            liveliness_callback=self.sub_liveliness_callback
        )

        self.subscription = self.create_subscription(
            String, 'qos_events',
            self.message_callback,
            qos_profile=QoSProfile(
                depth=10,
                deadline=Duration(seconds=1),
                liveliness=LivelinessPolicy.MANUAL_BY_TOPIC,
                liveliness_lease_duration=Duration(seconds=2)
            ),
            event_callbacks=subscription_event_callbacks
        )

        # Timer to periodically assert liveliness
        self.liveliness_timer = self.create_timer(
            1.0, self.assert_liveliness)

    def deadline_callback(self, event):
        self.get_logger().warn('Publisher missed deadline!')

    def liveliness_callback(self, event):
        self.get_logger().info('Publisher liveliness changed')

    def sub_deadline_callback(self, event):
        self.get_logger().warn('Subscriber detected deadline miss')

    def sub_liveliness_callback(self, event):
        self.get_logger().info('Subscriber detected liveliness change')

    def message_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

    def assert_liveliness(self):
        # Manually assert liveliness if using MANUAL_BY_TOPIC
        self.publisher.assert_liveliness()
```

## DDS Implementation Differences

### Available DDS Implementations

ROS2 supports multiple DDS implementations:

1. **Fast DDS (eProsima)**: Default in recent ROS2 versions
2. **Cyclone DDS (Eclipse)**: Lightweight, efficient
3. **RTI Connext DDS**: Commercial, feature-rich
4. **OpenSplice DDS**: Open source (deprecated)

### Selecting DDS Implementation

```bash
# Set DDS implementation via environment variable
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# or
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
# or
export RMW_IMPLEMENTATION=rmw_connext_cpp
```

### Implementation-Specific Configuration

```python
# Some implementations allow additional configuration
import os

class DDSConfigurableNode(Node):
    def __init__(self):
        # Check which implementation is being used
        rmw_implementation = os.environ.get('RMW_IMPLEMENTATION', 'default')

        super().__init__('dds_configurable_node')

        if 'fastrtps' in rmw_implementation:
            # Fast DDS specific optimizations
            self.get_logger().info('Using Fast DDS')
        elif 'cyclonedx' in rmw_implementation:
            # Cyclone DDS specific optimizations
            self.get_logger().info('Using Cyclone DDS')
        else:
            self.get_logger().info(f'Using {rmw_implementation}')
```

## Advanced QoS Scenarios

### Real-Time QoS Configuration

For time-critical applications:

```python
class RealTimeNode(Node):
    def __init__(self):
        super().__init__('real_time_node')

        # Ultra-low latency configuration
        self.rt_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            deadline=Duration(nanoseconds=10000000),  # 10ms deadline
            lifespan=Duration(nanoseconds=50000000)   # 50ms lifespan
        )

        self.rt_publisher = self.create_publisher(
            Twist, 'cmd_vel_rt', self.rt_qos)

    def publish_real_time_command(self, linear_vel, angular_vel):
        """Publish time-critical velocity commands"""
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel

        # Record publication time
        pub_time = self.get_clock().now()
        self.rt_publisher.publish(cmd)

        # Log for real-time analysis
        self.get_logger().debug(f'Published at {pub_time.nanoseconds}')
```

### Bandwidth-Conscious QoS

For network-constrained environments:

```python
class BandwidthConsciousNode(Node):
    def __init__(self):
        super().__init__('bandwidth_conscious_node')

        # Low-bandwidth configuration
        self.low_bw_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # High-frequency but low-importance data
        self.telemetry_pub = self.create_publisher(
            String, 'telemetry',
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                deadline=Duration(seconds=5)  # Data is stale after 5 seconds
            )
        )

        # Low-frequency but critical data
        self.critical_pub = self.create_publisher(
            String, 'critical',
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
        )
```

### Multi-Robot QoS Considerations

For coordinating multiple robots:

```python
class MultiRobotNode(Node):
    def __init__(self, robot_id):
        super().__init__(f'multi_robot_node_{robot_id}')
        self.robot_id = robot_id

        # Robot-specific topics with appropriate QoS
        self.status_pub = self.create_publisher(
            RobotStatus, f'robot_{robot_id}/status',
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE
            )
        )

        # Global coordination with transient durability
        self.coordination_sub = self.create_subscription(
            CoordinationMsg, 'coordination',
            self.coordination_callback,
            qos_profile=QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL  # New robots get initial coordination data
            )
        )

    def coordination_callback(self, msg):
        """Handle coordination messages from other robots"""
        if msg.target_robot == self.robot_id or msg.target_robot == 'all':
            # Process coordination message
            pass
```

## QoS Monitoring and Diagnostics

### QoS Status Monitoring

```python
class QoSMonitorNode(Node):
    def __init__(self):
        super().__init__('qos_monitor_node')

        # Publisher with monitoring
        self.publisher = self.create_publisher(
            String, 'monitored_topic',
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                liveliness=LivelinessPolicy.AUTOMATIC
            )
        )

        # Monitor QoS status
        self.qos_monitor_timer = self.create_timer(
            1.0, self.monitor_qos_status)

    def monitor_qos_status(self):
        """Monitor and log QoS status"""
        # Get publisher information
        pub_info = self.publisher.get_info()

        # Get subscription information
        # Note: This is conceptual - ROS2 doesn't directly expose all DDS info
        # In practice, you'd use tools like ros2 topic info

        self.get_logger().info('QoS monitoring active')
```

## Best Practices

### QoS Selection Guidelines

1. **Sensor Data**: Use BEST_EFFORT, VOLATILE, KEEP_LAST with small depth
2. **Command Data**: Use RELIABLE, VOLATILE, KEEP_LAST with appropriate depth
3. **Configuration Data**: Use RELIABLE, TRANSIENT_LOCAL, KEEP_LAST with depth=1
4. **Log Data**: Use RELIABLE, VOLATILE, KEEP_ALL with large depth

### Performance Considerations

- Higher QoS settings (RELIABLE, TRANSIENT_LOCAL) consume more resources
- BEST_EFFORT reduces network overhead but may lose data
- Large depth values consume more memory
- DEADLINE and LIFESPAN add processing overhead

### Troubleshooting QoS Issues

Common QoS problems and solutions:

```python
# Problem: No communication between nodes
# Solution: Check QoS compatibility
def check_qos_compatibility():
    pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    sub_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    # This is INCOMPATIBLE - subscriber expects RELIABLE but publisher is BEST_EFFORT

# Problem: Memory issues with large depth
# Solution: Use appropriate depth values
def appropriate_depth_selection():
    # For high-frequency sensor data
    sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

    # For low-frequency important data
    config_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
```

## Summary

This section covered Quality of Service (QoS) policies and Data Distribution Service (DDS) concepts in ROS2. QoS allows fine-tuning of communication behavior for different application requirements, while DDS provides the underlying communication infrastructure. Understanding these concepts is crucial for building robust, efficient robotic communication systems that meet real-time and reliability requirements.

Continue with [Examples](./examples/) to see practical implementations of these concepts.