---
title: "Module 2: Robotic Nervous System (ROS2) - Nodes, Topics, Services"
description: "Core communication mechanisms in ROS2: nodes, topics, and services"
sidebar_position: 4
slug: /module-2/nodes-topics-services
keywords: [robotics, ros2, nodes, topics, services, communication, middleware]
---

# Module 2: Robotic Nervous System (ROS2) - Nodes, Topics, Services

## Introduction

This section covers the three fundamental communication mechanisms in ROS2: nodes, topics, and services. These mechanisms form the backbone of robotic communication, enabling different components to interact and coordinate effectively.

## Nodes

### Node Fundamentals

A node is an executable process that works as part of a ROS2 system. It's the basic unit of computation in ROS2:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        # Initialize node with name
        super().__init__('my_node_name')

        # Log messages to console
        self.get_logger().info('Node initialized successfully')

        # Node cleanup callback
        self.on_destruction = self.cleanup

    def cleanup(self):
        self.get_logger().info('Cleaning up node resources')
```

### Node Lifecycle

Nodes follow a specific lifecycle that allows for more sophisticated management:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleManager(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_manager')

    def on_configure(self, state):
        self.get_logger().info('Configuring node')
        # Initialize resources here
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node')
        # Start operations
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating node')
        # Pause operations
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up node')
        # Release resources
        return TransitionCallbackReturn.SUCCESS
```

### Node Parameters

Nodes can be configured using parameters:

```python
class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'turtlebot4')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('sensors_enabled', True)
        self.declare_parameter('topics_to_subscribe', ['camera', 'lidar'])

        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.sensors_enabled = self.get_parameter('sensors_enabled').value
        self.topics = self.get_parameter('topics_to_subscribe').value

        # Add parameter callback for runtime changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name == 'max_velocity':
                if param.value <= 0:
                    return SetParametersResult(successful=False,
                                             reason='Max velocity must be positive')

        return SetParametersResult(successful=True)

    def update_parameters(self):
        """Method to update parameters at runtime"""
        self.set_parameters([Parameter('max_velocity', Parameter.Type.PARAMETER_DOUBLE, 2.0)])
```

## Topics

### Publisher Implementation

Publishers send messages to topics:

```python
from std_msgs.msg import String, Int32, Float64
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np

class DataPublisher(Node):
    def __init__(self):
        super().__init__('data_publisher')

        # Create publishers with different message types
        self.string_pub = self.create_publisher(String, 'chatter', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.laser_pub = self.create_publisher(LaserScan, 'scan', 10)

        # Timer to publish data periodically
        self.timer = self.create_timer(0.1, self.publish_data)
        self.counter = 0

    def publish_data(self):
        """Publish various types of data"""
        # Publish string message
        string_msg = String()
        string_msg.data = f'Hello World: {self.counter}'
        self.string_pub.publish(string_msg)

        # Publish velocity command
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd_msg.angular.z = 0.2  # Rotate at 0.2 rad/s
        self.cmd_vel_pub.publish(cmd_msg)

        # Publish laser scan (simulated)
        laser_msg = LaserScan()
        laser_msg.angle_min = -np.pi/2
        laser_msg.angle_max = np.pi/2
        laser_msg.angle_increment = np.pi/180  # 1 degree
        laser_msg.ranges = [2.0] * 181  # 181 points, 2m range
        laser_msg.range_min = 0.1
        laser_msg.range_max = 10.0
        self.laser_pub.publish(laser_msg)

        self.counter += 1
```

### Subscriber Implementation

Subscribers receive messages from topics:

```python
class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')

        # Create subscribers with callbacks
        self.string_sub = self.create_subscription(
            String, 'chatter', self.string_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

    def string_callback(self, msg):
        """Callback for string messages"""
        self.get_logger().info(f'Received string: {msg.data}')

    def cmd_vel_callback(self, msg):
        """Callback for velocity commands"""
        self.get_logger().info(
            f'Received velocity: linear={msg.linear.x}, angular={msg.angular.z}')

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        if len(msg.ranges) > 0:
            min_range = min([r for r in msg.ranges if r > msg.range_min])
            self.get_logger().info(f'Min obstacle distance: {min_range:.2f}m')
```

### Custom Message Types

Creating and using custom messages:

```python
# Custom message definition (RobotStatus.msg):
# ---
# string robot_name
# float64 battery_level
# bool is_charging
# geometry_msgs/Point position
# string[] active_sensors

from std_msgs.msg import Header
from geometry_msgs.msg import Point
from my_robot_msgs.msg import RobotStatus  # Custom message

class StatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')
        self.status_pub = self.create_publisher(RobotStatus, 'robot_status', 10)
        self.timer = self.create_timer(1.0, self.publish_status)

    def publish_status(self):
        """Publish custom robot status message"""
        msg = RobotStatus()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.robot_name = 'turtlebot4'
        msg.battery_level = 85.5
        msg.is_charging = False
        msg.position = Point(x=1.0, y=2.0, z=0.0)
        msg.active_sensors = ['camera', 'lidar', 'imu']

        self.status_pub.publish(msg)
```

### Topic Management

Advanced topic operations:

```python
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import ListParameters

class TopicManager(Node):
    def __init__(self):
        super().__init__('topic_manager')

        # Get information about topics
        self.list_topics_client = self.create_client(
            ListParameters, 'list_parameters')

        # Topic monitoring
        self.topic_monitor = self.create_timer(5.0, self.monitor_topics)

        # Dynamic topic creation
        self.dynamic_publishers = {}

    def get_topic_info(self):
        """Get information about current topics"""
        topic_names_and_types = self.get_topic_names_and_types()
        for name, types in topic_names_and_types:
            self.get_logger().info(f'Topic: {name}, Type: {types}')

    def create_dynamic_publisher(self, topic_name, msg_type):
        """Create publisher dynamically"""
        if topic_name not in self.dynamic_publishers:
            publisher = self.create_publisher(msg_type, topic_name, 10)
            self.dynamic_publishers[topic_name] = publisher
            return publisher
        return self.dynamic_publishers[topic_name]

    def monitor_topics(self):
        """Monitor topic health and connections"""
        topic_names_and_types = self.get_topic_names_and_types()

        # Count publishers and subscribers for each topic
        for topic_name, _ in topic_names_and_types:
            publishers = self.get_publishers_info_by_topic(topic_name)
            subscribers = self.get_subscriptions_info_by_topic(topic_name)

            self.get_logger().info(
                f'Topic {topic_name}: {len(publishers)} pub, {len(subscribers)} sub')
```

## Services

### Service Server Implementation

Services provide request-response communication:

```python
from example_interfaces.srv import AddTwoInts, Trigger
from my_robot_msgs.srv import NavigateToPose, SetMode
from geometry_msgs.msg import Pose

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')

        # Create multiple services
        self.add_srv = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_callback)
        self.nav_srv = self.create_service(
            NavigateToPose, 'navigate_to_pose', self.navigate_callback)
        self.mode_srv = self.create_service(
            SetMode, 'set_mode', self.set_mode_callback)

        self.current_mode = 'idle'

    def add_callback(self, request, response):
        """Add two integers service"""
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response

    def navigate_callback(self, request, response):
        """Navigation service"""
        target_pose = request.pose
        self.get_logger().info(
            f'Navigating to ({target_pose.position.x}, {target_pose.position.y})')

        # Simulate navigation
        import time
        time.sleep(2)  # Simulate navigation time

        response.success = True
        response.message = f'Reached pose ({target_pose.position.x}, {target_pose.position.y})'
        return response

    def set_mode_callback(self, request, response):
        """Mode setting service"""
        old_mode = self.current_mode
        self.current_mode = request.mode

        if self.current_mode in ['idle', 'navigation', 'manipulation', 'charging']:
            response.success = True
            response.message = f'Mode changed from {old_mode} to {self.current_mode}'
        else:
            response.success = False
            response.message = f'Invalid mode: {request.mode}'

        return response
```

### Service Client Implementation

Clients call services:

```python
class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')

        # Create clients for different services
        self.add_client = self.create_client(AddTwoInts, 'add_two_ints')
        self.nav_client = self.create_client(NavigateToPose, 'navigate_to_pose')
        self.mode_client = self.create_client(SetMode, 'set_mode')

        # Wait for services to be available
        while not self.add_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Add service not available, waiting...')

        while not self.nav_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Navigation service not available, waiting...')

        # Call services
        self.call_add_service()
        self.call_navigation_service()
        self.call_mode_service()

    def call_add_service(self):
        """Call the add service"""
        request = AddTwoInts.Request()
        request.a = 10
        request.b = 20

        future = self.add_client.call_async(request)
        # Add callback for async response
        future.add_done_callback(self.add_response_callback)

    def call_navigation_service(self):
        """Call the navigation service"""
        request = NavigateToPose.Request()
        request.pose.position.x = 5.0
        request.pose.position.y = 3.0
        request.pose.orientation.w = 1.0

        future = self.nav_client.call_async(request)
        future.add_done_callback(self.nav_response_callback)

    def call_mode_service(self):
        """Call the mode service"""
        request = SetMode.Request()
        request.mode = 'navigation'

        future = self.mode_client.call_async(request)
        future.add_done_callback(self.mode_response_callback)

    def add_response_callback(self, future):
        """Handle add service response"""
        try:
            response = future.result()
            self.get_logger().info(f'Add service result: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def nav_response_callback(self, future):
        """Handle navigation service response"""
        try:
            response = future.result()
            self.get_logger().info(f'Navigation result: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Navigation service failed: {e}')

    def mode_response_callback(self, future):
        """Handle mode service response"""
        try:
            response = future.result()
            self.get_logger().info(f'Mode change result: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Mode service failed: {e}')
```

### Asynchronous Service Handling

Advanced service patterns:

```python
import asyncio
from rclpy.executors import MultiThreadedExecutor

class AsyncServiceServer(Node):
    def __init__(self):
        super().__init__('async_service_server')

        # Use callback group for handling services
        self.service_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.long_running_srv = self.create_service(
            Trigger, 'long_running_task',
            self.long_running_callback,
            callback_group=self.service_callback_group)

    def long_running_callback(self, request, response):
        """Handle long-running service call asynchronously"""
        self.get_logger().info('Starting long-running task...')

        # Simulate long-running task
        import time
        time.sleep(5)  # This would block other service calls

        response.success = True
        response.message = 'Long-running task completed'
        self.get_logger().info('Long-running task completed')
        return response

# For truly asynchronous processing, use background threads
import threading

class ThreadedServiceServer(Node):
    def __init__(self):
        super().__init__('threaded_service_server')
        self.service_queue = queue.Queue()
        self.result_dict = {}

        self.complex_srv = self.create_service(
            SetMode, 'complex_computation', self.complex_callback)

        # Start background thread for processing
        self.processing_thread = threading.Thread(target=self.process_service_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def complex_callback(self, request, response):
        """Queue complex computation for background processing"""
        task_id = f'task_{int(time.time() * 1000)}'

        # Add task to queue
        self.service_queue.put((task_id, request))

        # Wait for result (with timeout)
        timeout = 10.0  # seconds
        start_time = time.time()

        while task_id in self.result_dict and time.time() - start_time < timeout:
            time.sleep(0.1)

        if task_id in self.result_dict:
            response = self.result_dict.pop(task_id)
        else:
            response.success = False
            response.message = 'Task timed out'

        return response

    def process_service_queue(self):
        """Background thread to process service queue"""
        while rclpy.ok():
            try:
                task_id, request = self.service_queue.get(timeout=1.0)

                # Perform complex computation
                result = self.perform_complex_computation(request)

                # Store result
                self.result_dict[task_id] = result
            except queue.Empty:
                continue
```

## Communication Patterns

### Producer-Consumer Pattern

Using topics for producer-consumer relationships:

```python
class DataProducer(Node):
    def __init__(self):
        super().__init__('data_producer')
        self.data_pub = self.create_publisher(String, 'data_stream', 100)
        self.timer = self.create_timer(0.5, self.produce_data)
        self.data_counter = 0

    def produce_data(self):
        """Produce data and publish"""
        msg = String()
        msg.data = f'Data item {self.data_counter}'
        self.data_pub.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.data_counter += 1

class DataConsumer(Node):
    def __init__(self):
        super().__init__('data_consumer')
        self.data_sub = self.create_subscription(
            String, 'data_stream', self.consume_data, 100)
        self.processed_count = 0

    def consume_data(self, msg):
        """Consume and process data"""
        self.get_logger().info(f'Consumed: {msg.data}')
        self.processed_count += 1
        # Process the data here
```

### Request-Reply Pattern

Using services for request-reply interactions:

```python
class RequestReplyServer(Node):
    def __init__(self):
        super().__init__('request_reply_server')
        self.query_srv = self.create_service(
            Trigger, 'query_status', self.handle_query)
        self.robot_status = {'battery': 80.0, 'position': (1.0, 2.0)}

    def handle_query(self, request, response):
        """Handle status query"""
        import json
        status_json = json.dumps(self.robot_status)
        response.success = True
        response.message = status_json
        return response

class RequestReplyClient(Node):
    def __init__(self):
        super().__init__('request_reply_client')
        self.query_client = self.create_client(Trigger, 'query_status')

        while not self.query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Query service not available...')

        self.query_timer = self.create_timer(5.0, self.send_query)

    def send_query(self):
        """Send query request"""
        request = Trigger.Request()
        future = self.query_client.call_async(request)
        future.add_done_callback(self.query_response_callback)

    def query_response_callback(self, future):
        """Handle query response"""
        try:
            response = future.result()
            if response.success:
                import json
                status = json.loads(response.message)
                self.get_logger().info(f'Robot status: {status}')
        except Exception as e:
            self.get_logger().error(f'Query failed: {e}')
```

## Performance Considerations

### Topic Performance Optimization

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class OptimizedNode(Node):
    def __init__(self):
        super().__init__('optimized_node')

        # Different QoS profiles for different data types

        # High-frequency sensor data (lose some data if needed)
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Critical command data (must not lose)
        cmd_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Historical data (keep many samples)
        log_qos = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL
        )

        self.sensor_pub = self.create_publisher(LaserScan, 'scan', sensor_qos)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', cmd_qos)
        self.log_pub = self.create_publisher(String, 'log', log_qos)
```

## Summary

This section covered the core communication mechanisms in ROS2: nodes as computational units, topics for publish-subscribe communication, and services for request-response interactions. We explored implementation patterns, advanced features like lifecycle nodes and custom messages, and performance optimization techniques. These mechanisms enable the creation of complex, distributed robotic systems where components can communicate effectively.

Continue with [QoS and DDS](./qos-dds) to learn about Quality of Service and Data Distribution Service concepts.