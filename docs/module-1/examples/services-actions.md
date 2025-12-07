---
title: Services and Actions Example
sidebar_label: Services and Actions
---

# Services and Actions Example

This example demonstrates the synchronous communication pattern using Services and the long-running task pattern using Actions in ROS2.

## Overview

ROS2 provides three main communication patterns:
1. **Topics (Publish/Subscribe)**: Asynchronous, continuous data flow
2. **Services**: Synchronous request/response pattern
3. **Actions**: Asynchronous long-running tasks with feedback

## Service Example

Services provide a request/response communication pattern. The client sends a request and waits for a response from the server.

### Service Definition

First, create a service definition file `srv/AddTwoInts.srv`:

```
int64 a
int64 b
---
int64 sum
```

### Service Server

```python
#!/usr/bin/env python3

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Request received: {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Client

```python
#!/usr/bin/env python3

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


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
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Action Example

Actions are used for long-running tasks that may take significant time to complete. They provide feedback during execution and can be canceled.

### Action Definition

Create an action definition file `action/Fibonacci.action`:

```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

### Action Server

```python
#!/usr/bin/env python3

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class MinimalActionServer(Node):

    def __init__(self):
        super().__init__('minimal_action_server')
        self._goal_handle = None
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
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

        # Append the seeds for the sequence
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]
        feedback_msg.sequence = feedback_msg.partial_sequence

        goal_handle.publish_feedback(feedback_msg)

        # Start executing the action
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            if not goal_handle.is_active:
                self.get_logger().info('Goal aborted')
                return Fibonacci.Result()

            if feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1] > 1000:
                # If the next number would be too large, break
                break

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            self.get_logger().info(f'Publishing feedback: {feedback_msg.partial_sequence}')

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence

        self.get_logger().info(f'Returning result: {result.sequence}')

        return result


def main(args=None):
    rclpy.init(args=args)

    minimal_action_server = MinimalActionServer()

    executor = MultiThreadedExecutor()
    rclpy.spin(minimal_action_server, executor=executor)

    minimal_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Action Client

```python
#!/usr/bin/env python3

import time

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class MinimalActionClient(Node):

    def __init__(self):
        super().__init__('minimal_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci',
            callback_group=ReentrantCallbackGroup())

    def send_goal(self, order):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info(f'Sending goal request with order: {order}')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.partial_sequence}')


def main(args=None):
    rclpy.init(args=args)

    action_client = MinimalActionClient()

    # We create a MultiThreadedExecutor to handle both the action client and any other callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(action_client)

    # We send the goal and wait for a result
    action_client.send_goal(5)

    # We spin the executor in a separate thread to handle callbacks
    spin_thread = action_client.create_rate(1)  # Placeholder - in real code we'd use threading
    executor.spin()

    action_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Running the Examples

### For Services:

1. **Run the service server**:
   ```bash
   ros2 run your_package_name service_server
   ```

2. **In another terminal, run the service client**:
   ```bash
   ros2 run your_package_name service_client
   ```

### For Actions:

1. **Run the action server**:
   ```bash
   ros2 run your_package_name action_server
   ```

2. **In another terminal, run the action client**:
   ```bash
   ros2 run your_package_name action_client
   ```

## Comparison of Communication Patterns

| Pattern | Type | Use Case | Example |
|---------|------|----------|---------|
| Topics | Asynchronous | Continuous data flow | Sensor data, robot state |
| Services | Synchronous | Request/response | Get robot position, set parameter |
| Actions | Asynchronous with feedback | Long-running tasks | Navigation, trajectory execution |

## Key Concepts Demonstrated

1. **Service Creation**: How to create service servers and clients
2. **Action Creation**: How to create action servers and clients with feedback
3. **Request/Response**: Synchronous communication pattern
4. **Long-Running Tasks**: Asynchronous tasks with feedback and cancellation
5. **Goal Handling**: Managing action goals, results, and cancellation
6. **Feedback Mechanisms**: Providing progress updates during long operations

## When to Use Each Pattern

- **Use Topics** when you need continuous data flow or event notifications
- **Use Services** for simple request/response interactions that complete quickly
- **Use Actions** for tasks that take time to complete and may need to provide feedback or be cancellable

This example shows how to implement the three main communication patterns in ROS2, demonstrating when and how to use each one for different types of robot communication needs.