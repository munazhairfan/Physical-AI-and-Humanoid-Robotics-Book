---
title: ROS2 Assignments
sidebar_label: Assignments
---

# ROS2 Assignments

This section contains assignments organized by difficulty level: beginner, intermediate, and advanced. Each assignment includes expected output or validation checkpoints.

## Beginner Tasks

### 1. Basic Publisher/Subscriber
Create a simple publisher that publishes "Hello, ROS2!" messages every second and a subscriber that prints received messages to the console.

**Requirements:**
- Create a publisher node that sends String messages to a topic called "hello_ros2"
- Create a subscriber node that listens to the "hello_ros2" topic and prints received messages
- Both nodes should run in separate terminals
- Use proper ROS2 node lifecycle management

**Expected Output:**
- Publisher should print "Publishing: Hello, ROS2! #n" every second
- Subscriber should print "I heard: Hello, ROS2! #n" for each message received

**Validation Checkpoints:**
- [ ] Both nodes can be run independently
- [ ] Messages are successfully published and received
- [ ] Nodes can be stopped gracefully with Ctrl+C

### 2. Parameter Server Usage
Create a node that uses ROS2 parameters to control its behavior (e.g., message frequency or content).

**Requirements:**
- Create a node that accepts a parameter for message content
- Create a node that accepts a parameter for publishing frequency
- Use the `declare_parameter()` method to define parameters
- Allow parameters to be set at launch time

**Expected Output:**
- Node behavior changes based on parameter values
- Parameters can be set via command line during launch

**Validation Checkpoints:**
- [ ] Parameters can be declared and accessed
- [ ] Node behavior changes when parameters are modified
- [ ] Default parameter values work correctly

### 3. Basic Service Implementation
Create a simple service that adds two numbers provided by a client.

**Requirements:**
- Implement a service server that adds two integers
- Implement a service client that sends requests to the server
- Use the `example_interfaces/srv/AddTwoInts` service type
- Handle the request/response cycle properly

**Expected Output:**
- Client sends two numbers (e.g., 5 and 3)
- Server responds with the sum (e.g., 8)
- Both client and server handle the communication properly

**Validation Checkpoints:**
- [ ] Service server can be reached by client
- [ ] Correct sum is returned for various inputs
- [ ] Communication is successful and synchronous

## Intermediate Tasks

### 4. Custom Message Type
Create and use a custom message type to send structured data between nodes.

**Requirements:**
- Define a custom message with at least 3 different field types (e.g., int, float, string)
- Create a publisher that sends custom messages
- Create a subscriber that receives and processes custom messages
- Include the message definition in a `msg/` directory
- Update package.xml and CMakeLists.txt appropriately

**Expected Output:**
- Custom message can be sent between nodes
- Data is correctly serialized and deserialized
- All fields of the custom message are accessible

**Validation Checkpoints:**
- [ ] Custom message compiles successfully
- [ ] Messages can be sent and received correctly
- [ ] All message fields are properly transmitted

### 5. Quality of Service (QoS) Comparison
Create nodes that demonstrate different QoS profiles and their effects.

**Requirements:**
- Create publishers with different QoS profiles (reliable vs. best effort)
- Create subscribers with matching and mismatching QoS profiles
- Compare message delivery rates and reliability
- Document the differences in behavior

**Expected Output:**
- Reliable QoS ensures all messages are delivered
- Best effort QoS may drop messages but has lower latency
- Mismatched QoS profiles may result in no communication

**Validation Checkpoints:**
- [ ] Different QoS profiles behave as expected
- [ ] Message delivery characteristics match QoS settings
- [ ] Incompatible QoS is properly handled

### 6. Launch File with Parameters
Create a launch file that starts multiple nodes with different configurations.

**Requirements:**
- Create a Python launch file that starts at least 2 different nodes
- Use launch arguments to parameterize the launch
- Include nodes with different namespaces
- Add conditional launching based on arguments

**Expected Output:**
- All nodes start successfully from a single launch command
- Nodes run with specified parameters and namespaces
- Launch arguments correctly modify node behavior

**Validation Checkpoints:**
- [ ] Launch file executes without errors
- [ ] All specified nodes start correctly
- [ ] Parameters are properly passed to nodes
- [ ] Namespacing works as expected

## Advanced Tasks

### 7. Action Server for Robot Movement
Implement an action server that controls a simulated robot's movement with feedback.

**Requirements:**
- Create an action server that accepts movement goals (position, orientation)
- Implement feedback mechanism showing progress toward goal
- Handle goal preemption and cancellation
- Create a client that sends goals and processes feedback
- Use appropriate action definition with goal, result, and feedback

**Expected Output:**
- Action server receives movement goals
- Client receives continuous feedback during execution
- Goal can be canceled if needed
- Result is returned upon completion

**Validation Checkpoints:**
- [ ] Action server accepts and processes goals
- [ ] Feedback is provided during execution
- [ ] Goals can be canceled appropriately
- [ ] Results are returned correctly

### 8. Multi-Robot Coordination System
Design and implement a system where multiple robots coordinate to achieve a common goal.

**Requirements:**
- Create at least 2 robot nodes with unique namespaces
- Implement communication between robots to avoid collisions
- Create a central coordinator node that assigns tasks
- Use proper namespacing and topic remapping
- Implement a shared resource management system

**Expected Output:**
- Multiple robots operate simultaneously without conflicts
- Tasks are distributed efficiently among robots
- Robots coordinate to avoid collisions
- Central coordinator manages the system effectively

**Validation Checkpoints:**
- [ ] Multiple robots can operate simultaneously
- [ ] Namespacing prevents topic conflicts
- [ ] Coordination logic prevents collisions
- [ ] Task distribution works correctly

### 9. ROS2 Package for Humanoid Robot Control
Develop a complete ROS2 package that interfaces with humanoid robot systems.

**Requirements:**
- Create a package with nodes for sensor processing, control, and communication
- Implement message types specific to humanoid robot control
- Include launch files for different operational modes
- Create services for high-level robot commands
- Implement proper error handling and safety mechanisms

**Expected Output:**
- Package compiles and runs without errors
- All nodes communicate properly using ROS2 patterns
- Control commands are processed correctly
- Safety mechanisms prevent dangerous operations

**Validation Checkpoints:**
- [ ] Package builds successfully
- [ ] All nodes communicate correctly
- [ ] Control system responds to commands
- [ ] Safety mechanisms are in place
- [ ] Launch files work as expected
- [ ] Custom messages are properly defined and used

## Submission Guidelines

For each assignment, students should submit:
1. Source code files
2. A brief report explaining the implementation
3. Screenshots or logs demonstrating successful execution
4. Any configuration files (e.g., launch files, URDF)
5. A summary of challenges encountered and solutions implemented

## Evaluation Criteria

- **Functionality**: Does the implementation work as specified?
- **Code Quality**: Is the code well-structured and documented?
- **ROS2 Best Practices**: Are ROS2 conventions and patterns followed correctly?
- **Error Handling**: Are errors and edge cases handled appropriately?
- **Documentation**: Is the code and approach well-documented?