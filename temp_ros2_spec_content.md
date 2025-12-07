### sp.specify("module.ros2")

# Module 1: Robotic Nervous System (ROS2)
## Specification

### Goal
Create a balanced, technically rich, and practically applicable module that teaches ROS2 as the "central nervous system" of robotics. The module must combine theory + examples + code + diagrams, and produce content that plugs cleanly into a Docusaurus documentation site.

### Output Format (Docusaurus-optimized)
The module must output content in the following Docusaurus-ready structure:

- `/docs/module-1/overview.md`
- `/docs/module-1/core-concepts.md`
- `/docs/module-1/architecture.md`
- `/docs/module-1/nodes-topics-services.md`
- `/docs/module-1/qos-dds.md`
- `/docs/module-1/examples/`
  - `publisher-subscriber.md`
  - `services-actions.md`
  - `launch-files.md`
- `/docs/module-1/diagrams/`
  - Architecture diagrams (Mermaid)
  - Topic flow diagrams (Mermaid)
- `/docs/module-1/assignments.md`
- `/docs/module-1/summary.md`

All text must be Markdown with:
- Headings (`#`, `##`, `###`)
- Code blocks (```bash, ```python, ```xml)
- Mermaid diagrams
- Docusaurus frontmatter (e.g. `--- title: ROS2 Overview ---`)

### Scope & Content Requirements

#### 1. **High-level Understanding**
- What ROS2 is and why it is the "nervous system" of robots
- Comparison: ROS1 → ROS2
- Real-world applications (drones, AMRs, humanoids, manipulators)
- ROS2 in safety-critical & distributed robotic systems

#### 2. **Architecture & Communication**
- ROS2 computation graph
- rclcpp & rclpy
- Middleware (DDS) overview
- Executors
- Launch architecture
- Namespaces & remapping

#### 3. **Core Concepts**
- Nodes
- Topics
- Publishers & Subscribers
- Services
- Actions
- Parameters
- QoS profiles (RELIABLE, BEST_EFFORT)
- DDS layers (discovery, data writers/readers)

#### 4. **Practical Hands-On Sections**
Each practical page must include:
- Step-by-step instructions
- Screenshots or terminal output excerpts
- Code in Python + C++ (both)
- Explanation before code
- Diagrams explaining message flow

Required examples:
- Simple publisher/subscriber
- Custom message creation
- Service + client example
- Action server + client
- Multi-node launch file
- Namespacing & remapping
- QoS effect demonstration

#### 5. **Mermaid Diagrams**
Must include:
- ROS2 architecture overview
- Topic communication flow
- Service handshake
- Action workflow
- Multi-robot communication graph

#### 6. **Assignments**
Include:
- 3 beginner tasks
- 3 intermediate tasks
- 2 advanced projects (mini robot systems)
- All tasks must include expected output or validation checkpoints

#### 7. **Integration Requirements**
- All files must use **Docusaurus frontmatter**
- Each page must include:
  - Summary block
  - Learning objectives
  - Code + explanation
  - Diagram
- Links between pages must use relative linking (`../`)
- Keep content modular — each section stands alone

### Tone & Style
- Balanced: technical + practical + visual
- Beginner-friendly wording but not oversimplified
- Explanations must be intuitive AND accurate
- Avoid unnecessary jargon
- Always introduce the concept → example → explanation → diagram

### Completion Definition
The module is complete when:
- All specified pages are generated
- The content compiles cleanly in a Docusaurus site
- All code examples are runnable
- All diagrams render
- Assignments are included
- Summary page connects the module into the larger curriculum

### Return
Produce a plan using `sp.plan()` next, breaking this module into tasks.