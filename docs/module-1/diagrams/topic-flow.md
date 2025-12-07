---
title: Topic Communication Flow Diagram
sidebar_label: Topic Communication Flow
---

# Topic Communication Flow Diagram

This diagram illustrates the communication flow between publishers and subscribers in ROS2 using topics.

```mermaid
graph TB
    subgraph "ROS2 System"
        subgraph "Node A (Publisher)"
            A1["Publisher Node"]
            A2["Create Publisher<br/>for 'topic_name'"]
            A3["Publish Message<br/>to topic"]
        end

        subgraph "ROS2 Middleware (DDS)"
            DDS["DDS Implementation<br/>(CycloneDDS, FastDDS, etc.)"]
            Discovery["Discovery Service<br/>- Node Registration<br/>- Topic Matching"]
            Transport["Message Transport<br/>- Network Layer<br/>- Serialization"]
        end

        subgraph "Node B (Subscriber)"
            B1["Subscriber Node"]
            B2["Create Subscriber<br/>for 'topic_name'"]
            B3["Receive Message<br/>from topic"]
        end

        subgraph "Node C (Additional Subscriber)"
            C1["Another Subscriber"]
            C2["Also Subscribed<br/>to 'topic_name'"]
            C3["Receives Same<br/>Message"]
        end
    end

    %% Publisher flow
    A1 --> A2
    A2 --> A3

    %% Discovery process
    A2 -.-> Discovery
    B2 -.-> Discovery
    C2 -.-> Discovery

    %% Message flow through DDS
    A3 --> Transport
    Transport --> DDS
    DDS --> B3
    DDS --> C3

    %% Subscriber flows
    B2 --> B1
    B1 --> B3
    C2 --> C1
    C1 --> C3

    %% Labels
    style A1 fill:#e1f5fe
    style B1 fill:#e8f5e8
    style C1 fill:#fff3e0
    style DDS fill:#f3e5f5
    style Discovery fill:#e0e0e0
```

## Topic Communication Process

1. **Node Creation**: Publisher and subscriber nodes are created
2. **Entity Creation**: Nodes create publishers/subscribers for specific topics
3. **Discovery**: ROS2 discovery mechanism matches publishers with subscribers
4. **Message Publishing**: Publisher sends messages to the topic
5. **Middleware Transport**: DDS handles message transport and delivery
6. **Message Reception**: All subscribers receive copies of the message
7. **Callback Execution**: Subscribers process received messages

## Quality of Service (QoS) Considerations

The communication flow can be affected by QoS settings:

```mermaid
graph LR
    Publisher["Publisher with QoS Profile"]
    Middleware["DDS Middleware"]
    Subscriber["Subscriber with QoS Profile"]

    Publisher -->|Reliability: RELIABLE| Middleware
    Publisher -->|Durability: TRANSIENT_LOCAL| Middleware
    Publisher -->|History: KEEP_LAST| Middleware

    Middleware -->|Matching QoS| Subscriber
    Middleware -->|Incompatible QoS| X["Messages Dropped"]

    style Publisher fill:#e1f5fe
    style Middleware fill:#f3e5f5
    style Subscriber fill:#e8f5e8
    style X fill:#ffebee
```

## Multi-Node Topic Communication

```mermaid
graph TB
    subgraph "Robot System"
        subgraph "Sensor Node"
            S1["Lidar Node"]
            S2["Publish /scan"]
        end

        subgraph "Processing Node"
            P1["Perception Node"]
            P2["Subscribe /scan"]
            P3["Process Data"]
        end

        subgraph "Navigation Node"
            N1["Navigation Node"]
            N2["Subscribe /scan"]
            N3["Path Planning"]
        end

        subgraph "Visualization Node"
            V1["RViz Node"]
            V2["Subscribe /scan"]
            V3["Visualize Data"]
        end
    end

    S2 --> P2
    S2 --> N2
    S2 --> V2

    P2 --> P3
    N2 --> N3
    V2 --> V3

    style S1 fill:#e1f5fe
    style P1 fill:#e8f5e8
    style N1 fill:#fff3e0
    style V1 fill:#fce4ec
```

This diagram shows how a single publisher can send messages to multiple subscribers simultaneously, demonstrating the publish-subscribe pattern that makes ROS2 ideal for distributed robotic systems.