---
title: Multi-Robot Communication Graph Diagram
sidebar_label: Multi-Robot Communication
---

# Multi-Robot Communication Graph Diagram

This diagram illustrates how multiple robots can communicate with each other and with centralized systems in a ROS2 environment.

```mermaid
graph TB
    subgraph "Central Control System"
        CC["Control Station<br/>- Fleet Management<br/>- Task Allocation<br/>- Data Aggregation"]
        RM["Resource Manager<br/>- Map Sharing<br/>- Path Coordination"]
        DB["Database<br/>- Logs<br/>- Maps<br/>- Task History"]
    end

    subgraph "Robot 1 (Delivery)"
        R1["Robot 1<br/>Delivery Bot"]
        R1N["Navigation<br/>Node"]
        R1S["Sensor<br/>Processing"]
        R1C["Control<br/>Node"]
    end

    subgraph "Robot 2 (Security)"
        R2["Robot 2<br/>Security Bot"]
        R2N["Navigation<br/>Node"]
        R2S["Sensor<br/>Processing"]
        R2C["Control<br/>Node"]
    end

    subgraph "Robot 3 (Maintenance)"
        R3["Robot 3<br/>Maintenance Bot"]
        R3N["Navigation<br/>Node"]
        R3S["Sensor<br/>Processing"]
        R3C["Control<br/>Node"]
    end

    %% Central system connections
    CC <-->|"Task Assignments<br/>Status Queries"| RM
    CC <-->|"Data Storage<br/>Retrieval"| DB

    %% Robot internal connections
    R1S <-->|"Sensor Data<br/>Processed Info"| R1N
    R1N <-->|"Goals<br/>Commands"| R1C
    R1C <-->|"Motor Commands<br/>Feedback"| R1S

    R2S <-->|"Sensor Data<br/>Processed Info"| R2N
    R2N <-->|"Goals<br/>Commands"| R2C
    R2C <-->|"Motor Commands<br/>Feedback"| R2S

    R3S <-->|"Sensor Data<br/>Processed Info"| R3N
    R3N <-->|"Goals<br/>Commands"| R3C
    R3C <-->|"Motor Commands<br/>Feedback"| R3S

    %% Inter-robot communication
    R1 <-->|"Collision Avoidance<br/>Coordination"| R2
    R2 <-->|"Task Coordination<br/>Area Sharing"| R3
    R1 <-->|"Resource Sharing<br/>Path Updates"| R3

    %% Robot to central communication
    R1 <-->|"Status Updates<br/>Sensor Data"| CC
    R2 <-->|"Status Updates<br/>Alerts"| CC
    R3 <-->|"Status Updates<br/>Maintenance Reports"| CC

    %% Resource sharing
    R1 <-->|"Map Updates"| RM
    R2 <-->|"Map Updates"| RM
    R3 <-->|"Map Updates"| RM

    style CC fill:#e1f5fe
    style RM fill:#f3e5f5
    style DB fill:#e8f5e8
    style R1 fill:#fff3e0
    style R2 fill:#fce4ec
    style R3 fill:#f1f8e9
```

## Multi-Robot Communication Patterns

```mermaid
graph LR
    subgraph "Decentralized Communication"
        A["Robot A"]
        B["Robot B"]
        C["Robot C"]
        A <-->|"Direct Communication"| B
        B <-->|"Direct Communication"| C
        A <-->|"Direct Communication"| C
    end

    subgraph "Centralized Communication"
        D["Robot D"]
        E["Robot E"]
        F["Robot F"]
        CS["Central Station"]
        D <-->|"Through Central"| CS
        E <-->|"Through Central"| CS
        F <-->|"Through Central"| CS
    end

    subgraph "Hybrid Communication"
        G["Robot G"]
        H["Robot H"]
        I["Robot I"]
        Z["Zone Controller"]
        G <-->|"Local Coordination"| Z
        H <-->|"Local Coordination"| Z
        Z <-->|"Global Coordination"| I
    end
```

## Namespace Organization in Multi-Robot Systems

```mermaid
graph TB
    subgraph "/tf (Transform Tree)"
        T1["/robot1/tf"]
        T2["/robot2/tf"]
        T3["/robot3/tf"]
        T1 -.->|"No Conflict"| T2
        T2 -.->|"No Conflict"| T3
        T1 -.->|"No Conflict"| T3
    end

    subgraph "/sensor_data"
        S1["/robot1/laser_scan"]
        S2["/robot2/laser_scan"]
        S3["/robot3/laser_scan"]
    end

    subgraph "/navigation"
        N1["/robot1/cmd_vel"]
        N2["/robot2/cmd_vel"]
        N3["/robot3/cmd_vel"]
    end

    style T1 fill:#e1f5fe
    style T2 fill:#e8f5e8
    style T3 fill:#fff3e0
```

## Communication Middleware Architecture

```mermaid
graph TB
    subgraph "Network Layer"
        NET["Network Infrastructure<br/>- WiFi<br/>- Ethernet<br/>- 5G"]
    end

    subgraph "DDS Domain"
        DDS1["DDS Instance 1<br/>Robot 1 Domain"]
        DDS2["DDS Instance 2<br/>Robot 2 Domain"]
        DDSC["DDS Instance C<br/>Central System Domain"]
    end

    subgraph "ROS2 Nodes"
        R1N["Robot 1<br/>ROS2 Nodes"]
        R2N["Robot 2<br/>ROS2 Nodes"]
        R3N["Robot 3<br/>ROS2 Nodes"]
        CSN["Control Station<br/>ROS2 Nodes"]
    end

    NET <--> DDS1
    NET <--> DDS2
    NET <--> DDSC

    DDS1 <--> R1N
    DDS2 <--> R2N
    DDSC <--> R3N
    DDSC <--> CSN

    R1N <-->|"Shared Topics"| CSN
    R2N <-->|"Shared Topics"| CSN
    R3N <-->|"Shared Topics"| CSN

    style NET fill:#e3f2fd
    style DDS1 fill:#e8eaf6
    style DDS2 fill:#e8eaf6
    style DDSC fill:#e8eaf6
    style R1N fill:#e1f5fe
    style R2N fill:#e8f5e8
    style R3N fill:#fff3e0
    style CSN fill:#f3e5f5
```

## Real-World Multi-Robot Scenario

```mermaid
sequenceDiagram
    participant R1 as Robot 1 (Warehouse)
    participant R2 as Robot 2 (Warehouse)
    participant R3 as Robot 3 (Warehouse)
    participant CC as Central Controller
    participant DB as Database

    Note over R1,R3: Warehouse Coordination Scenario

    CC->>R1: Assign Task (Pick Item A)
    CC->>R2: Assign Task (Pick Item B)
    CC->>R3: Assign Task (Transport Items)

    R1->>CC: Request Path to Item A
    R2->>CC: Request Path to Item B
    CC->>R1: Return Path (with obstacles)
    CC->>R2: Return Path (with obstacles)

    alt Path Conflict Detected
        CC->>R1: Wait (R2 has priority)
        R2->>CC: Path in use
        R1->>R1: Wait for clearance
    end

    R1->>DB: Update status (Item A acquired)
    R2->>DB: Update status (Item B acquired)
    R3->>R1: Request item pickup
    R1->>R3: Transfer item
    R3->>R2: Request item pickup
    R2->>R3: Transfer item

    R3->>CC: Transport complete
    CC->>DB: Update task completion
```

This diagram shows how multiple robots can effectively communicate in a ROS2 environment using proper namespacing, centralized coordination, and appropriate communication patterns to avoid conflicts while maintaining efficient operation.