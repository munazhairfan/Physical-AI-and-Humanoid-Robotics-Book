---
title: Action Workflow Diagram
sidebar_label: Action Workflow
---

# Action Workflow Diagram

This diagram illustrates the workflow of ROS2 actions, which are used for long-running tasks that may provide feedback and can be canceled.

```mermaid
sequenceDiagram
    participant C as Action Client
    participant RMW as ROS2 Middleware
    participant S as Action Server

    Note over C,S: Action Goal Request
    C->>S: Send Goal Request
    activate S
    S->>C: Goal Accepted Response
    C->>S: Request Feedback (optional)
    S->>C: Feedback Response (continuous)

    Note over C,S: Task Execution with Feedback
    loop Task Execution
        S->>C: Send Feedback
        C->>C: Process Feedback
    end

    Note over C,S: Goal Completion
    S->>C: Result Response
    deactivate S
    C->>C: Process Result

    Note over C,S: Optional Cancellation
    C->>S: Cancel Goal Request
    S->>C: Cancel Response
```

## Action Architecture

```mermaid
graph TB
    subgraph "ROS2 Action System"
        subgraph "Client Node"
            C1["Action Client"]
            C2["Create Action Client<br/>for 'action_name'"]
            C3["Send Goal Request"]
            C4["Receive Feedback"]
            C5["Receive Result or Cancel Response"]
        end

        subgraph "ROS2 Middleware (DDS)"
            DDS["DDS Implementation"]
            GoalTransport["Goal Transport"]
            FeedbackTransport["Feedback Transport"]
            ResultTransport["Result Transport"]
            CancelTransport["Cancel Transport"]
        end

        subgraph "Server Node"
            S1["Action Server"]
            S2["Create Action Server<br/>for 'action_name'"]
            S3["Process Goals"]
            S4["Send Feedback Continuously"]
            S5["Send Result or Handle Cancel"]
        end
    end

    %% Client flow
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5

    %% Server flow
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5

    %% Action communication flows
    C3 --> GoalTransport
    GoalTransport --> DDS
    DDS --> S3

    S4 --> FeedbackTransport
    FeedbackTransport --> DDS
    DDS --> C4

    S5 --> ResultTransport
    ResultTransport --> DDS
    DDS --> C5

    %% Cancel flow
    C1 -.->|"Cancel Request"| CancelTransport
    CancelTransport -.-> DDS
    DDS -.-> S1

    style C1 fill:#e1f5fe
    style S1 fill:#e8f5e8
    style DDS fill:#f3e5f5
```

## Action States and Transitions

```mermaid
stateDiagram-v2
    [*] --> PENDING : Goal Sent
    PENDING --> ACTIVE : Goal Accepted
    ACTIVE --> EXECUTING : Goal Processing
    EXECUTING --> SUCCEEDED : Task Completed
    EXECUTING --> CANCELLED : Goal Cancelled
    EXECUTING --> ABORTED : Task Failed
    PENDING --> RECALLING : Goal Recalled
    ACTIVE --> RECALLING : Goal Recalled
    RECALLING --> RECALLED : Goal Recalled
    CANCELLED --> [*]
    SUCCEEDED --> [*]
    ABORTED --> [*]
    RECALLED --> [*]

    note right of EXECUTING
        Feedback sent
        continuously during
        execution
    end
```

## Real-World Action Example: Navigation

```mermaid
sequenceDiagram
    participant NavClient as Navigation Client
    participant RMW as ROS2 Middleware
    participant NavServer as Navigation Server

    Note over NavClient,NavServer: Navigation to Goal
    NavClient->>NavServer: Send Navigation Goal (x, y, theta)
    activate NavServer
    NavServer->>NavClient: Goal Accepted
    loop Navigation Execution
        NavServer->>NavClient: Feedback (progress, distance)
        alt Obstacle Detected
            NavServer->>NavServer: Recalculate Path
        end
    end
    NavServer->>NavClient: Result (success/failure)
    deactivate NavServer

    Note over NavClient,NavServer: Optional Cancel
    NavClient->>NavServer: Cancel Navigation
    NavServer->>NavClient: Cancellation Confirmed
```

## Action vs Service vs Topic Comparison

```mermaid
graph TB
    subgraph "Topics"
        T["Continuous Data Flow<br/>- No Response<br/>- Async"]
    end

    subgraph "Services"
        S["Request/Response<br/>- Sync<br/>- Quick Operations"]
    end

    subgraph "Actions"
        A["Long-Running Tasks<br/>- Feedback<br/>- Cancelable<br/>- Async"]
    end

    T -->|"Simple data"| Simple["Sensor Data"]
    S -->|"Quick request"| Quick["Get Parameter"]
    A -->|"Complex task"| Complex["Navigation, Trajectory Execution"]

    style T fill:#e3f2fd
    style S fill:#e8f5e8
    style A fill:#fff3e0
    style Simple fill:#bbdefb
    style Quick fill:#c8e6c9
    style Complex fill:#ffccbc
```

This diagram shows how actions in ROS2 provide a sophisticated communication pattern for long-running tasks that require feedback, progress updates, and the ability to be canceled, making them ideal for robot navigation, manipulation, and other complex operations.