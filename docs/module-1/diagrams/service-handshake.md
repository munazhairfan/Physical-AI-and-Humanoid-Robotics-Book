---
title: Service Handshake Diagram
sidebar_label: Service Handshake
---

# Service Handshake Diagram

This diagram illustrates the request/response communication pattern in ROS2 using services.

```mermaid
sequenceDiagram
    participant C as Client Node
    participant RMW as ROS2 Middleware
    participant S as Service Server Node

    Note over C,S: Service Discovery Phase
    C->>RMW: Register Client for service
    S->>RMW: Register Service Server
    RMW->>C: Service discovery complete
    RMW->>S: Service discovery complete

    Note over C,S: Request/Response Phase
    C->>S: Service Request (blocking)
    activate S
    S->>S: Process Request
    S->>C: Service Response
    deactivate S

    Note over C,S: Another Request
    C->>S: Service Request (blocking)
    activate S
    S->>S: Process Request
    S->>C: Service Response
    deactivate S
```

## Service Communication Process

1. **Service Discovery**: Server registers service, client discovers it
2. **Request Initiation**: Client sends request to server
3. **Request Processing**: Server processes the request
4. **Response Delivery**: Server sends response back to client
5. **Synchronous Wait**: Client waits for response (blocking)

## Service Architecture

```mermaid
graph TB
    subgraph "ROS2 Service System"
        subgraph "Client Node"
            C1["Service Client"]
            C2["Create Service Client<br/>for 'service_name'"]
            C3["Send Request<br/>& Wait for Response"]
        end

        subgraph "ROS2 Middleware (DDS)"
            DDS["DDS Implementation"]
            Discovery["Service Discovery<br/>- Client Registration<br/>- Server Registration"]
            RequestTransport["Request Transport"]
            ResponseTransport["Response Transport"]
        end

        subgraph "Server Node"
            S1["Service Server"]
            S2["Create Service Server<br/>for 'service_name'"]
            S3["Wait for Requests<br/>& Process Them"]
            S4["Send Response"]
        end
    end

    %% Client flow
    C1 --> C2
    C2 --> C3

    %% Server flow
    S1 --> S2
    S2 --> S3
    S3 --> S4

    %% Discovery
    C2 -.-> Discovery
    S2 -.-> Discovery

    %% Request/Response flow
    C3 --> RequestTransport
    RequestTransport --> DDS
    DDS --> S3

    S4 --> ResponseTransport
    ResponseTransport --> DDS
    DDS --> C3

    style C1 fill:#e1f5fe
    style S1 fill:#e8f5e8
    style DDS fill:#f3e5f5
```

## Service vs Topic Comparison

```mermaid
graph LR
    subgraph "Topics (Async)"
        T1["Publisher"]
        T2["Publish Data"]
        T3["Subscribers Receive"]
    end

    subgraph "Services (Sync)"
        S1["Client"]
        S2["Send Request"]
        S3["Wait for Response"]
        S4["Server Processes"]
        S5["Send Response"]
    end

    T1 --> T2
    T2 --> T3

    S1 --> S2
    S2 --> S3
    S1 -.-> S4
    S4 --> S5
    S5 --> S1

    style T1 fill:#e1f5fe
    style T3 fill:#e8f5e8
    style S1 fill:#fff3e0
    style S4 fill:#f3e5f5
```

## Real-World Service Example

```mermaid
sequenceDiagram
    participant NavClient as Navigation Client
    participant RMW as ROS2 Middleware
    participant MapServer as Map Server

    Note over NavClient,MapServer: Get Map Service
    NavClient->>MapServer: GetMap Request
    activate MapServer
    MapServer->>MapServer: Load map from storage
    MapServer->>NavClient: Return map data
    deactivate MapServer

    Note over NavClient,MapServer: Set Parameters Service
    NavClient->>MapServer: SetParameters Request
    activate MapServer
    MapServer->>MapServer: Update internal parameters
    MapServer->>NavClient: Confirmation response
    deactivate MapServer
```

This diagram shows how services provide synchronous request/response communication in ROS2, which is ideal for operations that need a guaranteed response, such as getting robot state, setting parameters, or triggering specific actions.