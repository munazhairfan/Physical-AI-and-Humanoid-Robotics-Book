# Data Model: Module 6: Motion Planning Algorithms (A*, RRT, PRM, Trajectory Optimization)

**Feature**: `007-motion-planning` | **Date**: 2025-12-11

## Core Entities for Motion Planning Systems

### Environment Entity
- `id`: string (unique identifier for the environment)
- `name`: string (display name for the environment)
- `dimensions`: object (x, y, z bounds of the space)
- `obstacles`: array (geometric descriptions of obstacles)
- `start_position`: object (starting coordinates for planning)
- `goal_position`: object (goal coordinates for planning)
- `resolution`: number (discretization resolution for grid-based planning)

### Path Entity
- `id`: string (unique identifier for the path)
- `waypoints`: array (sequence of (x, y, z) coordinates)
- `cost`: number (total path cost/distance)
- `smoothed`: boolean (whether path has been smoothed)
- `computed_at`: datetime (timestamp when path was computed)
- `algorithm_used`: string (A*, RRT, PRM, etc.)

### Node Entity (for graph-based algorithms)
- `id`: string (unique identifier for the node)
- `position`: object (x, y, z coordinates)
- `parent_id`: string (id of parent node in search tree)
- `cost_from_start`: number (cost from start node)
- `heuristic_to_goal`: number (estimated cost to goal)
- `total_cost`: number (f = g + h for A*)
- `connections`: array (adjacent nodes)

### Planning Algorithm Configuration Entity
- `id`: string (configuration unique identifier)
- `algorithm_type`: string (A*, RRT, PRM, etc.)
- `parameters`: object (algorithm-specific parameters)
  - For A*: `heuristic_weight`, `allow_diagonal`
  - For RRT*: `step_size`, `max_iterations`, `goal_bias`
  - For PRM: `num_samples`, `connection_radius`
- `collision_checker`: object (configuration for collision detection)
- `termination_criteria`: object (when to stop planning)

### Robot Configuration Entity
- `id`: string (robot configuration unique identifier)
- `type`: string (differential, omni, manipulator, etc.)
- `dimensions`: object (size and shape of the robot)
- `kinematic_constraints`: object (movement limitations)
- `dynamic_constraints`: object (acceleration, velocity limits)
- `configuration_space`: object (C-space representation)

### Trajectory Entity
- `id`: string (trajectory unique identifier)
- `waypoints`: array (sequence of positions with timing)
- `velocities`: array (velocity at each waypoint)
- `accelerations`: array (acceleration at each waypoint)
- `timing`: array (time at each waypoint)
- `optimization_cost`: number (cost of trajectory optimization)
- `valid`: boolean (whether trajectory satisfies all constraints)

## Relationships
- Environment contains multiple Obstacles and defines start/goal positions
- Path is computed within an Environment using a Planning Algorithm Configuration
- Path consists of multiple Waypoint positions
- Trajectory includes timing information for a Path
- Robot Configuration defines kinematic and dynamic constraints for planning

## Data Flow
1. Environment is defined with obstacles and start/goal positions
2. Planning Algorithm Configuration is specified
3. Motion planning algorithm computes a Path
4. Path may be optimized to create a Trajectory
5. Robot follows the trajectory considering its Configuration constraints