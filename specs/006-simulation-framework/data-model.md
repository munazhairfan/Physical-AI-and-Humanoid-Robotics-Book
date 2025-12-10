# Data Model: Module 5: Robotics Simulation Frameworks (Gazebo, Webots, PyBullet)

**Feature**: `006-simulation-framework` | **Date**: 2025-12-11

## Core Entities for Simulation Systems

### Robot Model Entity
- `id`: string (unique identifier for the robot model)
- `name`: string (display name for the robot)
- `type`: string (differential, omni, manipulator, etc.)
- `urdf_file`: string (path to URDF definition)
- `sdf_file`: string (path to SDF definition)
- `visual_meshes`: array (3D model files for visualization)
- `collision_meshes`: array (3D model files for collision detection)
- `joints`: array (definitions of joint properties)
- `links`: array (definitions of link properties)

### Simulation Environment Entity
- `id`: string (unique identifier for the environment)
- `name`: string (display name for the environment)
- `world_file`: string (path to world/SDF file)
- `gravity`: object (gravity vector x, y, z)
- `physics_engine`: string (ODE, Bullet, Simbody)
- `time_step`: number (simulation time step in seconds)
- `real_time_factor`: number (real-time simulation speed)
- `objects`: array (static and dynamic objects in the environment)

### Physics Configuration Entity
- `id`: string (configuration unique identifier)
- `engine`: string (ODE, Bullet, Simbody)
- `solver_type`: string (quick, world, etc.)
- `iterations`: number (solver iterations)
- `cfm`: number (constraint force mixing parameter)
- `erp`: number (error reduction parameter)
- `contact_surface_layer`: number (contact surface layer thickness)

### Sensor Configuration Entity
- `id`: string (sensor unique identifier)
- `type`: string (lidar, camera, imu, gps, etc.)
- `parent_link`: string (which robot link the sensor is attached to)
- `pose`: object (position and orientation relative to parent)
- `update_rate`: number (sensor update frequency in Hz)
- `parameters`: object (sensor-specific parameters)
- `noise_model`: object (noise characteristics)

### Simulation State Entity
- `simulation_id`: string (unique identifier for the simulation run)
- `timestamp`: datetime
- `robot_states`: array (position, orientation, velocities for all robots)
- `sensor_readings`: array (current readings from all sensors)
- `physics_properties`: object (current physics parameters)
- `environment_state`: object (state of environment objects)

## Relationships
- Simulation Environment contains multiple Robot Models
- Robot Model has associated Physics Configuration
- Robot Model has multiple Sensor Configurations
- Simulation State captures the state of all entities at a given time

## Data Flow
1. Robot models are loaded into simulation environment
2. Physics and sensor configurations are applied
3. Simulation runs with specific time stepping
4. Sensor data is generated based on physics simulation
5. Robot states and sensor readings are captured as Simulation State