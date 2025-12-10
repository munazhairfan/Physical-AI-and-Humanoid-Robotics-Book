# Quickstart Guide: Module 6: Motion Planning Algorithms (A*, RRT, PRM, Trajectory Optimization)

**Feature**: `007-motion-planning` | **Date**: 2025-12-11

## Getting Started with Motion Planning Algorithms

This quickstart guide will help you set up the basic environment for implementing and testing motion planning algorithms.

### Prerequisites
- Python 3.6+
- pip package manager
- Basic understanding of graph algorithms and geometry

### Setup Instructions

1. **Install Required Dependencies**
   ```bash
   pip install numpy matplotlib scipy
   # Optional: For advanced motion planning
   pip install ompl
   # For visualization
   pip install plotly networkx
   ```

2. **Basic A* Implementation Example**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from queue import PriorityQueue

   def a_star(grid, start, goal):
       # Define possible movements (8 directions)
       movements = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
       
       # Initialize open and closed lists
       open_list = PriorityQueue()
       open_list.put((0, start))
       came_from = {}
       g_score = {start: 0}
       f_score = {start: heuristic(start, goal)}
       
       # Main loop
       while not open_list.empty():
           current = open_list.get()[1]
           
           if current == goal:
               return reconstruct_path(came_from, current)
           
           for dx, dy in movements:
               neighbor = (current[0] + dx, current[1] + dy)
               
               # Check bounds and obstacles
               if (0 <= neighbor[0] < grid.shape[0] and 
                   0 <= neighbor[1] < grid.shape[1] and 
                   grid[neighbor[0], neighbor[1]] == 0):
                   
                   tentative_g_score = g_score[current] + euclidean_distance(current, neighbor)
                   
                   if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                       came_from[neighbor] = current
                       g_score[neighbor] = tentative_g_score
                       f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                       open_list.put((f_score[neighbor], neighbor))
       
       return []  # No path found

   def heuristic(a, b):
       # Euclidean distance
       return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

   def euclidean_distance(a, b):
       return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

   def reconstruct_path(came_from, current):
       path = [current]
       while current in came_from:
           current = came_from[current]
           path.append(current)
       return path[::-1]
   ```

3. **Basic Visualization Example**
   ```python
   def visualize_path(grid, path, start, goal):
       plt.figure(figsize=(10, 10))
       plt.imshow(grid, cmap='Greys', origin='lower')
       
       # Extract path coordinates
       path_x, path_y = zip(*path)
       
       # Plot path
       plt.plot(path_y, path_x, 'r-', linewidth=2, label='Path')
       plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
       plt.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
       
       plt.title('A* Path Planning')
       plt.legend()
       plt.grid(True)
       plt.show()
   ```

### Key Concepts to Master
1. Grid-based vs continuous space planning
2. Heuristic functions and their properties
3. Sampling strategies for high-dimensional spaces
4. Trajectory optimization techniques
5. Collision detection algorithms

### Next Steps
- Complete the full module documentation in `/docs/module-6/`
- Try the hands-on examples in `/docs/module-6/examples/`
- Work through the assignments in `/docs/module-6/assignments.md`

### Troubleshooting
- Ensure proper bounds checking in your grid implementations
- Verify that your heuristic function is admissible (never overestimates)
- Check that your collision detection is accurate
- For RRT-based algorithms, adjust sampling parameters if paths are not found efficiently