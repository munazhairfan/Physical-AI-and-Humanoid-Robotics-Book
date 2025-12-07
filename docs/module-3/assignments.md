---
title: "Module 3 Assignments"
description: "Practical exercises for AI Perception & Sensor Fusion"
sidebar_label: "Assignments"
---

# Module 3 Assignments

## Overview

This page contains practical assignments designed to reinforce the concepts learned in the AI Perception & Sensor Fusion module. The assignments are organized by difficulty level: beginner, intermediate, and advanced.

## Beginner Assignments

### Assignment 1: Sensor Characteristics Analysis
**Difficulty**: Beginner
**Estimated Time**: 2 hours

#### Objective
Understand and compare the characteristics of different sensor types used in robotics.

#### Tasks
1. Research and create a comparison table of camera, LiDAR, and IMU specifications
2. Explain the advantages and disadvantages of each sensor type
3. Identify appropriate use cases for each sensor type
4. Discuss the complementary nature of different sensors

#### Validation Checkpoints
- [ ] Comparison table includes resolution, range, accuracy, and cost for each sensor
- [ ] Advantages and disadvantages are clearly explained
- [ ] At least 3 use cases identified for each sensor type
- [ ] Explanation of sensor complementarity is accurate

#### Expected Output
A written report with the comparison table and explanations.

### Assignment 2: Perception Pipeline Understanding
**Difficulty**: Beginner
**Estimated Time**: 2.5 hours

#### Objective
Map out and understand the components of a typical perception pipeline.

#### Tasks
1. Draw a block diagram of a perception pipeline
2. Explain the function of each stage
3. Identify inputs and outputs for each stage
4. Discuss potential failure modes at each stage

#### Validation Checkpoints
- [ ] Block diagram includes all major stages (data acquisition, preprocessing, feature extraction, etc.)
- [ ] Function of each stage is clearly explained
- [ ] Inputs and outputs are correctly identified for each stage
- [ ] Failure modes are reasonable and well-explained

#### Expected Output
A diagram with accompanying explanations for each component.

### Assignment 3: Feature Extraction Basics
**Difficulty**: Beginner
**Estimated Time**: 3 hours

#### Objective
Implement basic feature extraction techniques and understand their applications.

#### Tasks
1. Implement HOG (Histogram of Oriented Gradients) feature extraction
2. Apply HOG to sample images and visualize the results
3. Compare HOG features with simple color histogram features
4. Discuss when each feature type is most appropriate

#### Validation Checkpoints
- [ ] HOG implementation runs without errors
- [ ] Results are correctly visualized
- [ ] Comparison between HOG and color histograms is meaningful
- [ ] Discussion of appropriate use cases is accurate

#### Expected Output
Code implementation with visualizations and analysis report.

## Intermediate Assignments

### Assignment 4: Kalman Filter Implementation
**Difficulty**: Intermediate
**Estimated Time**: 6 hours

#### Objective
Implement and test a Kalman Filter for sensor fusion.

#### Tasks
1. Implement a basic Kalman Filter from scratch
2. Create a simulation environment with noisy sensor data
3. Test the filter with different noise levels
4. Compare the filtered results with ground truth
5. Analyze the filter's performance under various conditions

#### Validation Checkpoints
- [ ] Kalman Filter implementation is mathematically correct
- [ ] Simulation environment generates realistic noisy data
- [ ] Filter reduces noise effectively
- [ ] Performance analysis includes metrics like RMSE
- [ ] Results are compared with ground truth

#### Expected Output
Implementation code, test results, and performance analysis.

### Assignment 5: Camera-LiDAR Calibration
**Difficulty**: Intermediate
**Estimated Time**: 8 hours

#### Objective
Perform extrinsic calibration between camera and LiDAR sensors.

#### Tasks
1. Understand the mathematical model for camera-LiDAR calibration
2. Implement or use existing tools for calibration
3. Apply calibration to sample data
4. Validate the calibration accuracy
5. Demonstrate the impact of calibration on fusion performance

#### Validation Checkpoints
- [ ] Mathematical model is correctly understood and implemented
- [ ] Calibration process runs successfully
- [ ] Accuracy validation shows reasonable results
- [ ] Impact on fusion performance is demonstrated
- [ ] Error metrics are calculated and reported

#### Expected Output
Calibration implementation, validation results, and impact analysis.

### Assignment 6: Multi-Object Tracking
**Difficulty**: Intermediate
**Estimated Time**: 10 hours

#### Objective
Implement a multi-object tracking system using detection data.

#### Tasks
1. Implement a tracking-by-detection system
2. Use Kalman Filters for object prediction
3. Implement data association using Hungarian Algorithm
4. Test the system on sample detection data
5. Evaluate tracking performance using appropriate metrics

#### Validation Checkpoints
- [ ] Tracking system handles multiple objects correctly
- [ ] Kalman Filter prediction is implemented properly
- [ ] Data association works correctly
- [ ] Performance is evaluated using standard metrics (MOTA, MOTP)
- [ ] Results are visualized clearly

#### Expected Output
Implementation code, tracking results, and performance evaluation.

## Advanced Assignments

### Assignment 7: Deep Learning Sensor Fusion
**Difficulty**: Advanced
**Estimated Time**: 20 hours

#### Objective
Design and implement a deep learning approach for sensor fusion.

#### Tasks
1. Design a neural network architecture for sensor fusion
2. Prepare multi-modal data for training
3. Implement the fusion network
4. Train the network on appropriate datasets
5. Evaluate fusion performance against individual sensors
6. Analyze the learned fusion strategy

#### Validation Checkpoints
- [ ] Network architecture is appropriate for fusion task
- [ ] Data preparation handles multi-modal inputs correctly
- [ ] Training process converges and learns effectively
- [ ] Fusion performance exceeds individual sensor performance
- [ ] Analysis provides insights into learned fusion strategy

#### Expected Output
Network implementation, training results, performance comparison, and analysis.

### Assignment 8: Real-World Perception System
**Difficulty**: Advanced
**Estimated Time**: 30 hours

#### Objective
Design and implement a complete perception system for a specific robotic application.

#### Tasks
1. Choose a specific robotic application (e.g., autonomous driving, warehouse robot)
2. Design a perception system architecture for the application
3. Implement key components of the system
4. Integrate multiple sensors and fusion techniques
5. Test the system on real or simulated data
6. Evaluate system performance comprehensively
7. Analyze robustness to various conditions

#### Validation Checkpoints
- [ ] System architecture is appropriate for chosen application
- [ ] Multiple sensors are effectively integrated
- [ ] Fusion techniques are properly implemented
- [ ] Testing covers various scenarios
- [ ] Performance evaluation is comprehensive
- [ ] Robustness analysis identifies key failure modes
- [ ] System meets real-time requirements if applicable

#### Expected Output
Complete system implementation, comprehensive evaluation, and analysis report.

## Resources and Tools

### Recommended Software
- Python 3.7+ with NumPy, SciPy, OpenCV, Pandas
- ROS/ROS2 for robotics simulation
- PyTorch or TensorFlow for deep learning assignments
- Point Cloud Library (PCL) for 3D processing
- Open3D for 3D data visualization

### Datasets
- KITTI Vision Benchmark Suite
- nuScenes Dataset
- Waymo Open Dataset
- Custom simulation environments

### Libraries and Frameworks
- OpenCV for computer vision tasks
- PCL for point cloud processing
- scikit-learn for classical ML approaches
- PyTorch/TensorFlow for deep learning

## Submission Guidelines

### Format
- Code should be well-documented with clear comments
- Results should be clearly presented with appropriate visualizations
- Analysis should be thorough and well-reasoned
- Report should follow academic writing standards

### Evaluation Criteria
- Technical correctness of implementation
- Quality of results and analysis
- Understanding of underlying concepts
- Clarity of presentation
- Proper use of validation checkpoints

## Next Steps

After completing these assignments, you should have a solid understanding of AI perception and sensor fusion techniques and their practical implementation. Consider applying these skills to real robotics projects or further study in advanced topics.