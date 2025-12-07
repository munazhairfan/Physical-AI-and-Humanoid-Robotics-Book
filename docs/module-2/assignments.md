---
title: Assignments - Perception & Computer Vision
sidebar_label: Assignments
description: Practical assignments for the Perception & Computer Vision module
slug: /module-2/assignments
---

# Assignments - Perception & Computer Vision

## Summary

This section contains practical assignments designed to reinforce the concepts learned in the Perception & Computer Vision module. The assignments are organized by difficulty level and include expected outputs and validation checkpoints to help students assess their understanding.

## Learning Objectives

After completing these assignments, students will be able to:
- Apply perception concepts to practical problems
- Implement basic perception algorithms
- Evaluate the performance of perception systems
- Design perception pipelines for specific applications

## Assignment Structure

- **3 Beginner Tasks**: Focus on fundamental concepts and basic implementations
- **3 Intermediate Tasks**: Require integration of multiple concepts and techniques
- **2 Advanced Projects**: Comprehensive projects that simulate real-world perception challenges

---

## Beginner Assignments

### Assignment B1: Sensor Identification and Characteristics

**Objective**: Understand different sensor types and their characteristics.

**Task**:
1. Identify the following sensors from provided images/data:
   - RGB camera output
   - Depth camera output
   - LiDAR point cloud
   - Stereo camera pair
   - IMU readings

2. For each sensor, list:
   - Primary use cases
   - Advantages and limitations
   - Typical specifications (resolution, range, accuracy)

**Expected Output**:
- A table with sensor types and their characteristics
- Brief explanations of when to use each sensor type
- Examples of robotics applications for each sensor

**Validation Checkpoints**:
- [ ] Correctly identify all sensor types
- [ ] Accurately describe advantages and limitations
- [ ] Provide appropriate use cases for each sensor
- [ ] Include relevant specifications for each sensor type

**Estimated Time**: 2-3 hours

**Additional Resources**:
- Sample sensor data files are provided in the `data/sensors/` directory
- Reference specifications for common sensors are available in the appendix

---

### Assignment B2: Basic Image Filtering

**Objective**: Implement and compare basic image filtering techniques.

**Task**:
1. Load a sample image using OpenCV
2. Apply the following filters:
   - Gaussian blur (different kernel sizes)
   - Sobel edge detection (X and Y directions)
   - Canny edge detection (different thresholds)
3. Compare the results and explain when to use each filter

**Expected Output**:
- Python code implementing all filters
- Visual comparison of filtered images
- Written explanation of filter applications
- Performance analysis (computation time)

**Validation Checkpoints**:
- [ ] Successfully load and process images
- [ ] Implement all required filters correctly
- [ ] Provide visual comparison of results
- [ ] Explain appropriate use cases for each filter
- [ ] Include performance analysis

**Required Libraries**: OpenCV, NumPy, Matplotlib

**Estimated Time**: 3-4 hours

**Additional Resources**:
- Sample images are provided in the `data/images/` directory
- Reference implementations are available in the appendix

---

### Assignment B3: Coordinate Frame Transformations

**Objective**: Understand and implement coordinate transformations for camera systems.

**Task**:
1. Create a 3D coordinate system visualization
2. Implement transformations between:
   - World coordinates to camera coordinates
   - Camera coordinates to image coordinates
   - Image coordinates to pixel coordinates
3. Demonstrate with sample points
4. Calibrate a camera using provided images and apply transformations

**Expected Output**:
- Python code implementing all transformations
- Visualizations of coordinate systems
- Sample transformations with explanations
- Error analysis of transformation accuracy
- Camera calibration results with intrinsic/extrinsic parameters

**Validation Checkpoints**:
- [ ] Correctly implement all transformation matrices
- [ ] Visualize coordinate systems clearly
- [ ] Demonstrate transformations with examples
- [ ] Include error analysis
- [ ] Validate transformation accuracy
- [ ] Successfully calibrate camera parameters
- [ ] Apply transformations using calibrated parameters

**Required Libraries**: NumPy, Matplotlib, OpenCV

**Estimated Time**: 4-5 hours

**Additional Resources**:
- Calibration images are provided in the `data/calibration/` directory
- Sample 3D points are available in the `data/points/` directory

---

## Intermediate Assignments

### Assignment I1: Feature Detection and Matching

**Objective**: Implement a complete feature detection and matching pipeline.

**Task**:
1. Implement feature detection using multiple algorithms:
   - SIFT
   - ORB
   - Custom feature detector
2. Match features between image pairs
3. Evaluate matching performance
4. Apply to a simple object recognition task

**Expected Output**:
- Complete feature detection and matching implementation
- Performance comparison of different algorithms
- Visualization of matched features
- Object recognition results with accuracy metrics

**Validation Checkpoints**:
- [ ] Successfully detect features with all algorithms
- [ ] Implement feature matching correctly
- [ ] Compare algorithm performance quantitatively
- [ ] Apply to object recognition task
- [ ] Achieve minimum accuracy threshold

**Required Libraries**: OpenCV, NumPy, Matplotlib

**Estimated Time**: 6-8 hours

---

### Assignment I2: Camera Calibration and Undistortion

**Objective**: Calibrate a camera and correct lens distortion.

**Task**:
1. Generate synthetic calibration data or use provided calibration images
2. Implement camera calibration procedure
3. Extract intrinsic and extrinsic parameters
4. Apply undistortion to sample images
5. Evaluate calibration accuracy

**Expected Output**:
- Calibration implementation with parameter extraction
- Before/after comparison of distorted/undistorted images
- Quantitative evaluation of calibration quality
- Visualization of reprojection errors

**Validation Checkpoints**:
- [ ] Successfully calibrate camera parameters
- [ ] Apply undistortion correctly
- [ ] Quantify calibration accuracy
- [ ] Visualize reprojection errors
- [ ] Validate with multiple test images

**Required Libraries**: OpenCV, NumPy, Matplotlib

**Estimated Time**: 5-7 hours

---

### Assignment I3: Simple Object Detection Pipeline

**Objective**: Build a basic object detection pipeline using traditional computer vision.

**Task**:
1. Implement a color-based object detector
2. Add shape-based detection
3. Combine multiple detection methods
4. Evaluate detection performance
5. Test on provided image sequences

**Expected Output**:
- Complete object detection pipeline
- Performance metrics (precision, recall, F1-score)
- Visualization of detection results
- Comparison of different detection methods

**Validation Checkpoints**:
- [ ] Implement color-based detection
- [ ] Implement shape-based detection
- [ ] Combine detection methods effectively
- [ ] Achieve minimum performance thresholds
- [ ] Test on multiple image sequences

**Required Libraries**: OpenCV, NumPy, Matplotlib, scikit-image

**Estimated Time**: 6-8 hours

**Additional Resources**:
- Sample images for detection are provided in the `data/detection/` directory
- Ground truth annotations are available for evaluation

---

## Advanced Projects

### Project A1: Multi-Sensor Perception System

**Objective**: Design and implement a complete multi-sensor perception system for a mobile robot.

**Task**:
1. Integrate data from at least two sensor modalities (e.g., camera + LiDAR)
2. Implement sensor fusion for improved perception
3. Create a perception pipeline for obstacle detection and classification
4. Evaluate system performance in simulation or with provided datasets
5. Analyze the benefits of multi-sensor fusion

**Expected Output**:
- Complete multi-sensor perception implementation
- Performance comparison with single-sensor approaches
- Detailed analysis of fusion benefits and limitations
- Technical report on system design and evaluation

**Validation Checkpoints**:
- [ ] Successfully integrate multiple sensor types
- [ ] Implement effective sensor fusion
- [ ] Achieve improved performance over single sensors
- [ ] Provide comprehensive performance analysis
- [ ] Document system design and trade-offs

**Required Libraries**: OpenCV, NumPy, PCL (Point Cloud Library), scikit-learn

**Estimated Time**: 15-20 hours

**Additional Resources**:
- Multi-modal datasets are provided in the `data/multi-modal/` directory
- Sensor calibration files are available for camera-LiDAR alignment
- Evaluation tools are provided for performance comparison

### Project A2: Deep Learning-Based Perception System

**Objective**: Implement a deep learning-based perception system for a specific robotic task.

**Task**:
1. Select a perception task (e.g., semantic segmentation, object detection, pose estimation)
2. Prepare training data (or use existing datasets)
3. Implement and train a deep neural network
4. Evaluate performance on test data
5. Deploy the system and test in a simulated environment
6. Analyze the advantages and limitations of deep learning approaches

**Expected Output**:
- Trained deep learning model for perception task
- Performance evaluation on test data
- Deployment in simulated environment
- Analysis of deep learning vs. traditional approaches
- Technical documentation of implementation

**Validation Checkpoints**:
- [ ] Successfully train deep learning model
- [ ] Achieve acceptable performance on test data
- [ ] Deploy and test in simulated environment
- [ ] Compare with traditional approaches
- [ ] Document implementation and results

**Required Libraries**: PyTorch/TensorFlow, OpenCV, NumPy, Matplotlib

**Estimated Time**: 20-25 hours

**Additional Resources**:
- Training datasets are provided in the `data/training/` directory
- Pre-trained models are available for transfer learning
- Computing resources documentation is available in the appendix

---

## Submission Guidelines

### Format Requirements
- Code must be well-commented and follow standard practices
- Include a README with setup instructions
- Provide visualizations and results as specified
- Submit performance metrics where required

### Evaluation Criteria
- **Correctness** (40%): Implementation produces correct results
- **Code Quality** (25%): Clean, well-structured, and documented code
- **Analysis** (20%): Thorough analysis of results and performance
- **Creativity** (15%): Innovative approaches and extensions

### Late Submission Policy
- Submissions up to 24 hours late: 10% penalty
- Submissions 24-48 hours late: 25% penalty
- Submissions over 48 hours late: Not accepted without prior approval

---

## Resources and References

### Datasets
- [COCO Dataset](https://cocodataset.org/) for object detection
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) for autonomous driving perception
- [NYU Depth Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) for depth estimation
- [RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/) for visual SLAM

### Tools
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [ROS Perception Tutorials](http://wiki.ros.org/perception/Tutorials)
- [PyTorch Computer Vision](https://pytorch.org/vision/stable/index.html)

### Libraries Documentation
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

## Support and Questions

For questions about assignments:
- Post in the discussion forum with tag `#module-2-assignments`
- Office hours: Tuesdays and Thursdays, 2-4 PM
- TA support: Available via email within 24 hours

**Previous**: [Object Detection Examples](./examples/object-detection.md) | **Next**: [Summary](./summary.md)