---
title: "Module 1: Perception & Computer Vision - Core Concepts"
description: "Core concepts in perception and computer vision for robotics"
sidebar_position: 2
slug: /module-1/core-concepts
keywords: [robotics, perception, computer vision, core concepts, image processing]
---

# Module 1: Perception & Computer Vision - Core Concepts

## Introduction

This section covers the fundamental concepts that form the basis of robotic perception and computer vision. Understanding these concepts is essential for developing effective visual perception systems in robotics.

## Image Formation and Camera Models

### Pinhole Camera Model

The pinhole camera model describes the mathematical relationship between a 3D point in the world and its 2D projection on the image plane:

$$
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
=
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
$$

Where $(u, v)$ are image coordinates, $(X, Y, Z)$ are world coordinates, and $f_x, f_y$ are focal lengths.

### Distortion Models

Real cameras introduce distortions that need to be corrected:

- Radial distortion: Barrel and pincushion effects
- Tangential distortion: Due to lens and sensor misalignment

## Feature Detection and Description

### Key Concepts

- **Features**: Distinctive points in an image that can be reliably detected
- **Descriptors**: Mathematical representations of the local image patch around a feature
- **Matching**: Finding corresponding features between images

### Common Feature Detectors

- Harris corner detector
- SIFT (Scale-Invariant Feature Transform)
- SURF (Speeded Up Robust Features)
- ORB (Oriented FAST and Rotated BRIEF)

## Image Filtering and Enhancement

### Linear Filters

- Gaussian blur for noise reduction
- Edge detection filters (Sobel, Canny)
- Sharpening filters

### Non-linear Filters

- Median filter for salt-and-pepper noise
- Bilateral filter for edge-preserving smoothing

## Color Spaces and Representations

### RGB Color Space

- Red, Green, Blue channels
- Additive color mixing
- Device-dependent representation

### Alternative Color Spaces

- HSV (Hue, Saturation, Value): Better for color-based segmentation
- YUV: Separates luminance from chrominance
- Lab: Perceptually uniform color space

## Stereo Vision and Depth Perception

### Triangulation

Using two cameras to determine depth through triangulation:

$$
Z = \frac{fB}{x_1 - x_2}
$$

Where $f$ is focal length, $B$ is baseline, and $x_1 - x_2$ is disparity.

### Depth Estimation Techniques

- Stereo matching algorithms
- Structure from motion
- Time-of-flight sensors

## Object Detection and Recognition

### Traditional Approaches

- Template matching
- Haar cascades
- HOG (Histogram of Oriented Gradients)

### Deep Learning Approaches

- Convolutional Neural Networks (CNNs)
- Region-based CNNs (R-CNN, Fast R-CNN, Faster R-CNN)
- Single shot detectors (SSD, YOLO)

## Sensor Fusion in Perception

### Multi-Sensor Integration

Combining data from multiple sensors (cameras, LiDAR, radar) to improve perception robustness:

- Kalman filters for state estimation
- Particle filters for non-linear systems
- Bayesian networks for uncertainty modeling

## Performance Metrics

### Accuracy Measures

- Precision and recall
- F1-score
- Intersection over Union (IoU)
- Mean Average Precision (mAP)

### Computational Considerations

- Processing speed (frames per second)
- Memory usage
- Power consumption
- Real-time constraints

## Applications in Robotics

### Navigation

- Obstacle detection and avoidance
- Path planning based on visual input
- SLAM (Simultaneous Localization and Mapping)

### Manipulation

- Object recognition for grasping
- Pose estimation for manipulation
- Visual servoing

## Summary

This section covered the core concepts of perception and computer vision for robotics, including image formation, feature detection, filtering techniques, color spaces, stereo vision, object detection, and sensor fusion. These concepts form the foundation for more advanced perception systems in robotics.

Continue with [Architecture](./architecture) to learn about system design for perception pipelines.