---
title: "Module 1: Perception & Computer Vision - Algorithms"
description: "Key algorithms in perception and computer vision for robotics"
sidebar_position: 4
slug: /module-1/algorithms
keywords: [robotics, perception, computer vision, algorithms, image processing, detection]
---

# Module 1: Perception & Computer Vision - Algorithms

## Introduction

This section covers the fundamental algorithms used in perception and computer vision for robotics. Understanding these algorithms is essential for implementing effective visual perception systems.

## Image Processing Algorithms

### Filtering Algorithms

#### Gaussian Blur
Reduces image noise and detail using a Gaussian kernel:

```python
import numpy as np
import cv2

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to an image

    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred image
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    # Create Gaussian kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)

    # Apply convolution
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred
```

#### Edge Detection

##### Canny Edge Detection
Multi-stage algorithm to detect edges:

```python
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Canny edge detection algorithm

    Args:
        image: Input grayscale image
        low_threshold: Low threshold for edge linking
        high_threshold: High threshold for edge detection

    Returns:
        Binary edge image
    """
    # 1. Noise reduction with Gaussian filter
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    # 2. Gradient calculation
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_angle = np.arctan2(grad_y, grad_x)

    # 3. Non-maximum suppression
    # (Simplified version - full implementation requires angle quantization)

    # 4. Double threshold and edge tracking
    edges = np.zeros_like(image)
    strong_edges = (gradient_magnitude > high_threshold)
    weak_edges = ((gradient_magnitude >= low_threshold) &
                  (gradient_magnitude <= high_threshold))

    edges[strong_edges] = 255

    # Edge tracking by hysteresis
    # (Simplified - full implementation connects weak edges to strong edges)

    return edges.astype(np.uint8)
```

### Morphological Operations

#### Erosion and Dilation
Basic morphological operations for binary image processing:

```python
def morphological_erosion(image, kernel):
    """
    Morphological erosion operation

    Args:
        image: Binary input image
        kernel: Structuring element

    Returns:
        Eroded image
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    eroded = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            # Erosion: minimum value in the region where kernel is 1
            eroded[i, j] = np.min(region[kernel == 1])

    return eroded

def morphological_dilation(image, kernel):
    """
    Morphological dilation operation

    Args:
        image: Binary input image
        kernel: Structuring element

    Returns:
        Dilated image
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    dilated = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            # Dilation: maximum value in the region where kernel is 1
            dilated[i, j] = np.max(region[kernel == 1])

    return dilated
```

## Feature Detection Algorithms

### Harris Corner Detection
Detects corners in images:

```python
def harris_corner_detection(image, k=0.04, window_size=3, threshold=0.01):
    """
    Harris corner detection algorithm

    Args:
        image: Input grayscale image
        k: Harris detector free parameter
        window_size: Size of the window for corner detection
        threshold: Threshold for corner response

    Returns:
        Image with corners marked
    """
    # Calculate gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate products of gradients
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Apply Gaussian window
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)

    # Calculate Harris response
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * trace * trace

    # Threshold and return corner locations
    corners = np.zeros_like(image)
    corners[R > threshold * R.max()] = 255

    return corners
```

### SIFT (Scale-Invariant Feature Transform)
Scale-invariant feature detection:

```python
class SIFTDetector:
    """
    Simplified SIFT-like feature detector
    """
    def __init__(self, num_octaves=4, num_scales=5, sigma=1.6):
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.scale_factor = 2**(1/num_scales)

    def generate_gaussian_pyramid(self, image):
        """
        Generate Gaussian pyramid for scale-space representation
        """
        pyramid = []
        current_image = image.astype(np.float32)

        for octave in range(self.num_octaves):
            scale_images = []
            sigma_current = self.sigma

            for scale in range(self.num_scales):
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(current_image, (0, 0), sigma_current)
                scale_images.append(blurred)

                # Increase sigma for next scale
                sigma_current *= self.scale_factor

            pyramid.append(scale_images)

            # Downsample for next octave
            if octave < self.num_octaves - 1:
                current_image = cv2.resize(current_image,
                                         (current_image.shape[1]//2,
                                          current_image.shape[0]//2))

        return pyramid

    def detect_keypoints(self, image):
        """
        Detect keypoints using difference of Gaussians
        """
        pyramid = self.generate_gaussian_pyramid(image)
        keypoints = []

        for octave_idx, octave in enumerate(pyramid):
            for scale_idx in range(1, len(octave)-1):
                # Compute Difference of Gaussians (DoG)
                prev_scale = octave[scale_idx - 1]
                curr_scale = octave[scale_idx]
                next_scale = octave[scale_idx + 1]

                # Find extrema in 3D space (x, y, scale)
                # This is a simplified version - full SIFT has more steps
                dog_current = curr_scale - prev_scale
                dog_next = next_scale - curr_scale

                # Find local maxima/minima
                # (Simplified implementation)
                local_max = (curr_scale > prev_scale) & (curr_scale > next_scale)
                local_min = (curr_scale < prev_scale) & (curr_scale < next_scale)

                # Combine and find significant points
                extrema = local_max | local_min
                extrema = extrema & (np.abs(curr_scale) > 0.03)  # Threshold

                # Get coordinates of extrema
                y_coords, x_coords = np.where(extrema)

                for y, x in zip(y_coords, x_coords):
                    keypoints.append((x, y, octave_idx, scale_idx))

        return keypoints
```

## Object Detection Algorithms

### Template Matching
Find template in image:

```python
def template_matching(image, template, method='sq_diff'):
    """
    Template matching algorithm

    Args:
        image: Input image
        template: Template image to find
        method: Matching method ('sq_diff', 'ccorr', 'ccoeff')

    Returns:
        Matching result and location
    """
    h, w = template.shape
    ih, iw = image.shape

    result = np.zeros((ih - h + 1, iw - w + 1))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            region = image[i:i+h, j:j+w]

            if method == 'sq_diff':
                # Sum of Squared Differences (lower is better)
                result[i, j] = np.sum((region - template) ** 2)
            elif method == 'ccorr':
                # Cross Correlation (higher is better)
                result[i, j] = np.sum(region * template)
            elif method == 'ccoeff':
                # Correlation Coefficient (higher is better)
                region_norm = region - np.mean(region)
                template_norm = template - np.mean(template)
                result[i, j] = np.sum(region_norm * template_norm) / (
                    np.sqrt(np.sum(region_norm**2) * np.sum(template_norm**2)) + 1e-10)

    # Find best match location
    if method == 'sq_diff':
        min_loc = np.unravel_index(np.argmin(result), result.shape)
        return result, min_loc
    else:
        max_loc = np.unravel_index(np.argmax(result), result.shape)
        return result, max_loc
```

### Hough Transform
Detect lines and circles:

```python
def hough_lines(image, rho_resolution=1, theta_resolution=1*np.pi/180, threshold=100):
    """
    Hough transform for line detection

    Args:
        image: Binary edge image
        rho_resolution: Distance resolution in pixels
        theta_resolution: Angle resolution in radians
        threshold: Minimum votes for a line

    Returns:
        Detected lines in (rho, theta) format
    """
    # Get edge coordinates
    y_coords, x_coords = np.where(image > 0)

    # Define parameter space
    max_rho = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    rhos = np.arange(-max_rho, max_rho, rho_resolution)
    thetas = np.arange(0, np.pi, theta_resolution)

    # Accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)))

    # Vote for each edge point
    for x, y in zip(x_coords, y_coords):
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.where(rhos == rho)[0]
            if len(rho_idx) > 0:
                accumulator[rho_idx[0], theta_idx] += 1

    # Find peaks in accumulator
    lines = []
    for rho_idx in range(len(rhos)):
        for theta_idx in range(len(thetas)):
            if accumulator[rho_idx, theta_idx] >= threshold:
                lines.append((rhos[rho_idx], thetas[theta_idx]))

    return lines
```

## Stereo Vision Algorithms

### Block Matching
Simple stereo matching algorithm:

```python
def block_matching_stereo(left_image, right_image, block_size=15, max_disparity=64):
    """
    Block matching algorithm for stereo vision

    Args:
        left_image: Left camera image
        right_image: Right camera image
        block_size: Size of matching blocks
        max_disparity: Maximum disparity to search

    Returns:
        Disparity map
    """
    h, w = left_image.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    half_block = block_size // 2

    for y in range(half_block, h - half_block):
        for x in range(half_block, w - half_block):
            # Get block from left image
            left_block = left_image[y-half_block:y+half_block+1,
                                   x-half_block:x+half_block+1]

            min_ssd = float('inf')
            best_disparity = 0

            # Search along the epipolar line in right image
            search_start = max(half_block, x - max_disparity)

            for d in range(search_start, x + 1):
                if d - half_block >= 0 and d + half_block < w:
                    right_block = right_image[y-half_block:y+half_block+1,
                                             d-half_block:d+half_block+1]

                    # Calculate Sum of Squared Differences
                    ssd = np.sum((left_block.astype(np.float32) -
                                 right_block.astype(np.float32)) ** 2)

                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_disparity = x - d

            disparity_map[y, x] = best_disparity

    return disparity_map
```

## Deep Learning Approaches

### Basic CNN for Classification
Simple convolutional neural network:

```python
import tensorflow as tf
from tensorflow import keras

def create_basic_cnn(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a basic CNN for image classification

    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # Classification head
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
```

## Performance Optimization

### Algorithm Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|----------------|------------------|----------|
| Gaussian Blur | O(n²m²) | O(1) | Noise reduction |
| Canny Edge Detection | O(nm) | O(nm) | Edge detection |
| Harris Corner Detection | O(nm) | O(nm) | Feature detection |
| Template Matching | O(n²m²) | O(1) | Object localization |
| Hough Transform | O(nm) | O(k) | Line detection |

Where n×m is image size and k is parameter space size.

## Summary

This section covered key algorithms in perception and computer vision for robotics, including image processing techniques, feature detection methods, object detection algorithms, stereo vision approaches, and deep learning methods. These algorithms form the computational foundation for robotic perception systems.

Continue with [Examples](./examples/) to see practical implementations of these algorithms.