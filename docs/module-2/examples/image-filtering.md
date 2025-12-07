---
title: Image Filtering Pipelines
sidebar_label: Image Filtering
description: Complete image filtering pipelines for robotic vision applications
slug: /module-2/examples/image-filtering
---

# Image Filtering Pipelines

## Summary

This example demonstrates complete image filtering pipelines for robotic vision applications. Students will learn to implement and combine various filtering techniques to enhance image quality, extract features, and prepare images for further processing in robotic perception systems.

## Learning Objectives

By the end of this example, students will be able to:
- Design and implement complete image filtering pipelines
- Combine multiple filtering techniques effectively
- Optimize filtering for specific robotic applications
- Evaluate filtering performance and quality
- Integrate filtering into perception workflows

## Table of Contents

1. [Basic Filtering Techniques](#basic-filtering-techniques)
2. [Advanced Filtering Pipelines](#advanced-filtering-pipelines)
3. [Application-Specific Filtering](#application-specific-filtering)
4. [Performance Optimization](#performance-optimization)
5. [Quality Assessment](#quality-assessment)

## Basic Filtering Techniques

### Gaussian Filtering

Gaussian filtering is commonly used for noise reduction while preserving edges:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter_pipeline(image, sigma=1.0, kernel_size=5):
    """
    Apply Gaussian filtering to reduce noise

    Args:
        image: Input image
        sigma: Standard deviation for Gaussian kernel
        kernel_size: Size of the Gaussian kernel

    Returns:
        Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered

def compare_gaussian_filters():
    """Compare different Gaussian filter parameters"""
    # Load sample image
    img = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Create a sample image if file doesn't exist
        img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        # Add some noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    # Different filter parameters
    sigmas = [0.5, 1.0, 2.0, 3.0]
    kernel_sizes = [3, 5, 7, 9]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    idx = 1
    for i in range(2):
        for j in range(1, 3):
            if idx < 5:
                filtered = gaussian_filter_pipeline(img, sigma=sigmas[idx-1], kernel_size=kernel_sizes[idx-1])
                axes[i, j].imshow(filtered, cmap='gray')
                axes[i, j].set_title(f'Sigma={sigmas[idx-1]}, Size={kernel_sizes[idx-1]}')
                axes[i, j].axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    compare_gaussian_filters()
```

### Edge Detection Filters

Edge detection is crucial for feature extraction:

```python
def sobel_edge_pipeline(image):
    """
    Apply Sobel edge detection

    Args:
        image: Input image (grayscale)

    Returns:
        Tuple of (magnitude, direction) of edges
    """
    # Calculate gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    # Normalize magnitude to 0-255 range
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude, direction

def canny_edge_pipeline(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection

    Args:
        image: Input image (grayscale)
        low_threshold: Low threshold for hysteresis
        high_threshold: High threshold for hysteresis

    Returns:
        Binary edge image
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return edges

def edge_detection_comparison():
    """Compare different edge detection methods"""
    # Load sample image
    img = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Create a sample image with geometric shapes
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        cv2.circle(img, (200, 200), 50, 255, -1)
        cv2.line(img, (0, 0), (300, 300), 255, 2)

    # Apply different edge detection methods
    sobel_mag, _ = sobel_edge_pipeline(img)
    canny_edges = canny_edge_pipeline(img)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_mag, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    edge_detection_comparison()
```

### Morphological Operations

Morphological operations are useful for binary image processing:

```python
def morphological_pipeline(image, operation='open', kernel_size=3, iterations=1):
    """
    Apply morphological operations

    Args:
        image: Input binary image
        operation: Type of operation ('open', 'close', 'erode', 'dilate')
        kernel_size: Size of structuring element
        iterations: Number of times to apply the operation

    Returns:
        Processed image
    """
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'open':
        # Opening: erosion followed by dilation
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        # Closing: dilation followed by erosion
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'erode':
        result = cv2.erode(image, kernel, iterations=iterations)
    elif operation == 'dilate':
        result = cv2.dilate(image, kernel, iterations=iterations)
    else:
        result = image

    return result

def morphological_pipeline_demo():
    """Demonstrate morphological operations"""
    # Create sample binary image with noise
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

    # Add noise
    noise = np.random.choice([0, 255], size=img.shape, p=[0.95, 0.05]).astype(np.uint8)
    img = cv2.bitwise_or(img, noise)

    # Apply different morphological operations
    opened = morphological_pipeline(img, 'open', kernel_size=3, iterations=1)
    closed = morphological_pipeline(img, 'close', kernel_size=3, iterations=1)
    cleaned = morphological_pipeline(opened, 'close', kernel_size=3, iterations=1)

    # Display results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original with Noise')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(opened, cmap='gray')
    plt.title('After Opening (Noise Removal)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(closed, cmap='gray')
    plt.title('After Closing (Gap Filling)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cleaned, cmap='gray')
    plt.title('Opening + Closing (Noise Removal + Gap Filling)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    morphological_pipeline_demo()
```

## Advanced Filtering Pipelines

### Multi-Stage Filtering Pipeline

Combine multiple filtering stages for enhanced results:

```python
class ImageFilteringPipeline:
    def __init__(self):
        self.pipeline = []
        self.results = {}

    def add_noise_reduction(self, method='gaussian', **kwargs):
        """Add noise reduction stage"""
        self.pipeline.append(('noise_reduction', method, kwargs))
        return self

    def add_edge_detection(self, method='canny', **kwargs):
        """Add edge detection stage"""
        self.pipeline.append(('edge_detection', method, kwargs))
        return self

    def add_morphology(self, method='open', **kwargs):
        """Add morphological operation stage"""
        self.pipeline.append(('morphology', method, kwargs))
        return self

    def add_custom_filter(self, func, **kwargs):
        """Add custom filter function"""
        self.pipeline.append(('custom', func, kwargs))
        return self

    def execute(self, image):
        """Execute the complete pipeline"""
        current_image = image.copy()

        for stage_idx, (stage_type, method, params) in enumerate(self.pipeline):
            if stage_type == 'noise_reduction':
                if method == 'gaussian':
                    sigma = params.get('sigma', 1.0)
                    kernel_size = params.get('kernel_size', 5)
                    current_image = gaussian_filter_pipeline(current_image, sigma, kernel_size)
                elif method == 'bilateral':
                    d = params.get('d', 9)
                    sigma_color = params.get('sigma_color', 75)
                    sigma_space = params.get('sigma_space', 75)
                    current_image = cv2.bilateralFilter(current_image, d, sigma_color, sigma_space)

            elif stage_type == 'edge_detection':
                if method == 'canny':
                    low_thresh = params.get('low_threshold', 50)
                    high_thresh = params.get('high_threshold', 150)
                    current_image = canny_edge_pipeline(current_image, low_thresh, high_thresh)
                elif method == 'sobel':
                    current_image, _ = sobel_edge_pipeline(current_image)

            elif stage_type == 'morphology':
                kernel_size = params.get('kernel_size', 3)
                iterations = params.get('iterations', 1)
                current_image = morphological_pipeline(current_image, method, kernel_size, iterations)

            elif stage_type == 'custom':
                current_image = method(current_image, **params)

            # Store intermediate result
            self.results[f'stage_{stage_idx}'] = current_image.copy()

        return current_image

def advanced_pipeline_example():
    """Example of advanced multi-stage pipeline"""
    # Create sample image
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.circle(img, (200, 200), 50, 255, -1)

    # Add noise
    noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Create pipeline
    pipeline = (ImageFilteringPipeline()
                .add_noise_reduction('gaussian', sigma=1.0, kernel_size=5)
                .add_edge_detection('canny', low_threshold=50, high_threshold=150)
                .add_morphology('close', kernel_size=3, iterations=1))

    # Execute pipeline
    result = pipeline.execute(img)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original with Noise')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pipeline.results['stage_0'], cmap='gray')
    plt.title('After Noise Reduction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='gray')
    plt.title('Final Result (Noise Reduction + Edge Detection + Closing)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    advanced_pipeline_example()
```

### Adaptive Filtering Pipeline

Create filters that adapt to image characteristics:

```python
def adaptive_threshold_pipeline(image, method='gaussian', block_size=11, c=2):
    """
    Apply adaptive thresholding

    Args:
        image: Input image (grayscale)
        method: 'gaussian' or 'mean'
        block_size: Size of the pixel neighborhood
        c: Constant subtracted from the mean or weighted mean

    Returns:
        Binary image
    """
    if method == 'gaussian':
        result = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, c)
    else:  # mean
        result = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, block_size, c)

    return result

def otsu_threshold_pipeline(image):
    """
    Apply Otsu's thresholding method

    Args:
        image: Input image (grayscale)

    Returns:
        Binary image and optimal threshold value
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def illumination_invariant_pipeline(image):
    """
    Create illumination invariant representation

    Args:
        image: Input color image (BGR)

    Returns:
        Illumination invariant image
    """
    # Convert to different color spaces for illumination invariance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Use the L channel (lightness) and normalize
    l_channel = lab[:,:,0]

    # Apply histogram equalization to enhance contrast
    enhanced = cv2.equalizeHist(l_channel)

    # Create new LAB image with enhanced L channel
    enhanced_lab = cv2.merge([enhanced, lab[:,:,1], lab[:,:,2]])

    # Convert back to BGR
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return result

def adaptive_filtering_demo():
    """Demonstrate adaptive filtering techniques"""
    # Create sample image with varying illumination
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (200, 200), 50, (128, 128, 128), -1)

    # Simulate varying illumination
    illumination = np.zeros((300, 300), dtype=np.float32)
    cv2.circle(illumination, (150, 150), 200, 1.0, -1)
    illumination = np.clip(illumination, 0.3, 1.0)  # Ensure minimum illumination

    # Apply illumination to each channel
    for i in range(3):
        img[:,:,i] = (img[:,:,i].astype(np.float32) * illumination).astype(np.uint8)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply different thresholding methods
    global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    adaptive_thresh = adaptive_threshold_pipeline(gray, method='gaussian', block_size=11, c=2)
    otsu_thresh = otsu_threshold_pipeline(gray)

    # Apply illumination invariant processing
    illum_invariant = illumination_invariant_pipeline(img)
    illum_invariant_gray = cv2.cvtColor(illum_invariant, cv2.COLOR_BGR2GRAY)
    illum_invariant_thresh = cv2.threshold(illum_invariant_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Display results
    plt.figure(figsize=(16, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image (Varying Illumination)')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(illum_invariant, cv2.COLOR_BGR2RGB))
    plt.title('Illumination Invariant')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(global_thresh, cmap='gray')
    plt.title('Global Threshold')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('Adaptive Threshold')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title('Otsu Threshold')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(illum_invariant_gray, cmap='gray')
    plt.title('Illumination Invariant Grayscale')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(illum_invariant_thresh, cmap='gray')
    plt.title('Otsu on Illumination Invariant')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    comparison = np.hstack([
        np.vstack([global_thresh, adaptive_thresh]),
        np.vstack([otsu_thresh, illum_invariant_thresh])
    ])
    plt.imshow(comparison, cmap='gray')
    plt.title('Comparison of Methods')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    adaptive_filtering_demo()
```

## Application-Specific Filtering

### Robot Navigation Filtering

Filtering optimized for robot navigation tasks:

```python
def navigation_preprocessing_pipeline(image):
    """
    Preprocessing pipeline optimized for robot navigation

    Args:
        image: Input navigation image

    Returns:
        Preprocessed image suitable for navigation
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)

    # Detect edges that are likely to be obstacles or landmarks
    edges = cv2.Canny(enhanced, 50, 150)

    # Enhance edges with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return enhanced, edges

def obstacle_detection_pipeline(image):
    """
    Pipeline for detecting obstacles in navigation

    Args:
        image: Input navigation image

    Returns:
        Binary image with obstacles highlighted
    """
    # Preprocess for navigation
    enhanced, edges = navigation_preprocessing_pipeline(image)

    # Find contours of potential obstacles
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create obstacle map
    obstacle_map = np.zeros_like(enhanced)

    # Filter contours by size and shape to identify obstacles
    for contour in contours:
        area = cv2.contourArea(contour)
        # Only consider contours of significant size
        if area > 100 and area < 0.3 * enhanced.shape[0] * enhanced.shape[1]:
            # Draw filled contour on obstacle map
            cv2.fillPoly(obstacle_map, [contour], 255)

    return obstacle_map

def navigation_filtering_demo():
    """Demonstrate navigation-specific filtering"""
    # Create sample navigation image
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add some obstacles
    cv2.rectangle(img, (100, 100), (200, 200), (100, 100, 100), -1)  # Obstacle
    cv2.circle(img, (400, 300), 50, (150, 150, 150), -1)  # Obstacle

    # Add some landmarks
    cv2.circle(img, (300, 150), 20, (255, 0, 0), -1)  # Landmark
    cv2.rectangle(img, (450, 100), (480, 130), (0, 255, 0), -1)  # Landmark

    # Apply navigation preprocessing
    enhanced, edges = navigation_preprocessing_pipeline(img)
    obstacles = obstacle_detection_pipeline(img)

    # Display results
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Navigation Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Navigation Edges')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(obstacles, cmap='gray')
    plt.title('Detected Obstacles')
    plt.axis('off')

    # Overlay obstacles on original
    overlay = img.copy()
    overlay[obstacles > 0] = [0, 0, 255]  # Red for obstacles
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Obstacles Overlay')
    plt.axis('off')

    # Show all processing steps
    combined = np.vstack([
        np.hstack([img, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)]),
        np.hstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.cvtColor(obstacles, cv2.COLOR_GRAY2BGR)])
    ])
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title('Processing Steps')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    navigation_filtering_demo()
```

### Object Recognition Filtering

Filtering optimized for object recognition tasks:

```python
def object_recognition_pipeline(image):
    """
    Preprocessing pipeline optimized for object recognition

    Args:
        image: Input image for object recognition

    Returns:
        Preprocessed image and feature descriptors
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply noise reduction while preserving important features
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Enhance important features with unsharp masking
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # Normalize the image
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

def feature_enhancement_pipeline(image):
    """
    Pipeline to enhance features for object recognition

    Args:
        image: Input image

    Returns:
        Enhanced image with highlighted features
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Laplacian for edge enhancement
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # Combine with original for enhancement
    enhanced = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final = clahe.apply(enhanced)

    return final

def object_recognition_demo():
    """Demonstrate object recognition filtering"""
    # Create sample image with objects
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # Add objects with different textures
    cv2.rectangle(img, (50, 50), (150, 150), (200, 200, 200), -1)  # Smooth object
    cv2.circle(img, (220, 80), 40, (150, 150, 150), -1)  # Round object

    # Add texture patterns
    for i in range(60, 140, 10):
        cv2.line(img, (i, 60), (i, 140), (100, 100, 100), 1)

    for i in range(180, 260, 5):
        cv2.line(img, (220, i-40), (260, i), (100, 100, 100), 1)

    # Apply object recognition preprocessing
    recognition_ready = object_recognition_pipeline(img)
    feature_enhanced = feature_enhancement_pipeline(img)

    # Compare with standard preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    standard = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Display results
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(standard, cmap='gray')
    plt.title('Standard Preprocessing')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(recognition_ready, cmap='gray')
    plt.title('Object Recognition Preprocessing')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(feature_enhanced, cmap='gray')
    plt.title('Feature Enhancement')
    plt.axis('off')

    # Show feature detection comparison
    # Detect features in both preprocessed versions
    orb = cv2.ORB_create()

    kp1 = orb.detect(recognition_ready, None)
    kp2 = orb.detect(feature_enhanced, None)

    img1_features = cv2.drawKeypoints(cv2.cvtColor(recognition_ready, cv2.COLOR_GRAY2BGR), kp1, None, color=(0, 255, 0))
    img2_features = cv2.drawKeypoints(cv2.cvtColor(feature_enhanced, cv2.COLOR_GRAY2BGR), kp2, None, color=(0, 255, 0))

    plt.subplot(2, 3, 6)
    combined_features = np.hstack([img1_features, img2_features])
    plt.imshow(cv2.cvtColor(combined_features, cv2.COLOR_BGR2RGB))
    plt.title('Feature Detection Comparison')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    object_recognition_demo()
```

## Performance Optimization

### Fast Filtering Techniques

Optimize filtering for real-time applications:

```python
import time

def fast_gaussian_blur(image, kernel_size=5):
    """Fast Gaussian blur using separable filters"""
    # Apply Gaussian blur with separable kernel (faster)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def fast_median_filter(image, kernel_size=3):
    """Fast median filter"""
    return cv2.medianBlur(image, kernel_size)

def optimized_pipeline(image):
    """Optimized pipeline for speed"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Fast noise reduction
    denoised = fast_median_filter(gray, 3)

    # Fast edge detection using Scharr (faster than Sobel for some cases)
    grad_x = cv2.Scharr(denoised, cv2.CV_16S, 1, 0)
    grad_y = cv2.Scharr(denoised, cv2.CV_16S, 0, 1)
    edges = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
    edges = np.uint8(edges / edges.max() * 255)

    return edges

def performance_comparison():
    """Compare performance of different filtering approaches"""
    # Create a larger test image
    img = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

    methods = {
        'Standard Gaussian': lambda x: cv2.GaussianBlur(x, (5, 5), 1.0),
        'Fast Gaussian': lambda x: fast_gaussian_blur(x, 5),
        'Optimized Pipeline': lambda x: optimized_pipeline(x)
    }

    results = {}

    for name, method in methods.items():
        start_time = time.time()

        # Run method 10 times to get average
        for _ in range(10):
            result = method(img)

        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        results[name] = avg_time

    # Print performance results
    print("Performance Comparison (average time per operation):")
    for name, avg_time in results.items():
        print(f"{name}: {avg_time*1000:.2f} ms")

    # Display a sample result
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    result = optimized_pipeline(img)
    plt.subplot(1, 3, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Optimized Pipeline Result')
    plt.axis('off')

    # Show timing results as a bar chart
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    times = [results[name] * 1000 for name in names]  # Convert to ms
    plt.bar(names, times)
    plt.title('Processing Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

# Example usage
if __name__ == "__main__":
    performance_comparison()
```

## Quality Assessment

### Filter Quality Metrics

Evaluate the quality of filtering operations:

```python
def calculate_psnr(original, filtered):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, filtered):
    """Calculate Structural Similarity Index"""
    # Simple implementation of SSIM (in practice, use scikit-image)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = np.mean(original)
    mu2 = np.mean(filtered)

    sigma1_sq = np.var(original)
    sigma2_sq = np.var(filtered)
    sigma12 = np.mean((original - mu1) * (filtered - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator
    return ssim

def filter_quality_assessment():
    """Assess quality of different filtering approaches"""
    # Create a test image with known characteristics
    original = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(original, (50, 50), (150, 150), 200, -1)
    cv2.circle(original, (200, 200), 50, 150, -1)

    # Add noise to create a degraded version
    noise = np.random.normal(0, 15, original.shape).astype(np.uint8)
    noisy = cv2.add(original, noise)

    # Apply different filters
    gaussian_filtered = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    bilateral_filtered = cv2.bilateralFilter(noisy, 9, 75, 75)
    median_filtered = cv2.medianBlur(noisy, 5)

    # Calculate quality metrics
    filters = {
        'Noisy': noisy,
        'Gaussian': gaussian_filtered,
        'Bilateral': bilateral_filtered,
        'Median': median_filtered
    }

    results = {}
    for name, filtered in filters.items():
        psnr = calculate_psnr(original, filtered)
        ssim = calculate_ssim(original, filtered)
        results[name] = {'PSNR': psnr, 'SSIM': ssim}

    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Noisy (PSNR: {results["Noisy"]["PSNR"]:.2f}, SSIM: {results["Noisy"]["SSIM"]:.3f})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gaussian_filtered, cmap='gray')
    axes[0, 2].set_title(f'Gaussian (PSNR: {results["Gaussian"]["PSNR"]:.2f}, SSIM: {results["Gaussian"]["SSIM"]:.3f})')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(bilateral_filtered, cmap='gray')
    axes[1, 0].set_title(f'Bilateral (PSNR: {results["Bilateral"]["PSNR"]:.2f}, SSIM: {results["Bilateral"]["SSIM"]:.3f})')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(median_filtered, cmap='gray')
    axes[1, 1].set_title(f'Median (PSNR: {results["Median"]["PSNR"]:.2f}, SSIM: {results["Median"]["SSIM"]:.3f})')
    axes[1, 1].axis('off')

    # Bar chart of metrics
    axes[1, 2].axis('off')
    ax_inset = fig.add_subplot(2, 3, 6)

    names = list(results.keys())
    psnr_values = [results[name]['PSNR'] for name in names]
    ssim_values = [results[name]['SSIM'] for name in names]

    x = np.arange(len(names))
    width = 0.35

    ax_inset.bar(x - width/2, psnr_values, width, label='PSNR', alpha=0.8)
    ax_inset.bar(x + width/2, ssim_values, width, label='SSIM', alpha=0.8)

    ax_inset.set_xlabel('Filter Type')
    ax_inset.set_ylabel('Metric Value')
    ax_inset.set_title('Filter Quality Metrics')
    ax_inset.set_xticks(x)
    ax_inset.set_xticklabels(names)
    ax_inset.legend()

    plt.tight_layout()
    plt.show()

    # Print detailed results
    print("Filter Quality Assessment:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:12} - PSNR: {metrics['PSNR']:6.2f} dB, SSIM: {metrics['SSIM']:5.3f}")

# Example usage
if __name__ == "__main__":
    filter_quality_assessment()
```

## Exercises

1. Create a filtering pipeline that enhances text in images for OCR applications
2. Build an adaptive filtering system that selects the best filter based on image content
3. Implement a real-time filtering pipeline optimized for embedded robotic systems
4. Develop a filtering pipeline that works with different lighting conditions

---

**Previous**: [Camera Stream](./camera-stream.md) | **Next**: [Object Detection](./object-detection.md)