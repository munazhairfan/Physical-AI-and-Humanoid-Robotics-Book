---
title: Image Processing Techniques
sidebar_label: Image Processing
description: Fundamental image processing techniques for robotic vision
slug: /module-2/image-processing
---

# Image Processing Techniques

## Summary

This section covers fundamental image processing techniques essential for robotic vision systems. Students will learn about image acquisition, preprocessing, filtering, feature extraction, and coordinate transformations that form the foundation of computer vision applications in robotics.

## Learning Objectives

By the end of this section, students will be able to:
- Understand the fundamentals of digital image formation and representation
- Apply various filtering techniques for image enhancement and noise reduction
- Extract meaningful features from images for robotic perception
- Perform camera coordinate transformations and understand geometric relationships
- Implement basic image processing pipelines for robotic applications

## Table of Contents

1. [Image Acquisition and Representation](#image-acquisition-and-representation)
2. [Image Preprocessing](#image-preprocessing)
3. [Filtering Techniques](#filtering-techniques)
4. [Feature Extraction and Matching](#feature-extraction-and-matching)
5. [Camera Transformations and Coordinate Frames](#camera-transformations-and-coordinate-frames)
6. [Practical Examples](#practical-examples)
7. [Real-World Applications](#real-world-applications)

## Image Acquisition and Representation

### Digital Image Formation

A digital image is a discrete representation of a continuous scene, formed by sampling and quantizing light intensity at regular intervals.

**Key Parameters:**
- **Resolution**: Number of pixels (width × height)
- **Bit Depth**: Number of bits per pixel (8-bit, 16-bit, etc.)
- **Color Space**: Representation of color information (RGB, HSV, etc.)
- **Dynamic Range**: Range of light intensities captured

### Image Formats in Robotics

**Common Formats:**
- **RAW**: Unprocessed sensor data, maximum information
- **BMP**: Uncompressed, simple format
- **PNG**: Lossless compression, supports transparency
- **JPEG**: Lossy compression, smaller file sizes
- **TIFF**: Flexible format, supports multiple images and metadata

### Image Coordinate Systems

**Pixel Coordinates (u, v):**
- Origin typically at top-left corner
- u: column index (horizontal)
- v: row index (vertical)

**World Coordinates (X, Y, Z):**
- 3D coordinates in the real world
- Depends on chosen reference frame

## Image Preprocessing

### Color Space Conversions

Different color spaces are useful for different image processing tasks:

**RGB to Grayscale:**
```
Gray = 0.299×R + 0.587×G + 0.114×B
```

**RGB to HSV:**
- **Hue (H)**: Color type (0-360°)
- **Saturation (S)**: Color purity (0-1)
- **Value (V)**: Brightness (0-1)

### Histogram Equalization

Enhances image contrast by redistributing intensity values:

**Cumulative Distribution Function:**
```
CDF(i) = Σ(j=0 to i) h(j)
```

**Equalized Histogram:**
```
f(i) = round((CDF(i) - CDF_min) / (M×N - CDF_min) × (L-1))
```

Where:
- h(j) is the histogram value at intensity j
- M, N are image dimensions
- L is the number of intensity levels

### Geometric Transformations

**Scaling:**
```
u' = s_x × u
v' = s_y × v
```

**Rotation:**
```
[u']   [cos(θ)  -sin(θ)] [u]
[v'] = [sin(θ)   cos(θ)] [v]
```

**Translation:**
```
u' = u + t_x
v' = v + t_y
```

## Filtering Techniques

### Linear Filters

Linear filters are implemented as convolution operations with kernel matrices.

**General Convolution:**
```
g(i,j) = Σ(Σ f(m,n) × h(i-m, j-n))
```

Where f is the input image, h is the kernel, and g is the output.

### Gaussian Filtering

Gaussian filters are used for smoothing and noise reduction:

**2D Gaussian Kernel:**
```
G(x,y) = (1/(2πσ²)) × exp(-(x² + y²)/(2σ²))
```

**Properties:**
- Separable (can be applied as two 1D filters)
- Removes high-frequency noise
- Preserves edges better than simple averaging

**Python Implementation:**
```python
import cv2
import numpy as np

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian filter to image"""
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel * kernel.T

    # Apply filter
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered

# Example usage
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
smoothed = gaussian_filter(image, kernel_size=5, sigma=1.0)
```

### Edge Detection Filters

Edge detection identifies significant changes in image intensity.

#### Sobel Operator

The Sobel operator computes gradients in x and y directions:

**X-gradient kernel:**
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

**Y-gradient kernel:**
```
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```

**Gradient magnitude:**
```
G = √(G_x² + G_y²)
```

**Gradient direction:**
```
θ = arctan(G_y / G_x)
```

**Python Implementation:**
```python
def sobel_edge_detection(image):
    """Apply Sobel edge detection"""
    # Calculate gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    return magnitude, direction

# Example usage
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
edges, directions = sobel_edge_detection(image)
```

#### Canny Edge Detection

The Canny edge detector is a multi-stage algorithm:

1. **Noise Reduction**: Apply Gaussian filter
2. **Gradient Calculation**: Use Sobel operators
3. **Non-maximum Suppression**: Thin edges to single-pixel width
4. **Double Thresholding**: Identify strong and weak edges
5. **Edge Tracking**: Connect weak edges to strong edges

**Python Implementation:**
```python
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection"""
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# Example usage
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
edges = canny_edge_detection(image, low_threshold=50, high_threshold=150)
```

### Morphological Operations

Morphological operations process binary or grayscale images based on shape:

**Structuring Element:**
```
[0 1 0]
[1 1 1]
[0 1 0]
```

**Erosion:**
```
(f ⊗ b)(x,y) = min{f(x+i, y+j) - b(i,j)}
```

**Dilation:**
```
(f ⊕ b)(x,y) = max{f(x-i, y-j) + b(i,j)}
```

**Opening (Erosion + Dilation):**
Removes small objects and smooths boundaries

**Closing (Dilation + Erosion):**
Fills small holes and smooths boundaries

## Feature Extraction and Matching

### Corner Detection

Corners are points where image intensity changes significantly in multiple directions.

#### Harris Corner Detector

The Harris detector identifies corners using the autocorrelation matrix:

**Autocorrelation Matrix:**
```
M = Σ(w) [Ix²   IxIy]
          [IxIy  Iy² ]
```

Where w is a window function, and Ix, Iy are image gradients.

**Harris Response:**
```
R = det(M) - k×trace(M)²
```

Where k is an empirical constant (typically 0.04-0.06).

**Python Implementation:**
```python
def harris_corners(image, k=0.04, threshold=0.01):
    """Detect corners using Harris corner detector"""
    # Calculate gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Apply Gaussian smoothing
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)

    # Calculate Harris response
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * trace * trace

    # Apply threshold
    corners = np.where(R > threshold * R.max())
    return corners

# Example usage
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
corner_points = harris_corners(image)
```

### Feature Descriptors

Feature descriptors provide distinctive representations of image regions.

#### SIFT (Scale-Invariant Feature Transform)

SIFT features are invariant to scale, rotation, and illumination changes:

1. **Scale-space extrema detection**: Find keypoints at different scales
2. **Keypoint localization**: Refine keypoint locations
3. **Orientation assignment**: Assign canonical orientation
4. **Descriptor computation**: Create 128-dimensional descriptor

**Python Implementation:**
```python
def compute_sift_features(image):
    """Compute SIFT features"""
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors

# Example usage (Note: SIFT is patented, use ORB for open-source alternative)
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
kp, desc = compute_sift_features(image)
```

#### ORB (Oriented FAST and Rotated BRIEF)

ORB is a fast, efficient alternative to SIFT:

**Python Implementation:**
```python
def compute_orb_features(image):
    """Compute ORB features"""
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors

# Example usage
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
kp, desc = compute_orb_features(image)
```

### Feature Matching

Feature matching finds correspondences between images:

**Brute-Force Matching:**
```python
def match_features(desc1, desc2, method='BF'):
    """Match features between two sets of descriptors"""
    if method == 'BF':
        # Brute force matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches
    else:
        raise ValueError("Method not implemented")
```

## Camera Transformations and Coordinate Frames

### Pinhole Camera Model

The pinhole camera model describes the geometric relationship between 3D world points and 2D image points:

**Projection Equation:**
```
s[u]   [fx  0  cx  0] [X]
s[v] = [0  fy  cy  0] [Y]
[1]    [0   0   1  0] [Z]
                        [1]
```

Where:
- (u, v) are image coordinates
- (X, Y, Z) are world coordinates
- fx, fy are focal lengths in pixels
- cx, cy are principal point coordinates

### Camera Calibration

Camera calibration determines intrinsic and extrinsic parameters:

**Intrinsic Parameters:**
- **Focal lengths**: fx, fy
- **Principal point**: cx, cy
- **Skew coefficient**: s (usually 0)
- **Distortion coefficients**: k1, k2, p1, p2, k3

**Extrinsic Parameters:**
- **Rotation matrix**: R (3×3)
- **Translation vector**: t (3×1)

**Python Implementation:**
```python
def calibrate_camera(object_points, image_points, image_size):
    """Calibrate camera using known object points"""
    # Camera matrix initialization
    camera_matrix = np.zeros((3, 3))
    camera_matrix[0, 0] = image_size[0]  # fx
    camera_matrix[1, 1] = image_size[1]  # fy
    camera_matrix[0, 2] = image_size[0] / 2  # cx
    camera_matrix[1, 2] = image_size[1] / 2  # cy
    camera_matrix[2, 2] = 1

    # Distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.zeros((5, 1))

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, camera_matrix, dist_coeffs
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

# Example usage with chessboard pattern
def calibrate_with_chessboard(images, pattern_size=(9, 6)):
    """Calibrate camera using chessboard images"""
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane

    # Prepare object points (real world coordinates)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            object_points.append(objp)
            image_points.append(corners)

    if len(object_points) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            object_points, image_points, gray.shape[::-1]
        )
        return ret, camera_matrix, dist_coeffs
    else:
        return False, None, None
```

### Undistortion

Lens distortion can be corrected using calibration parameters:

**Python Implementation:**
```python
def undistort_image(image, camera_matrix, dist_coeffs):
    """Remove lens distortion from image"""
    h, w = image.shape[:2]

    # Get optimal camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort image
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop image to remove black regions
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    return undistorted
```

## Practical Examples

### Image Filtering Pipeline

Here's a complete example of an image filtering pipeline:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_filtering_pipeline(image_path):
    """Complete image filtering pipeline"""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian smoothing to reduce noise
    smoothed = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Apply Canny edge detection
    edges = cv2.Canny(smoothed, 50, 150)

    # Apply morphological operations to clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return image, smoothed, edges, edges_cleaned

# Example usage
original, smoothed, edges, cleaned = image_filtering_pipeline('input.jpg')

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(smoothed, cmap='gray')
axes[0, 1].set_title('Gaussian Smoothed')
axes[1, 0].imshow(edges, cmap='gray')
axes[1, 0].set_title('Canny Edges')
axes[1, 1].imshow(cleaned, cmap='gray')
axes[1, 1].set_title('Morphologically Cleaned')
plt.tight_layout()
plt.show()
```

### Feature Detection and Matching

```python
def feature_matching_example(img1_path, img2_path):
    """Example of feature detection and matching"""
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Detect ORB features
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result, len(good_matches)

# Example usage
matches_result, num_matches = feature_matching_example('image1.jpg', 'image2.jpg')
print(f"Found {num_matches} good matches")
```

## Real-World Applications

### Autonomous Navigation

Image processing enables robots to navigate safely:
- Obstacle detection using edge detection
- Lane following using line detection
- Path planning using feature extraction

### Quality Inspection

Industrial robots use image processing for:
- Defect detection using texture analysis
- Dimension measurement using edge detection
- Color matching using color space analysis

### Human-Robot Interaction

Robots use image processing to:
- Recognize human gestures using feature extraction
- Detect faces using template matching
- Track humans using motion analysis

## Key Takeaways

1. **Image preprocessing** is crucial for robust vision systems
2. **Filtering techniques** enhance image quality and extract relevant features
3. **Feature extraction** enables object recognition and scene understanding
4. **Camera calibration** is essential for accurate 3D measurements
5. **Coordinate transformations** connect image space to world space

## Exercises

1. Implement a complete image filtering pipeline with Gaussian smoothing and Canny edge detection
2. Detect and match features between two similar images using different feature detectors
3. Calibrate a camera using a chessboard pattern and apply undistortion to images

---

**Previous**: [Sensors](./sensors.md) | **Next**: [Deep Vision](./deep-vision.md)