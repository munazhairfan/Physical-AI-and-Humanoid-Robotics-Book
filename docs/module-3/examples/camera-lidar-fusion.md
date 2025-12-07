---
title: "Camera-LiDAR Fusion Example"
description: "Practical implementation of camera and LiDAR sensor fusion for robotic perception"
sidebar_label: "Camera-LiDAR Fusion"
---

# Camera-LiDAR Fusion Example

## Introduction

Camera-LiDAR fusion is one of the most important multi-sensor fusion techniques in robotics, combining the rich appearance information from cameras with the accurate geometric information from LiDAR sensors. This example demonstrates practical implementation of this fusion approach.

## Why Camera-LiDAR Fusion?

### Complementary Information
- **Cameras**: Provide color, texture, and appearance information
- **LiDAR**: Provide accurate 3D geometric information
- **Combined**: Rich 3D understanding with appearance context

### Applications
- Autonomous vehicles for obstacle detection
- Robotics for navigation and manipulation
- Augmented reality systems
- Surveillance and security systems

## Mathematical Foundation

### Pinhole Camera Model

The relationship between a 3D point in the world and its 2D projection on the image plane is given by:

```
s * [u]   [fx  0  cx  0] [Xw]
s * [v] = [0  fy  cy  0] [Yw]
    [1]   [0   0   1  0] [Zw]
          [0   0   0  0] [1]
```

Where:
- (u, v) are pixel coordinates
- (Xw, Yw, Zw) are world coordinates
- fx, fy are focal lengths
- cx, cy are principal point coordinates
- s is a scaling factor

### Extrinsic Calibration

To transform LiDAR points to camera coordinates:
```
P_camera = R * P_lidar + T
```

Where R is the rotation matrix and T is the translation vector.

## Implementation Approaches

### 1. Early Fusion (Image Level)

Project LiDAR points onto the camera image and combine information at the pixel level.

#### Python Implementation

```python
import numpy as np
import cv2

class CameraLidarFusion:
    def __init__(self, camera_matrix, dist_coeffs, extrinsic_matrix):
        """
        Initialize the fusion module
        :param camera_matrix: 3x3 camera intrinsic matrix
        :param dist_coeffs: Distortion coefficients
        :param extrinsic_matrix: 4x4 extrinsic matrix [R|T]
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.extrinsic_matrix = extrinsic_matrix
        self.rotation = extrinsic_matrix[:3, :3]
        self.translation = extrinsic_matrix[:3, 3]

    def project_lidar_to_image(self, lidar_points):
        """
        Project 3D LiDAR points to 2D image coordinates
        :param lidar_points: Nx3 array of 3D points [X, Y, Z]
        :return: Nx2 array of 2D image coordinates [u, v], and valid mask
        """
        # Transform LiDAR points to camera coordinate system
        points_cam = self.rotation @ lidar_points.T + self.translation.reshape(3, 1)
        points_cam = points_cam.T  # Shape: Nx3

        # Filter points that are in front of the camera
        valid_mask = points_cam[:, 2] > 0

        if not np.any(valid_mask):
            return np.array([]).reshape(0, 2), np.array([], dtype=bool)

        # Project to image coordinates
        u = points_cam[:, 0] / points_cam[:, 2]
        v = points_cam[:, 1] / points_cam[:, 2]

        # Apply camera matrix
        u = self.camera_matrix[0, 0] * u + self.camera_matrix[0, 2]
        v = self.camera_matrix[1, 1] * v + self.camera_matrix[1, 2]

        projected_points = np.column_stack((u, v))

        return projected_points, valid_mask

    def create_birds_eye_view(self, lidar_points, image, bev_range=40, bev_resolution=0.1):
        """
        Create a bird's eye view representation combining LiDAR and image information
        :param lidar_points: Nx3 array of 3D points [X, Y, Z]
        :param image: Input camera image
        :param bev_range: Range in meters for BEV (e.g., 40 means -40 to +40)
        :param bev_resolution: Resolution in meters per pixel
        :return: Bird's eye view image
        """
        bev_width = int(2 * bev_range / bev_resolution)
        bev_height = int(2 * bev_range / bev_resolution)

        bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

        # Convert LiDAR coordinates to BEV coordinates
        x_coords = lidar_points[:, 0]
        y_coords = lidar_points[:, 1]

        # Filter points within range
        valid_x = np.logical_and(x_coords >= -bev_range, x_coords <= bev_range)
        valid_y = np.logical_and(y_coords >= -bev_range, y_coords <= bev_range)
        valid_mask = np.logical_and(valid_x, valid_y)

        x_bev = ((x_coords[valid_mask] + bev_range) / bev_resolution).astype(int)
        y_bev = ((y_coords[valid_mask] + bev_range) / bev_resolution).astype(int)

        # Ensure indices are within bounds
        valid_indices = np.logical_and(
            np.logical_and(x_bev >= 0, x_bev < bev_width),
            np.logical_and(y_bev >= 0, y_bev < bev_height)
        )

        x_bev = x_bev[valid_indices]
        y_bev = y_bev[valid_indices]

        # Color based on height
        z_vals = lidar_points[valid_mask][valid_indices][:, 2]
        colors = self._height_to_color(z_vals)

        bev_image[y_bev, x_bev] = colors

        return bev_image

    def _height_to_color(self, heights):
        """
        Convert height values to colors for visualization
        :param heights: Array of height values
        :return: Array of RGB colors
        """
        # Normalize heights to 0-255 range
        min_h, max_h = np.min(heights), np.max(heights)
        if max_h == min_h:
            norm_heights = np.zeros_like(heights)
        else:
            norm_heights = 255 * (heights - min_h) / (max_h - min_h)

        # Create color map (blue to red)
        colors = np.zeros((len(heights), 3), dtype=np.uint8)
        colors[:, 0] = 255 - norm_heights.astype(int)  # Blue
        colors[:, 2] = norm_heights.astype(int)        # Red

        return colors

# Example usage
def example_usage():
    # Define camera parameters (example values)
    camera_matrix = np.array([
        [700, 0, 640],  # fx, 0, cx
        [0, 700, 360],  # 0, fy, cy
        [0, 0, 1]       # 0, 0, 1
    ])

    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

    # Define extrinsic parameters (example values)
    # This is a 4x4 transformation matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = np.array([
        [0.998, 0.015, -0.061],
        [-0.014, 0.999, 0.026],
        [0.061, -0.027, 0.998]
    ])
    extrinsic_matrix[:3, 3] = np.array([0.1, 0.2, 0.3])  # Translation

    # Create fusion instance
    fusion = CameraLidarFusion(camera_matrix, dist_coeffs, extrinsic_matrix)

    # Generate example LiDAR points (in LiDAR coordinate system)
    # This would typically come from a LiDAR sensor
    lidar_points = np.random.rand(1000, 3) * 20  # Random points in 20m cube
    lidar_points[:, 0] -= 10  # Center around 0
    lidar_points[:, 1] -= 10
    lidar_points[:, 2] -= 5   # Ground plane around z=-5

    # Simulate an image (in practice, this would be a real camera image)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Project LiDAR points to image
    projected_points, valid_mask = fusion.project_lidar_to_image(lidar_points)

    print(f"Projected {np.sum(valid_mask)} LiDAR points to image coordinates")
    print(f"Valid points: {projected_points.shape[0]}")

    # Create bird's eye view
    bev_image = fusion.create_birds_eye_view(lidar_points, image)
    print(f"Bird's eye view shape: {bev_image.shape}")

    return fusion, projected_points, bev_image

if __name__ == "__main__":
    fusion, points, bev = example_usage()
    print("Camera-LiDAR fusion example completed successfully!")
```

### 2. Late Fusion (Object Level)

Detect objects separately in camera and LiDAR data, then combine the detection results.

#### Python Implementation

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

class LateFusionDetector:
    def __init__(self, iou_threshold=0.3):
        """
        Initialize late fusion detector
        :param iou_threshold: Threshold for matching camera and LiDAR detections
        """
        self.iou_threshold = iou_threshold

    def match_detections(self, camera_detections, lidar_detections):
        """
        Match camera and LiDAR detections based on 2D IoU after projection
        :param camera_detections: List of (bbox, confidence, class) from camera
        :param lidar_detections: List of (bbox_3d, confidence, class) from LiDAR
        :return: Matched pairs and unmatched detections
        """
        if len(camera_detections) == 0 or len(lidar_detections) == 0:
            return [], camera_detections, lidar_detections

        # Create cost matrix based on IoU
        cost_matrix = np.zeros((len(camera_detections), len(lidar_detections)))

        for i, cam_det in enumerate(camera_detections):
            cam_bbox = cam_det[0]  # [x1, y1, x2, y2]
            for j, lidar_det in enumerate(lidar_detections):
                lidar_bbox_3d = lidar_det[0]  # [x, y, z, l, w, h]

                # Project 3D bbox to 2D (simplified - just center and size)
                projected_bbox = self._project_3d_bbox_to_2d(lidar_bbox_3d)

                # Calculate IoU
                iou = self._calculate_iou(cam_bbox, projected_bbox)
                cost_matrix[i, j] = 1 - iou  # Use 1-IoU as cost

        # Use Hungarian algorithm to find optimal assignment
        cam_indices, lidar_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_camera = []
        unmatched_lidar = []

        # Process matches
        for i, j in zip(cam_indices, lidar_indices):
            if cost_matrix[i, j] < (1 - self.iou_threshold):  # IoU > threshold
                # Combine detections
                combined_det = self._combine_detections(
                    camera_detections[i], lidar_detections[j]
                )
                matches.append(combined_det)
            else:
                unmatched_camera.append(camera_detections[i])
                unmatched_lidar.append(lidar_detections[j])

        # Add remaining unmatched detections
        for i in range(len(camera_detections)):
            if i not in cam_indices:
                unmatched_camera.append(camera_detections[i])

        for j in range(len(lidar_detections)):
            if j not in lidar_indices:
                unmatched_lidar.append(lidar_detections[j])

        return matches, unmatched_camera, unmatched_lidar

    def _project_3d_bbox_to_2d(self, bbox_3d):
        """
        Project 3D bounding box to 2D image coordinates (simplified)
        :param bbox_3d: [x, y, z, length, width, height]
        :return: [x1, y1, x2, y2] in image coordinates
        """
        x, y, z, l, w, h = bbox_3d

        # Simplified projection - in real implementation, this would use
        # camera intrinsic and extrinsic parameters
        # For now, just return a 2D bounding box based on 3D center and size
        x_2d = x * 100 + 640  # Scale and shift to image center
        y_2d = y * 100 + 360

        w_2d = l * 50  # Scale factor
        h_2d = w * 50  # Scale factor

        return [x_2d - w_2d/2, y_2d - h_2d/2,
                x_2d + w_2d/2, y_2d + h_2d/2]

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two 2D bounding boxes
        :param bbox1: [x1, y1, x2, y2]
        :param bbox2: [x1, y1, x2, y2]
        :return: IoU value
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _combine_detections(self, cam_det, lidar_det):
        """
        Combine camera and LiDAR detection information
        :param cam_det: (bbox, confidence, class) from camera
        :param lidar_det: (bbox_3d, confidence, class) from LiDAR
        :return: Combined detection with both 2D and 3D information
        """
        # Average the confidence scores
        combined_conf = (cam_det[1] + lidar_det[1]) / 2

        # Combine class information (for simplicity, use camera class)
        combined_class = cam_det[2]

        # Return combined detection with both 2D and 3D info
        return {
            'bbox_2d': cam_det[0],
            'bbox_3d': lidar_det[0],
            'confidence': combined_conf,
            'class': combined_class
        }

# Example usage
def late_fusion_example():
    # Create detector
    detector = LateFusionDetector(iou_threshold=0.3)

    # Example camera detections: (bbox_2d, confidence, class)
    camera_detections = [
        ([100, 100, 200, 200], 0.8, 'car'),
        ([300, 150, 400, 250], 0.7, 'person'),
        ([500, 100, 600, 200], 0.9, 'bicycle')
    ]

    # Example LiDAR detections: (bbox_3d, confidence, class)
    lidar_detections = [
        ([5, 2, -1, 4, 2, 1.5], 0.85, 'car'),      # [x, y, z, l, w, h]
        ([10, -1, 0, 0.8, 0.5, 1.8], 0.75, 'person'),
        ([15, 3, 0, 2, 1, 1.2], 0.8, 'bicycle')
    ]

    # Perform matching
    matches, unmatched_cam, unmatched_lidar = detector.match_detections(
        camera_detections, lidar_detections
    )

    print(f"Found {len(matches)} matched detections")
    print(f"{len(unmatched_cam)} unmatched camera detections")
    print(f"{len(unmatched_lidar)} unmatched LiDAR detections")

    for i, match in enumerate(matches):
        print(f"Match {i+1}: {match['class']} with confidence {match['confidence']:.2f}")
        print(f"  2D bbox: [{match['bbox_2d'][0]:.1f}, {match['bbox_2d'][1]:.1f}, "
              f"{match['bbox_2d'][2]:.1f}, {match['bbox_2d'][3]:.1f}]")
        print(f"  3D bbox: center=({match['bbox_3d'][0]:.1f}, {match['bbox_3d'][1]:.1f}, "
              f"{match['bbox_3d'][2]:.1f}), size=({match['bbox_3d'][3]:.1f}, "
              f"{match['bbox_3d'][4]:.1f}, {match['bbox_3d'][5]:.1f})")

    return matches, unmatched_cam, unmatched_lidar

if __name__ == "__main__":
    matches, unmatched_cam, unmatched_lidar = late_fusion_example()
    print("\nLate fusion example completed successfully!")
```

## Deep Learning Fusion Approach

### PointPainting

PointPainting is a method that colors LiDAR points with semantic information from camera images.

```python
import torch
import torch.nn as nn
import numpy as np

class PointPaintingFusion(nn.Module):
    def __init__(self, num_classes=10):
        """
        PointPainting fusion implementation
        :param num_classes: Number of semantic classes
        """
        super(PointPaintingFusion, self).__init__()
        self.num_classes = num_classes

        # Simple classifier for fused features
        self.classifier = nn.Sequential(
            nn.Linear(3 + num_classes, 128),  # 3D coordinates + semantic features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def project_points_to_image(self, points, camera_matrix, extrinsic_matrix):
        """
        Project 3D LiDAR points to 2D image coordinates
        :param points: Nx3 tensor of 3D points
        :param camera_matrix: 3x3 camera intrinsic matrix
        :param extrinsic_matrix: 4x4 extrinsic matrix
        :return: Nx2 tensor of 2D coordinates and valid mask
        """
        # Transform to camera coordinates
        points_homo = torch.cat([points, torch.ones((points.shape[0], 1))], dim=1)
        points_cam = torch.matmul(points_homo, extrinsic_matrix.t())
        points_cam = points_cam[:, :3]  # Remove homogeneous coordinate

        # Filter points in front of camera
        valid_mask = points_cam[:, 2] > 0

        # Project to image
        u = points_cam[:, 0] / points_cam[:, 2]
        v = points_cam[:, 1] / points_cam[:, 2]

        u = camera_matrix[0, 0] * u + camera_matrix[0, 2]
        v = camera_matrix[1, 1] * v + camera_matrix[1, 2]

        projected_points = torch.stack([u, v], dim=1)

        return projected_points, valid_mask

    def forward(self, lidar_points, semantic_map):
        """
        Forward pass of PointPainting fusion
        :param lidar_points: Nx3 tensor of 3D LiDAR points
        :param semantic_map: HxWxC tensor of semantic predictions from camera
        :return: Fused features
        """
        # This is a simplified implementation
        # In practice, you would:
        # 1. Project LiDAR points to image coordinates
        # 2. Sample semantic features at those coordinates
        # 3. Combine with original 3D coordinates

        # For this example, we'll create dummy semantic features
        batch_size = lidar_points.shape[0]
        semantic_features = torch.randn(batch_size, self.num_classes)

        # Concatenate 3D coordinates with semantic features
        fused_features = torch.cat([lidar_points, semantic_features], dim=1)

        # Apply classification
        output = self.classifier(fused_features)

        return output

# Example usage
def pointpainting_example():
    # Create model
    model = PointPaintingFusion(num_classes=10)

    # Example LiDAR points
    lidar_points = torch.randn(1000, 3)  # 1000 points with x, y, z coordinates

    # Example semantic map (simplified)
    semantic_map = torch.randn(480, 640, 10)  # H, W, C

    # Forward pass
    output = model(lidar_points, semantic_map)

    print(f"Input LiDAR points shape: {lidar_points.shape}")
    print(f"Output shape: {output.shape}")
    print("PointPainting fusion example completed successfully!")

    return output

if __name__ == "__main__":
    result = pointpainting_example()
```

## Practical Considerations

### Calibration

Proper calibration is essential for effective fusion:

```python
def calibrate_camera_lidar(camera_matrix, distortion_coeffs, initial_extrinsics):
    """
    Function to refine camera-LiDAR extrinsic calibration
    This is a simplified example - in practice, you would use
    optimization techniques like bundle adjustment
    """
    # In practice, this would involve:
    # 1. Collecting synchronized camera and LiDAR data
    # 2. Detecting calibration targets (e.g., checkerboard) in both modalities
    # 3. Optimizing the transformation to minimize reprojection error

    # Return refined extrinsics
    return initial_extrinsics
```

### Synchronization

Time synchronization between sensors is crucial:

```python
def synchronize_sensors(camera_timestamps, lidar_timestamps, max_offset=0.1):
    """
    Synchronize camera and LiDAR data based on timestamps
    :param camera_timestamps: Array of camera capture times
    :param lidar_timestamps: Array of LiDAR sweep times
    :param max_offset: Maximum acceptable time offset
    :return: Indices of synchronized pairs
    """
    sync_pairs = []

    for cam_idx, cam_time in enumerate(camera_timestamps):
        # Find closest LiDAR timestamp
        time_diffs = np.abs(lidar_timestamps - cam_time)
        closest_idx = np.argmin(time_diffs)

        if time_diffs[closest_idx] <= max_offset:
            sync_pairs.append((cam_idx, closest_idx))

    return sync_pairs
```

## Performance Evaluation

### Metrics for Fusion Quality

```python
def evaluate_fusion_performance(ground_truth, fused_detections, camera_detections, lidar_detections):
    """
    Evaluate the performance improvement from fusion
    """
    # Calculate metrics for each modality separately
    cam_metrics = calculate_detection_metrics(ground_truth, camera_detections)
    lidar_metrics = calculate_detection_metrics(ground_truth, lidar_detections)
    fused_metrics = calculate_detection_metrics(ground_truth, fused_detections)

    print("Performance Comparison:")
    print(f"Camera-only mAP: {cam_metrics['mAP']:.3f}")
    print(f"LiDAR-only mAP: {lidar_metrics['mAP']:.3f}")
    print(f"Fused mAP: {fused_metrics['mAP']:.3f}")
    print(f"Fusion improvement: {fused_metrics['mAP'] - max(cam_metrics['mAP'], lidar_metrics['mAP']):.3f}")

    return {
        'camera': cam_metrics,
        'lidar': lidar_metrics,
        'fused': fused_metrics
    }

def calculate_detection_metrics(ground_truth, detections, iou_threshold=0.5):
    """
    Calculate detection metrics (simplified)
    """
    # Implementation would involve matching detections to ground truth
    # and calculating precision, recall, mAP, etc.
    pass
```

## Kalman Filter Implementation

### Linear Kalman Filter for Sensor Fusion

The Kalman Filter is an optimal recursive estimator that combines predictions from a motion model with measurements from sensors to estimate the state of a system.

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class KalmanFilter:
    def __init__(self, state_dim: int, measurement_dim: int):
        """
        Initialize Kalman Filter
        :param state_dim: Dimension of the state vector
        :param measurement_dim: Dimension of the measurement vector
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector (will be set by user)
        self.x = np.zeros((state_dim, 1))

        # Error covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance
        self.Q = np.eye(state_dim)

        # Measurement noise covariance
        self.R = np.eye(measurement_dim)

        # State transition model
        self.F = np.eye(state_dim)

        # Control input model
        self.B = np.zeros((state_dim, 1)) if state_dim > 0 else np.array([]).reshape(0, 0)

        # Measurement model
        self.H = np.zeros((measurement_dim, state_dim))

    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prediction step of Kalman Filter
        :param u: Control input (optional)
        :return: Predicted state
        """
        # State prediction: x = F * x + B * u
        if u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x

        # Covariance prediction: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step of Kalman Filter
        :param z: Measurement vector
        :return: Updated state
        """
        # Innovation: y = z - H * x
        y = z - self.H @ self.x

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(len(self.P))
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

class ObjectTracker:
    def __init__(self, dt: float = 0.1):
        """
        Object tracker using Kalman Filter
        :param dt: Time step between measurements
        """
        # State: [x, y, vx, vy] (position and velocity)
        self.kf = KalmanFilter(state_dim=4, measurement_dim=2)

        # Time step
        self.dt = dt

        # Initialize state transition matrix (constant velocity model)
        # x_{k+1} = x_k + vx_k * dt
        # y_{k+1} = y_k + vy_k * dt
        # vx_{k+1} = vx_k
        # vy_{k+1} = vy_k
        self.kf.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])

        # Initialize measurement matrix (we only measure position, not velocity)
        # z = [x, y]
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise (assumes random acceleration)
        # Q = G * G^T * sigma_a^2
        # where G = [dt^2/2, dt^2/2, dt, dt]^T for 2D motion
        sigma_a = 5.0  # acceleration uncertainty
        self.kf.Q = np.array([
            [dt**4/4,      0,    dt**3/2,         0],
            [     0, dt**4/4,         0,   dt**3/2],
            [dt**3/2,      0,      dt**2,         0],
            [     0, dt**3/2,         0,     dt**2]
        ]) * sigma_a**2

        # Measurement noise
        sigma_z = 0.5  # measurement uncertainty
        self.kf.R = np.eye(2) * sigma_z**2

        # Error covariance (initialize with high uncertainty)
        self.kf.P = np.eye(4) * 1000

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state
        :return: (predicted state, predicted covariance)
        """
        predicted_state = self.kf.predict()
        return predicted_state, self.kf.P.copy()

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the state with a new measurement
        :param measurement: [x, y] measurement
        :return: (updated state, updated covariance)
        """
        self.kf.x[:2] = measurement.reshape(2, 1)  # Initialize position with first measurement
        updated_state = self.kf.update(measurement.reshape(-1, 1))
        return updated_state, self.kf.P.copy()

    def track_sequence(self, measurements: list) -> list:
        """
        Track a sequence of measurements
        :param measurements: List of [x, y] measurements
        :return: List of tracked states [x, y, vx, vy]
        """
        tracked_states = []

        for i, meas in enumerate(measurements):
            if i == 0:
                # Initialize state with first measurement
                self.kf.x[:2] = meas.reshape(2, 1)
                tracked_states.append(self.kf.x.flatten())
            else:
                # Predict
                self.kf.predict()
                # Update with measurement
                self.kf.update(meas.reshape(-1, 1))
                tracked_states.append(self.kf.x.flatten())

        return tracked_states

# Example usage: Simulate tracking of an object with noisy measurements
def simulate_tracking_example():
    """
    Example: Track a moving object with noisy measurements using Kalman Filter
    """
    # Create tracker
    tracker = ObjectTracker(dt=0.1)

    # Simulate true trajectory (moving in a straight line)
    dt = 0.1
    time_steps = 100
    true_positions = []
    measurements = []

    # Initial state: position [10, 5], velocity [2, 1]
    pos = np.array([10.0, 5.0])
    vel = np.array([2.0, 1.0])

    for t in range(time_steps):
        # True position
        true_pos = pos + vel * t * dt
        true_positions.append(true_pos.copy())

        # Noisy measurement
        noise = np.random.normal(0, 0.5, size=2)  # measurement noise
        meas = true_pos + noise
        measurements.append(meas.copy())

    # Track the object
    tracked_states = tracker.track_sequence(measurements)

    # Extract tracked positions and velocities
    tracked_positions = [state[:2] for state in tracked_states]
    tracked_velocities = [state[2:] for state in tracked_states]

    # Plot results
    true_x = [p[0] for p in true_positions]
    true_y = [p[1] for p in true_positions]
    meas_x = [m[0] for m in measurements]
    meas_y = [m[1] for m in measurements]
    track_x = [p[0] for p in tracked_positions]
    track_y = [p[1] for p in tracked_positions]

    plt.figure(figsize=(12, 8))
    plt.plot(true_x, true_y, 'g-', label='True Trajectory', linewidth=2)
    plt.scatter(meas_x, meas_y, c='r', alpha=0.5, label='Noisy Measurements')
    plt.plot(track_x, track_y, 'b-', label='Kalman Filter Estimate', linewidth=2)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Object Tracking with Kalman Filter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    # Print tracking performance
    true_positions = np.array(true_positions)
    measurements = np.array(measurements)
    tracked_positions = np.array(tracked_positions)

    # Calculate RMSE
    meas_rmse = np.sqrt(np.mean((true_positions - measurements)**2))
    track_rmse = np.sqrt(np.mean((true_positions - tracked_positions)**2))

    print(f"Performance Comparison:")
    print(f"RMSE without filter (raw measurements): {meas_rmse:.3f}")
    print(f"RMSE with Kalman filter: {track_rmse:.3f}")
    print(f"Improvement: {((meas_rmse - track_rmse) / meas_rmse * 100):.1f}%")

    return tracker, tracked_states

if __name__ == "__main__":
    tracker, states = simulate_tracking_example()
```

### Extended Kalman Filter (EKF)

For non-linear systems, we use the Extended Kalman Filter which linearizes the system around the current state estimate.

```python
class ExtendedKalmanFilter:
    def __init__(self, state_dim: int, measurement_dim: int):
        """
        Initialize Extended Kalman Filter
        :param state_dim: Dimension of the state vector
        :param measurement_dim: Dimension of the measurement vector
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector
        self.x = np.zeros((state_dim, 1))

        # Error covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance
        self.Q = np.eye(state_dim)

        # Measurement noise covariance
        self.R = np.eye(measurement_dim)

    def predict(self, f_func, F_func, dt: float = 1.0, u: Optional[np.ndarray] = None):
        """
        Prediction step for EKF
        :param f_func: Non-linear state transition function: x_{k+1} = f(x_k, u_k)
        :param F_func: Jacobian of f with respect to state
        :param dt: Time step
        :param u: Control input
        """
        # Predict state: x = f(x, u)
        self.x = f_func(self.x, u, dt)

        # Compute Jacobian of f at current state
        F = F_func(self.x, u, dt)

        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, h_func, H_func, z: np.ndarray):
        """
        Update step for EKF
        :param h_func: Non-linear measurement function: z = h(x)
        :param H_func: Jacobian of h with respect to state
        :param z: Measurement vector
        """
        # Compute Jacobian of h at current state
        H = H_func(self.x)

        # Innovation: y = z - h(x)
        y = z - h_func(self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state: x = x + K * y
        self.x = self.x + K @ y

        # Update covariance: P = (I - K * H) * P
        I = np.eye(len(self.P))
        self.P = (I - K @ H) @ self.P

        return self.x.copy(), K

# Example: EKF for tracking in polar coordinates (bearings-only tracking)
def bearings_only_tracking_example():
    """
    Example: Track an object using only bearing measurements (e.g., from camera)
    This is a non-linear problem as bearing = atan2(y-y_sensor, x-x_sensor)
    """
    # State: [x, y, vx, vy] (position and velocity)
    state_dim = 4
    measurement_dim = 1  # Only bearing measurement

    ekf = ExtendedKalmanFilter(state_dim, measurement_dim)

    # Initialize state (position and velocity)
    ekf.x = np.array([[10.0], [5.0], [1.0], [0.5]])  # [x, y, vx, vy]

    # Covariance initialization
    ekf.P = np.diag([100, 100, 10, 10])  # High uncertainty in initial position/velocity

    # Process noise (random acceleration model)
    dt = 0.1
    sigma_a = 2.0
    ekf.Q = np.array([
        [dt**4/4,      0,    dt**3/2,         0],
        [     0, dt**4/4,         0,   dt**3/2],
        [dt**3/2,      0,      dt**2,         0],
        [     0, dt**3/2,         0,     dt**2]
    ]) * sigma_a**2

    # Measurement noise (bearing uncertainty)
    sigma_bearing = 0.05  # 0.05 radians (~3 degrees)
    ekf.R = np.array([[sigma_bearing**2]])

    # Sensor position (e.g., camera position)
    sensor_pos = np.array([0.0, 0.0])

    def state_transition_function(x, u, dt):
        """Constant velocity model: x_{k+1} = f(x_k)"""
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])
        return F @ x

    def state_transition_jacobian(x, u, dt):
        """Jacobian of state transition function"""
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])

    def measurement_function(x):
        """Convert state to bearing measurement"""
        pos_x = x[0, 0]
        pos_y = x[1, 0]

        # Calculate bearing from sensor to object
        dx = pos_x - sensor_pos[0]
        dy = pos_y - sensor_pos[1]

        bearing = np.arctan2(dy, dx)
        return np.array([[bearing]])

    def measurement_jacobian(x):
        """Jacobian of measurement function"""
        pos_x = x[0, 0]
        pos_y = x[1, 0]

        dx = pos_x - sensor_pos[0]
        dy = pos_y - sensor_pos[1]
        r_squared = dx**2 + dy**2

        # Partial derivatives of h(x) = arctan2(dy, dx)
        H = np.array([[
            -dy / r_squared,  # dh/dx
            dx / r_squared,   # dh/dy
            0,                # dh/dvx
            0                 # dh/dvy
        ]])

        return H

    # Simulate a trajectory and measurements
    true_states = []
    measurements = []

    # True initial state
    true_x = np.array([10.0, 5.0, 1.0, 0.5])

    for k in range(50):
        # True state evolution
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])
        true_x = F @ true_x

        # Add process noise to simulate real world
        process_noise = np.random.multivariate_normal(np.zeros(4), ekf.Q/10)
        true_x_with_noise = true_x + process_noise

        # Generate measurement (bearing with noise)
        dx = true_x_with_noise[0] - sensor_pos[0]
        dy = true_x_with_noise[1] - sensor_pos[1]
        true_bearing = np.arctan2(dy, dx)
        measurement_noise = np.random.normal(0, sigma_bearing)
        z = np.array([[true_bearing + measurement_noise]])

        true_states.append(true_x_with_noise.copy())
        measurements.append(z.copy())

        # EKF prediction and update
        ekf.predict(state_transition_function, state_transition_jacobian, dt)
        ekf.update(measurement_function, measurement_jacobian, z)

    print("Bearings-only tracking with EKF completed")
    print(f"Final estimated state: {ekf.x.flatten()}")
    print(f"Final true state: {true_x}")

    return ekf, true_states, measurements

if __name__ == "__main__":
    # Run both examples
    print("Running Kalman Filter example...")
    tracker, states = simulate_tracking_example()

    print("\nRunning Extended Kalman Filter example...")
    ekf, true_states, measurements = bearings_only_tracking_example()

## C++ Implementation of Kalman Filter

Here's a C++ implementation of the Kalman Filter for sensor fusion applications:

```cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>

class KalmanFilter {
private:
    int state_dim;
    int measurement_dim;

    // State vector
    Eigen::VectorXd x;

    // Error covariance matrix
    Eigen::MatrixXd P;

    // Process noise covariance
    Eigen::MatrixXd Q;

    // Measurement noise covariance
    Eigen::MatrixXd R;

    // State transition model
    Eigen::MatrixXd F;

    // Control input model
    Eigen::MatrixXd B;

    // Measurement model
    Eigen::MatrixXd H;

public:
    KalmanFilter(int state_dim, int measurement_dim)
        : state_dim(state_dim), measurement_dim(measurement_dim) {
        x = Eigen::VectorXd::Zero(state_dim);
        P = Eigen::MatrixXd::Identity(state_dim, state_dim);
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
        R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
        F = Eigen::MatrixXd::Identity(state_dim, state_dim);
        B = Eigen::MatrixXd::Zero(state_dim, 1);
        H = Eigen::MatrixXd::Zero(measurement_dim, state_dim);
    }

    void setInitialState(const Eigen::VectorXd& initial_state) {
        x = initial_state;
    }

    void setStateTransitionMatrix(const Eigen::MatrixXd& F_matrix) {
        F = F_matrix;
    }

    void setMeasurementMatrix(const Eigen::MatrixXd& H_matrix) {
        H = H_matrix;
    }

    void setProcessNoiseCovariance(const Eigen::MatrixXd& Q_matrix) {
        Q = Q_matrix;
    }

    void setMeasurementNoiseCovariance(const Eigen::MatrixXd& R_matrix) {
        R = R_matrix;
    }

    void setControlInputMatrix(const Eigen::MatrixXd& B_matrix) {
        B = B_matrix;
    }

    void predict(const Eigen::VectorXd& u = Eigen::VectorXd()) {
        // State prediction: x = F * x + B * u
        if (u.size() > 0 && B.cols() == u.size()) {
            x = F * x + B * u;
        } else {
            x = F * x;
        }

        // Covariance prediction: P = F * P * F^T + Q
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd& z) {
        // Innovation: y = z - H * x
        Eigen::VectorXd y = z - H * x;

        // Innovation covariance: S = H * P * H^T + R
        Eigen::MatrixXd S = H * P * H.transpose() + R;

        // Kalman gain: K = P * H^T * S^(-1)
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        // State update: x = x + K * y
        x = x + K * y;

        // Covariance update: P = (I - K * H) * P
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
        P = (I - K * H) * P;
    }

    Eigen::VectorXd getState() const {
        return x;
    }

    Eigen::MatrixXd getCovariance() const {
        return P;
    }
};

class ObjectTracker {
private:
    KalmanFilter kf;
    double dt;

public:
    ObjectTracker(double dt = 0.1) : kf(4, 2), dt(dt) {
        // Initialize state transition matrix (constant velocity model)
        // x_{k+1} = x_k + vx_k * dt
        // y_{k+1} = y_k + vy_k * dt
        // vx_{k+1} = vx_k
        // vy_{k+1} = vy_k
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(4, 4);
        F << 1, 0, dt,  0,
             0, 1,  0, dt,
             0, 0,  1,  0,
             0, 0,  0,  1;
        kf.setStateTransitionMatrix(F);

        // Initialize measurement matrix (we only measure position, not velocity)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
        H << 1, 0, 0, 0,
             0, 1, 0, 0;
        kf.setMeasurementMatrix(H);

        // Process noise (assumes random acceleration)
        double sigma_a = 5.0;  // acceleration uncertainty
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 4);
        Q << std::pow(dt,4)/4,           0,    std::pow(dt,3)/2,           0,
                    0, std::pow(dt,4)/4,           0,    std::pow(dt,3)/2,
             std::pow(dt,3)/2,           0, std::pow(dt,2),           0,
                    0,    std::pow(dt,3)/2,           0, std::pow(dt,2);
        Q *= std::pow(sigma_a, 2);
        kf.setProcessNoiseCovariance(Q);

        // Measurement noise
        double sigma_z = 0.5;  // measurement uncertainty
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * std::pow(sigma_z, 2);
        kf.setMeasurementNoiseCovariance(R);

        // Error covariance (initialize with high uncertainty)
        Eigen::MatrixXd P = Eigen::MatrixXd::Identity(4, 4) * 1000;
        kf.setCovariance(P);
    }

    void initializeState(const Eigen::Vector2d& position) {
        Eigen::VectorXd state(4);
        state << position(0), position(1), 0.0, 0.0;  // Initialize with zero velocity
        kf.setInitialState(state);
    }

    Eigen::VectorXd predict() {
        kf.predict();
        return kf.getState();
    }

    Eigen::VectorXd update(const Eigen::Vector2d& measurement) {
        kf.update(measurement);
        return kf.getState();
    }

    std::vector<Eigen::VectorXd> trackSequence(const std::vector<Eigen::Vector2d>& measurements) {
        std::vector<Eigen::VectorXd> trackedStates;

        for (size_t i = 0; i < measurements.size(); ++i) {
            if (i == 0) {
                // Initialize state with first measurement
                initializeState(measurements[i]);
                trackedStates.push_back(kf.getState());
            } else {
                // Predict
                predict();
                // Update with measurement
                Eigen::VectorXd state = update(measurements[i]);
                trackedStates.push_back(state);
            }
        }

        return trackedStates;
    }
};

// Example usage
int main() {
    // Create tracker
    ObjectTracker tracker(0.1);  // dt = 0.1 seconds

    // Simulate true trajectory (moving in a straight line)
    double dt = 0.1;
    int timeSteps = 100;

    std::vector<Eigen::Vector2d> truePositions;
    std::vector<Eigen::Vector2d> measurements;

    // Initial state: position [10, 5], velocity [2, 1]
    Eigen::Vector2d pos(10.0, 5.0);
    Eigen::Vector2d vel(2.0, 1.0);

    // Random number generator for noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noiseDist(0.0, 0.5);  // measurement noise

    for (int t = 0; t < timeSteps; ++t) {
        // True position
        Eigen::Vector2d truePos = pos + vel * t * dt;
        truePositions.push_back(truePos);

        // Noisy measurement
        Eigen::Vector2d noise(noiseDist(gen), noiseDist(gen));
        Eigen::Vector2d meas = truePos + noise;
        measurements.push_back(meas);
    }

    // Track the object
    std::vector<Eigen::VectorXd> trackedStates = tracker.trackSequence(measurements);

    // Calculate and print performance metrics
    double measRMSE = 0.0;
    double trackRMSE = 0.0;

    for (int i = 0; i < timeSteps; ++i) {
        Eigen::Vector2d truePos = truePositions[i];
        Eigen::Vector2d meas = measurements[i];
        Eigen::Vector2d trackedPos(trackedStates[i](0), trackedStates[i](1));

        measRMSE += (truePos - meas).squaredNorm();
        trackRMSE += (truePos - trackedPos).squaredNorm();
    }

    measRMSE = std::sqrt(measRMSE / timeSteps);
    trackRMSE = std::sqrt(trackRMSE / timeSteps);

    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "RMSE without filter (raw measurements): " << measRMSE << std::endl;
    std::cout << "RMSE with Kalman filter: " << trackRMSE << std::endl;
    std::cout << "Improvement: " << ((measRMSE - trackRMSE) / measRMSE * 100) << "%" << std::endl;

    return 0;
}
```

### Compilation and Usage

To compile and run the C++ Kalman Filter example:

```bash
# Install Eigen library (if not already installed)
# On Ubuntu/Debian: sudo apt-get install libeigen3-dev
# On macOS with Homebrew: brew install eigen

# Compile
g++ -std=c++11 kalman_filter.cpp -o kalman_filter

# Run
./kalman_filter
```

The C++ implementation provides the same functionality as the Python version but with better performance for real-time applications. The Eigen library is used for efficient matrix operations, which are essential for Kalman Filter computations.
```

## Next Steps

1. **Implementation Practice**: Try implementing the fusion techniques on sample datasets
2. **Real Data**: Apply to real camera and LiDAR data from datasets like KITTI
3. **Advanced Techniques**: Explore learning-based fusion approaches
4. **Evaluation**: Test performance under various conditions (weather, lighting)

This example provides a foundation for understanding and implementing camera-LiDAR fusion techniques. The code can be extended and adapted for specific applications and datasets.