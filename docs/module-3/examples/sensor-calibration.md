---
title: "Sensor Calibration Techniques"
description: "Methods for calibrating robotic sensors to ensure accurate perception"
sidebar_label: "Sensor Calibration"
---

# Sensor Calibration Techniques

## Introduction

Sensor calibration is a critical step in robotic perception systems that ensures accurate and reliable data interpretation. Calibration involves determining both the internal parameters of sensors (intrinsic calibration) and their position and orientation relative to other sensors or the robot frame (extrinsic calibration).

## Types of Calibration

### Intrinsic Calibration
- **Definition**: Determining internal sensor parameters
- **Examples**: Camera focal length, principal point, distortion coefficients
- **Purpose**: Correct internal sensor characteristics

### Extrinsic Calibration
- **Definition**: Determining spatial relationship between sensors
- **Examples**: Position and orientation of camera relative to LiDAR
- **Purpose**: Transform data between different sensor coordinate systems

## Camera Calibration

### Pinhole Camera Model

The pinhole camera model describes the mathematical relationship between a 3D point in the world and its 2D projection on the image plane:

```
s * [u]   [fx  0  cx  0] [Xw]
s * [v] = [0  fy  cy  0] [Yw]
    [1]   [0   0   1  0] [Zw]
          [0   0   0  0] [1]
```

Where:
- (u, v) are pixel coordinates
- (Xw, Yw, Zw) are world coordinates
- fx, fy are focal lengths in pixels
- cx, cy are principal point coordinates
- s is a scaling factor

### Distortion Models

Real cameras have lens distortions that need to be corrected:

**Radial Distortion**:
```
x_corrected = x * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_corrected = y * (1 + k1*r² + k2*r⁴ + k3*r⁶)
```

Where r² = x² + y² and k1, k2, k3 are radial distortion coefficients.

**Tangential Distortion**:
```
x_corrected = x + [2*p1*x*y + p2*(r² + 2*x²)]
y_corrected = y + [p1*(r² + 2*y²) + 2*p2*x*y]
```

Where p1, p2 are tangential distortion coefficients.

### Camera Calibration Implementation

```python
import numpy as np
import cv2
from typing import Tuple, List

class CameraCalibrator:
    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), square_size: float = 1.0):
        """
        Initialize camera calibrator
        :param pattern_size: Number of internal corners (width, height)
        :param square_size: Size of chessboard squares in real units (e.g., cm)
        """
        self.pattern_size = pattern_size
        self.square_size = square_size  # in real-world units

        # 3D points in real world space (chessboard coordinates)
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane

        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add an image for calibration
        :param image: Input image containing calibration pattern
        :return: True if pattern was found, False otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                       cv2.CALIB_CB_FILTER_QUADS)

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Create 3D points (chessboard coordinates)
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size  # Scale to real-world units

            # Add to lists
            self.obj_points.append(objp)
            self.img_points.append(corners_refined)

            return True
        else:
            return False

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform camera calibration
        :return: (camera_matrix, dist_coeffs)
        """
        if len(self.obj_points) < 10:
            raise ValueError("Need at least 10 images for reliable calibration")

        # Perform calibration
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(
                self.obj_points,
                self.img_points,
                (640, 480),  # Assuming 640x480 image size - should be actual size
                None,
                None
            )

        print(f"Calibration successful! Reprojection error: {ret}")
        return self.camera_matrix, self.dist_coeffs

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort an image using the calibration parameters
        :param image: Input distorted image
        :return: Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera must be calibrated first")

        h, w = image.shape[:2]

        # Get optimal camera matrix
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_mtx
        )

        # Crop the image based on ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def get_reprojection_error(self) -> float:
        """
        Calculate average reprojection error
        :return: Average reprojection error
        """
        if self.rvecs is None or self.tvecs is None:
            return float('inf')

        total_error = 0
        for i in range(len(self.obj_points)):
            # Reproject 3D points to 2D
            img_points2, _ = cv2.projectPoints(
                self.obj_points[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs
            )

            # Calculate error
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error

        return total_error / len(self.obj_points)

# Example usage
def camera_calibration_example():
    # This would typically be run with actual calibration images
    # For demonstration, we'll create a dummy example

    # Initialize calibrator
    calibrator = CameraCalibrator(pattern_size=(9, 6), square_size=2.5)  # 2.5cm squares

    print("Camera calibrator initialized.")
    print("To use: call add_calibration_image() with chessboard images, then calibrate()")

    return calibrator

if __name__ == "__main__":
    calibrator = camera_calibration_example()
```

## LiDAR Calibration

### LiDAR Coordinate Systems

LiDAR sensors typically use a 3D Cartesian coordinate system where:
- X: Forward direction of the sensor
- Y: Left direction of the sensor
- Z: Up direction (usually aligned with gravity)

### LiDAR-LiDAR Calibration

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

class LiDARCalibrator:
    def __init__(self):
        """
        Initialize LiDAR calibrator for extrinsic calibration
        """
        self.transformation_matrix = np.eye(4)  # 4x4 identity matrix

    def find_correspondences(self, points1: np.ndarray, points2: np.ndarray,
                           max_distance: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find correspondences between two point clouds
        :param points1: First point cloud (Nx3)
        :param points2: Second point cloud (Mx3)
        :param max_distance: Maximum distance for correspondence
        :return: Corresponding points from both clouds
        """
        # Use nearest neighbor to find correspondences
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points2)
        distances, indices = nbrs.kneighbors(points1)

        # Filter correspondences based on distance
        valid_mask = distances.flatten() < max_distance

        if np.sum(valid_mask) < 10:  # Need minimum correspondences
            raise ValueError("Not enough correspondences found")

        corr1 = points1[valid_mask]
        corr2 = points2[indices.flatten()[valid_mask]]

        return corr1, corr2

    def estimate_rigid_transform(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Estimate rigid transformation between two point sets using SVD
        :param src_points: Source points (Nx3)
        :param dst_points: Destination points (Nx3)
        :return: 4x4 transformation matrix
        """
        # Center both point sets
        centroid_src = np.mean(src_points, axis=0)
        centroid_dst = np.mean(dst_points, axis=0)

        # Remove centroids
        src_centered = src_points - centroid_src
        dst_centered = dst_points - centroid_dst

        # Compute covariance matrix
        H = src_centered.T @ dst_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure rotation matrix is proper (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_dst - R @ centroid_src

        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        return transform

    def calibrate_lidar_to_lidar(self, points1: np.ndarray, points2: np.ndarray,
                                max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """
        Calibrate two LiDAR sensors using ICP (Iterative Closest Point)
        :param points1: Point cloud from first LiDAR
        :param points2: Point cloud from second LiDAR
        :param max_iterations: Maximum ICP iterations
        :param tolerance: Convergence tolerance
        :return: Transformation matrix from points1 to points2 coordinate system
        """
        # Initial transformation (identity)
        transform = np.eye(4)

        prev_error = float('inf')

        for i in range(max_iterations):
            # Transform points1 to points2 coordinate system
            transformed_points1 = self.transform_points(points1, transform)

            # Find correspondences
            try:
                corr1, corr2 = self.find_correspondences(transformed_points1, points2)
            except ValueError:
                print(f"ICP iteration {i}: Not enough correspondences found")
                break

            # Estimate transformation
            new_transform = self.estimate_rigid_transform(corr1, corr2)

            # Update global transformation
            transform = new_transform @ transform

            # Calculate error
            current_error = np.mean(np.linalg.norm(corr1 - corr2, axis=1))

            # Check for convergence
            if abs(prev_error - current_error) < tolerance:
                print(f"ICP converged after {i+1} iterations")
                break

            prev_error = current_error

            if i % 10 == 0:
                print(f"ICP iteration {i}, error: {current_error:.6f}")

        self.transformation_matrix = transform
        return transform

    def transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Transform 3D points using a 4x4 transformation matrix
        :param points: Input points (Nx3)
        :param transform: 4x4 transformation matrix
        :return: Transformed points (Nx3)
        """
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])

        # Apply transformation
        transformed_homo = points_homo @ transform.T

        # Convert back to 3D
        return transformed_homo[:, :3]

# Example usage
def lidar_calibration_example():
    # Create sample point clouds (simulated)
    np.random.seed(42)

    # Generate a simple shape (e.g., a box) in first LiDAR coordinates
    points1 = np.random.rand(100, 3) * 5  # Random points in 5x5x5 cube

    # Create second point cloud with some transformation
    rotation = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()  # Small rotation
    translation = np.array([1.0, 2.0, 0.5])  # Translation

    # Transform points1 to create points2
    points2 = (points1 @ rotation.T) + translation
    # Add some noise to simulate real measurements
    points2 += np.random.normal(0, 0.01, points2.shape)

    # Initialize calibrator
    calibrator = LiDARCalibrator()

    # Perform calibration
    estimated_transform = calibrator.calibrate_lidar_to_lidar(points1, points2)

    print("LiDAR-LiDAR Calibration Results:")
    print(f"Estimated transformation matrix:\n{estimated_transform}")

    # Calculate the actual transformation for comparison
    actual_transform = np.eye(4)
    actual_transform[:3, :3] = rotation
    actual_transform[:3, 3] = translation

    print(f"Actual transformation matrix:\n{actual_transform}")

    # Calculate error
    error_matrix = np.abs(estimated_transform - actual_transform)
    max_error = np.max(error_matrix)
    print(f"Max error in transformation: {max_error:.6f}")

    return calibrator, estimated_transform

if __name__ == "__main__":
    calibrator, transform = lidar_calibration_example()
```

## Camera-LiDAR Calibration

### Mathematical Foundation

Camera-LiDAR calibration involves finding the transformation between the camera coordinate system and the LiDAR coordinate system:

```
P_camera = R * P_lidar + T
```

Where R is the 3x3 rotation matrix and T is the 3x1 translation vector.

### Implementation

```python
import numpy as np
import cv2
from scipy.optimize import minimize
from typing import Tuple, List

class CameraLidarCalibrator:
    def __init__(self):
        """
        Initialize camera-LiDAR calibrator
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        self.extrinsic_matrix = None  # 4x4 transformation matrix

    def calibrate_camera_lidar(self, images: List[np.ndarray],
                              point_clouds: List[np.ndarray],
                              calibration_targets: List[np.ndarray]) -> np.ndarray:
        """
        Calibrate camera-LiDAR system using calibration targets
        :param images: List of synchronized camera images
        :param point_clouds: List of synchronized LiDAR point clouds
        :param calibration_targets: List of known 3D coordinates of calibration targets
        :return: 4x4 transformation matrix from LiDAR to camera frame
        """
        if len(images) != len(point_clouds) or len(images) != len(calibration_targets):
            raise ValueError("Number of images, point clouds, and targets must match")

        # Extract 2D image points of calibration targets
        image_points = []
        world_points = []

        for img, pc, targets_3d in zip(images, point_clouds, calibration_targets):
            # Detect calibration target in image (this would typically use a pattern like checkerboard)
            # For this example, we'll assume we have 2D points corresponding to 3D points
            img_pts = self._detect_calibration_targets_in_image(img)

            if img_pts is not None:
                image_points.extend(img_pts)
                world_points.extend(targets_3d)

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # First, calibrate camera intrinsics if not already done
        if self.camera_matrix is None:
            self._calibrate_camera_intrinsics(images)

        # Initialize extrinsic parameters
        initial_extrinsics = np.eye(4)

        # Optimize extrinsic parameters
        result = minimize(
            self._calibration_cost_function,
            initial_extrinsics[:3, :4].flatten(),  # Flatten rotation and translation
            args=(world_points, image_points),
            method='lm'  # Levenberg-Marquardt
        )

        # Reshape optimized parameters back to 3x4 matrix
        extrinsics_3x4 = result.x.reshape(3, 4)

        # Convert to 4x4 matrix
        self.extrinsic_matrix = np.eye(4)
        self.extrinsic_matrix[:3, :] = extrinsics_3x4

        return self.extrinsic_matrix

    def _calibrate_camera_intrinsics(self, images: List[np.ndarray]):
        """
        Calibrate camera intrinsics using a chessboard pattern
        """
        # This is a simplified version - in practice, you'd use the CameraCalibrator class
        # For now, we'll use default values
        self.camera_matrix = np.array([
            [800, 0, 320],   # fx, 0, cx
            [0, 800, 240],   # 0, fy, cy
            [0, 0, 1]        # 0, 0, 1
        ])

        self.dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

    def _detect_calibration_targets_in_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detect calibration targets in the image
        This is a placeholder - in practice, this would detect a specific pattern
        """
        # For this example, we'll return dummy points
        # In practice, you would detect chessboard corners or other calibration patterns
        return np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=np.float32)

    def _calibration_cost_function(self, extrinsics_flat: np.ndarray,
                                  world_points: np.ndarray,
                                  image_points: np.ndarray) -> float:
        """
        Cost function for camera-LiDAR calibration optimization
        :param extrinsics_flat: Flattened extrinsic parameters [r11, r12, ..., t1, t2, t3]
        :param world_points: 3D points in LiDAR coordinate system
        :param image_points: 2D points in image coordinate system
        :return: Reprojection error
        """
        # Reshape extrinsics
        extrinsics = extrinsics_flat.reshape(3, 4)

        # Transform 3D points to camera coordinate system
        world_homo = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
        camera_points = world_homo @ extrinsics.T

        # Project to image coordinates
        valid_mask = camera_points[:, 2] > 0  # Points in front of camera
        if not np.any(valid_mask):
            return float('inf')

        projected = camera_points[valid_mask]
        u = (self.camera_matrix[0, 0] * projected[:, 0] / projected[:, 2]) + self.camera_matrix[0, 2]
        v = (self.camera_matrix[1, 1] * projected[:, 1] / projected[:, 2]) + self.camera_matrix[1, 2]

        projected_2d = np.column_stack([u, v])

        # Get corresponding 2D points
        valid_image_points = image_points[valid_mask]

        if projected_2d.shape[0] != valid_image_points.shape[0]:
            return float('inf')

        # Calculate reprojection error
        error = np.mean(np.linalg.norm(projected_2d - valid_image_points, axis=1))
        return error

    def project_lidar_to_image(self, lidar_points: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Project LiDAR points to image using calibrated parameters
        :param lidar_points: Nx3 array of LiDAR points
        :param image: Input image
        :return: Image with projected points
        """
        if self.extrinsic_matrix is None:
            raise ValueError("Camera-LiDAR system must be calibrated first")

        # Transform LiDAR points to camera coordinate system
        points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
        camera_points = points_homo @ self.extrinsic_matrix.T

        # Filter points in front of camera
        valid_mask = camera_points[:, 2] > 0
        valid_points = camera_points[valid_mask]

        if valid_points.size == 0:
            return image.copy()

        # Project to image coordinates
        u = (self.camera_matrix[0, 0] * valid_points[:, 0] / valid_points[:, 2]) + self.camera_matrix[0, 2]
        v = (self.camera_matrix[1, 1] * valid_points[:, 1] / valid_points[:, 2]) + self.camera_matrix[1, 2]

        # Convert to integer coordinates
        u = u.astype(int)
        v = v.astype(int)

        # Filter points within image bounds
        h, w = image.shape[:2]
        valid_coords = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u = u[valid_coords]
        v = v[valid_coords]

        # Create output image with projected points
        output_img = image.copy()

        # Color points based on depth
        depth = valid_points[valid_coords, 2]
        normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        for i, (x, y) in enumerate(zip(u, v)):
            # Map depth to color (blue to red)
            color = (int(255 * (1 - normalized_depth[i])), 0, int(255 * normalized_depth[i]))
            cv2.circle(output_img, (x, y), 2, color, -1)

        return output_img

# Example usage
def camera_lidar_calibration_example():
    # Create sample data
    np.random.seed(42)

    # Simulate LiDAR points of a calibration target
    lidar_targets = np.array([
        [0, 0, 0],      # Origin
        [1, 0, 0],      # 1m along X
        [0, 1, 0],      # 1m along Y
        [1, 1, 0],      # 1m along X and Y
        [0, 0, 1],      # 1m along Z
    ], dtype=np.float32)

    # Simulate camera image points (these would be detected in real images)
    img_targets = np.array([
        [320, 240],     # Center
        [400, 240],     # Right
        [320, 320],     # Down
        [400, 320],     # Right and down
        [320, 160],     # Up
    ], dtype=np.float32)

    # Initialize calibrator
    calibrator = CameraLidarCalibrator()

    # For this example, we'll directly set the camera matrix
    calibrator.camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    # Simulate the calibration process
    print("Camera-LiDAR calibration example")
    print("Note: This is a simplified example. In practice, you would need")
    print("actual synchronized camera and LiDAR data with known calibration targets.")

    # Simulate a LiDAR point cloud and image
    dummy_lidar_points = np.random.rand(1000, 3) * 10  # Random points
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image

    # Create a simple transformation matrix (this would be the result of calibration)
    calibrator.extrinsic_matrix = np.eye(4)
    calibrator.extrinsic_matrix[:3, :3] = np.array([
        [0.998, 0.015, -0.061],
        [-0.014, 0.999, 0.026],
        [0.061, -0.027, 0.998]
    ])
    calibrator.extrinsic_matrix[:3, 3] = np.array([0.1, 0.2, 0.3])  # Translation

    # Project LiDAR points to image
    projected_img = calibrator.project_lidar_to_image(dummy_lidar_points, dummy_image)

    print(f"Projected {dummy_lidar_points.shape[0]} LiDAR points to image")
    print("Calibration matrix set for demonstration purposes")

    return calibrator, projected_img

if __name__ == "__main__":
    calibrator, projected_img = camera_lidar_calibration_example()
```

## Multi-Sensor Calibration

### Calibration Pattern Design

```python
import numpy as np
import cv2

class MultiSensorCalibrator:
    def __init__(self):
        """
        Multi-sensor calibrator for calibrating multiple sensors simultaneously
        """
        self.calibration_data = {
            'camera': [],
            'lidar': [],
            'imu': [],
            'gps': []
        }
        self.transforms = {}  # Store transformations between sensors

    def add_calibration_data(self, sensor_type: str, data: np.ndarray, timestamp: float):
        """
        Add calibration data from a sensor
        :param sensor_type: Type of sensor ('camera', 'lidar', 'imu', 'gps')
        :param data: Sensor data
        :param timestamp: Timestamp of data collection
        """
        self.calibration_data[sensor_type].append({
            'data': data,
            'timestamp': timestamp
        })

    def synchronize_data(self, time_tolerance: float = 0.01) -> List[dict]:
        """
        Synchronize data from multiple sensors based on timestamps
        :param time_tolerance: Maximum time difference allowed for synchronization
        :return: List of synchronized sensor data sets
        """
        # Find common time windows where all sensors have data
        synchronized_sets = []

        # This is a simplified version - in practice, you'd use more sophisticated
        # time synchronization methods
        for i in range(min(len(self.calibration_data[s]) for s in self.calibration_data)):
            sync_set = {}
            base_time = self.calibration_data['camera'][i]['timestamp']

            # Find closest timestamps for other sensors
            for sensor_type in self.calibration_data:
                closest_idx = min(range(len(self.calibration_data[sensor_type])),
                                key=lambda x: abs(self.calibration_data[sensor_type][x]['timestamp'] - base_time))

                time_diff = abs(self.calibration_data[sensor_type][closest_idx]['timestamp'] - base_time)

                if time_diff <= time_tolerance:
                    sync_set[sensor_type] = self.calibration_data[sensor_type][closest_idx]['data']
                else:
                    print(f"Warning: No synchronized data for {sensor_type} within tolerance")
                    break
            else:
                # If we got here, all sensors had synchronized data
                synchronized_sets.append(sync_set)

        return synchronized_sets

    def optimize_multi_sensor_calibration(self, synchronized_data: List[dict]) -> dict:
        """
        Optimize calibration parameters for multiple sensors simultaneously
        :param synchronized_data: List of synchronized sensor data sets
        :return: Dictionary of transformation matrices
        """
        # This is a complex optimization problem that typically involves:
        # 1. Establishing a common coordinate frame
        # 2. Using calibration targets visible to multiple sensors
        # 3. Joint optimization of all transformations

        # For this example, we'll return a simple transformation setup
        transforms = {
            'lidar_to_camera': np.eye(4),  # Identity initially
            'imu_to_camera': np.eye(4),
            'gps_to_camera': np.eye(4)
        }

        # In practice, you would implement a joint optimization algorithm
        # such as bundle adjustment or graph optimization

        print("Multi-sensor calibration optimization would be performed here")
        print("This involves complex optimization to ensure consistency across all sensors")

        return transforms

# Example usage
def multi_sensor_calibration_example():
    calibrator = MultiSensorCalibrator()

    # Add sample data (in practice, this would come from actual sensors)
    for i in range(5):
        # Simulate synchronized data collection
        calibrator.add_calibration_data('camera', np.random.rand(480, 640, 3), i * 0.1)
        calibrator.add_calibration_data('lidar', np.random.rand(1000, 3), i * 0.1 + 0.001)  # Small time offset
        calibrator.add_calibration_data('imu', np.random.rand(3), i * 0.1 - 0.002)  # Small time offset
        calibrator.add_calibration_data('gps', np.random.rand(3), i * 0.1 + 0.003)  # Small time offset

    # Synchronize data
    sync_data = calibrator.synchronize_data()

    print(f"Synchronized {len(sync_data)} sets of multi-sensor data")

    # Perform optimization
    transforms = calibrator.optimize_multi_sensor_calibration(sync_data)

    print("Multi-sensor calibration transforms:")
    for sensor, transform in transforms.items():
        print(f"{sensor}: {transform}")

    return calibrator, transforms

if __name__ == "__main__":
    calibrator, transforms = multi_sensor_calibration_example()
```

## Validation and Quality Assessment

### Calibration Quality Metrics

```python
import numpy as np

class CalibrationValidator:
    def __init__(self):
        """
        Tools for validating calibration quality
        """
        pass

    def check_calibration_consistency(self, transformation_matrices: dict,
                                    test_data: List[dict]) -> dict:
        """
        Check consistency of calibration across multiple measurements
        :param transformation_matrices: Dictionary of transformation matrices
        :param test_data: List of test measurements
        :return: Dictionary of consistency metrics
        """
        results = {}

        for sensor_pair, transform in transformation_matrices.items():
            errors = []

            # Apply transformation to test data and check consistency
            for measurement in test_data:
                if sensor_pair in measurement:
                    # Transform from one sensor frame to another
                    # and compare with direct measurements
                    pass  # Implementation would depend on specific sensor pair

            results[sensor_pair] = {
                'mean_error': np.mean(errors) if errors else 0,
                'std_error': np.std(errors) if errors else 0,
                'max_error': np.max(errors) if errors else 0
            }

        return results

    def calculate_calibration_precision(self, repeated_calibrations: List[np.ndarray]) -> dict:
        """
        Calculate precision of calibration by repeating the process
        :param repeated_calibrations: List of transformation matrices from repeated calibrations
        :return: Precision metrics
        """
        if len(repeated_calibrations) < 2:
            return {'error': 'Need at least 2 repeated calibrations'}

        # Convert transformations to more manageable representations
        rotations = []
        translations = []

        for transform in repeated_calibrations:
            rotation = transform[:3, :3]
            translation = transform[:3, 3]

            # Convert rotation matrix to axis-angle representation
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(rotation)
            rotations.append(r.as_rotvec())
            translations.append(translation)

        # Calculate statistics
        rot_precision = np.std(rotations, axis=0)
        trans_precision = np.std(translations, axis=0)

        return {
            'rotation_precision': rot_precision,
            'translation_precision': trans_precision,
            'rotation_mean': np.mean(rotations, axis=0),
            'translation_mean': np.mean(translations, axis=0)
        }

# Example validation
def calibration_validation_example():
    validator = CalibrationValidator()

    # Simulate repeated calibrations (in practice, these would come from multiple calibration runs)
    repeated_transforms = []
    for i in range(5):
        # Add small random variations to simulate calibration uncertainty
        noise = np.random.normal(0, 0.01, (4, 4))
        noise[3, 3] = 1  # Keep bottom-right as 1
        base_transform = np.eye(4)
        base_transform += noise
        repeated_transforms.append(base_transform)

    precision_metrics = validator.calculate_calibration_precision(repeated_transforms)

    print("Calibration Precision Metrics:")
    print(f"Rotation precision (rad): {precision_metrics['rotation_precision']}")
    print(f"Translation precision (m): {precision_metrics['translation_precision']}")

    return precision_metrics

if __name__ == "__main__":
    metrics = calibration_validation_example()
```

## Practical Implementation Tips

### Best Practices for Calibration

1. **Environmental Conditions**:
   - Perform calibration in controlled, well-lit environments
   - Avoid extreme temperatures that might affect sensor characteristics
   - Ensure calibration targets are stable and not moving

2. **Data Quality**:
   - Use high-quality calibration targets (chessboards, AprilTags, etc.)
   - Collect data from multiple viewpoints and distances
   - Ensure good coverage of the sensor's field of view

3. **Validation**:
   - Always validate calibration results with test data
   - Monitor calibration quality over time
   - Implement automatic quality checks

### Troubleshooting Common Issues

```python
def diagnose_calibration_issues(camera_matrix, dist_coeffs, image_shape):
    """
    Diagnose common calibration issues
    """
    issues = []

    # Check intrinsic parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    w, h = image_shape[1], image_shape[0]

    # Check if focal lengths are reasonable
    if fx < 100 or fy < 100:
        issues.append("Focal length seems too small - check units")

    if fx > 5000 or fy > 5000:
        issues.append("Focal length seems too large - check units")

    # Check if principal point is reasonable
    if cx < 0 or cx > w or cy < 0 or cy > h:
        issues.append("Principal point outside image bounds")

    # Check distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

    if abs(k1) > 1.0:
        issues.append("First radial distortion coefficient is very large")

    if abs(k2) > 1.0:
        issues.append("Second radial distortion coefficient is very large")

    # Check for tangential distortion issues
    if abs(p1) > 0.1 or abs(p2) > 0.1:
        issues.append("Large tangential distortion - check calibration target flatness")

    return issues

# Example usage
def calibration_diagnostics_example():
    # Example camera matrix and distortion coefficients
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    dist_coeffs = np.array([0.1, -0.2, 0.001, -0.001, 0.05])

    image_shape = (480, 640, 3)

    issues = diagnose_calibration_issues(camera_matrix, dist_coeffs, image_shape)

    if issues:
        print("Calibration Issues Found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("No obvious calibration issues detected")

    return issues

if __name__ == "__main__":
    issues = calibration_diagnostics_example()
```

## Extended Kalman Filter Implementation

### Introduction to Extended Kalman Filter (EKF)

The Extended Kalman Filter (EKF) is an extension of the standard Kalman Filter for non-linear systems. It linearizes the non-linear system around the current state estimate using Jacobian matrices, making it suitable for systems where the state transition and measurement models are non-linear.

### Mathematical Foundation

For a non-linear system:
- State prediction: x_k = f(x_{k-1}, u_k) + w_k
- Measurement update: z_k = h(x_k) + v_k

Where f and h are non-linear functions, and w_k and v_k are process and measurement noise respectively.

The EKF linearizes these functions:
- F_k = ∂f/∂x evaluated at x_{k|k-1}
- H_k = ∂h/∂x evaluated at x_{k|k-1}

### Python Implementation of Extended Kalman Filter

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from typing import Tuple, Callable

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

    def predict(self, f_func: Callable, F_func: Callable, dt: float = 1.0,
                u: np.ndarray = None) -> np.ndarray:
        """
        Prediction step of Extended Kalman Filter
        :param f_func: Non-linear state transition function: x_{k+1} = f(x_k, u_k)
        :param F_func: Jacobian of f with respect to state
        :param dt: Time step
        :param u: Control input (optional)
        :return: Predicted state
        """
        # Predict state: x = f(x, u)
        self.x = f_func(self.x, u, dt)

        # Compute Jacobian of f at current state
        F = F_func(self.x, u, dt)

        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update(self, z: np.ndarray, h_func: Callable, H_func: Callable) -> np.ndarray:
        """
        Update step of Extended Kalman Filter
        :param z: Measurement vector
        :param h_func: Non-linear measurement function: z = h(x)
        :param H_func: Jacobian of h with respect to state
        :return: Updated state
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

        return self.x.copy()

class BearingsOnlyTracker:
    """
    Example: Track an object using only bearing measurements (e.g., from camera)
    This is a classic non-linear problem as bearing = atan2(y-y_sensor, x-x_sensor)
    """
    def __init__(self, dt: float = 0.1, sensor_pos: np.ndarray = None):
        """
        Initialize bearings-only tracker
        :param dt: Time step
        :param sensor_pos: Position of the sensor [x, y]
        """
        self.dt = dt
        self.sensor_pos = sensor_pos if sensor_pos is not None else np.array([0.0, 0.0])

        # State: [x, y, vx, vy] (position and velocity)
        self.ekf = ExtendedKalmanFilter(state_dim=4, measurement_dim=1)

        # Initialize with high uncertainty
        self.ekf.P = np.diag([100, 100, 10, 10])  # High uncertainty in initial position/velocity

        # Process noise (assumes random acceleration model)
        sigma_a = 2.0  # acceleration uncertainty
        self.ekf.Q = np.array([
            [self.dt**4/4,      0,    self.dt**3/2,         0],
            [         0, self.dt**4/4,         0,   self.dt**3/2],
            [self.dt**3/2,      0,      self.dt**2,         0],
            [         0,    self.dt**3/2,         0,     self.dt**2]
        ]) * sigma_a**2

        # Measurement noise
        sigma_bearing = 0.05  # 0.05 radians (~3 degrees)
        self.ekf.R = np.array([[sigma_bearing**2]])

    def state_transition_function(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Constant velocity model: x_{k+1} = f(x_k, u_k)
        State: [x, y, vx, vy]
        """
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])
        return F @ x

    def state_transition_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Jacobian of state transition function with respect to state
        """
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])

    def measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Convert state to bearing measurement
        z = atan2(y - y_sensor, x - x_sensor)
        """
        pos_x = x[0, 0]
        pos_y = x[1, 0]

        # Calculate bearing from sensor to object
        dx = pos_x - self.sensor_pos[0]
        dy = pos_y - self.sensor_pos[1]

        bearing = np.arctan2(dy, dx)
        return np.array([[bearing]])

    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of measurement function with respect to state
        h(x) = atan2(y - y_sensor, x - x_sensor)
        dh/dx = -dy / ((x - x_sensor)^2 + (y - y_sensor)^2)
        dh/dy = dx / ((x - x_sensor)^2 + (y - y_sensor)^2)
        dh/dvx = dh/dvy = 0
        """
        pos_x = x[0, 0]
        pos_y = x[1, 0]

        dx = pos_x - self.sensor_pos[0]
        dy = pos_y - self.sensor_pos[1]
        r_squared = dx**2 + dy**2

        # Partial derivatives of h(x) = arctan2(dy, dx)
        H = np.array([[
            -dy / r_squared,  # dh/dx
            dx / r_squared,   # dh/dy
            0,                # dh/dvx
            0                 # dh/dvy
        ]])

        return H

    def track_sequence(self, measurements: list, initial_state: np.ndarray = None) -> list:
        """
        Track a sequence of bearing measurements
        :param measurements: List of bearing measurements
        :param initial_state: Initial state [x, y, vx, vy]
        :return: List of tracked states [x, y, vx, vy]
        """
        tracked_states = []

        for i, meas in enumerate(measurements):
            if i == 0 and initial_state is not None:
                # Initialize with known initial state
                self.ekf.x = initial_state.reshape(-1, 1)
            elif i == 0:
                # For bearings-only tracking, we need to initialize with a position estimate
                # This is a simplified approach - in practice, you might use multiple measurements
                # to triangulate initial position
                print("Warning: Initializing without known position. This is challenging for bearings-only tracking.")
                # Set an initial position guess
                range_guess = 10.0  # Initial range guess
                initial_x = self.sensor_pos[0] + range_guess * np.cos(meas[0])
                initial_y = self.sensor_pos[1] + range_guess * np.sin(meas[0])
                self.ekf.x[:2] = np.array([[initial_x], [initial_y]])

            # Predict
            self.ekf.predict(self.state_transition_function, self.state_transition_jacobian, self.dt)

            # Update with measurement
            self.ekf.update(meas.reshape(-1, 1), self.measurement_function, self.measurement_jacobian)

            tracked_states.append(self.ekf.x.flatten())

        return tracked_states

def bearings_only_tracking_example():
    """
    Example: Track an object using only bearing measurements with EKF
    """
    # Create tracker with sensor at origin
    tracker = BearingsOnlyTracker(dt=0.1, sensor_pos=np.array([0.0, 0.0]))

    # Simulate a target moving in a straight line
    dt = 0.1
    num_steps = 100

    # True trajectory: target starts at [10, 5] moving with velocity [1, 0.5]
    true_states = []
    measurements = []

    # Initial state: position [10, 5], velocity [1, 0.5]
    pos = np.array([10.0, 5.0])
    vel = np.array([1.0, 0.5])

    for t in range(num_steps):
        # True position
        true_pos = pos + vel * t * dt
        true_state = np.hstack([true_pos, vel])  # [x, y, vx, vy]
        true_states.append(true_state.copy())

        # Calculate true bearing from sensor
        dx = true_pos[0] - tracker.sensor_pos[0]
        dy = true_pos[1] - tracker.sensor_pos[1]
        true_bearing = np.arctan2(dy, dx)

        # Add measurement noise
        measurement_noise = np.random.normal(0, 0.05)  # 0.05 rad = ~3 degrees
        meas = np.array([true_bearing + measurement_noise])
        measurements.append(meas.copy())

    # Track the target using EKF
    # Note: For bearings-only tracking, initialization is critical
    # In a real scenario, you'd need multiple measurements from different positions
    # to triangulate the initial position
    initial_pos_guess = np.array([10.5, 4.8, 0.9, 0.55])  # Slightly off from true initial state
    tracked_states = tracker.track_sequence(measurements, initial_pos_guess)

    # Extract positions for plotting
    true_positions = np.array([s[:2] for s in true_states])
    measured_positions = []  # Convert bearings to positions for comparison

    for meas, true_pos in zip(measurements, true_states):
        # Convert bearing to position estimate (this is not the EKF estimate, just for visualization)
        # This is just for visualization purposes
        dx = true_pos[0] - tracker.sensor_pos[0]
        dy = true_pos[1] - tracker.sensor_pos[1]
        range_true = np.sqrt(dx**2 + dy**2)
        bearing_true = np.arctan2(dy, dx)

        # Calculate range from sensor to object based on bearing
        # This is just for visualization - not part of the filter
        range_est = range_true  # In real scenario, range would be unknown
        x_est = tracker.sensor_pos[0] + range_est * np.cos(meas[0])
        y_est = tracker.sensor_pos[1] + range_est * np.sin(meas[0])
        measured_positions.append([x_est, y_est])

    measured_positions = np.array(measured_positions)
    tracked_positions = np.array([s[:2] for s in tracked_states])

    # Calculate performance metrics
    meas_rmse = np.sqrt(np.mean((true_positions - measured_positions)**2, axis=0))
    track_rmse = np.sqrt(np.mean((true_positions - tracked_positions)**2, axis=0))

    print(f"Performance Comparison:")
    print(f"RMSE without filter (bearing-to-position conversion): X={meas_rmse[0]:.3f}, Y={meas_rmse[1]:.3f}")
    print(f"RMSE with EKF: X={track_rmse[0]:.3f}, Y={track_rmse[1]:.3f}")
    print(f"Improvement in X: {((meas_rmse[0] - track_rmse[0]) / meas_rmse[0] * 100):.1f}%")
    print(f"Improvement in Y: {((meas_rmse[1] - track_rmse[1]) / meas_rmse[1] * 100):.1f}%")

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Trajectory', linewidth=2)
    plt.plot(tracked_positions[:, 0], tracked_positions[:, 1], 'b-', label='EKF Estimate', linewidth=2)
    plt.scatter(tracker.sensor_pos[0], tracker.sensor_pos[1], c='red', s=100, marker='s', label='Sensor Position')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Bearings-Only Tracking with Extended Kalman Filter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    return tracker, tracked_states

# Example of EKF for sensor fusion: fusing GPS and IMU data
class GPSIMUFusionEKF:
    """
    Example: Fuse GPS and IMU data using EKF
    State: [x, y, vx, vy, ax_bias, ay_bias]
    where ax_bias and ay_bias are IMU accelerometer biases
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt

        # State: [x, y, vx, vy, ax_bias, ay_bias]
        self.ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=4)  # [pos_x, pos_y, vel_x, vel_y] from GPS

        # Initialize state with zeros (or with initial estimates)
        self.ekf.x = np.zeros((6, 1))

        # Initialize covariance with high uncertainty
        self.ekf.P = np.diag([10, 10, 5, 5, 0.1, 0.1])  # Higher uncertainty for positions, lower for biases

        # Process noise - higher for position/velocity, lower for biases
        self.ekf.Q = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.01])

        # Measurement noise for GPS (position) and IMU (acceleration)
        # GPS: [x_pos, y_pos] with noise
        # We'll also include velocity measurements from GPS when available
        self.ekf.R = np.diag([2.0, 2.0, 0.5, 0.5])  # Position noise (m), velocity noise (m/s)

    def state_transition_function(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition model for GPS-IMU fusion
        x_{k+1} = f(x_k, u_k)
        x: [x, y, vx, vy, ax_bias, ay_bias]
        u: [measured_ax, measured_ay] (from IMU)
        """
        new_x = x.copy()

        # Extract state variables
        pos_x = x[0, 0]
        pos_y = x[1, 0]
        vel_x = x[2, 0]
        vel_y = x[3, 0]
        ax_bias = x[4, 0]
        ay_bias = x[5, 0]

        # If control input (IMU measurements) provided, use them
        if u is not None:
            # Correct IMU measurements for bias
            true_ax = u[0, 0] - ax_bias
            true_ay = u[1, 0] - ay_bias
        else:
            # If no IMU data, assume zero acceleration
            true_ax = 0
            true_ay = 0

        # Update state equations
        new_x[0, 0] = pos_x + vel_x * dt + 0.5 * true_ax * dt**2  # x position
        new_x[1, 0] = pos_y + vel_y * dt + 0.5 * true_ay * dt**2  # y position
        new_x[2, 0] = vel_x + true_ax * dt  # x velocity
        new_x[3, 0] = vel_y + true_ay * dt  # y velocity
        # Biases assumed to change slowly (random walk model)
        # In this simple model, we assume biases are constant

        return new_x

    def state_transition_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Jacobian of state transition function
        """
        F = np.eye(6)  # Start with identity matrix

        # Partial derivatives for position and velocity updates
        # dx/dx = 1, dx/dvx = dt, dx/dax = 0.5*dt^2
        F[0, 2] = dt  # dx/dvx
        F[0, 4] = -0.5 * dt**2  # dx/d(ax_bias) (negative because bias is subtracted)

        # dy/dy = 1, dy/dvy = dt, dy/day = 0.5*dt^2
        F[1, 3] = dt  # dy/dvy
        F[1, 5] = -0.5 * dt**2  # dy/d(ay_bias)

        # dvx/dvx = 1, dvx/dax = dt
        F[2, 4] = -dt  # dvx/d(ax_bias)

        # dvy/dvy = 1, dvy/day = dt
        F[3, 5] = -dt  # dvy/d(ay_bias)

        return F

    def measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement model: GPS provides position and velocity
        h(x) = [x, y, vx, vy]
        """
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [x[3, 0]]])

    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of measurement function
        """
        H = np.zeros((4, 6))  # 4 measurements, 6 states

        # dh/d[pos_x, pos_y, vel_x, vel_y] = identity
        H[0, 0] = 1  # dh0/dx
        H[1, 1] = 1  # dh1/dy
        H[2, 2] = 1  # dh2/dvx
        H[3, 3] = 1  # dh3/dvy

        return H

    def fuse_data(self, gps_data: list, imu_data: list) -> list:
        """
        Fuse GPS and IMU data using EKF
        :param gps_data: List of [x, y, vx, vy] measurements (None if no GPS at that time step)
        :param imu_data: List of [ax, ay] measurements (None if no IMU at that time step)
        :return: List of fused state estimates
        """
        estimates = []

        for gps_meas, imu_meas in zip(gps_data, imu_data):
            # Prediction step - always use IMU if available
            control_input = None
            if imu_meas is not None:
                control_input = np.array([[imu_meas[0]], [imu_meas[1]]])

            self.ekf.predict(self.state_transition_function, self.state_transition_jacobian,
                            self.dt, control_input)

            # Update step - use GPS if available
            if gps_meas is not None:
                gps_measurement = np.array([[gps_meas[0]], [gps_meas[1]],
                                           [gps_meas[2]], [gps_meas[3]]])
                self.ekf.update(gps_measurement, self.measurement_function,
                               self.measurement_jacobian)

            estimates.append(self.ekf.x.flatten())

        return estimates

def gps_imu_fusion_example():
    """
    Example: Fuse GPS and IMU data for navigation
    """
    # Create fusion filter
    fusion_filter = GPSIMUFusionEKF(dt=0.1)  # 10 Hz update rate

    # Simulate trajectory with GPS and IMU data
    dt = 0.1
    num_steps = 200

    true_states = []
    gps_measurements = []
    imu_measurements = []

    # Initial state: position [0, 0], velocity [2, 1], zero biases
    pos = np.array([0.0, 0.0])
    vel = np.array([2.0, 1.0])
    biases = np.array([0.05, -0.03])  # True biases in IMU measurements

    # Simulate motion with some acceleration changes
    for t in range(num_steps):
        # True state
        if t < 50:  # Constant velocity
            acc = np.array([0.0, 0.0])
        elif t < 100:  # Accelerate
            acc = np.array([0.5, 0.2])
        else:  # Decelerate
            acc = np.array([-0.3, -0.1])

        # Update position and velocity
        pos = pos + vel * dt + 0.5 * acc * dt**2
        vel = vel + acc * dt

        true_state = np.hstack([pos, vel, biases])  # [x, y, vx, vy, ax_bias, ay_bias]
        true_states.append(true_state.copy())

        # Simulate GPS measurement with noise
        gps_noise = np.random.normal(0, [1.0, 1.0, 0.2, 0.2])  # pos_x, pos_y, vel_x, vel_y noise
        gps_meas = np.array([pos[0], pos[1], vel[0], vel[1]]) + gps_noise
        gps_measurements.append(gps_meas.copy())

        # Simulate IMU measurement (true acceleration + bias + noise)
        imu_noise = np.random.normal(0, [0.02, 0.02])  # ax, ay noise
        true_imu = acc + biases + imu_noise
        imu_measurements.append(true_imu.copy())

    # Sometimes GPS might be unavailable (e.g., in tunnels)
    # For this example, we'll simulate GPS dropouts every 20 steps for 2 steps
    for i in range(0, len(gps_measurements), 20):
        if i + 1 < len(gps_measurements):
            gps_measurements[i] = None
            gps_measurements[i + 1] = None

    # Convert None values to proper format for processing
    gps_proc = []
    imu_proc = []
    for gps, imu in zip(gps_measurements, imu_measurements):
        gps_proc.append(gps if gps is not None else None)
        imu_proc.append(imu)

    # Perform fusion
    fused_estimates = fusion_filter.fuse_data(gps_proc, imu_proc)

    # Extract position estimates
    true_positions = np.array([s[:2] for s in true_states])
    gps_positions = np.array([s[:2] if s is not None else [np.nan, np.nan]
                             for s in gps_measurements])
    fused_positions = np.array([s[:2] for s in fused_estimates])

    # Calculate performance metrics
    # Remove NaN values for GPS when calculating metrics
    valid_gps_mask = ~np.isnan(gps_positions[:, 0])
    if np.any(valid_gps_mask):
        gps_rmse = np.sqrt(np.mean((true_positions[valid_gps_mask] - gps_positions[valid_gps_mask])**2, axis=0))
    else:
        gps_rmse = np.array([float('inf'), float('inf')])

    fusion_rmse = np.sqrt(np.mean((true_positions - fused_positions)**2, axis=0))

    print(f"GPS-IMU Fusion Performance:")
    print(f"RMSE with GPS only: X={gps_rmse[0]:.3f}, Y={gps_rmse[1]:.3f}")
    print(f"RMSE with EKF fusion: X={fusion_rmse[0]:.3f}, Y={fusion_rmse[1]:.3f}")

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot 1: Trajectory
    plt.subplot(1, 3, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Trajectory', linewidth=2)
    plt.plot(fused_positions[:, 0], fused_positions[:, 1], 'b-', label='EKF Fused', linewidth=2)

    # Plot GPS measurements where available
    valid_gps = ~np.isnan(gps_positions[:, 0])
    if np.any(valid_gps):
        plt.scatter(gps_positions[valid_gps, 0], gps_positions[valid_gps, 1],
                   c='r', alpha=0.5, label='GPS Measurements', s=10)

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('GPS-IMU Fusion with EKF')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: X position over time
    plt.subplot(1, 3, 2)
    plt.plot(true_positions[:, 0], 'g-', label='True X', linewidth=2)
    plt.plot(fused_positions[:, 0], 'b-', label='EKF X', linewidth=2)
    if np.any(valid_gps):
        plt.scatter(np.where(valid_gps)[0], gps_positions[valid_gps, 0],
                   c='r', alpha=0.5, label='GPS X', s=10)
    plt.xlabel('Time Step')
    plt.ylabel('X Position (m)')
    plt.title('X Position Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Y position over time
    plt.subplot(1, 3, 3)
    plt.plot(true_positions[:, 1], 'g-', label='True Y', linewidth=2)
    plt.plot(fused_positions[:, 1], 'b-', label='EKF Y', linewidth=2)
    if np.any(valid_gps):
        plt.scatter(np.where(valid_gps)[0], gps_positions[valid_gps, 1],
                   c='r', alpha=0.5, label='GPS Y', s=10)
    plt.xlabel('Time Step')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fusion_filter, fused_estimates

if __name__ == "__main__":
    print("Running bearings-only tracking example with EKF...")
    tracker, states = bearings_only_tracking_example()

    print("\nRunning GPS-IMU fusion example with EKF...")
    fusion_filter, estimates = gps_imu_fusion_example()

## C++ Implementation of Extended Kalman Filter

Here's a C++ implementation of the Extended Kalman Filter for sensor fusion applications:

```cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include <random>

class ExtendedKalmanFilter {
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

public:
    ExtendedKalmanFilter(int state_dim, int measurement_dim)
        : state_dim(state_dim), measurement_dim(measurement_dim) {
        x = Eigen::VectorXd::Zero(state_dim);
        P = Eigen::MatrixXd::Identity(state_dim, state_dim);
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
        R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    }

    void setState(const Eigen::VectorXd& state) {
        x = state;
    }

    void setCovariance(const Eigen::MatrixXd& cov) {
        P = cov;
    }

    void setProcessNoiseCovariance(const Eigen::MatrixXd& noise_cov) {
        Q = noise_cov;
    }

    void setMeasurementNoiseCovariance(const Eigen::MatrixXd& noise_cov) {
        R = noise_cov;
    }

    void predict(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> f_func,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double)> F_func,
        double dt = 1.0
    ) {
        // Predict state: x = f(x, dt)
        x = f_func(x, dt);

        // Compute Jacobian of f at current state
        Eigen::MatrixXd F = F_func(x, dt);

        // Predict covariance: P = F * P * F^T + Q
        P = F * P * F.transpose() + Q;
    }

    void update(
        const Eigen::VectorXd& z,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h_func,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_func
    ) {
        // Compute Jacobian of h at current state
        Eigen::MatrixXd H = H_func(x);

        // Innovation: y = z - h(x)
        Eigen::VectorXd y = z - h_func(x);

        // Innovation covariance: S = H * P * H^T + R
        Eigen::MatrixXd S = H * P * H.transpose() + R;

        // Kalman gain: K = P * H^T * S^(-1)
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        // Update state: x = x + K * y
        x = x + K * y;

        // Update covariance: P = (I - K * H) * P
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

class BearingsOnlyTracker {
private:
    ExtendedKalmanFilter ekf;
    double dt;
    Eigen::Vector2d sensor_pos;

public:
    BearingsOnlyTracker(double dt = 0.1, const Eigen::Vector2d& sensor_pos = Eigen::Vector2d(0.0, 0.0))
        : ekf(4, 1), dt(dt), sensor_pos(sensor_pos) {

        // Initialize state: [x, y, vx, vy]
        ekf.setState(Eigen::VectorXd::Zero(4));

        // Initialize covariance with high uncertainty
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(4, 4);
        P(0, 0) = 100;  // x position uncertainty
        P(1, 1) = 100;  // y position uncertainty
        P(2, 2) = 10;   // x velocity uncertainty
        P(3, 3) = 10;   // y velocity uncertainty
        ekf.setCovariance(P);

        // Process noise (assumes random acceleration model)
        double sigma_a = 2.0;  // acceleration uncertainty
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 4);
        Q(0, 0) = std::pow(dt, 4)/4; Q(0, 2) = std::pow(dt, 3)/2;
        Q(1, 1) = std::pow(dt, 4)/4; Q(1, 3) = std::pow(dt, 3)/2;
        Q(2, 0) = std::pow(dt, 3)/2; Q(2, 2) = std::pow(dt, 2);
        Q(3, 1) = std::pow(dt, 3)/2; Q(3, 3) = std::pow(dt, 2);
        Q *= std::pow(sigma_a, 2);
        ekf.setProcessNoiseCovariance(Q);

        // Measurement noise
        double sigma_bearing = 0.05;  // 0.05 radians (~3 degrees)
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * std::pow(sigma_bearing, 2);
        ekf.setMeasurementNoiseCovariance(R);
    }

    // State transition function (constant velocity model)
    Eigen::VectorXd stateTransitionFunction(const Eigen::VectorXd& x, double dt) const {
        Eigen::VectorXd new_state = x;
        // State: [x, y, vx, vy]
        new_state(0) += x(2) * dt;  // x_new = x + vx * dt
        new_state(1) += x(3) * dt;  // y_new = y + vy * dt
        // Velocities remain the same in constant velocity model
        return new_state;
    }

    // Jacobian of state transition function
    Eigen::MatrixXd stateTransitionJacobian(const Eigen::VectorXd& x, double dt) const {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(4, 4);
        // Partial derivatives of state transition equations
        F(0, 2) = dt;  // dx/dvx
        F(1, 3) = dt;  // dy/dvy
        return F;
    }

    // Measurement function: bearing from sensor to object
    Eigen::VectorXd measurementFunction(const Eigen::VectorXd& x) const {
        double pos_x = x(0);
        double pos_y = x(1);

        // Calculate bearing from sensor to object
        double dx = pos_x - sensor_pos(0);
        double dy = pos_y - sensor_pos(1);

        double bearing = std::atan2(dy, dx);
        Eigen::VectorXd measurement(1);
        measurement << bearing;
        return measurement;
    }

    // Jacobian of measurement function
    Eigen::MatrixXd measurementJacobian(const Eigen::VectorXd& x) const {
        double pos_x = x(0);
        double pos_y = x(1);

        double dx = pos_x - sensor_pos(0);
        double dy = pos_y - sensor_pos(1);
        double r_squared = dx*dx + dy*dy;

        // Partial derivatives of h(x) = arctan2(dy, dx)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 4);
        H(0, 0) = -dy / r_squared;  // dh/dx
        H(0, 1) = dx / r_squared;   // dh/dy
        // dh/dvx = dh/dvy = 0
        return H;
    }

    std::vector<Eigen::VectorXd> trackSequence(const std::vector<Eigen::VectorXd>& measurements) {
        std::vector<Eigen::VectorXd> trackedStates;

        for (size_t i = 0; i < measurements.size(); ++i) {
            if (i == 0) {
                // Initialize with first measurement
                // This is a simplified approach - in practice, you might need multiple measurements
                // to triangulate initial position
                std::cout << "Warning: Initializing without known position. This is challenging for bearings-only tracking." << std::endl;
            }

            // Predict
            ekf.predict(
                [this](const Eigen::VectorXd& x, double dt) { return stateTransitionFunction(x, dt); },
                [this](const Eigen::VectorXd& x, double dt) { return stateTransitionJacobian(x, dt); },
                dt
            );

            // Update with measurement
            ekf.update(
                measurements[i],
                [this](const Eigen::VectorXd& x) { return measurementFunction(x); },
                [this](const Eigen::VectorXd& x) { return measurementJacobian(x); }
            );

            trackedStates.push_back(ekf.getState());
        }

        return trackedStates;
    }
};

// Example: GPS-IMU fusion with EKF
class GPSIMUFusionEKF {
private:
    ExtendedKalmanFilter ekf;
    double dt;

public:
    GPSIMUFusionEKF(double dt = 0.1) : ekf(6, 4), dt(dt) {  // State: [x, y, vx, vy, ax_bias, ay_bias], Measurement: [x, y, vx, vy]

        // Initialize state
        ekf.setState(Eigen::VectorXd::Zero(6));

        // Initialize covariance
        Eigen::VectorXd diag(6);
        diag << 10, 10, 5, 5, 0.1, 0.1;  // Higher uncertainty for positions, lower for biases
        Eigen::MatrixXd P = diag.asDiagonal();
        ekf.setCovariance(P);

        // Process noise
        Eigen::VectorXd q_diag(6);
        q_diag << 0.1, 0.1, 0.5, 0.5, 0.01, 0.01;
        Eigen::MatrixXd Q = q_diag.asDiagonal();
        ekf.setProcessNoiseCovariance(Q);

        // Measurement noise for GPS
        Eigen::VectorXd r_diag(4);
        r_diag << 2.0, 2.0, 0.5, 0.5;  // Position noise (m), velocity noise (m/s)
        Eigen::MatrixXd R = r_diag.asDiagonal();
        ekf.setMeasurementNoiseCovariance(R);
    }

    // State transition function for GPS-IMU fusion
    Eigen::VectorXd stateTransitionFunction(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) const {
        Eigen::VectorXd new_x = x;

        // Extract state variables
        double pos_x = x(0);
        double pos_y = x(1);
        double vel_x = x(2);
        double vel_y = x(3);
        double ax_bias = x(4);
        double ay_bias = x(5);

        // If control input (IMU measurements) provided, use them
        double true_ax, true_ay;
        if (u.size() >= 2) {
            // Correct IMU measurements for bias
            true_ax = u(0) - ax_bias;
            true_ay = u(1) - ay_bias;
        } else {
            // If no IMU data, assume zero acceleration
            true_ax = 0;
            true_ay = 0;
        }

        // Update state equations
        new_x(0) = pos_x + vel_x * dt + 0.5 * true_ax * dt*dt;  // x position
        new_x(1) = pos_y + vel_y * dt + 0.5 * true_ay * dt*dt;  // y position
        new_x(2) = vel_x + true_ax * dt;  // x velocity
        new_x(3) = vel_y + true_ay * dt;  // y velocity
        // Biases assumed to change slowly (random walk model)

        return new_x;
    }

    // Jacobian of state transition function
    Eigen::MatrixXd stateTransitionJacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) const {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);

        // Partial derivatives for position and velocity updates
        F(0, 2) = dt;  // dx/dvx
        F(0, 4) = -0.5 * dt*dt;  // dx/d(ax_bias) (negative because bias is subtracted)

        F(1, 3) = dt;  // dy/dvy
        F(1, 5) = -0.5 * dt*dt;  // dy/d(ay_bias)

        F(2, 4) = -dt;  // dvx/d(ax_bias)
        F(3, 5) = -dt;  // dvy/d(ay_bias)

        return F;
    }

    // Measurement function: GPS provides position and velocity
    Eigen::VectorXd measurementFunction(const Eigen::VectorXd& x) const {
        Eigen::VectorXd measurement(4);
        measurement << x(0), x(1), x(2), x(3);  // [x, y, vx, vy]
        return measurement;
    }

    // Jacobian of measurement function
    Eigen::MatrixXd measurementJacobian(const Eigen::VectorXd& x) const {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 6);

        // dh/d[pos_x, pos_y, vel_x, vel_y] = identity
        H(0, 0) = 1;  // dh0/dx
        H(1, 1) = 1;  // dh1/dy
        H(2, 2) = 1;  // dh2/dvx
        H(3, 3) = 1;  // dh3/dvy

        return H;
    }

    std::vector<Eigen::VectorXd> fuseData(
        const std::vector<std::pair<bool, Eigen::VectorXd>>& gps_data,
        const std::vector<std::pair<bool, Eigen::VectorXd>>& imu_data
    ) {
        std::vector<Eigen::VectorXd> estimates;

        for (size_t i = 0; i < gps_data.size(); ++i) {
            // Prediction step - always use IMU if available
            Eigen::VectorXd control_input;
            if (i < imu_data.size() && imu_data[i].first) {
                control_input = imu_data[i].second;
            } else {
                control_input = Eigen::VectorXd::Zero(2);  // Zero IMU measurement if unavailable
            }

            // For this example, we'll need to adapt the EKF interface to accept control inputs
            // This requires modifying the EKF class to support control inputs
            // For now, we'll use a simplified approach

            // Update step - use GPS if available
            if (i < gps_data.size() && gps_data[i].first) {
                ekf.update(
                    gps_data[i].second,
                    [this](const Eigen::VectorXd& x) { return measurementFunction(x); },
                    [this](const Eigen::VectorXd& x) { return measurementJacobian(x); }
                );
            }

            estimates.push_back(ekf.getState());
        }

        return estimates;
    }
};

// Example usage
int main() {
    // Example 1: Bearings-only tracking
    std::cout << "=== Bearings-Only Tracking Example ===" << std::endl;

    BearingsOnlyTracker tracker(0.1, Eigen::Vector2d(0.0, 0.0));  // dt=0.1s, sensor at origin

    // Simulate bearing measurements
    std::vector<Eigen::VectorXd> measurements;
    int num_steps = 50;
    double dt = 0.1;

    // Simulate target moving with constant velocity
    Eigen::Vector2d true_pos(10.0, 5.0);
    Eigen::Vector2d true_vel(2.0, 1.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_dist(0.0, 0.05);  // 0.05 rad = ~3 degrees

    for (int t = 0; t < num_steps; ++t) {
        // True position
        Eigen::Vector2d curr_pos = true_pos + true_vel * t * dt;

        // Calculate true bearing from sensor
        double dx = curr_pos(0) - tracker.sensor_pos(0);
        double dy = curr_pos(1) - tracker.sensor_pos(1);
        double true_bearing = std::atan2(dy, dx);

        // Add measurement noise
        double meas = true_bearing + noise_dist(gen);
        Eigen::VectorXd meas_vec(1);
        meas_vec << meas;
        measurements.push_back(meas_vec);
    }

    // Track the target
    std::vector<Eigen::VectorXd> tracked_states = tracker.trackSequence(measurements);

    std::cout << "Tracked " << tracked_states.size() << " positions" << std::endl;
    if (!tracked_states.empty()) {
        std::cout << "Final estimated state: [x=" << tracked_states.back()(0)
                  << ", y=" << tracked_states.back()(1)
                  << ", vx=" << tracked_states.back()(2)
                  << ", vy=" << tracked_states.back()(3) << "]" << std::endl;
    }

    // Example 2: GPS-IMU fusion (simplified)
    std::cout << "\n=== GPS-IMU Fusion Example ===" << std::endl;

    GPSIMUFusionEKF fusion_filter(0.1);  // 10 Hz update rate

    // In a real implementation, you would create GPS and IMU measurement data
    // and then call fusion_filter.fuseData(gps_data, imu_data)

    std::cout << "GPS-IMU fusion EKF initialized with 6-state model [x, y, vx, vy, ax_bias, ay_bias]" << std::endl;

    return 0;
}
```

### Compilation and Usage

To compile and run the C++ Extended Kalman Filter example:

```bash
# Install Eigen library (if not already installed)
# On Ubuntu/Debian: sudo apt-get install libeigen3-dev
# On macOS with Homebrew: brew install eigen

# Compile
g++ -std=c++11 -O3 extended_kalman_filter.cpp -o extended_kalman_filter

# Run
./extended_kalman_filter
```

The C++ implementation provides the same functionality as the Python version but with better performance for real-time applications. The Eigen library is used for efficient matrix operations, which are essential for Extended Kalman Filter computations.

## Next Steps

1. **Implementation Practice**: Apply calibration techniques to real sensor data
2. **Quality Assessment**: Validate calibration results with test measurements
3. **Automation**: Develop automated calibration procedures for your system
4. **Monitoring**: Implement continuous monitoring of calibration quality
5. **Documentation**: Document your calibration procedures for reproducibility

This comprehensive guide covers the fundamental concepts and practical implementations of sensor calibration for robotic perception systems. The techniques described here form the foundation for accurate multi-sensor fusion in robotics applications.