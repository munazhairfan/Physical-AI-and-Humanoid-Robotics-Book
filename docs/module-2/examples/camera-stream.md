---
title: Camera Stream Processing
sidebar_label: Camera Stream
description: Processing real-time camera streams for robotic perception
slug: /module-2/examples/camera-stream
---

# Camera Stream Processing

## Summary

This example demonstrates how to process real-time camera streams for robotic perception applications. Students will learn to capture, process, and analyze video streams from various camera types, implementing real-time filtering and feature extraction techniques.

## Learning Objectives

By the end of this example, students will be able to:
- Capture and process real-time camera streams
- Implement real-time image processing techniques
- Apply filtering and feature extraction to video streams
- Optimize performance for real-time applications
- Integrate camera processing with robotic systems

## Table of Contents

1. [Basic Camera Stream Capture](#basic-camera-stream-capture)
2. [Real-time Image Processing](#real-time-image-processing)
3. [Performance Optimization](#performance-optimization)
4. [Integration with Robotics](#integration-with-robotics)
5. [Advanced Applications](#advanced-applications)

## Basic Camera Stream Capture

### Simple Camera Stream

The most basic camera stream implementation captures frames from a webcam:

```python
import cv2
import numpy as np

def basic_camera_stream():
    """Basic camera stream capture and display"""
    # Initialize camera (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Can't receive frame")
                break

            # Display the resulting frame
            cv2.imshow('Camera Stream', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release everything
        cap.release()
        cv2.destroyAllWindows()

# Run the basic camera stream
if __name__ == "__main__":
    basic_camera_stream()
```

### Camera Configuration

Different cameras may require specific configuration:

```python
def configure_camera():
    """Configure camera properties"""
    cap = cv2.VideoCapture(0)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
    cap.set(cv2.CAP_PROP_SATURATION, 0.5)

    # Print actual values
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    return cap

def camera_with_config():
    """Camera stream with configuration"""
    cap = configure_camera()

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add frame information
            height, width = frame.shape[:2]
            cv2.putText(frame, f'{width}x{height}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Configured Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_with_config()
```

## Real-time Image Processing

### Real-time Filtering

Apply filters to camera streams in real-time:

```python
import cv2
import numpy as np

class RealTimeProcessor:
    def __init__(self):
        self.current_filter = 'original'
        self.filters = {
            'original': self.original,
            'grayscale': self.grayscale,
            'gaussian': self.gaussian_blur,
            'edges': self.canny_edges,
            'threshold': self.threshold
        }

    def original(self, frame):
        """Return original frame"""
        return frame

    def grayscale(self, frame):
        """Convert to grayscale"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def gaussian_blur(self, frame):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def canny_edges(self, frame):
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def threshold(self, frame):
        """Apply binary threshold"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def process_frame(self, frame):
        """Apply current filter to frame"""
        return self.filters[self.current_filter](frame)

def real_time_filtering():
    """Real-time filtering of camera stream"""
    cap = cv2.VideoCapture(0)
    processor = RealTimeProcessor()

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Controls:")
    print("1 - Original, 2 - Grayscale, 3 - Gaussian Blur, 4 - Canny Edges, 5 - Threshold")
    print("q - Quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = processor.process_frame(frame)

            # Add filter info
            cv2.putText(processed_frame, f'Filter: {processor.current_filter}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Real-time Processing', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                processor.current_filter = 'original'
            elif key == ord('2'):
                processor.current_filter = 'grayscale'
            elif key == ord('3'):
                processor.current_filter = 'gaussian'
            elif key == ord('4'):
                processor.current_filter = 'edges'
            elif key == ord('5'):
                processor.current_filter = 'threshold'

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_filtering()
```

### Feature Detection in Real-time

Detect features in real-time camera streams:

```python
import cv2
import numpy as np

class RealTimeFeatureDetector:
    def __init__(self):
        self.detector_type = 'orb'
        self.detector = cv2.ORB_create()
        self.detectors = {
            'orb': cv2.ORB_create(),
            'sift': cv2.xfeatures2d.SIFT_create() if hasattr(cv2.xfeatures2d, 'SIFT_create') else cv2.ORB_create(),
            'harris': self.harris_detector
        }

    def harris_detector(self, frame):
        """Custom Harris corner detector"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # Dilate to mark corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]  # Red dots for corners
        return frame

    def detect_features(self, frame):
        """Detect features in frame"""
        if self.detector_type in ['orb', 'sift']:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = self.detectors[self.detector_type].detect(gray, None)
            output_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
        elif self.detector_type == 'harris':
            output_frame = self.harris_detector(frame.copy())
        else:
            output_frame = frame

        return output_frame

def real_time_feature_detection():
    """Real-time feature detection"""
    cap = cv2.VideoCapture(0)
    detector = RealTimeFeatureDetector()

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Controls:")
    print("o - ORB, s - SIFT, h - Harris, q - Quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect features
            output_frame = detector.detect_features(frame)

            # Add detector info
            cv2.putText(output_frame, f'Detector: {detector.detector_type}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Feature Detection', output_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                detector.detector_type = 'orb'
            elif key == ord('s'):
                detector.detector_type = 'sift'
            elif key == ord('h'):
                detector.detector_type = 'harris'

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_feature_detection()
```

## Performance Optimization

### Threading for Better Performance

Use threading to improve camera stream performance:

```python
import cv2
import threading
import time
import queue

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Continuously capture frames in background thread"""
        while self.running:
            ret, frame = self.capture.read()
            if not self.q.empty():
                self.q.get()  # Remove old frame to prevent queue buildup
            if ret:
                self.q.put(frame)

    def read(self):
        """Read the latest frame"""
        if not self.q.empty():
            return True, self.q.get()
        else:
            return False, None

    def stop(self):
        """Stop the camera"""
        self.running = False
        self.thread.join()
        self.capture.release()

def threaded_camera_stream():
    """Camera stream using threading for better performance"""
    threaded_cam = ThreadedCamera(0)

    try:
        while True:
            ret, frame = threaded_cam.read()
            if ret:
                cv2.imshow('Threaded Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        threaded_cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    threaded_camera_stream()
```

### FPS Measurement and Optimization

Measure and optimize frames per second:

```python
import cv2
import time

def measure_fps():
    """Measure and display FPS of camera stream"""
    cap = cv2.VideoCapture(0)

    # Initialize FPS calculation
    prev_time = time.time()
    fps = 0

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time > 0:
                fps = 1 / elapsed_time
            prev_time = current_time

            # Display FPS on frame
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('FPS Measurement', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def optimized_camera_stream():
    """Optimized camera stream with reduced processing"""
    cap = cv2.VideoCapture(0)

    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Skip frames to reduce processing load
    frame_count = 0
    skip_frames = 2  # Process every 3rd frame

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame only
            if frame_count % (skip_frames + 1) == 0:
                # Apply minimal processing
                processed_frame = cv2.GaussianBlur(frame, (3, 3), 0)
            else:
                processed_frame = frame  # Just pass through

            cv2.imshow('Optimized Stream', processed_frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Running FPS measurement...")
    measure_fps()
```

## Integration with Robotics

### ROS2 Camera Integration

Integrate camera streams with ROS2 for robotic applications:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()

        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open camera')
            return

        # Timer for publishing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(ros_image)
        else:
            self.get_logger().error('Could not read frame from camera')

def ros2_camera_publisher():
    """ROS2 camera publisher example"""
    rclpy.init()
    camera_publisher = CameraPublisher()

    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.cap.release()
        camera_publisher.destroy_node()
        rclpy.shutdown()

# Note: This example requires ROS2 to be installed and sourced
```

### Object Detection in Camera Stream

Combine camera streaming with object detection:

```python
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np

class CameraObjectDetector:
    def __init__(self):
        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

        # COCO dataset class names
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect_objects(self, frame):
        """Detect objects in frame"""
        # Preprocess frame
        img_tensor = self.transform(frame).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        threshold = 0.5
        mask = scores > threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Draw bounding boxes
        output_frame = frame.copy()
        for i in range(len(boxes)):
            box = boxes[i]
            label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(output_frame,
                         (int(box[0]), int(box[1])),
                         (int(box[2]), int(box[3])),
                         (0, 255, 0), 2)

            # Draw label and score
            label_text = f'{label}: {score:.2f}'
            cv2.putText(output_frame, label_text,
                       (int(box[0]), int(box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_frame

def camera_with_object_detection():
    """Camera stream with real-time object detection"""
    cap = cv2.VideoCapture(0)
    detector = CameraObjectDetector()

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera with object detection running... Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects in frame
            output_frame = detector.detect_objects(frame)

            cv2.imshow('Camera with Object Detection', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_with_object_detection()
```

## Advanced Applications

### Multiple Camera Streams

Handle multiple camera streams simultaneously:

```python
import cv2
import threading
import time

class MultiCameraStream:
    def __init__(self, camera_ids=[0]):
        self.camera_ids = camera_ids
        self.caps = {}
        self.frames = {}
        self.running = True

        # Initialize all cameras
        for cam_id in camera_ids:
            self.caps[cam_id] = cv2.VideoCapture(cam_id)
            self.frames[cam_id] = None

        # Start threads for each camera
        self.threads = []
        for cam_id in camera_ids:
            thread = threading.Thread(target=self.update, args=(cam_id,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def update(self, cam_id):
        """Update frames for specific camera"""
        while self.running:
            ret, frame = self.caps[cam_id].read()
            if ret:
                self.frames[cam_id] = frame
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def get_frames(self):
        """Get latest frames from all cameras"""
        return {cam_id: self.frames[cam_id] for cam_id in self.camera_ids if self.frames[cam_id] is not None}

    def stop(self):
        """Stop all cameras"""
        self.running = False
        for thread in self.threads:
            thread.join()
        for cap in self.caps.values():
            cap.release()

def multi_camera_display():
    """Display multiple camera streams"""
    # For this example, we'll use the same camera multiple times (or different cameras if available)
    multi_cam = MultiCameraStream([0])  # Change to [0, 1] if you have multiple cameras

    try:
        while True:
            frames = multi_cam.get_frames()

            for i, (cam_id, frame) in enumerate(frames.items()):
                # Apply different processing to each camera view
                if i == 0:
                    display_frame = frame  # Original
                elif i == 1:
                    display_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # Grayscale
                else:
                    display_frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Blurred

                cv2.imshow(f'Camera {cam_id}', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        multi_cam.stop()
        cv2.destroyAllWindows()

# Note: This example assumes you have multiple cameras.
# If you only have one camera, it will just show the same feed with different processing.
```

## Troubleshooting and Best Practices

### Common Issues

1. **Camera not opening**: Check if camera is being used by another application
2. **Low FPS**: Reduce resolution or processing load
3. **Memory issues**: Release resources properly with `cap.release()` and `cv2.destroyAllWindows()`

### Best Practices

1. Always release camera resources when done
2. Use threading for better performance
3. Optimize processing for real-time applications
4. Handle errors gracefully
5. Consider using GPU acceleration for heavy processing

## Exercises

1. Implement a camera stream that saves frames when motion is detected
2. Create a multi-camera setup that stitches frames together
3. Build a camera stream that performs real-time face detection and blurring
4. Develop a camera stream with adjustable parameters via a GUI

---

**Previous**: [Deep Vision](../deep-vision.md) | **Next**: [Image Filtering](./image-filtering.md)