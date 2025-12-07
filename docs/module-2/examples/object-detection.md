---
title: Object Detection Workflows
sidebar_label: Object Detection
description: Using pre-trained models for object detection in robotic applications
slug: /module-2/examples/object-detection
---

# Object Detection Workflows

## Summary

This example demonstrates how to use pre-trained models for object detection in robotic applications. Students will learn to implement object detection pipelines, evaluate detection performance, and integrate detection results into robotic perception systems.

## Learning Objectives

By the end of this example, students will be able to:
- Load and use pre-trained object detection models
- Implement real-time object detection pipelines
- Evaluate detection performance and accuracy
- Integrate object detection with robotic systems
- Optimize detection for specific robotic tasks

## Table of Contents

1. [Pre-trained Model Loading](#pre-trained-model-loading)
2. [Basic Object Detection](#basic-object-detection)
3. [Real-time Detection](#real-time-detection)
4. [Performance Evaluation](#performance-evaluation)
5. [Robotic Integration](#robotic-integration)
6. [Advanced Workflows](#advanced-workflows)

## Pre-trained Model Loading

### Loading PyTorch Models

PyTorch provides several pre-trained object detection models:

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, retinanet_resnet50_fpn
import torchvision.transforms as T
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='fasterrcnn'):
        """
        Initialize object detector with pre-trained model

        Args:
            model_name: Name of the model ('fasterrcnn', 'ssd', 'retinanet')
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model based on name
        if model_name == 'fasterrcnn':
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_name == 'ssd':
            self.model = ssd300_vgg16(pretrained=True)
        elif model_name == 'retinanet':
            self.model = retinanet_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

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

        # Transform for input images
        self.transform = T.Compose([T.ToTensor()])

    def load_image(self, image_path):
        """Load and preprocess image for detection"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_rgb

    def detect_objects(self, image_rgb, confidence_threshold=0.5):
        """
        Detect objects in image

        Args:
            image_rgb: Input image in RGB format
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary with boxes, labels, and scores
        """
        # Preprocess image
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }

def load_and_test_model():
    """Load model and test with sample image"""
    # Initialize detector
    detector = ObjectDetector('fasterrcnn')

    # Create a sample image if file doesn't exist
    sample_img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    cv2.rectangle(sample_img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(sample_img, (300, 150), 50, (0, 255, 0), -1)  # Green circle

    # Detect objects
    results = detector.detect_objects(sample_img, confidence_threshold=0.3)

    print(f"Detected {len(results['boxes'])} objects")
    for i in range(len(results['boxes'])):
        box = results['boxes'][i]
        label = detector.COCO_INSTANCE_CATEGORY_NAMES[results['labels'][i]]
        score = results['scores'][i]
        print(f"  {label}: {score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

# Example usage
if __name__ == "__main__":
    load_and_test_model()
```

### Loading YOLO Models

YOLO models are popular for real-time detection:

```python
import torch
import cv2
import numpy as np

def load_yolo_model():
    """Load YOLOv5 model using PyTorch Hub"""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def yolo_detection(image_path, model=None):
    """Perform YOLO detection on image"""
    if model is None:
        model = load_yolo_model()
        if model is None:
            return None

    # Perform inference
    results = model(image_path)

    # Get predictions
    predictions = results.pandas().xyxy[0]  # Results in pandas format

    # Convert to standard format
    boxes = []
    labels = []
    scores = []

    for _, row in predictions.iterrows():
        boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        labels.append(row['name'])
        scores.append(row['confidence'])

    return {
        'boxes': np.array(boxes),
        'labels': labels,
        'scores': np.array(scores)
    }

def compare_model_loading():
    """Compare different model loading approaches"""
    print("Loading different models...")

    # Test PyTorch models
    try:
        faster_rcnn = ObjectDetector('fasterrcnn')
        print("✓ Faster R-CNN loaded successfully")
    except Exception as e:
        print(f"✗ Faster R-CNN failed: {e}")

    try:
        yolo_model = load_yolo_model()
        if yolo_model:
            print("✓ YOLOv5 loaded successfully")
        else:
            print("✗ YOLOv5 failed to load")
    except Exception as e:
        print(f"✗ YOLOv5 failed: {e}")

# Example usage
if __name__ == "__main__":
    compare_model_loading()
```

## Basic Object Detection

### Simple Detection Pipeline

Create a basic object detection pipeline:

```python
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

class BasicDetectionPipeline:
    def __init__(self, model_name='fasterrcnn', confidence_threshold=0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        if model_name == 'fasterrcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_name == 'ssd':
            self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # COCO class names
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

        self.transform = T.Compose([T.ToTensor()])

    def detect(self, image):
        """
        Detect objects in image

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Image with bounding boxes drawn and detection results
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Draw bounding boxes on image
        output_image = image.copy()
        for i in range(len(boxes)):
            box = boxes[i]
            label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(output_image,
                         (int(box[0]), int(box[1])),
                         (int(box[2]), int(box[3])),
                         (0, 255, 0), 2)

            # Draw label and score
            label_text = f'{label}: {score:.2f}'
            cv2.putText(output_image, label_text,
                       (int(box[0]), int(box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_image, {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'class_names': [self.COCO_INSTANCE_CATEGORY_NAMES[l] for l in labels]
        }

def basic_detection_demo():
    """Demonstrate basic object detection"""
    # Create sample image
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add some objects
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Red rectangle (might be detected as person/bench)
    cv2.circle(img, (300, 150), 40, (0, 255, 0), -1)  # Green circle (might be detected as person/bottle)
    cv2.rectangle(img, (400, 250), (500, 350), (0, 0, 255), -1)  # Blue rectangle

    # Initialize detector
    detector = BasicDetectionPipeline(confidence_threshold=0.3)

    # Perform detection
    result_img, detections = detector.detect(img)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detections (Found {len(detections["boxes"])} objects)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print detection results
    print("Detection Results:")
    for i, (label, score) in enumerate(zip(detections['class_names'], detections['scores'])):
        box = detections['boxes'][i]
        print(f"  {label}: {score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

# Example usage
if __name__ == "__main__":
    basic_detection_demo()
```

### Custom Detection with Post-processing

Add custom post-processing to detection results:

```python
import cv2
import torch
import torchvision.transforms as T
import numpy as np

class CustomDetectionPipeline:
    def __init__(self, model_name='fasterrcnn', confidence_threshold=0.5, iou_threshold=0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        if model_name == 'fasterrcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # COCO class names
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

        self.transform = T.Compose([T.ToTensor()])

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        # Calculate intersection area
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def non_max_suppression(self, boxes, scores, labels):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(boxes) == 0:
            return [], [], []

        # Sort by confidence scores
        sorted_indices = np.argsort(scores)[::-1]

        keep = []
        for i in sorted_indices:
            # Check if this box should be kept
            is_duplicate = False
            for j in keep:
                iou = self.calculate_iou(boxes[i], boxes[j])
                if iou > self.iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(i)

        # Return filtered results
        return boxes[keep], scores[keep], labels[keep]

    def detect_with_postprocessing(self, image):
        """Detect objects with custom post-processing"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            boxes, scores, labels = self.non_max_suppression(boxes, scores, labels)

        # Draw bounding boxes on image
        output_image = image.copy()
        for i in range(len(boxes)):
            box = boxes[i]
            label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(output_image,
                         (int(box[0]), int(box[1])),
                         (int(box[2]), int(box[3])),
                         (0, 255, 0), 2)

            # Draw label and score
            label_text = f'{label}: {score:.2f}'
            cv2.putText(output_image, label_text,
                       (int(box[0]), int(box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_image, {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'class_names': [self.COCO_INSTANCE_CATEGORY_NAMES[l] for l in labels]
        }

def custom_postprocessing_demo():
    """Demonstrate custom post-processing"""
    # Create sample image with overlapping objects to test NMS
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add overlapping rectangles
    cv2.rectangle(img, (100, 100), (250, 200), (255, 0, 0), -1)
    cv2.rectangle(img, (120, 120), (270, 220), (0, 255, 0), -1)  # Overlapping
    cv2.rectangle(img, (300, 100), (400, 200), (0, 0, 255), -1)

    # Initialize detector with NMS
    detector = CustomDetectionPipeline(confidence_threshold=0.3, iou_threshold=0.5)

    # Perform detection with post-processing
    result_img, detections = detector.detect_with_postprocessing(img)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Overlapping Objects')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detections with NMS (Found {len(detections["boxes"])} objects)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print detection results
    print("Detection Results with NMS:")
    for i, (label, score) in enumerate(zip(detections['class_names'], detections['scores'])):
        box = detections['boxes'][i]
        print(f"  {label}: {score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

# Example usage
if __name__ == "__main__":
    custom_postprocessing_demo()
```

## Real-time Detection

### Real-time Object Detection

Implement real-time object detection for robotic applications:

```python
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import time

class RealTimeDetector:
    def __init__(self, model_name='fasterrcnn', confidence_threshold=0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        if model_name == 'fasterrcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_name == 'mobile':
            # Use a lighter model for real-time applications
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # COCO class names
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

        self.transform = T.Compose([T.ToTensor()])

    def preprocess_frame(self, frame, target_size=(300, 300)):
        """Preprocess frame for detection"""
        # Resize frame to target size while maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to target size
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return padded, (x_offset, y_offset, scale)

    def detect_frame(self, frame):
        """Detect objects in a single frame"""
        # Preprocess frame
        processed_frame, (x_offset, y_offset, scale) = self.preprocess_frame(frame)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Adjust boxes for padding and scaling
        boxes[:, 0] = (boxes[:, 0] - x_offset) / scale  # x1
        boxes[:, 2] = (boxes[:, 2] - x_offset) / scale  # x2
        boxes[:, 1] = (boxes[:, 1] - y_offset) / scale  # y1
        boxes[:, 3] = (boxes[:, 3] - y_offset) / scale  # y2

        # Keep boxes within frame bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, frame.shape[1])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, frame.shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, frame.shape[0])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, frame.shape[0])

        # Remove invalid boxes (where x2 <= x1 or y2 <= y1)
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_mask]
        labels = labels[valid_mask]
        scores = scores[valid_mask]

        return boxes, labels, scores

    def draw_detections(self, frame, boxes, labels, scores):
        """Draw detection results on frame"""
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

def real_time_detection():
    """Real-time object detection from camera"""
    # Initialize detector
    detector = RealTimeDetector(model_name='mobile', confidence_threshold=0.5)

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for processing time

    print("Real-time object detection started. Press 'q' to quit.")

    # Initialize FPS calculation
    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            start_time = time.time()
            boxes, labels, scores = detector.detect_frame(frame)
            detection_time = time.time() - start_time

            # Draw detections
            output_frame = detector.draw_detections(frame, boxes, labels, scores)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Display FPS and detection time
            cv2.putText(output_frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f'Detect: {detection_time*1000:.1f}ms', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f'Objects: {len(boxes)}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Real-time Object Detection', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # For demonstration purposes, we'll just show the function
    # Uncomment the line below to run real-time detection
    # real_time_detection()
    print("Real-time detection function defined. Uncomment to run with camera.")
```

### Optimized Real-time Pipeline

Create an optimized pipeline for embedded robotic systems:

```python
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import threading
import queue
import time

class OptimizedRealTimeDetector:
    def __init__(self, model_name='mobile', confidence_threshold=0.4):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load optimized model
        if model_name == 'mobile':
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # Use torch.jit to optimize model
        self.model = torch.jit.script(self.model)

        # COCO class names
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

        self.transform = T.Compose([T.ToTensor()])

    def preprocess_for_speed(self, frame):
        """Fast preprocessing for speed"""
        # Resize to smaller size for faster processing
        target_size = (320, 320)  # Smaller size for speed
        resized = cv2.resize(frame, target_size)
        return resized

    def detect_fast(self, frame):
        """Fast detection optimized for speed"""
        # Preprocess frame
        processed_frame = self.preprocess_for_speed(frame)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        # Perform inference with optimized model
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        return boxes, labels, scores

class ThreadedRealTimeDetector:
    def __init__(self, detector, max_queue_size=2):
        self.detector = detector
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def _process_frames(self):
        """Process frames in background thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:  # Stop signal
                    break

                # Perform detection
                boxes, labels, scores = self.detector.detect_fast(frame)

                # Put results in queue
                if not self.result_queue.full():
                    self.result_queue.put((frame, boxes, labels, scores))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue

    def submit_frame(self, frame):
        """Submit frame for processing"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_latest_result(self):
        """Get the most recent detection result"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        return results[-1] if results else None

    def stop(self):
        """Stop the detector"""
        self.running = False
        self.frame_queue.put(None)  # Stop signal
        self.process_thread.join()

def optimized_real_time_demo():
    """Demonstrate optimized real-time detection"""
    # Initialize optimized detector
    detector = OptimizedRealTimeDetector(confidence_threshold=0.4)
    threaded_detector = ThreadedRealTimeDetector(detector)

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Optimized real-time detection started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Submit frame for processing
            threaded_detector.submit_frame(frame)

            # Get latest results
            result = threaded_detector.get_latest_result()
            if result:
                prev_frame, boxes, labels, scores = result

                # Draw detections on the frame that was processed
                output_frame = detector.draw_detections(prev_frame, boxes, labels, scores)

                # Add performance info
                cv2.putText(output_frame, f'Objects: {len(boxes)}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Optimized Real-time Detection', output_frame)
            else:
                # Show original frame if no results yet
                cv2.imshow('Optimized Real-time Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        threaded_detector.stop()
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # For demonstration, just show the function definition
    # Uncomment to run optimized real-time detection
    # optimized_real_time_demo()
    print("Optimized real-time detection functions defined. Uncomment to run with camera.")
```

## Performance Evaluation

### Detection Metrics

Evaluate object detection performance:

```python
import cv2
import torch
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_detection_performance(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    Evaluate detection performance

    Args:
        gt_boxes: Ground truth boxes [N, 4]
        gt_labels: Ground truth labels [N]
        pred_boxes: Predicted boxes [M, 4]
        pred_labels: Predicted labels [M]
        pred_scores: Predicted scores [M]
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with performance metrics
    """
    # Initialize metrics
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Track which ground truth boxes have been matched
    gt_matched = [False] * len(gt_boxes)

    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_label = pred_labels[pred_idx]
        pred_score = pred_scores[pred_idx]

        # Find the best matching ground truth box
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_matched[gt_idx]:
                continue  # Skip already matched ground truth

            if gt_label != pred_label:
                continue  # Only match same class

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if match is good enough
        if best_iou >= iou_threshold:
            # True positive: correct detection
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            # False positive: incorrect detection
            fp += 1

    # False negatives: ground truth boxes that weren't detected
    fn = len(gt_boxes) - sum(gt_matched)

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
    }

def create_synthetic_data():
    """Create synthetic ground truth and prediction data for evaluation"""
    # Create synthetic ground truth
    gt_boxes = np.array([
        [50, 50, 150, 150],   # Person
        [200, 100, 300, 200], # Car
        [350, 50, 450, 150]   # Chair
    ])
    gt_labels = np.array([1, 3, 56])  # person, car, chair in COCO

    # Create synthetic predictions (some correct, some incorrect)
    pred_boxes = np.array([
        [52, 48, 148, 152],   # Good detection for person
        [190, 95, 310, 210],  # Good detection for car
        [345, 55, 455, 145],  # Good detection for chair
        [250, 250, 350, 350], # False positive (no object there)
        [400, 400, 500, 500]  # False positive (no object there)
    ])
    pred_labels = np.array([1, 3, 56, 1, 79])  # person, car, chair, person, pizza
    pred_scores = np.array([0.9, 0.85, 0.8, 0.7, 0.6])

    return gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores

def performance_evaluation_demo():
    """Demonstrate performance evaluation"""
    # Create synthetic data
    gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores = create_synthetic_data()

    # Evaluate performance at different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    results = {}

    for iou_thresh in iou_thresholds:
        metrics = evaluate_detection_performance(
            gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            iou_threshold=iou_thresh
        )
        results[iou_thresh] = metrics

    # Print results
    print("Detection Performance Evaluation")
    print("=" * 50)
    print(f"{'IoU Threshold':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10}")
    print("-" * 50)

    for iou_thresh, metrics in results.items():
        print(f"{iou_thresh:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} {metrics['accuracy']:<10.3f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot metrics vs IoU threshold
    ious = list(results.keys())
    precisions = [results[i]['precision'] for i in ious]
    recalls = [results[i]['recall'] for i in ious]
    f1_scores = [results[i]['f1_score'] for i in ious]

    axes[0].plot(ious, precisions, 'b-o', label='Precision', linewidth=2)
    axes[0].plot(ious, recalls, 'r-s', label='Recall', linewidth=2)
    axes[0].plot(ious, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    axes[0].set_xlabel('IoU Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance vs IoU Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Create a sample detection visualization
    img = np.ones((500, 500, 3), dtype=np.uint8) * 240  # Light gray background

    # Draw ground truth boxes in blue
    for box in gt_boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    # Draw prediction boxes in green
    for i, box in enumerate(pred_boxes):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        cv2.putText(img, f'{pred_scores[i]:.2f}', (int(box[0]), int(box[1])-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Ground Truth (Blue) vs Predictions (Green)\nText shows confidence scores')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    performance_evaluation_demo()
```

## Robotic Integration

### Robot Perception Integration

Integrate object detection with robotic perception systems:

```python
import cv2
import torch
import numpy as np
import math

class RobotPerceptionSystem:
    def __init__(self, detection_threshold=0.5):
        self.detection_threshold = detection_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load detection model
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.to(self.device)
        self.detector.eval()

        # COCO class names
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

        # Robot coordinate system (example: camera at origin, forward is +X)
        self.robot_position = np.array([0, 0, 0])  # Robot position in world
        self.camera_height = 1.0  # Camera height above ground (meters)

        # Camera intrinsic parameters (example values)
        self.fx = 600  # Focal length x
        self.fy = 600  # Focal length y
        self.cx = 320  # Principal point x
        self.cy = 240  # Principal point y

    def detect_and_localize(self, image, camera_pose=None):
        """
        Detect objects and estimate their 3D positions relative to robot

        Args:
            image: Input image from robot camera
            camera_pose: Camera pose relative to robot (rotation matrix + translation)

        Returns:
            List of detected objects with 2D and 3D information
        """
        # Convert image to RGB and preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)

        # Perform detection
        with torch.no_grad():
            predictions = self.detector(input_tensor)

        # Process results
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Filter by confidence
        mask = scores > self.detection_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Convert 2D detections to 3D positions
        detected_objects = []
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            score = scores[i]

            # Calculate 2D center of bounding box
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # Estimate depth using object size heuristics (simplified)
            # In practice, you'd use stereo vision, depth sensor, or monocular depth estimation
            bbox_height = box[3] - box[1]
            # Assume average object height of 1.7m (for people) and calculate distance
            # This is a simplified depth estimation
            estimated_depth = (self.fy * 1.7) / bbox_height if bbox_height > 0 else 10.0

            # Convert to 3D coordinates in robot frame
            # This assumes the camera is level and pointing forward
            x_3d = estimated_depth  # Forward direction
            y_3d = (center_x - self.cx) * estimated_depth / self.fx  # Lateral offset
            z_3d = self.camera_height - (center_y - self.cy) * estimated_depth / self.fy  # Height

            # Apply camera pose transformation if provided
            if camera_pose is not None:
                # Apply rotation and translation
                pos_3d = np.array([x_3d, y_3d, z_3d])
                # This is a simplified transformation - in practice you'd use the full pose matrix
                pass  # For this example, we'll skip the transformation

            detected_objects.append({
                'class_id': label,
                'class_name': self.COCO_INSTANCE_CATEGORY_NAMES[label],
                'confidence': score,
                'bbox': box,
                'center_2d': (center_x, center_y),
                'position_3d': (x_3d, y_3d, z_3d),
                'distance': math.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
            })

        return detected_objects

    def create_obstacle_map(self, image, max_distance=10.0):
        """
        Create an obstacle map from object detection results

        Args:
            image: Input image from robot camera
            max_distance: Maximum distance to consider for obstacles

        Returns:
            Binary obstacle map in robot coordinate system
        """
        detections = self.detect_and_localize(image)

        # Create a 2D grid for obstacle map (e.g., 20m x 20m with 0.1m resolution)
        grid_size = 200  # 20m / 0.1m per cell
        obstacle_map = np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Robot is at the center of the map (index 100, 100 for 20m x 20m)
        robot_idx = grid_size // 2

        for detection in detections:
            x, y, z = detection['position_3d']
            distance = detection['distance']

            # Only consider obstacles within max distance
            if distance <= max_distance:
                # Convert world coordinates to grid indices
                # x is forward, y is left-right, z is up-down
                grid_x = int(robot_idx + x / 0.1)  # Forward/backward
                grid_y = int(robot_idx + y / 0.1)  # Left/right

                # Mark as obstacle if within grid bounds
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    obstacle_map[grid_x, grid_y] = 255  # Obstacle detected

        return obstacle_map, detections

def robot_perception_demo():
    """Demonstrate robot perception integration"""
    # Initialize perception system
    perception = RobotPerceptionSystem(detection_threshold=0.5)

    # Create a sample image with objects
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some objects
    cv2.rectangle(img, (200, 150), (300, 300), (255, 0, 0), -1)  # Person-like object
    cv2.circle(img, (400, 200), 50, (0, 255, 0), -1)  # Car-like object
    cv2.rectangle(img, (100, 250), (180, 400), (0, 0, 255), -1)  # Chair-like object

    # Perform detection and localization
    detections = perception.detect_and_localize(img)

    # Create obstacle map
    obstacle_map, _ = perception.create_obstacle_map(img)

    # Display results
    plt.figure(figsize=(18, 6))

    # Original image with detections
    plt.subplot(1, 3, 1)
    result_img = img.copy()
    for detection in detections:
        box = detection['bbox']
        cv2.rectangle(result_img,
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     (0, 255, 0), 2)
        cv2.putText(result_img,
                   f"{detection['class_name']}: {detection['confidence']:.2f}",
                   (int(box[0]), int(box[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Objects ({len(detections)})')
    plt.axis('off')

    # Obstacle map
    plt.subplot(1, 3, 2)
    plt.imshow(obstacle_map, cmap='gray', origin='lower')
    plt.title('Obstacle Map\n(White = Obstacle, Black = Free Space)')
    plt.xlabel('Right (0.1m cells)')
    plt.ylabel('Forward (0.1m cells)')

    # Add robot position indicator
    robot_pos = obstacle_map.shape[0] // 2
    plt.plot(robot_pos, robot_pos, 'ro', markersize=10, label='Robot')
    plt.legend()

    # Detection statistics
    plt.subplot(1, 3, 3)
    class_counts = {}
    distances = []

    for detection in detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        distances.append(detection['distance'])

    if class_counts:
        classes, counts = zip(*class_counts.items())
        plt.bar(classes, counts)
        plt.title('Detected Object Classes')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No objects detected',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Detected Object Classes')

    plt.tight_layout()
    plt.show()

    # Print detection details
    print("Robot Perception Results:")
    print("=" * 40)
    for i, detection in enumerate(detections):
        print(f"Object {i+1}:")
        print(f"  Class: {detection['class_name']}")
        print(f"  Confidence: {detection['confidence']:.3f}")
        print(f"  2D Position: ({detection['center_2d'][0]:.1f}, {detection['center_2d'][1]:.1f})")
        print(f"  3D Position: ({detection['position_3d'][0]:.2f}, {detection['position_3d'][1]:.2f}, {detection['position_3d'][2]:.2f})")
        print(f"  Distance: {detection['distance']:.2f}m")
        print()

# Example usage
if __name__ == "__main__":
    robot_perception_demo()
```

## Advanced Workflows

### Multi-Model Ensemble Detection

Combine multiple models for improved detection:

```python
import cv2
import torch
import torchvision.transforms as T
import numpy as np

class EnsembleDetector:
    def __init__(self, confidence_threshold=0.4, iou_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load multiple models
        self.models = {
            'faster_rcnn': torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
            'ssd': torchvision.models.detection.ssd300_vgg16(pretrained=True),
            'retinanet': torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        }

        # Move models to device and set to eval mode
        for name, model in self.models.items():
            model.to(self.device)
            model.eval()

        # COCO class names
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

        self.transform = T.Compose([T.ToTensor()])

    def detect_with_all_models(self, image_rgb):
        """Get detections from all models"""
        results = {}

        for name, model in self.models.items():
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = model(input_tensor)

            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            # Filter by confidence
            mask = scores > self.confidence_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            results[name] = {
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            }

        return results

    def ensemble_nms(self, all_results, weights=None):
        """
        Perform Non-Maximum Suppression across all model results

        Args:
            all_results: Dictionary with results from each model
            weights: Weight for each model's predictions (for weighted NMS)
        """
        if weights is None:
            weights = {name: 1.0 for name in all_results.keys()}

        # Combine all detections
        all_boxes = []
        all_labels = []
        all_scores = []
        all_source = []  # Track which model each detection came from

        for model_name, results in all_results.items():
            for i in range(len(results['boxes'])):
                all_boxes.append(results['boxes'][i])
                all_labels.append(results['labels'][i])
                all_scores.append(results['scores'][i] * weights[model_name])
                all_source.append(model_name)

        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        all_boxes = np.array(all_boxes)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        # Sort by score
        sorted_indices = np.argsort(all_scores)[::-1]

        keep = []
        suppressed = np.zeros(len(all_boxes), dtype=bool)

        for i in sorted_indices:
            if suppressed[i]:
                continue

            keep.append(i)

            # Calculate IoU with all remaining boxes
            for j in sorted_indices:
                if j == i or suppressed[j] or all_labels[i] != all_labels[j]:
                    continue

                iou = self.calculate_iou(all_boxes[i], all_boxes[j])
                if iou > self.iou_threshold:
                    suppressed[j] = True

        # Return filtered results
        keep = np.array(keep)
        return all_boxes[keep], all_labels[keep], all_scores[keep]

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        # Calculate intersection area
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def detect_ensemble(self, image):
        """Perform ensemble detection"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get detections from all models
        all_results = self.detect_with_all_models(image_rgb)

        # Perform ensemble NMS
        boxes, labels, scores = self.ensemble_nms(all_results)

        # Draw results
        output_image = image.copy()
        for i in range(len(boxes)):
            box = boxes[i]
            label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(output_image,
                         (int(box[0]), int(box[1])),
                         (int(box[2]), int(box[3])),
                         (0, 255, 0), 2)

            # Draw label and score
            label_text = f'{label}: {score:.2f}'
            cv2.putText(output_image, label_text,
                       (int(box[0]), int(box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_image, {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'class_names': [self.COCO_INSTANCE_CATEGORY_NAMES[l] for l in labels],
            'model_results': all_results
        }

def ensemble_detection_demo():
    """Demonstrate ensemble detection"""
    # Initialize ensemble detector
    ensemble_detector = EnsembleDetector(confidence_threshold=0.3)

    # Create sample image
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add some objects
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Red object
    cv2.circle(img, (300, 150), 50, (0, 255, 0), -1)  # Green object
    cv2.rectangle(img, (400, 250), (500, 350), (0, 0, 255), -1)  # Blue object

    # Perform ensemble detection
    result_img, detections = ensemble_detector.detect_ensemble(img)

    # Display results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Ensemble Detection ({len(detections["boxes"])} objects)')
    plt.axis('off')

    # Show individual model results
    individual_results = detections['model_results']

    plt.subplot(2, 2, 3)
    model_names = list(individual_results.keys())
    model_counts = [len(individual_results[name]['boxes']) for name in model_names]

    plt.bar(model_names, model_counts)
    plt.title('Detections by Model')
    plt.ylabel('Number of Detections')
    plt.xticks(rotation=45)

    # Show ensemble vs individual performance
    plt.subplot(2, 2, 4)
    all_counts = model_counts + [len(detections['boxes'])]
    methods = model_names + ['Ensemble']

    bars = plt.bar(methods, all_counts)
    plt.title('Detection Count Comparison')
    plt.ylabel('Number of Detections')
    plt.xticks(rotation=45)

    # Color the ensemble bar differently
    bars[-1].set_color('red')

    plt.tight_layout()
    plt.show()

    # Print detection results
    print("Ensemble Detection Results:")
    print("=" * 40)
    for i, (label, score) in enumerate(zip(detections['class_names'], detections['scores'])):
        box = detections['boxes'][i]
        print(f"  {label}: {score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

# Example usage
if __name__ == "__main__":
    ensemble_detection_demo()
```

## Exercises

1. Implement a real-time object detection system that can run on a Raspberry Pi
2. Create a detection pipeline that works with different camera types (RGB, thermal, depth)
3. Build a system that learns to adapt detection parameters based on environmental conditions
4. Develop a multi-camera detection system that fuses detections from multiple viewpoints

---

**Previous**: [Image Filtering](./image-filtering.md) | **Next**: [Assignments](../assignments.md)