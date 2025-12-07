---
title: "Object Detection in Robotics"
description: "Implementation of object detection algorithms for robotic perception"
sidebar_label: "Object Detection"
---

# Object Detection in Robotics

## Introduction

Object detection is a fundamental task in robotic perception that involves identifying and localizing objects within sensor data. This capability is essential for robots to understand their environment, navigate safely, and interact with objects. This example demonstrates various object detection techniques used in robotics, from classical computer vision approaches to modern deep learning methods.

## Object Detection Fundamentals

### Definition and Challenges

Object detection involves two main tasks:
1. **Classification**: Determining what class an object belongs to
2. **Localization**: Determining where the object is located in the image/space

**Key Challenges in Robotics**:
- Real-time processing requirements
- Varying lighting conditions
- Occlusions and cluttered scenes
- Scale variations
- Domain adaptation (simulation to reality)

### Detection Metrics

**Precision and Recall**:
- **Precision** = True Positives / (True Positives + False Positives)
- **Recall** = True Positives / (True Positives + False Negatives)

**IoU (Intersection over Union)**:
```
IoU = Area of Overlap / Area of Union
```

**mAP (mean Average Precision)**: Average precision across all classes and IoU thresholds.

## Classical Computer Vision Approaches

### Sliding Window Approach

The sliding window approach systematically moves a classifier across an image at multiple scales to detect objects.

```python
import numpy as np
import cv2
from scipy.ndimage import zoom

class SlidingWindowDetector:
    def __init__(self, classifier, scales=None, step_size=10):
        """
        Initialize sliding window detector
        :param classifier: Function that takes an image patch and returns probability
        :param scales: List of scale factors to try
        :param step_size: Step size for sliding window
        """
        self.classifier = classifier
        self.scales = scales or [0.5, 0.75, 1.0, 1.25, 1.5]
        self.step_size = step_size

    def detect(self, image, min_prob=0.5):
        """
        Detect objects using sliding window approach
        :param image: Input image
        :param min_prob: Minimum probability threshold
        :return: List of detections [(bbox, probability, class), ...]
        """
        detections = []

        for scale in self.scales:
            # Resize image
            scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
            h, w = scaled_image.shape[:2]

            # Slide window across image
            for y in range(0, h - 64, self.step_size):  # Assuming 64x64 window
                for x in range(0, w - 64, self.step_size):
                    # Extract patch
                    patch = scaled_image[y:y+64, x:x+64]

                    # Classify patch
                    prob = self.classifier(patch)

                    if prob > min_prob:
                        # Convert coordinates back to original image scale
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(64 / scale)
                        orig_h = int(64 / scale)

                        detections.append(
                            ([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h], prob, "object")
                        )

        return detections

# Example classifier function
def example_classifier(patch):
    """
    Example classifier that detects bright regions
    In practice, this would be a trained model
    """
    # Simple heuristic: return probability based on average brightness
    avg_brightness = np.mean(patch)
    return min(1.0, avg_brightness / 255.0)

# Example usage
def sliding_window_example():
    # Create a sample image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some bright regions (simulating objects)
    cv2.rectangle(image, (100, 100), (150, 150), (255, 255, 255), -1)
    cv2.circle(image, (300, 200), 30, (255, 255, 255), -1)

    # Create detector
    detector = SlidingWindowDetector(example_classifier)

    # Detect objects
    detections = detector.detect(image)

    print(f"Found {len(detections)} detections")
    for i, (bbox, prob, cls) in enumerate(detections[:5]):  # Show first 5
        print(f"Detection {i+1}: bbox={bbox}, prob={prob:.3f}, class={cls}")

    return detections

if __name__ == "__main__":
    detections = sliding_window_example()
```

### HOG + SVM Approach

Histogram of Oriented Gradients (HOG) combined with Support Vector Machines (SVM) was a popular approach before deep learning.

```python
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC

class HOGSVMDetector:
    def __init__(self, win_size=(64, 128), block_size=(16, 16), cell_size=(8, 8),
                 n_bins=9, svm_model=None):
        """
        Initialize HOG + SVM detector
        :param win_size: Detection window size
        :param block_size: Block size in pixels
        :param cell_size: Cell size in pixels
        :param n_bins: Number of orientation bins
        :param svm_model: Pre-trained SVM model (optional)
        """
        self.win_size = win_size
        self.block_size = block_size
        self.cell_size = cell_size
        self.n_bins = n_bins
        self.svm_model = svm_model

    def extract_hog_features(self, image):
        """
        Extract HOG features from an image
        :param image: Input image
        :return: HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to window size
        resized = cv2.resize(gray, self.win_size)

        # Extract HOG features
        features = hog(
            resized,
            orientations=self.n_bins,
            pixels_per_cell=self.cell_size,
            cells_per_block=self.block_size,
            block_norm='L2-Hys',
            feature_vector=True
        )

        return features

    def train(self, pos_samples, neg_samples):
        """
        Train the SVM classifier
        :param pos_samples: List of positive samples (images containing objects)
        :param neg_samples: List of negative samples (images without objects)
        """
        # Extract features from all samples
        all_features = []
        all_labels = []

        for sample in pos_samples:
            features = self.extract_hog_features(sample)
            all_features.append(features)
            all_labels.append(1)  # Positive label

        for sample in neg_samples:
            features = self.extract_hog_features(sample)
            all_features.append(features)
            all_labels.append(0)  # Negative label

        # Train SVM
        self.svm_model = SVC(kernel='linear', probability=True)
        self.svm_model.fit(all_features, all_labels)

    def detect(self, image, scale_step=1.1, min_scale=0.5, max_scale=2.0,
               threshold=0.7, overlap_threshold=0.3):
        """
        Detect objects in image using HOG + SVM
        :param image: Input image
        :param scale_step: Factor by which to scale image at each step
        :param min_scale: Minimum scale factor
        :param max_scale: Maximum scale factor
        :param threshold: Detection threshold
        :param overlap_threshold: Threshold for non-maximum suppression
        :return: List of detections
        """
        if self.svm_model is None:
            raise ValueError("Model must be trained before detection")

        detections = []
        scales = []

        # Generate scale factors
        scale = min_scale
        while scale <= max_scale:
            scales.append(scale)
            scale *= scale_step

        # Multi-scale detection
        for scale_factor in scales:
            # Resize image
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            scaled_img = cv2.resize(image, (new_width, new_height))

            # Slide window across scaled image
            for y in range(0, new_height - self.win_size[1], 10):
                for x in range(0, new_width - self.win_size[0], 10):
                    # Extract window
                    window = scaled_img[y:y+self.win_size[1], x:x+self.win_size[0]]

                    if window.shape[0] != self.win_size[1] or window.shape[1] != self.win_size[0]:
                        continue

                    # Extract HOG features
                    features = self.extract_hog_features(window)

                    # Classify
                    prob = self.svm_model.predict_proba([features])[0][1]  # Probability of positive class

                    if prob > threshold:
                        # Convert coordinates back to original image
                        orig_x = int(x / scale_factor)
                        orig_y = int(y / scale_factor)
                        orig_w = int(self.win_size[0] / scale_factor)
                        orig_h = int(self.win_size[1] / scale_factor)

                        detections.append([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h, prob])

        # Apply non-maximum suppression
        detections = self.non_max_suppression(detections, overlap_threshold)

        # Format detections
        formatted_detections = []
        for det in detections:
            bbox = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
            prob = det[4]
            formatted_detections.append((bbox, prob, "object"))

        return formatted_detections

    def non_max_suppression(self, detections, overlap_threshold):
        """
        Apply non-maximum suppression to remove overlapping detections
        :param detections: List of detections [x1, y1, x2, y2, prob]
        :param overlap_threshold: Threshold for IoU
        :return: Filtered detections
        """
        if len(detections) == 0:
            return []

        # Convert to numpy array
        boxes = np.array(detections)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        # Calculate areas
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by scores
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            # Keep the detection with highest score
            current_idx = indices[0]
            keep.append(detections[current_idx])

            # Calculate IoU with remaining detections
            xx1 = np.maximum(x1[current_idx], x1[indices[1:]])
            yy1 = np.maximum(y1[current_idx], y1[indices[1:]])
            xx2 = np.minimum(x2[current_idx], x2[indices[1:]])
            yy2 = np.minimum(y2[current_idx], y2[indices[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            intersection = w * h
            iou = intersection / (areas[current_idx] + areas[indices[1:]] - intersection)

            # Keep detections with IoU below threshold
            mask = iou < overlap_threshold
            indices = indices[1:][mask]

        return keep

# Example usage
def hog_svm_example():
    # Create sample positive and negative training data
    # In practice, you would use real training images
    pos_samples = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(10)]
    neg_samples = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(10)]

    # Create and train detector
    detector = HOGSVMDetector()
    detector.train(pos_samples, neg_samples)

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Detect objects
    detections = detector.detect(test_image)

    print(f"Found {len(detections)} detections using HOG+SVM")
    for i, (bbox, prob, cls) in enumerate(detections[:3]):  # Show first 3
        print(f"Detection {i+1}: bbox={bbox}, prob={prob:.3f}, class={cls}")

    return detections

if __name__ == "__main__":
    detections = hog_svm_example()
```

## Deep Learning Approaches

### YOLO (You Only Look Once) Implementation

Here's a simplified implementation of a YOLO-like single-stage detector:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class YOLOv1(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=20):
        """
        Simplified YOLO implementation
        :param grid_size: Size of the grid (SxS)
        :param num_bboxes: Number of bounding boxes per grid cell
        :param num_classes: Number of object classes
        """
        super(YOLOv1, self).__init__()
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        # YOLO architecture: simplified CNN backbone
        self.backbone = nn.Sequential(
            # Convolutional layers for feature extraction
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # More convolutional layers...
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Detection head
        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),  # Assuming input is 448x448 -> 7x7 after backbone
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * (num_bboxes * 5 + num_classes))
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Feature extraction
        x = self.backbone(x)
        x = x.view(batch_size, -1)

        # Detection
        x = self.fc(x)

        # Reshape to (batch, grid_size, grid_size, num_bboxes*5 + num_classes)
        x = x.view(batch_size, self.grid_size, self.grid_size,
                  self.num_bboxes * 5 + self.num_classes)

        return x

class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.5, nms_threshold=0.4):
        """
        YOLO detector wrapper
        :param model_path: Path to pre-trained model (optional)
        :param conf_threshold: Confidence threshold
        :param nms_threshold: Non-maximum suppression threshold
        """
        self.model = YOLOv1(grid_size=7, num_bboxes=2, num_classes=20)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def preprocess(self, image):
        """
        Preprocess image for YOLO
        :param image: Input image (H, W, C)
        :return: Preprocessed tensor
        """
        # Convert to tensor and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # Normalize to [0, 1] and permute to (C, H, W)
        image = image / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

        return image

    def postprocess(self, predictions):
        """
        Postprocess YOLO predictions
        :param predictions: Raw model output
        :return: List of detections [(bbox, conf, class), ...]
        """
        # This is a simplified postprocessing
        # In practice, you would implement the full YOLO postprocessing
        # including confidence thresholding, class prediction, and NMS

        detections = []

        # For each prediction in batch
        for pred in predictions:
            # Reshape to (grid_size, grid_size, num_bboxes*5 + num_classes)
            grid_size = self.model.grid_size
            num_bboxes = self.model.num_bboxes
            num_classes = self.model.num_classes

            pred = pred.view(grid_size, grid_size, num_bboxes * 5 + num_classes)

            # Process each grid cell
            for i in range(grid_size):
                for j in range(grid_size):
                    # Get class probabilities
                    class_probs = torch.softmax(pred[i, j, num_bboxes*5:], dim=0)
                    class_conf, class_idx = torch.max(class_probs, dim=0)

                    # Process each bounding box prediction
                    for b in range(num_bboxes):
                        box_offset = b * 5
                        conf = torch.sigmoid(pred[i, j, box_offset + 4])

                        # Apply confidence threshold
                        if conf.item() * class_conf.item() > self.conf_threshold:
                            # Extract bounding box (x, y, w, h in grid coordinates)
                            x = (j + torch.sigmoid(pred[i, j, box_offset])) / grid_size
                            y = (i + torch.sigmoid(pred[i, j, box_offset + 1])) / grid_size
                            w = torch.pow(torch.sigmoid(pred[i, j, box_offset + 2]), 2)
                            h = torch.pow(torch.sigmoid(pred[i, j, box_offset + 3]), 2)

                            # Convert to image coordinates
                            img_w, img_h = 448, 448  # Assuming input size
                            x *= img_w
                            y *= img_h
                            w *= img_w
                            h *= img_h

                            # Calculate bbox [x1, y1, x2, y2]
                            x1 = x - w/2
                            y1 = y - h/2
                            x2 = x + w/2
                            y2 = y + h/2

                            detections.append([
                                [int(x1), int(y1), int(x2), int(y2)],  # bbox
                                conf.item() * class_conf.item(),       # confidence
                                class_idx.item()                       # class
                            ])

        return detections

# Example usage
def yolo_example():
    # Create detector
    detector = YOLODetector(conf_threshold=0.3)

    # Create dummy input (448x448x3 as expected by YOLO)
    dummy_image = torch.randn(1, 3, 448, 448)

    # Forward pass
    with torch.no_grad():
        raw_predictions = detector.model(dummy_image)

    # Postprocess predictions
    detections = detector.postprocess(raw_predictions)

    print(f"Found {len(detections)} detections using YOLO")

    # Show first few detections
    for i, (bbox, conf, cls) in enumerate(detections[:3]):
        print(f"Detection {i+1}: bbox={bbox}, conf={conf:.3f}, class={cls}")

    return detections

if __name__ == "__main__":
    detections = yolo_example()
```

### Two-Stage Detector (Faster R-CNN Style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        """
        Region Proposal Network
        :param in_channels: Input feature channels
        :param num_anchors: Number of anchor boxes per location
        """
        super(RegionProposalNetwork, self).__init__()

        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, 1)  # 2 classes: object/background
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, 1)   # 4 coords per box

    def forward(self, features):
        x = F.relu(self.conv(features))
        cls_scores = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)

        return cls_scores, bbox_deltas

class ROIPool(nn.Module):
    def __init__(self, pooled_height=7, pooled_width=7):
        """
        ROI Pooling layer
        :param pooled_height: Output height
        :param pooled_width: Output width
        """
        super(ROIPool, self).__init__()
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

    def forward(self, features, rois):
        """
        :param features: Feature maps from backbone
        :param rois: Region of interest coordinates [batch_idx, x1, y1, x2, y2]
        :return: Pooled features for each ROI
        """
        # Simplified implementation
        # In practice, this would implement the actual ROI pooling operation
        pooled_features = []

        for roi in rois:
            batch_idx, x1, y1, x2, y2 = roi.int()
            roi_features = features[batch_idx, :, y1:y2, x1:x2]
            pooled = F.adaptive_avg_pool2d(roi_features,
                                         (self.pooled_height, self.pooled_width))
            pooled_features.append(pooled)

        return torch.stack(pooled_features, dim=0)

class TwoStageDetector(nn.Module):
    def __init__(self, num_classes=20):
        """
        Two-stage detector (simplified Faster R-CNN)
        :param num_classes: Number of object classes
        """
        super(TwoStageDetector, self).__init__()

        # Feature extractor (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        # Region Proposal Network
        self.rpn = RegionProposalNetwork(256)

        # ROI Pooling
        self.roi_pool = ROIPool()

        # Classification and regression head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes + 1)  # +1 for background
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)  # dx, dy, dw, dh
        )

    def forward(self, images):
        # Extract features
        features = self.backbone(images)

        # Generate region proposals
        cls_scores, bbox_deltas = self.rpn(features)

        # For simplicity, we'll skip the detailed RPN implementation
        # and just return the raw scores and deltas
        return cls_scores, bbox_deltas

# Example usage
def two_stage_example():
    detector = TwoStageDetector(num_classes=20)

    # Create dummy input
    dummy_images = torch.randn(2, 3, 224, 224)  # 2 images, 3 channels, 224x224

    # Forward pass
    cls_scores, bbox_deltas = detector(dummy_images)

    print(f"Classification scores shape: {cls_scores.shape}")
    print(f"Bounding box deltas shape: {bbox_deltas.shape}")

    return cls_scores, bbox_deltas

if __name__ == "__main__":
    scores, deltas = two_stage_example()
```

## 3D Object Detection

### PointNet-based Detection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()

        # Input transform network
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )

        # Feature transform network
        self.feature_transform = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
        )

        # Classification network
        self.classifier = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_points, 3)
        batch_size, num_points, _ = x.shape
        x = x.transpose(2, 1)  # (batch_size, 3, num_points)

        # Input transform
        input_transform = self.input_transform(x)

        # Feature transform
        features = self.feature_transform(input_transform)

        # Global feature
        global_feature = torch.max(features, dim=2, keepdim=True)[0]  # (batch_size, 1024, 1)
        global_feature = global_feature.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)

        # Combine local and global features
        combined_features = torch.cat([features, global_feature], dim=1)

        # Classification
        output = self.classifier(combined_features)
        output = F.log_softmax(output, dim=1)

        return output

class PointNetDetector:
    def __init__(self, num_classes=10):
        self.model = PointNet(num_classes=num_classes)
        self.model.eval()

    def detect(self, point_cloud):
        """
        Detect objects in point cloud
        :param point_cloud: Input point cloud (num_points, 3)
        :return: Classification probabilities for each point
        """
        # Add batch dimension
        if len(point_cloud.shape) == 2:
            point_cloud = point_cloud.unsqueeze(0)

        with torch.no_grad():
            output = self.model(point_cloud)

        # Convert to probabilities
        probs = torch.exp(output).squeeze(0)  # Remove batch dimension

        return probs

# Example usage
def pointnet_example():
    detector = PointNetDetector(num_classes=10)

    # Create dummy point cloud (1024 points, 3 coordinates each)
    dummy_points = torch.randn(1, 1024, 3)

    # Detect
    probs = detector.detect(dummy_points)

    print(f"Point classification probabilities shape: {probs.shape}")
    print(f"Max probability class: {torch.argmax(probs, dim=0)}")

    return probs

if __name__ == "__main__":
    probs = pointnet_example()
```

## Performance Evaluation

### Evaluation Metrics Implementation

```python
import numpy as np
from typing import List, Tuple

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes
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

def calculate_ap(precision: List[float], recall: List[float]) -> float:
    """
    Calculate Average Precision using 11-point interpolation
    :param precision: List of precision values
    :param recall: List of recall values
    :return: Average Precision
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        # Find maximum precision for recall >= t
        precisions_at_t = [p for p, r in zip(precision, recall) if r >= t]
        if precisions_at_t:
            ap += max(precisions_at_t)

    return ap / 11.0

def evaluate_detector(ground_truth: List[Tuple], detections: List[Tuple],
                     iou_threshold: float = 0.5) -> dict:
    """
    Evaluate object detector performance
    :param ground_truth: List of (bbox, class) tuples
    :param detections: List of (bbox, confidence, class) tuples
    :param iou_threshold: IoU threshold for matching
    :return: Dictionary of evaluation metrics
    """
    # Sort detections by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    # Track which ground truth objects have been matched
    gt_matched = [False] * len(ground_truth)

    tp = []  # True positives
    fp = []  # False positives

    for det_bbox, det_conf, det_class in detections:
        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx, (gt_bbox, gt_class) in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue  # Already matched

            if det_class != gt_class:
                continue  # Different classes

            iou = calculate_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            # True positive
            tp.append(det_conf)
            gt_matched[best_gt_idx] = True
        else:
            # False positive
            fp.append(det_conf)

    # Count false negatives (unmatched ground truth)
    fn = sum(1 for matched in gt_matched if not matched)

    # Calculate precision and recall at each confidence threshold
    all_conf = tp + fp
    all_conf.sort(reverse=True)

    precisions = []
    recalls = []

    for conf in all_conf:
        tp_count = sum(1 for p in tp if p >= conf)
        fp_count = sum(1 for p in fp if p >= conf)

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / len(ground_truth) if len(ground_truth) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Calculate Average Precision
    ap = calculate_ap(precisions, recalls) if precisions else 0.0

    # Calculate overall precision and recall
    total_tp = len(tp)
    total_fp = len(fp)
    total_gt = len(ground_truth)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0

    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'ap': ap,
        'tp': total_tp,
        'fp': total_fp,
        'fn': fn,
        'mAP': ap  # For single class
    }

# Example evaluation
def evaluation_example():
    # Example ground truth: (bbox, class)
    ground_truth = [
        ([100, 100, 200, 200], 'car'),
        ([300, 150, 400, 250], 'person'),
        ([500, 100, 600, 200], 'bicycle')
    ]

    # Example detections: (bbox, confidence, class)
    detections = [
        ([105, 105, 195, 195], 0.9, 'car'),      # Good detection
        ([290, 140, 410, 260], 0.8, 'person'),   # Good detection
        ([510, 110, 590, 190], 0.7, 'bicycle'),  # Good detection
        ([10, 10, 50, 50], 0.6, 'car'),          # False positive
        ([200, 200, 300, 300], 0.5, 'person')    # False positive
    ]

    # Evaluate
    metrics = evaluate_detector(ground_truth, detections)

    print("Detection Evaluation Results:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"AP: {metrics['ap']:.3f}")
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")

    return metrics

if __name__ == "__main__":
    metrics = evaluation_example()
```

## Practical Considerations

### Real-time Optimization

```python
class OptimizedDetector:
    def __init__(self, model_path=None):
        """
        Optimized detector for real-time applications
        """
        # Use smaller model or quantized model for speed
        # Implement multi-threading for input/output
        pass

    def preprocess_optimized(self, image):
        """
        Optimized preprocessing pipeline
        """
        # Resize to fixed size for consistent inference time
        # Use efficient data types (e.g., float16 instead of float32)
        # Batch processing when possible
        pass

    def postprocess_optimized(self, outputs):
        """
        Optimized postprocessing pipeline
        """
        # Use NMS with efficient algorithms
        # Limit number of detections processed
        # Use fixed-size output tensors
        pass
```

## Particle Filter Implementation

### Introduction to Particle Filters

Particle filters are a type of sequential Monte Carlo method used for estimating the state of a system. Unlike Kalman filters which assume Gaussian distributions, particle filters can represent arbitrary probability distributions making them suitable for non-linear, non-Gaussian systems.

### Python Implementation of Particle Filter

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Tuple, Callable

class ParticleFilter:
    def __init__(self, num_particles: int, state_dim: int,
                 process_noise_std: float = 1.0, measurement_noise_std: float = 1.0):
        """
        Initialize Particle Filter
        :param num_particles: Number of particles to use
        :param state_dim: Dimension of the state vector
        :param process_noise_std: Standard deviation of process noise
        :param measurement_noise_std: Standard deviation of measurement noise
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        # Initialize particles randomly
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, state_transition_func: Callable,
                control_input: np.ndarray = None, dt: float = 1.0):
        """
        Prediction step: propagate particles through state transition model
        :param state_transition_func: Function that takes (particles, control_input, dt) and returns new particles
        :param control_input: Control input to the system (optional)
        :param dt: Time step
        """
        # Propagate each particle through the state transition model
        for i in range(self.num_particles):
            self.particles[i] = state_transition_func(self.particles[i], control_input, dt)

        # Add process noise to particles
        noise = np.random.normal(0, self.process_noise_std, self.particles.shape)
        self.particles += noise

    def update(self, measurement: np.ndarray, measurement_model_func: Callable):
        """
        Update step: update particle weights based on measurement
        :param measurement: Actual measurement from sensor
        :param measurement_model_func: Function that takes a state and returns expected measurement
        """
        # Calculate likelihood of each particle given the measurement
        for i in range(self.num_particles):
            predicted_measurement = measurement_model_func(self.particles[i])
            # Calculate likelihood (probability of measurement given particle state)
            likelihood = norm.pdf(measurement, predicted_measurement, self.measurement_noise_std)
            # For multi-dimensional measurements, take product of likelihoods
            if measurement.ndim > 0:
                likelihood = np.prod(likelihood)
            self.weights[i] *= likelihood

        # Normalize weights
        if np.sum(self.weights) == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= np.sum(self.weights)

        # Avoid numerical issues
        self.weights = np.maximum(self.weights, 1e-300)

    def resample(self):
        """
        Resample particles based on their weights to avoid particle degeneracy
        """
        # Calculate cumulative sum of weights
        cumulative_weights = np.cumsum(self.weights)

        # Generate random numbers for resampling
        random_numbers = (np.random.rand(self.num_particles) + np.arange(self.num_particles)) / self.num_particles

        # Resample particles
        new_particles = np.zeros_like(self.particles)
        for i, rand_num in enumerate(random_numbers):
            idx = np.searchsorted(cumulative_weights, rand_num)
            new_particles[i] = self.particles[idx]

        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particles = new_particles

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state and covariance from particles
        :return: (estimated state, estimated covariance)
        """
        # Calculate weighted mean
        estimated_state = np.average(self.particles, axis=0, weights=self.weights)

        # Calculate weighted covariance
        diff = self.particles - estimated_state
        weighted_diff = diff * self.weights.reshape(-1, 1)
        estimated_covariance = weighted_diff.T @ diff / np.sum(self.weights)

        return estimated_state, estimated_covariance

    def effective_sample_size(self) -> float:
        """
        Calculate effective sample size to determine if resampling is needed
        :return: Effective sample size
        """
        return 1.0 / np.sum(self.weights ** 2)

class ObjectTrackerParticleFilter:
    def __init__(self, num_particles: int = 1000, dt: float = 0.1):
        """
        Object tracker using Particle Filter
        :param num_particles: Number of particles to use
        :param dt: Time step between measurements
        """
        self.dt = dt
        self.pf = ParticleFilter(
            num_particles=num_particles,
            state_dim=4,  # [x, y, vx, vy]
            process_noise_std=0.5,
            measurement_noise_std=0.5
        )

    def state_transition_model(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Constant velocity model for state transition
        :param state: Current state [x, y, vx, vy]
        :param control: Control input (not used in constant velocity model)
        :param dt: Time step
        :return: New state
        """
        new_state = state.copy()
        new_state[0] += state[2] * dt  # x_new = x + vx * dt
        new_state[1] += state[3] * dt  # y_new = y + vy * dt
        # vx and vy remain the same in constant velocity model
        return new_state

    def measurement_model(self, state: np.ndarray) -> np.ndarray:
        """
        Measurement model: only position is measured
        :param state: Current state [x, y, vx, vy]
        :return: Expected measurement [x, y]
        """
        return state[:2]  # Only return position

    def track(self, measurements: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Track object through a sequence of measurements
        :param measurements: List of measurements [x, y]
        :return: List of (estimated_state, estimated_covariance) tuples
        """
        results = []

        for i, measurement in enumerate(measurements):
            if i == 0:
                # Initialize particles around the first measurement
                self.pf.particles[:, 0] = measurement[0] + np.random.normal(0, 0.1, self.pf.num_particles)
                self.pf.particles[:, 1] = measurement[1] + np.random.normal(0, 0.1, self.pf.num_particles)
                # Initialize velocities randomly
                self.pf.particles[:, 2] = np.random.normal(0, 1, self.pf.num_particles)
                self.pf.particles[:, 3] = np.random.normal(0, 1, self.pf.num_particles)

            # Prediction step
            self.pf.predict(self.state_transition_model, dt=self.dt)

            # Update step
            self.pf.update(measurement, self.measurement_model)

            # Resample if effective sample size is too low
            if self.pf.effective_sample_size() < self.pf.num_particles / 2:
                self.pf.resample()

            # Get estimate
            estimated_state, estimated_cov = self.pf.estimate()
            results.append((estimated_state, estimated_cov))

        return results

# Example usage: Track an object with particle filter
def particle_filter_tracking_example():
    """
    Example: Track a moving object using particle filter
    """
    # Create tracker
    tracker = ObjectTrackerParticleFilter(num_particles=1000, dt=0.1)

    # Generate simulated data
    dt = 0.1
    num_steps = 100

    # True trajectory: moving in a straight line with constant velocity
    true_positions = []
    measurements = []

    # Initial state: position [10, 5], velocity [2, 1]
    pos = np.array([10.0, 5.0])
    vel = np.array([2.0, 1.0])

    for t in range(num_steps):
        # True position
        true_pos = pos + vel * t * dt
        true_positions.append(true_pos.copy())

        # Noisy measurement
        noise = np.random.normal(0, 0.5, size=2)
        meas = true_pos + noise
        measurements.append(meas.copy())

    # Convert to numpy arrays
    measurements = np.array(measurements)
    true_positions = np.array(true_positions)

    # Track the object
    results = tracker.track([meas for meas in measurements])
    estimated_states = [result[0] for result in results]
    estimated_positions = np.array([state[:2] for state in estimated_states])

    # Calculate performance metrics
    meas_rmse = np.sqrt(np.mean((true_positions - measurements)**2, axis=0))
    track_rmse = np.sqrt(np.mean((true_positions - estimated_positions)**2, axis=0))

    print(f"Performance Comparison:")
    print(f"RMSE without filter (raw measurements): X={meas_rmse[0]:.3f}, Y={meas_rmse[1]:.3f}")
    print(f"RMSE with Particle Filter: X={track_rmse[0]:.3f}, Y={track_rmse[1]:.3f}")

    # Plot results
    plt.figure(figsize=(12, 8))
    true_x = true_positions[:, 0]
    true_y = true_positions[:, 1]
    meas_x = measurements[:, 0]
    meas_y = measurements[:, 1]
    est_x = estimated_positions[:, 0]
    est_y = estimated_positions[:, 1]

    plt.plot(true_x, true_y, 'g-', label='True Trajectory', linewidth=2)
    plt.scatter(meas_x, meas_y, c='r', alpha=0.3, label='Noisy Measurements')
    plt.plot(est_x, est_y, 'b-', label='Particle Filter Estimate', linewidth=2)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Object Tracking with Particle Filter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    return tracker, results

if __name__ == "__main__":
    tracker, results = particle_filter_tracking_example()
```

### Advanced Particle Filter Example: Multi-Modal Tracking

```python
class MultiModalParticleFilter:
    """
    Example of particle filter handling multi-modal distributions
    This can track objects that switch between different motion models
    """
    def __init__(self, num_particles: int = 2000):
        self.num_particles = num_particles
        self.state_dim = 6  # [x, y, vx, vy, ax, ay] (with acceleration)

        # Initialize particles
        self.particles = np.random.randn(num_particles, self.state_dim)
        self.weights = np.ones(num_particles) / num_particles

        # Motion models: different dynamics for different object behaviors
        self.motion_models = {
            'constant_velocity': 0.7,    # 70% of particles follow constant velocity
            'constant_acceleration': 0.2, # 20% follow constant acceleration
            'random_walk': 0.1          # 10% follow random walk
        }

        # Assign particles to different motion models
        n_cv = int(0.7 * num_particles)
        n_ca = int(0.2 * num_particles)
        self.model_assignments = np.concatenate([
            np.full(n_cv, 0),      # constant velocity
            np.full(n_ca, 1),      # constant acceleration
            np.full(num_particles - n_cv - n_ca, 2)  # random walk
        ])
        np.random.shuffle(self.model_assignments)

    def state_transition_model(self, state: np.ndarray, model_idx: int, dt: float) -> np.ndarray:
        """
        Multiple motion models for different object behaviors
        """
        new_state = state.copy()

        if model_idx == 0:  # Constant velocity model
            new_state[0] += state[2] * dt  # x_new = x + vx * dt
            new_state[1] += state[3] * dt  # y_new = y + vy * dt
            # Velocities remain the same
        elif model_idx == 1:  # Constant acceleration model
            new_state[0] += state[2] * dt + 0.5 * state[4] * dt**2
            new_state[1] += state[3] * dt + 0.5 * state[5] * dt**2
            new_state[2] += state[4] * dt  # vx_new = vx + ax * dt
            new_state[3] += state[5] * dt  # vy_new = vy + ay * dt
            # Accelerations remain the same
        else:  # Random walk model
            # Add random changes to velocity
            new_state[2] += np.random.normal(0, 0.1)
            new_state[3] += np.random.normal(0, 0.1)
            new_state[0] += new_state[2] * dt
            new_state[1] += new_state[3] * dt

        return new_state

    def measurement_model(self, state: np.ndarray) -> np.ndarray:
        """
        Measurement model: only position is measured
        """
        return state[:2]  # Only return position

    def predict(self, dt: float = 1.0):
        """
        Prediction step with multiple motion models
        """
        for i in range(self.num_particles):
            model_idx = int(self.model_assignments[i])
            self.particles[i] = self.state_transition_model(self.particles[i], model_idx, dt)

        # Add process noise
        process_noise = np.random.normal(0, 0.1, self.particles.shape)
        self.particles += process_noise

    def update(self, measurement: np.ndarray):
        """
        Update step: calculate likelihood for each particle
        """
        for i in range(self.num_particles):
            predicted_measurement = self.measurement_model(self.particles[i])
            # Calculate likelihood
            likelihood = np.exp(-0.5 * np.sum(((measurement - predicted_measurement) / 0.5)**2))
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights = np.maximum(self.weights, 1e-300)  # Avoid numerical issues
        self.weights /= np.sum(self.weights)

    def estimate(self) -> np.ndarray:
        """
        Calculate weighted mean of particles
        """
        return np.average(self.particles, axis=0, weights=self.weights)

# Example: Multi-modal tracking
def multi_modal_tracking_example():
    """
    Example: Track an object that changes motion patterns
    """
    pf = MultiModalParticleFilter(num_particles=2000)

    # Simulate trajectory with different motion patterns
    dt = 0.1
    num_steps = 150

    true_states = []
    measurements = []

    # Start with constant velocity
    state = np.array([0.0, 0.0, 2.0, 0.5, 0.0, 0.0])  # [x, y, vx, vy, ax, ay]

    for t in range(num_steps):
        # Change motion pattern at certain points
        if 50 <= t < 100:  # Constant acceleration phase
            state[4] = 0.5  # ax
            state[5] = 0.2  # ay
        elif t >= 100:  # Random walk phase
            state[2] += np.random.normal(0, 0.1)  # Random velocity change
            state[3] += np.random.normal(0, 0.1)
            state[4] = 0.0  # Reset acceleration
            state[5] = 0.0

        # Update state based on motion model
        if t < 50:  # Constant velocity
            state[0] += state[2] * dt
            state[1] += state[3] * dt
        elif 50 <= t < 100:  # Constant acceleration
            state[0] += state[2] * dt + 0.5 * state[4] * dt**2
            state[1] += state[3] * dt + 0.5 * state[5] * dt**2
            state[2] += state[4] * dt
            state[3] += state[5] * dt
        else:  # Random walk
            state[0] += state[2] * dt
            state[1] += state[3] * dt

        true_states.append(state.copy())

        # Add measurement noise
        measurement = state[:2] + np.random.normal(0, 0.3, size=2)
        measurements.append(measurement.copy())

    # Track using particle filter
    estimates = []
    for measurement in measurements:
        pf.predict(dt)
        pf.update(measurement)

        # Resample if needed
        if pf.weights.var() > 0.01:  # High variance in weights
            # Simple resampling based on weights
            cumulative_weights = np.cumsum(pf.weights)
            random_numbers = (np.random.rand(pf.num_particles) + np.arange(pf.num_particles)) / pf.num_particles
            new_particles = np.zeros_like(pf.particles)
            for i, rand_num in enumerate(random_numbers):
                idx = np.searchsorted(cumulative_weights, rand_num)
                new_particles[i] = pf.particles[idx]
            pf.particles = new_particles
            pf.weights = np.ones(pf.num_particles) / pf.num_particles

        estimate = pf.estimate()
        estimates.append(estimate)

    # Convert to arrays for plotting
    true_positions = np.array([s[:2] for s in true_states])
    measurements = np.array(measurements)
    estimated_positions = np.array([e[:2] for e in estimates])

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot 1: Trajectory
    plt.subplot(1, 3, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Trajectory', linewidth=2)
    plt.scatter(measurements[:, 0], measurements[:, 1], c='r', alpha=0.3, label='Measurements')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b-', label='PF Estimate', linewidth=2)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Multi-Modal Tracking')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: X position over time
    plt.subplot(1, 3, 2)
    plt.plot(true_positions[:, 0], 'g-', label='True X', linewidth=2)
    plt.scatter(range(len(measurements)), measurements[:, 0], c='r', alpha=0.3, label='Measurements')
    plt.plot(estimated_positions[:, 0], 'b-', label='PF Estimate', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('X Position')
    plt.title('X Position Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Y position over time
    plt.subplot(1, 3, 3)
    plt.plot(true_positions[:, 1], 'g-', label='True Y', linewidth=2)
    plt.scatter(range(len(measurements)), measurements[:, 1], c='r', alpha=0.3, label='Measurements')
    plt.plot(estimated_positions[:, 1], 'b-', label='PF Estimate', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Y Position')
    plt.title('Y Position Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((true_positions - estimated_positions)**2, axis=0))
    print(f"Multi-modal tracking RMSE: X={rmse[0]:.3f}, Y={rmse[1]:.3f}")

    return pf, estimates

if __name__ == "__main__":
    print("Running basic particle filter example...")
    tracker, results = particle_filter_tracking_example()

    print("\nRunning multi-modal tracking example...")
    pf, estimates = multi_modal_tracking_example()

## C++ Implementation of Particle Filter

Here's a C++ implementation of the Particle Filter for robotic perception applications:

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>

class ParticleFilter {
private:
    int num_particles;
    int state_dim;
    double process_noise_std;
    double measurement_noise_std;

    // Particles: each row is a state vector
    Eigen::MatrixXd particles;
    // Weights for each particle
    std::vector<double> weights;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<double> noise_dist;

public:
    ParticleFilter(int num_particles, int state_dim,
                   double process_noise_std = 1.0,
                   double measurement_noise_std = 1.0)
        : num_particles(num_particles), state_dim(state_dim),
          process_noise_std(process_noise_std),
          measurement_noise_std(measurement_noise_std),
          gen(rd()), noise_dist(0.0, 1.0) {

        // Initialize particles randomly
        particles = Eigen::MatrixXd::Random(num_particles, state_dim);

        // Initialize weights uniformly
        weights.resize(num_particles, 1.0 / num_particles);
    }

    void predict(std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> state_transition_func,
                 double dt = 1.0) {
        for (int i = 0; i < num_particles; ++i) {
            Eigen::VectorXd current_state = particles.row(i);
            Eigen::VectorXd new_state = state_transition_func(current_state, dt);

            // Add process noise
            for (int j = 0; j < state_dim; ++j) {
                new_state(j) += noise_dist(gen) * process_noise_std;
            }

            particles.row(i) = new_state;
        }
    }

    void update(const Eigen::VectorXd& measurement,
                std::function<Eigen::VectorXd(const Eigen::VectorXd&)> measurement_model_func) {

        double weight_sum = 0.0;

        for (int i = 0; i < num_particles; ++i) {
            Eigen::VectorXd predicted_measurement = measurement_model_func(particles.row(i));

            // Calculate likelihood (assuming Gaussian noise)
            double diff_squared = (measurement - predicted_measurement).squaredNorm();
            double likelihood = std::exp(-0.5 * diff_squared / (measurement_noise_std * measurement_noise_std));

            weights[i] *= likelihood;
            weight_sum += weights[i];
        }

        // Normalize weights
        if (weight_sum > 0) {
            for (auto& w : weights) {
                w /= weight_sum;
            }
        } else {
            // If all weights are zero, reset to uniform
            std::fill(weights.begin(), weights.end(), 1.0 / num_particles);
        }
    }

    void resample() {
        // Create cumulative weights
        std::vector<double> cumulative_weights(num_particles);
        std::partial_sum(weights.begin(), weights.end(), cumulative_weights.begin());

        // Generate random numbers for resampling
        Eigen::MatrixXd new_particles(num_particles, state_dim);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

        for (int i = 0; i < num_particles; ++i) {
            double random_val = uniform_dist(gen);
            // Find particle index using binary search
            auto it = std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_val);
            int idx = std::min(static_cast<int>(it - cumulative_weights.begin()), num_particles - 1);

            new_particles.row(i) = particles.row(idx);
        }

        particles = new_particles;
        std::fill(weights.begin(), weights.end(), 1.0 / num_particles);
    }

    Eigen::VectorXd estimate() {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(state_dim);

        // Calculate weighted mean
        for (int i = 0; i < num_particles; ++i) {
            mean += weights[i] * particles.row(i).transpose();
        }

        return mean;
    }

    double effectiveSampleSize() {
        double sum_weights_sq = 0.0;
        for (double w : weights) {
            sum_weights_sq += w * w;
        }
        return 1.0 / sum_weights_sq;
    }
};

class ObjectTrackerParticleFilter {
private:
    ParticleFilter pf;
    double dt;

public:
    ObjectTrackerParticleFilter(int num_particles = 1000, double dt = 0.1)
        : pf(num_particles, 4, 0.5, 0.5), dt(dt) {}  // 4D state: [x, y, vx, vy]

    static Eigen::VectorXd stateTransitionModel(const Eigen::VectorXd& state, double dt) {
        Eigen::VectorXd new_state = state;
        // Constant velocity model
        new_state(0) += state(2) * dt;  // x_new = x + vx * dt
        new_state(1) += state(3) * dt;  // y_new = y + vy * dt
        // Velocities remain the same
        return new_state;
    }

    static Eigen::VectorXd measurementModel(const Eigen::VectorXd& state) {
        // Only position is measured
        Eigen::VectorXd measurement(2);
        measurement << state(0), state(1);  // [x, y]
        return measurement;
    }

    std::vector<Eigen::VectorXd> track(const std::vector<Eigen::VectorXd>& measurements) {
        std::vector<Eigen::VectorXd> estimates;

        for (size_t i = 0; i < measurements.size(); ++i) {
            if (i == 0) {
                // Initialize particles around the first measurement
                for (int p = 0; p < pf.num_particles; ++p) {
                    pf.particles(p, 0) = measurements[i](0) + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.2;
                    pf.particles(p, 1) = measurements[i](1) + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.2;
                    pf.particles(p, 2) = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;  // vx
                    pf.particles(p, 3) = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;  // vy
                }
            }

            // Prediction step
            pf.predict(stateTransitionModel, dt);

            // Update step
            pf.update(measurements[i], measurementModel);

            // Resample if effective sample size is too low
            if (pf.effectiveSampleSize() < pf.num_particles / 2.0) {
                pf.resample();
            }

            // Get estimate
            Eigen::VectorXd estimate = pf.estimate();
            estimates.push_back(estimate);
        }

        return estimates;
    }
};

// Example usage
int main() {
    // Create tracker
    ObjectTrackerParticleFilter tracker(1000, 0.1);  // 1000 particles, dt = 0.1

    // Generate simulated data
    int num_steps = 100;
    double dt = 0.1;

    std::vector<Eigen::VectorXd> true_positions;
    std::vector<Eigen::VectorXd> measurements;

    // Initial state: position [10, 5], velocity [2, 1]
    Eigen::Vector2d pos(10.0, 5.0);
    Eigen::Vector2d vel(2.0, 1.0);

    // Random number generator for noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_dist(0.0, 0.5);  // measurement noise

    for (int t = 0; t < num_steps; ++t) {
        // True position
        Eigen::Vector2d true_pos = pos + vel * t * dt;
        Eigen::VectorXd true_pos_vec(2);
        true_pos_vec << true_pos(0), true_pos(1);
        true_positions.push_back(true_pos_vec);

        // Noisy measurement
        Eigen::VectorXd noise(2);
        noise << noise_dist(gen), noise_dist(gen);
        Eigen::VectorXd meas = true_pos_vec + noise;
        measurements.push_back(meas);
    }

    // Track the object
    std::vector<Eigen::VectorXd> estimates = tracker.track(measurements);

    // Extract estimated positions
    std::vector<Eigen::VectorXd> estimated_positions;
    for (const auto& est : estimates) {
        Eigen::VectorXd pos_est(2);
        pos_est << est(0), est(1);  // Extract x, y from state
        estimated_positions.push_back(pos_est);
    }

    // Calculate performance metrics
    double meas_rmse_x = 0.0, meas_rmse_y = 0.0;
    double track_rmse_x = 0.0, track_rmse_y = 0.0;

    for (int i = 0; i < num_steps; ++i) {
        meas_rmse_x += std::pow(true_positions[i](0) - measurements[i](0), 2);
        meas_rmse_y += std::pow(true_positions[i](1) - measurements[i](1), 2);
        track_rmse_x += std::pow(true_positions[i](0) - estimated_positions[i](0), 2);
        track_rmse_y += std::pow(true_positions[i](1) - estimated_positions[i](1), 2);
    }

    meas_rmse_x = std::sqrt(meas_rmse_x / num_steps);
    meas_rmse_y = std::sqrt(meas_rmse_y / num_steps);
    track_rmse_x = std::sqrt(track_rmse_x / num_steps);
    track_rmse_y = std::sqrt(track_rmse_y / num_steps);

    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "RMSE without filter (raw measurements): X=" << meas_rmse_x << ", Y=" << meas_rmse_y << std::endl;
    std::cout << "RMSE with Particle Filter: X=" << track_rmse_x << ", Y=" << track_rmse_y << std::endl;

    return 0;
}
```

### Compilation and Usage

To compile and run the C++ Particle Filter example:

```bash
# Install Eigen library (if not already installed)
# On Ubuntu/Debian: sudo apt-get install libeigen3-dev
# On macOS with Homebrew: brew install eigen

# Compile
g++ -std=c++11 -O3 particle_filter.cpp -o particle_filter

# Run
./particle_filter
```

The C++ implementation provides the same functionality as the Python version but with better performance for real-time applications. The Eigen library is used for efficient matrix operations, which are essential for particle filter computations.
```

## Next Steps

1. **Implementation Practice**: Implement the detection algorithms on sample datasets
2. **Real Data**: Apply to real robotics datasets (KITTI, nuScenes, etc.)
3. **Evaluation**: Test performance under various conditions
4. **Optimization**: Optimize for your specific robotics platform
5. **Integration**: Integrate with robot control and navigation systems

This example provides a comprehensive overview of object detection techniques in robotics, from classical approaches to modern deep learning methods. The implementations can be adapted and extended for specific robotic applications.