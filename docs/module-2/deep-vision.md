---
title: Deep Vision and Neural Networks
sidebar_label: Deep Vision
description: Deep learning approaches to computer vision for robotic applications
slug: /module-2/deep-vision
---

# Deep Vision and Neural Networks

## Summary

This section covers deep learning approaches to computer vision with a focus on robotic applications. Students will learn about Convolutional Neural Networks (CNNs), object detection pipelines, semantic segmentation, and multi-modal fusion techniques that enable robots to understand complex visual scenes.

## Learning Objectives

By the end of this section, students will be able to:
- Understand the fundamentals of Convolutional Neural Networks and their applications
- Implement and deploy object detection models for robotic perception
- Apply semantic segmentation techniques to understand scene composition
- Design multi-modal fusion systems combining visual and other sensor data
- Evaluate the performance and limitations of deep vision systems in robotics

## Table of Contents

1. [Introduction to Deep Learning for Vision](#introduction-to-deep-learning-for-vision)
2. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
3. [Object Detection Pipelines](#object-detection-pipelines)
4. [Semantic Segmentation](#semantic-segmentation)
5. [Multi-Modal Fusion](#multi-modal-fusion)
6. [Practical Considerations](#practical-considerations)
7. [Real-World Applications](#real-world-applications)

## Introduction to Deep Learning for Vision

### Traditional vs. Deep Learning Approaches

Traditional computer vision relied on hand-crafted features and rule-based systems, while deep learning approaches learn features automatically from data.

**Traditional Approach:**
1. Feature extraction (SIFT, HOG, etc.)
2. Feature matching/classification
3. Post-processing

**Deep Learning Approach:**
1. End-to-end learning from raw pixels
2. Hierarchical feature learning
3. Direct output prediction

### Advantages of Deep Learning in Robotics

- **Robustness**: Better performance in diverse environments
- **Generalization**: Ability to handle novel situations
- **Adaptability**: Can learn from experience
- **Integration**: Unified framework for multiple tasks

### Challenges in Robotics Applications

- **Computational requirements**: Need for efficient models
- **Real-time constraints**: Latency requirements for control
- **Data efficiency**: Limited training data in robotics
- **Safety criticality**: Need for reliable predictions
- **Domain adaptation**: Differences between training and deployment environments

## Convolutional Neural Networks (CNNs)

### Architecture Fundamentals

CNNs are specialized neural networks designed for grid-like data such as images.

**Key Components:**
- **Convolutional Layers**: Extract local features
- **Pooling Layers**: Reduce spatial dimensions
- **Activation Functions**: Introduce non-linearity
- **Fully Connected Layers**: Perform classification

### Mathematical Foundation

**Convolution Operation:**
```
(I * K)(i, j) = Σ(Σ I(m, n) × K(i-m, j-n))
```

Where I is the input image and K is the kernel/filter.

**Feature Maps:**
After convolution, the output is a feature map that highlights specific patterns.

### Common CNN Architectures

#### LeNet-5
- One of the first CNNs
- Used for handwritten digit recognition
- Simple architecture: Conv → Pool → Conv → Pool → FC → FC

#### AlexNet
- Winner of ImageNet 2012
- Introduced ReLU activation
- Used dropout for regularization
- Utilized GPU acceleration

#### VGGNet
- Deep architecture with small 3×3 filters
- Demonstrated importance of depth
- Uniform architecture with repeated patterns

#### ResNet
- Introduced residual connections
- Enabled training of very deep networks
- Addressed vanishing gradient problem

**Residual Block:**
```
output = F(x) + x
```

Where F is the residual function and x is the input.

### Implementation with PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x4x4

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Example usage
model = SimpleCNN(num_classes=10)
print(model)
```

### Training Considerations

**Data Augmentation:**
- Random cropping, flipping, rotation
- Color jittering
- Mixup and CutMix techniques

**Regularization:**
- Dropout
- Batch normalization
- Weight decay

**Optimization:**
- Learning rate scheduling
- Adaptive optimizers (Adam, RMSprop)
- Transfer learning

## Object Detection Pipelines

### Problem Definition

Object detection combines object classification and localization in a single task.

**Output:** For each object instance:
- Class label
- Bounding box coordinates
- Confidence score

### Two-Stage Approaches

#### R-CNN (Region-based CNN)

**Process:**
1. Generate region proposals
2. Extract features for each region
3. Classify each region
4. Refine bounding boxes

#### Fast R-CNN

**Improvements over R-CNN:**
- Single network for feature extraction
- ROI pooling for fixed-size feature extraction
- Joint training of classification and bounding box regression

#### Faster R-CNN

**Key Innovation:** Region Proposal Network (RPN)

The RPN shares convolutional features with the detection network, making the process more efficient.

```python
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def create_faster_rcnn_model(num_classes, pretrained=True):
    """Create a Faster R-CNN model"""
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # Replace the classifier with a new one for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn FastRCNNPredictor(in_features, num_classes)

    return model

# Example usage
num_classes = 10  # background + 9 object classes
model = create_faster_rcnn_model(num_classes)
```

### One-Stage Approaches

#### YOLO (You Only Look Once)

**Advantages:**
- Fast inference
- Unified detection framework
- Good real-time performance

**YOLO v3 Architecture:**
- Feature extraction using Darknet-53
- Multi-scale prediction
- Residual connections

```python
import torch
import torch.nn as nn

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        # This is a simplified representation
        # Actual implementation would include the full Darknet-53 backbone
        # and the detection heads at 3 different scales

        self.num_classes = num_classes
        self.backbone = self._create_backbone()
        self.detection_heads = self._create_detection_heads()

    def _create_backbone(self):
        # Simplified backbone
        layers = []
        layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(0.1))
        # Additional layers would go here
        return nn.Sequential(*layers)

    def _create_detection_heads(self):
        # Detection heads for 3 scales
        heads = nn.ModuleList([
            nn.Conv2d(1024, 3 * (5 + self.num_classes), kernel_size=1),
            nn.Conv2d(512, 3 * (5 + self.num_classes), kernel_size=1),
            nn.Conv2d(256, 3 * (5 + self.num_classes), kernel_size=1)
        ])
        return heads

    def forward(self, x):
        # Forward pass through backbone and detection heads
        # This is a simplified version
        features = self.backbone(x)
        # Apply detection heads to different feature maps
        outputs = []
        for head in self.detection_heads:
            output = head(features)  # In reality, this would use features from different levels
            outputs.append(output)
        return outputs
```

#### SSD (Single Shot MultiBox Detector)

**Key Features:**
- Multi-scale feature maps
- Default boxes (priors) with different aspect ratios
- Single forward pass for detection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSD300(nn.Module):
    def __init__(self, num_classes=21):  # 20 classes + background
        super(SSD300, self).__init__()
        self.num_classes = num_classes

        # Feature extraction backbone (simplified)
        self.backbone = self._create_vgg_backbone()

        # Additional feature layers for multi-scale detection
        self.extras = self._create_extra_layers()

        # Location and confidence prediction layers
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        # Number of default boxes at each feature map scale
        num_defaults = [4, 6, 6, 6, 4, 4]

        for nd in num_defaults:
            self.loc_layers.append(nn.Conv2d(self._get_channels(), nd * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self._get_channels(), nd * num_classes, kernel_size=3, padding=1))

    def _create_vgg_backbone(self):
        # Simplified VGG-based backbone
        layers = []
        in_channels = 3

        # Standard VGG configuration with some modifications
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def _create_extra_layers(self):
        # Additional layers for smaller scale feature maps
        layers = []
        in_channels = 512

        # Additional conv layers for feature extraction
        for i in range(7):
            if i % 2 == 0:
                layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)]
            else:
                layers += [nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1)]
            in_channels = 256 if i % 2 == 0 else 512

        return nn.Sequential(*layers)

    def _get_channels(self):
        # Helper to get appropriate number of channels
        return 512  # Simplified

    def forward(self, x):
        # Implementation would involve passing through backbone,
        # extracting features at multiple scales, and applying
        # location and confidence prediction layers
        pass
```

### Evaluation Metrics

**Intersection over Union (IoU):**
```
IoU = Area of Overlap / Area of Union
```

**Mean Average Precision (mAP):**
- Precision-Recall curve for each class
- Average precision across all classes
- Standard evaluation metric for object detection

**Average Precision at IoU=0.5:**
Commonly used threshold for detection evaluation.

## Semantic Segmentation

### Problem Definition

Semantic segmentation assigns a class label to each pixel in an image.

**Output:** A segmentation mask where each pixel has a class label.

### Fully Convolutional Networks (FCNs)

**Key Innovation:** Replace fully connected layers with convolutional layers to enable dense prediction.

**Upsampling:**
- Transposed convolution (deconvolution)
- Skip connections to preserve spatial details

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes

        # VGG16 backbone (simplified)
        self.features = self._make_vgg_layers()

        # Classifier (convolutionalized)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Upsampling to full resolution
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes,
                                          kernel_size=64, stride=32,
                                          bias=False)

    def _make_vgg_layers(self):
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        h = x
        h = self.features(h)
        h = self.classifier(h)
        h = self.upsample(h)

        # Crop to original size if needed
        if h.size()[2:] != x.size()[2:]:
            h = h[:, :, :x.size(2), :x.size(3)]

        return h
```

### U-Net Architecture

**Encoder-Decoder Structure:**
- Contracting path (encoder): Captures context
- Expansive path (decoder): Enables precise localization
- Skip connections: Preserve spatial information

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

### DeepLab Architecture

**Key Components:**
- **Atrous Convolution:** Increases receptive field without increasing parameters
- **Atrous Spatial Pyramid Pooling (ASPP):** Captures multi-scale context
- **Decoder Module:** Improves object boundary segmentation

## Multi-Modal Fusion

### Motivation

Different sensor modalities provide complementary information:

- **Cameras:** Rich color and texture information
- **LiDAR:** Accurate depth and geometry
- **RADAR:** All-weather capability
- **Thermal:** Different object properties

### Fusion Strategies

#### Early Fusion

Combine raw data or low-level features:
- Concatenate sensor data
- Joint processing
- Captures sensor correlations

**Advantages:**
- Maximum information preservation
- Joint optimization possible

**Disadvantages:**
- Computational complexity
- Requires synchronized data

#### Late Fusion

Combine high-level decisions or predictions:
- Individual sensor processing
- Combine final outputs
- Fusion at decision level

**Advantages:**
- Computationally efficient
- Modular design

**Disadvantages:**
- Information loss during processing

#### Deep Fusion

Learn fusion in a deep learning framework:
- Learnable fusion weights
- Adaptive to input conditions
- End-to-end training

### Camera-LiDAR Fusion Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraLidarFusion(nn.Module):
    def __init__(self, camera_features=256, lidar_features=128, output_features=512):
        super(CameraLidarFusion, self).__init__()

        # Feature extractors
        self.camera_encoder = self._create_camera_encoder(camera_features)
        self.lidar_encoder = self._create_lidar_encoder(lidar_features)

        # Attention mechanisms
        self.camera_attention = nn.Sequential(
            nn.Conv2d(camera_features, camera_features // 8, 1),
            nn.ReLU(),
            nn.Conv2d(camera_features // 8, camera_features, 1),
            nn.Sigmoid()
        )

        self.lidar_attention = nn.Sequential(
            nn.Conv2d(lidar_features, lidar_features // 8, 1),
            nn.ReLU(),
            nn.Conv2d(lidar_features // 8, lidar_features, 1),
            nn.Sigmoid()
        )

        # Fusion layer
        self.fusion = nn.Conv2d(camera_features + lidar_features,
                               output_features, 1)

    def _create_camera_encoder(self, features):
        # Simple encoder for camera features
        return nn.Sequential(
            nn.Conv2d(3, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def _create_lidar_encoder(self, features):
        # Simple encoder for LiDAR features (assuming BEV representation)
        return nn.Sequential(
            nn.Conv2d(5, features, 3, padding=1),  # 5 channels: x, y, z, intensity, range
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, camera_input, lidar_input):
        # Extract features
        cam_features = self.camera_encoder(camera_input)
        lidar_features = self.lidar_encoder(lidar_input)

        # Apply attention
        cam_attention = self.camera_attention(cam_features)
        lidar_attention = self.lidar_attention(lidar_features)

        # Weighted features
        cam_weighted = cam_features * cam_attention
        lidar_weighted = lidar_features * lidar_attention

        # Upsample LiDAR features to match camera resolution if needed
        if lidar_weighted.shape[2:] != cam_weighted.shape[2:]:
            lidar_weighted = F.interpolate(lidar_weighted,
                                         size=cam_weighted.shape[2:],
                                         mode='bilinear',
                                         align_corners=False)

        # Concatenate features
        combined_features = torch.cat([cam_weighted, lidar_weighted], dim=1)

        # Final fusion
        fused_features = self.fusion(combined_features)

        return fused_features
```

## Practical Considerations

### Model Optimization for Robotics

**Quantization:**
- Reduce precision (e.g., float32 to int8)
- Decrease model size and inference time
- Acceptable accuracy trade-off

**Pruning:**
- Remove unimportant weights/connections
- Reduce computational requirements
- Maintain key features

**Knowledge Distillation:**
- Train smaller "student" model from larger "teacher"
- Transfer knowledge while reducing size
- Maintain performance in smaller footprint

### Deployment on Edge Devices

**ONNX Conversion:**
```python
import torch
import torch.onnx

# Export model to ONNX format
model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model,
                  dummy_input,
                  "model.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])
```

**TensorRT Optimization:**
```python
import tensorrt as trt
import onnx

def optimize_with_tensorrt(onnx_model_path):
    """Optimize ONNX model with TensorRT"""
    # Create TensorRT builder
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt.Logger())

    # Parse ONNX model
    with open(onnx_model_path, 'rb') as model:
        parser.parse(model.read())

    # Configure optimization
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Build optimized engine
    engine = builder.build_engine(network, config)

    return engine
```

### Performance Evaluation

**Inference Speed:**
- Frames per second (FPS)
- Latency (ms per frame)
- Throughput (images per second)

**Memory Usage:**
- Model size (MB)
- Runtime memory (MB)
- Bandwidth requirements

**Accuracy:**
- Task-specific metrics (mAP, IoU, etc.)
- Robustness to environmental conditions
- Generalization to new scenarios

## Real-World Applications

### Autonomous Vehicles

- Object detection for traffic participants
- Lane detection and road marking recognition
- Traffic sign classification
- 3D object detection using camera-LiDAR fusion

### Service Robots

- Person detection and tracking
- Object recognition for manipulation
- Scene understanding for navigation
- Human activity recognition

### Industrial Automation

- Quality inspection and defect detection
- Object counting and sorting
- Assembly verification
- Safety monitoring

### Agricultural Robotics

- Crop and weed identification
- Disease detection in plants
- Fruit/vegetable recognition and harvesting
- Field mapping and monitoring

## Pre-trained Model Usage

### Transfer Learning

Transfer learning leverages pre-trained models on large datasets (like ImageNet) and adapts them to specific tasks:

**Feature Extraction:**
- Use pre-trained model as fixed feature extractor
- Train only the classifier on top
- Fast and requires less data

**Fine-tuning:**
- Start with pre-trained weights
- Continue training with smaller learning rate
- Better performance with sufficient data

```python
import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Replace the final classifier for your task
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # num_classes for your task

# For fine-tuning, keep gradients enabled
# For feature extraction, freeze early layers:
for param in model.parameters():
    param.requires_grad = False
# Only train the new classifier
for param in model.fc.parameters():
    param.requires_grad = True
```

### Using Pre-trained Detection Models

**PyTorch Hub Example:**
```python
import torch

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Perform inference
results = model('image.jpg')

# Show results
results.print()  # Print results to console
results.show()   # Display image with bounding boxes
```

**Torchvision Detection Models:**
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Prepare input
from PIL import Image
import torchvision.transforms as T

# Transform image
transform = T.Compose([T.ToTensor()])
img = Image.open('image.jpg')
img_tensor = transform(img).unsqueeze(0)

# Perform inference
with torch.no_grad():
    predictions = model(img_tensor)

# Process predictions
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
```

### Model Zoo and Pre-trained Models

**Common Pre-trained Models:**
- **Image Classification**: ResNet, EfficientNet, Vision Transformer
- **Object Detection**: YOLO, SSD, Faster R-CNN, DETR
- **Segmentation**: DeepLab, U-Net, Mask R-CNN

**Loading Custom Pre-trained Models:**
```python
def load_pretrained_model(model_path, model_class, num_classes):
    """Load a pre-trained model from file"""
    model = model_class(num_classes=num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Example usage
model = load_pretrained_model('path/to/checkpoint.pth', MyDetectionModel, 10)
```

## Key Takeaways

1. **Deep learning** has revolutionized computer vision with end-to-end learning
2. **CNNs** form the backbone of modern vision systems
3. **Object detection** enables understanding of object instances
4. **Semantic segmentation** provides pixel-level understanding
5. **Multi-modal fusion** combines different sensor inputs for robust perception
6. **Practical considerations** are crucial for deployment in robotics
7. **Pre-trained models** accelerate development and improve performance

## Exercises

1. Implement and compare different object detection models (YOLO, SSD) on a robotics dataset
2. Train a semantic segmentation model on a scene understanding task
3. Design and implement a camera-LiDAR fusion architecture for object detection
4. Optimize a deep vision model for deployment on an edge device
5. Fine-tune a pre-trained model for a specific robotic perception task

---

**Previous**: [Image Processing](./image-processing.md) | **Next**: [Assignments](./assignments.md)