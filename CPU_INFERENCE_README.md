# CPU Inference on Raspberry Pi 5

This guide explains how to use CPU inference instead of Hailo hardware acceleration for object detection on Raspberry Pi 5.

## Overview

The modified detection pipeline supports both Hailo hardware acceleration and CPU inference. CPU inference is useful when:
- Hailo hardware is not available
- You want to test with different model formats (ONNX, PyTorch, TensorFlow Lite)
- You need more flexibility in model selection

## Installation

1. Install CPU inference dependencies:
```bash
pip install -r requirements_cpu_inference.txt
```

2. For specific model formats, install the appropriate runtime:

**ONNX (Recommended for Raspberry Pi):**
```bash
pip install onnxruntime>=1.15.0
```

**PyTorch:**
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

**TensorFlow Lite:**
```bash
pip install tensorflow>=2.13.0
```

## Usage

### Basic CPU Inference

To use CPU inference instead of Hailo hardware, add the `--use-cpu-inference` flag and specify your model path:

```bash
python detection_pipeline_simple.py \
    --use-cpu-inference \
    --cpu-model-path /path/to/your/model.onnx \
    --input /dev/video0
```

### Command Line Arguments

- `--use-cpu-inference`: Enable CPU inference mode
- `--cpu-model-path`: Path to your CPU model file (.onnx, .pt, .pth, .tflite)
- `--input`: Video source (webcam, video file, etc.)
- `--labels-json`: Optional path to custom labels JSON file

### Example Commands

**Using ONNX YOLOv8 model with webcam:**
```bash
python detection_pipeline_simple.py \
    --use-cpu-inference \
    --cpu-model-path models/yolov8n.onnx \
    --input /dev/video0
```

**Using PyTorch model with video file:**
```bash
python detection_pipeline_simple.py \
    --use-cpu-inference \
    --cpu-model-path models/yolov5s.pt \
    --input videos/test_video.mp4
```

**Using TensorFlow Lite model:**
```bash
python detection_pipeline_simple.py \
    --use-cpu-inference \
    --cpu-model-path models/yolov8n.tflite \
    --input /dev/video0
```

## Supported Model Formats

### ONNX Models (.onnx)
- **Recommended** for Raspberry Pi 5
- Good performance with ONNX Runtime
- Wide compatibility with various model architectures
- Example: YOLOv8, YOLOv5 exported to ONNX

### PyTorch Models (.pt, .pth)
- Native PyTorch models
- Larger memory footprint
- Good for development and testing
- Example: YOLOv5, YOLOv8 PyTorch models

### TensorFlow Lite Models (.tflite)
- Optimized for edge devices
- Smaller model size
- Good inference speed on ARM processors
- Example: TensorFlow Lite optimized models

## Model Requirements

Your model should:
1. Accept RGB input images
2. Output detection results in a standard format (boxes, scores, classes)
3. Be compatible with the input resolution (default: 640x640)

## Performance Considerations

### Raspberry Pi 5 Optimization Tips:

1. **Use ONNX models** for best CPU performance
2. **Lower input resolution** (e.g., 416x416 instead of 640x640) for faster inference
3. **Adjust confidence thresholds** to reduce post-processing overhead
4. **Use lighter models** like YOLOv8n instead of YOLOv8x

### Expected Performance:
- **YOLOv8n ONNX**: ~2-5 FPS on Raspberry Pi 5
- **YOLOv5s ONNX**: ~1-3 FPS on Raspberry Pi 5
- **TensorFlow Lite**: ~3-7 FPS (depending on model)

## Getting Pre-trained Models

### YOLOv8 ONNX Export:
```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640)
```

### YOLOv5 ONNX Export:
```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Export to ONNX
python export.py --weights yolov5s.pt --include onnx --imgsz 640
```

## Troubleshooting

### Common Issues:

1. **Model loading errors**: Ensure the correct runtime is installed for your model format
2. **Memory issues**: Try smaller models or lower input resolution
3. **Slow performance**: Use ONNX models and optimize your Raspberry Pi settings
4. **Import errors**: Install missing dependencies from requirements_cpu_inference.txt

### Debug Mode:
The application prints detection results to console for debugging:
```
CPU Inference detected 2 objects:
  1: class_id=0, confidence=0.847, bbox=[120, 150, 300, 400]
  2: class_id=2, confidence=0.623, bbox=[450, 200, 580, 350]
```

## Architecture Changes

The CPU inference implementation:
1. Uses `CPU_INFERENCE_PIPELINE` instead of `INFERENCE_PIPELINE`
2. Processes frames via GStreamer `appsink` element
3. Runs inference in separate threads to avoid blocking
4. Supports multiple model formats through pluggable backends

## Comparison: Hailo vs CPU Inference

| Feature | Hailo Hardware | CPU Inference |
|---------|----------------|---------------|
| Performance | ~30+ FPS | ~2-7 FPS |
| Power Consumption | Low | Higher |
| Model Flexibility | HEF format only | ONNX, PyTorch, TFLite |
| Setup Complexity | Hailo-specific | Standard ML libraries |
| Cost | Requires Hailo hardware | Uses existing CPU |

Choose CPU inference when you need model flexibility or don't have Hailo hardware available.
