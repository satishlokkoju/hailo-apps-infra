"""
CPU-based object detection inference module for Raspberry Pi 5.
Supports ONNX, PyTorch, and TensorFlow Lite models.
"""

import cv2
import numpy as np
import threading
import time
from typing import List, Tuple, Optional, Dict, Any
import logging

# Optional imports - will be loaded based on model type
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")


class CPUDetector:
    """CPU-based object detector supporting multiple model formats."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.4, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize CPU detector.
        
        Args:
            model_path: Path to the model file
            confidence_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for filtering overlapping boxes
            input_size: Model input size (width, height)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.model = None
        self.model_type = None
        self.input_name = None
        self.output_names = None
        
        # Load model based on file extension
        # self._load_model()
        
    def _load_model(self):
        """Load model based on file extension."""
        if self.model_path.endswith('.onnx'):
            self._load_onnx_model()
        elif self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
            self._load_torch_model()
        elif self.model_path.endswith('.tflite'):
            self._load_tflite_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        self.model_type = 'onnx'
        # Use CPU provider for Raspberry Pi
        providers = ['CPUExecutionProvider']
        self.model = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [output.name for output in self.model.get_outputs()]
        
        logging.info(f"Loaded ONNX model: {self.model_path}")
        logging.info(f"Input: {self.input_name}, Outputs: {self.output_names}")
    
    def _load_torch_model(self):
        """Load PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.model_type = 'torch'
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()
        
        logging.info(f"Loaded PyTorch model: {self.model_path}")
    
    def _load_tflite_model(self):
        """Load TensorFlow Lite model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.model_type = 'tflite'
        self.model = tf.lite.Interpreter(model_path=self.model_path)
        self.model.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        
        logging.info(f"Loaded TensorFlow Lite model: {self.model_path}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame ready for inference
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(frame_rgb, self.input_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose for model input
        if self.model_type == 'onnx':
            # ONNX typically expects NCHW format
            input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        elif self.model_type == 'torch':
            # PyTorch expects NCHW format
            input_tensor = np.transpose(normalized, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
        else:  # tflite
            # TensorFlow Lite typically expects NHWC format
            input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
    
    def postprocess_detections(self, outputs: List[np.ndarray], 
                             original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to get final detections.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original frame shape (height, width)
            
        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id
        """
        detections = []
        
        # This is a simplified postprocessing for YOLO-style models
        # You may need to adjust this based on your specific model format
        if len(outputs) > 0:
            predictions = outputs[0]  # Assuming first output contains predictions
            
            # Handle different output shapes
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension
            
            # Filter by confidence threshold
            if predictions.shape[-1] >= 5:  # x, y, w, h, conf, class_scores...
                confidences = predictions[:, 4]
                valid_indices = confidences > self.confidence_threshold
                valid_predictions = predictions[valid_indices]
                
                if len(valid_predictions) > 0:
                    # Extract boxes and scores
                    boxes = valid_predictions[:, :4]
                    scores = valid_predictions[:, 4]
                    
                    # Get class predictions if available
                    if predictions.shape[-1] > 5:
                        class_scores = valid_predictions[:, 5:]
                        class_ids = np.argmax(class_scores, axis=1)
                    else:
                        class_ids = np.zeros(len(valid_predictions), dtype=int)
                    
                    # Convert boxes to original image coordinates
                    scale_x = original_shape[1] / self.input_size[0]
                    scale_y = original_shape[0] / self.input_size[1]
                    
                    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                        x_center, y_center, width, height = box
                        
                        # Convert from center format to corner format
                        x1 = int((x_center - width/2) * scale_x)
                        y1 = int((y_center - height/2) * scale_y)
                        x2 = int((x_center + width/2) * scale_x)
                        y2 = int((y_center + height/2) * scale_y)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(score),
                            'class_id': int(class_id)
                        })
        
        # Apply NMS to remove overlapping detections
        if detections:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to filter overlapping detections."""
        if not detections:
            return detections
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = []
        confidences = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to x, y, w, h
            confidences.append(det['confidence'])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                  self.confidence_threshold, self.nms_threshold)
        
        # Filter detections based on NMS results
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            List of detections
        """
        # Mock detection for testing - return some fake detections
        original_shape = frame.shape[:2]  # height, width
        height, width = original_shape
        
        # Generate mock detections
        mock_detections = [
            {
                'bbox': [int(width * 0.1), int(height * 0.1), int(width * 0.4), int(height * 0.4)],
                'confidence': 0.85,
                'class_id': 0  # person
            },
            {
                'bbox': [int(width * 0.6), int(height * 0.3), int(width * 0.9), int(height * 0.7)],
                'confidence': 0.72,
                'class_id': 2  # car
            }
        ]
        
        return mock_detections


class CPUInferenceHandler:
    """Handler for CPU inference in GStreamer pipeline."""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4, input_size: Tuple[int, int] = (640, 640)):
        """Initialize CPU inference handler."""
        # Use mock model path if None provided
        if model_path is None:
            model_path = "mock_model.onnx"
        
        self.detector = CPUDetector(model_path, confidence_threshold, 
                                   nms_threshold, input_size)
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process frame and return detections."""
        detections = self.detector.detect(frame)
        
        with self.detection_lock:
            self.latest_detections = detections
            
        return detections
    
    def get_latest_detections(self) -> List[Dict[str, Any]]:
        """Get the latest detections (thread-safe)."""
        with self.detection_lock:
            return self.latest_detections.copy()
