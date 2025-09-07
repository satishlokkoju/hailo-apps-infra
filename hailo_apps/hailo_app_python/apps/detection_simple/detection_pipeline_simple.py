# region imports
# Standard library imports
import setproctitle
import numpy as np
import cv2
import threading
from gi.repository import Gst, GLib

# Local application-specific imports
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import RESOURCES_VIDEOS_DIR_NAME, SIMPLE_DETECTION_VIDEO_NAME, SIMPLE_DETECTION_APP_TITLE, SIMPLE_DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME, SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME, SIMPLE_DETECTION_POSTPROCESS_FUNCTION
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import SOURCE_PIPELINE, INFERENCE_PIPELINE, CPU_INFERENCE_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps.hailo_app_python.core.cpu_inference import CPUInferenceHandler
# endregion imports

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser == None:
            parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )
        parser.add_argument(
            "--use-cpu-inference",
            action="store_true",
            help="Use CPU inference instead of Hailo hardware",
        )
        parser.add_argument(
            "--cpu-model-path",
            default=None,
            help="Path to CPU model file (ONNX, PyTorch, etc.)",
        )
        # Call the parent class constructor
        super().__init__(parser, user_data)

        # Additional initialization code can be added here
        self.video_width = 640
        self.video_height = 640

        # Set Hailo parameters - these parameters should be set based on the model used
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45
        if self.options_menu.input is None:  # Setting up a new application-specific default video (overrides the default video set in the GStreamerApp constructor)
            self.video_source = get_resource_path(
                pipeline_name=SIMPLE_DETECTION_PIPELINE,
                resource_type=RESOURCES_VIDEOS_DIR_NAME,
                model=SIMPLE_DETECTION_VIDEO_NAME
            )
        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            # if detected_arch is None:
            #     raise ValueError("Could not auto-detect Hailo architecture as No Hailo hardware is available. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        else:
            self.hef_path = get_resource_path(
                pipeline_name=SIMPLE_DETECTION_PIPELINE,
                resource_type=RESOURCES_MODELS_DIR_NAME,
            )

        print(f"Using HEF path: {self.hef_path}")
        self.post_process_so = get_resource_path(
            pipeline_name=SIMPLE_DETECTION_PIPELINE,
            resource_type=RESOURCES_SO_DIR_NAME,
            model=SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME
        )
        print(f"Using post-process shared object: {self.post_process_so}")

        self.post_function_name = SIMPLE_DETECTION_POSTPROCESS_FUNCTION

        # User-defined label JSON file
        self.labels_json = self.options_menu.labels_json
        
        # CPU inference settings
        self.use_cpu_inference = self.options_menu.use_cpu_inference
        self.cpu_model_path = self.options_menu.cpu_model_path
        
        print(f"Using cpu inference {self.use_cpu_inference}")
        print(f"Using cpu inference {self.cpu_model_path}")
       
        # For now, we'll allow CPU inference without a model path (using mock inference)
        # if self.use_cpu_inference and self.cpu_model_path is None:
        #     raise ValueError("CPU model path must be specified when using CPU inference. Use --cpu-model-path argument.")
        
        # Initialize CPU inference handler if needed
        self.cpu_inference_handler = None

        if self.use_cpu_inference:
            self.cpu_inference_handler = CPUInferenceHandler(
                model_path=self.cpu_model_path,
                confidence_threshold=nms_score_threshold,
                nms_threshold=nms_iou_threshold,
                input_size=(self.video_width, self.video_height)
            )
            print(f"Initialized CPU inference with model: {self.cpu_model_path}")

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle(SIMPLE_DETECTION_APP_TITLE)

        self.create_pipeline()
        
        # Set up CPU inference callback if needed
        if self.use_cpu_inference:
            self._setup_cpu_inference_callback()

    def get_pipeline_string(self):
        print(f"Creating Source pipeline String for source {self.video_source}")
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,
                                          video_width=self.video_width, video_height=self.video_height,
                                          frame_rate=self.frame_rate, sync=self.sync,
                                          no_webcam_compression=True)

        if self.use_cpu_inference:
            detection_pipeline = CPU_INFERENCE_PIPELINE(
                model_path=self.cpu_model_path,
                input_width=self.video_width,
                input_height=self.video_height,
                confidence_threshold=0.3,  # Using nms_score_threshold value
                nms_threshold=0.45  # Using nms_iou_threshold value
            )
        else:
            detection_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path,
                post_process_so=self.post_process_so,
                post_function_name=self.post_function_name,
                batch_size=self.batch_size,
                config_json=self.labels_json,
                additional_params=self.thresholds_str)

        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
        
        if self.use_cpu_inference:
            # For CPU inference, we need to split the pipeline:
            # Main path: source -> user_callback -> display
            # Side path: source -> cpu_inference (appsink for processing)
            pipeline_string = (
                f'{source_pipeline} ! '
                f'tee name=t ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline} '
                f't. ! {detection_pipeline}'
            )
        else:
            pipeline_string = (
                f'{source_pipeline} ! '
                #f'{detection_pipeline} ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline}'
            )
        return pipeline_string
    
    def _setup_cpu_inference_callback(self):
        """Set up callback for CPU inference appsink."""
        if self.pipeline is None:
            return
            
        # Get the appsink element
        appsink = self.pipeline.get_by_name('cpu_inference_appsink')
        if appsink:
            appsink.connect('new-sample', self._on_new_sample_cpu_inference)
            print("Connected CPU inference callback to appsink")
    
    def _on_new_sample_cpu_inference(self, appsink):
        """Callback for processing frames with CPU inference."""
        try:
            # Get the sample from appsink
            sample = appsink.emit('pull-sample')
            if sample is None:
                return Gst.FlowReturn.ERROR
            
            # Get buffer and caps
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Extract frame info from caps
            structure = caps.get_structure(0)
            width = structure.get_int('width')[1]
            height = structure.get_int('height')[1]
            
            # Map buffer to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR
            
            # Convert to numpy array (RGB format)
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Run CPU inference in a separate thread to avoid blocking
            threading.Thread(
                target=self._process_frame_async,
                args=(frame_bgr,),
                daemon=True
            ).start()
            
            # Unmap buffer
            buffer.unmap(map_info)
            
            return Gst.FlowReturn.OK
            
        except Exception as e:
            print(f"Error in CPU inference callback: {e}")
            return Gst.FlowReturn.ERROR
    
    def _process_frame_async(self, frame):
        """Process frame with CPU inference asynchronously."""
        try:
            if self.cpu_inference_handler:
                detections = self.cpu_inference_handler.process_frame(frame)
                
                # Print detections for debugging
                if detections:
                    print(f"CPU Inference detected {len(detections)} objects:")
                    for i, det in enumerate(detections):
                        print(f"  {i+1}: class_id={det['class_id']}, "
                              f"confidence={det['confidence']:.3f}, "
                              f"bbox={det['bbox']}")
                else:
                    print("CPU Inference: No objects detected")
                    
        except Exception as e:
            print(f"Error in async frame processing: {e}")

def main():
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    print("Starting Hailo Detection App...")
    main()
