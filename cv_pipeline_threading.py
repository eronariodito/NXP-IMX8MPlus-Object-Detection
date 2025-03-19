#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
import sys
import time
import ai_edge_litert.interpreter as litert
import os
import time
import yaml
import threading
import queue
import utils.yolo_v8_v11_postprocessing as postprocess

class GstOpenCVPipeline:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)

        self.conf = 0.25
        self.iou = 0.45
        self.image_width = 1280 
        self.image_height = 720

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Define the pipeline as a string
        pipeline_str = (
            f'v4l2src device=/dev/video2 ! video/x-raw,width={self.image_width},height={self.image_height} ! imxvideoconvert_g2d ! video/x-raw,format=BGRA,width={self.image_width},height={self.image_height} ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            "appsink name=opencv_sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        
        # Create the source pipeline
        self.source_pipeline = Gst.parse_launch(pipeline_str)
        
        # Get the appsink element
        self.appsink = self.source_pipeline.get_by_name("opencv_sink")
        self.appsink.connect("new-sample", self.on_new_sample)
        
        # Define the sink pipeline
        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=true do-timestamp=true block=false "
            f'caps=video/x-raw,format=BGRA,width={self.image_width},height={self.image_height} ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            'imxvideoconvert_g2d !  textoverlay name=text_overlay text=" " valignment=bottom halignment=center font-desc="Arial, 36" !'
            "fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=false signal-fps-measurements=true"
        )
        
        # Create the sink pipeline
        self.sink_pipeline = Gst.parse_launch(sink_pipeline_str)
        
        # Get the appsrc element
        self.appsrc = self.sink_pipeline.get_by_name("opencv_src")

        # Connect to display fps-measurements
        textoverlay = self.sink_pipeline.get_by_name("text_overlay")
        if textoverlay:
            # Font selection - professional, readable font
            textoverlay.set_property("font-desc", "Arial Bold 12")  
            
            # Text styling
            textoverlay.set_property("valignment", "top")  # Vertical alignment
            textoverlay.set_property("halignment", "left")    # Horizontal alignment
            textoverlay.set_property("line-alignment", "left")
            textoverlay.set_property("xpad", 20)              # Horizontal padding
            textoverlay.set_property("ypad", 20)              # Vertical padding
            
            # Colors and visibility
            textoverlay.set_property("color", 0xFFFFFFFF)     # White text (ARGB)
            textoverlay.set_property("outline-color", 0xFF000000)  # Black outline
            textoverlay.set_property("shaded-background", True)    # Background shading
            #textoverlay.set_property("shadow", True)          # Text shadow

        self.sink_pipeline.get_by_name("display").connect("fps-measurements", self.on_fps_measurement)

        path = os.getcwd() + "/models/yolov8n_full_integer_quant_416.tflite"

        delegate_lib = "/usr/lib/libvx_delegate.so"

        self.count = 0

        try:
            # Create the delegate
            delegate = litert.load_delegate(delegate_lib)
            
            # Initialize the interpreter with the delegate
            self.interpreter = litert.Interpreter(
                model_path=path,
                experimental_delegates=[delegate]
            )
            self.interpreter.allocate_tensors()
            
            print("Successfully loaded NPU delegate")
            
        except Exception as e:
            print(f"Error loading delegate: {e}")
            print("Falling back to CPU execution")
            self.interpreter = litert.Interpreter(model_path=path)
            self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.scale, self.zero_point = self.output_details[0]["quantization"]
        self.in_scale, self.in_zero_point = self.input_details[0]["quantization"]
        print(self.scale, self.zero_point)

        with open("coco128.yaml", "r") as f:
            data = yaml.safe_load(f)

         # Load the class names from the COCO dataset
        self.classes = data.get("names", {})

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        new_text = f"FPS: {fps:.2f}\nDroprate: {droprate:.2f}\nAvg FPS: {avgfps:.2f}"
        self.sink_pipeline.get_by_name("text_overlay").set_property("text", new_text)
        #print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True
                
    def on_new_sample(self, appsink):
        
	    # Get the sample from appsink
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        # Get buffer from sample
        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Map buffer for read access
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        # Get image dimensions from caps
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        # Create a numpy array directly from the mapped memory (no copy)
        frame = np.ndarray(
            shape=(height, width, 4), 
            dtype=np.uint8,
            buffer=map_info.data
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        input_data, pad = self.preprocess(frame)
        input_data = (input_data / self.in_scale + self.in_zero_point).astype(np.int8)

        self.input_queue.put((input_data, pad, frame, map_info, buf))
       
        # # Push buffer to appsrc
        return Gst.FlowReturn.OK

    def preprocess(self, frame):
        
        # frame_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        frame, pad = self.letterbox(frame, (self.input_shape[2], self.input_shape[1]))
        frame = frame[None]
        frame = np.ascontiguousarray(frame)
        frame = frame.astype(np.float32)
        return frame/255, pad

    def letterbox(
        self, frame, new_shape):
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (Tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        shape = frame.shape[:2]  # Current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # Resize if needed
            frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return frame, (top / frame.shape[0], left / frame.shape[1])


    def infer_thread(self, queue):
        while True:
            try:
                item = queue.get(timeout=5)

                if item is None:
                    break

                input_data, pad, frame, map_info, buf = item

                # set frame as input tensors
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

                # perform inference
                self.interpreter.invoke()

                outputs = self.interpreter.get_tensor(self.output_details[0]["index"])

                self.output_queue.put((outputs, pad, frame, map_info, buf))
            except Exception as e:
                print(f"Error in inference thread: {e}")
    
    def postprocess_thread(self, queue):
        while True:
            try:
                item = queue.get(timeout=5)

                if item is None:
                    break

                outputs, pad, frame, map_info, buf = item
                
                outputs = (outputs.astype(np.float32) - self.zero_point) * self.scale
                postprocess.postprocess(self, frame, outputs, pad)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            
                # Unmap the buffer (as soon as possible)
                buf.unmap(map_info)

                # Create a new Gst.Buffer from the flipped frame without copying the memory
                new_buffer = Gst.Buffer.new_wrapped(frame.tobytes())

                self.appsrc.emit("push-buffer", new_buffer)
            except Exception as e:
                print(f"Error in postprocessing thread: {e}")


    def run(self):
        # Start the pipelines
        self.source_pipeline.set_state(Gst.State.PLAYING)
        self.sink_pipeline.set_state(Gst.State.PLAYING)
        
        
        # Create GLib MainLoop
        self.loop = GLib.MainLoop()

        self.inference_thread = threading.Thread(target=self.infer_thread, args=(self.input_queue,), daemon=True )
        self.inference_thread.start()

        self.postprocess_thread = threading.Thread(target=self.postprocess_thread, args=(self.output_queue,), daemon=True )
        self.postprocess_thread.start()

        
        try:
            # Run the main loop
            self.loop.run()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Shutting down...")
        finally:
               
            # Clean up
            self.source_pipeline.set_state(Gst.State.NULL)
            self.sink_pipeline.set_state(Gst.State.NULL)

            self.input_queue.put(None)
            self.output_queue.put(None)

            if self.inference_thread.is_alive():
                self.inference_thread.join()
            if self.postprocess_thread.is_alive():
                self.postprocess_thread.join()

            self.loop.quit()
            print("Cleanup complete. Exiting...")
            

if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()
