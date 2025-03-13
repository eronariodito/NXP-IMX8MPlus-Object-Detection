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

class GstOpenCVPipeline:
    def __init__(self):
        # Create an empty (height, width, 4) array filled with 255
       	# self.frame_rgbx = np.full((1080, 1920, 4), 255, dtype=np.uint8)
        # Initialize GStreamer
        Gst.init(None)

        # Define the pipeline as a string
        pipeline_str = (
            "v4l2src device=/dev/video2 ! video/x-raw,width=640,height=480  ! imxvideoconvert_g2d ! video/x-raw,format=BGRA,width=1920,height=1080,framerate=30/1 ! "
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
            "caps=video/x-raw,format=BGRA,width=1920,height=1080,framerate=30/1 ! "
            "queue max-size-buffers=3 leaky=downstream ! "
            "imxvideoconvert_g2d ! fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=true signal-fps-measurements=true"
        )
        
        # Create the sink pipeline
        self.sink_pipeline = Gst.parse_launch(sink_pipeline_str)
        
        # Get the appsrc element
        self.appsrc = self.sink_pipeline.get_by_name("opencv_src")

        path = os.getcwd() + "/yolov4-tiny_416_quant.tflite"

        delegate_lib = "/usr/lib/libvx_delegate.so"

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
        
                
    def on_new_sample(self, appsink):
        tic = time.time()
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
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGRA2RGB)

        frame_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = np.array(input_data, dtype=np.int8)

        # set frame as input tensors
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # perform inference
        self.interpreter.invoke()

        print(self.interpreter.get_tensor(self.output_details[0]['index']))

        cv2.putText(frame, "DIDAAAAAAAAAAAAAa", (500, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Flip the frame using OpenCV (horizontal flip)
        #flipped_frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)

        # Copy RGB values from the original frame
        
        # Unmap the buffer (as soon as possible)
        buf.unmap(map_info)

        # Create a new Gst.Buffer from the flipped frame without copying the memory
        new_buffer = Gst.Buffer.new_wrapped(frame.tobytes())

        #print(buf)

        # # Copy timestamps for proper sync
        # new_buffer.pts = buf.pts
        # new_buffer.duration = buf.duration
        toc = time.time() - tic
        # print("PROCESSING TIME")
        # print(toc)
        # # Push buffer to appsrc
        return self.appsrc.emit("push-buffer", new_buffer)

    
    def run(self):
        # Start the pipelines
        self.source_pipeline.set_state(Gst.State.PLAYING)
        self.sink_pipeline.set_state(Gst.State.PLAYING)
        
        # Create GLib MainLoop
        self.loop = GLib.MainLoop()
        
        try:
            # Run the main loop
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up
            self.source_pipeline.set_state(Gst.State.NULL)
            self.sink_pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()
