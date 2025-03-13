#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
import sys
import time

# Initialize GStreamer
Gst.init(None)

class GstOpenCVPipeline:
    def __init__(self):
        # Define the pipeline as a string
        pipeline_str = (
            "v4l2src device=/dev/video2 ! imxvideoconvert_g2d ! video/x-raw,format=RGBx,width=1920,height=1080,framerate=30/1 ! "
            "queue max-size-buffers=3 leaky=2 ! "
            "appsink name=opencv_sink emit-signals=true max-buffers=100 drop=true sync=true "
        )
        
        # Define the sink pipeline
        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=false do-timestamp=true block=false "
            "caps=video/x-raw,format=RGBx,width=1920,height=1080,framerate=30/1 ! "
            #"videotestsrc ! "
            "queue max-size-buffers=3 leaky=2 ! "
            "imxvideoconvert_g2d ! fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=true signal-fps-measurements=true"
        )

        total_pipeline_str = (
            pipeline_str + sink_pipeline_str
        )
        
        self.pipeline = Gst.parse_launch(total_pipeline_str)
        
        # Get the appsrc element
        self.appsink = self.pipeline.get_by_name("opencv_sink")
        self.appsink.connect("new-sample", self.on_new_sample)
        self.appsrc = self.pipeline.get_by_name("opencv_src")
    
        
    def on_new_sample(self, appsink):
        print("DIDAAAAAAAAAAa")
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
            shape=(height, width, 4),  # Assuming 4 channels (BGRA or RGBA)
            dtype=np.uint8,
            buffer=map_info.data
        )

        # Flip the frame using OpenCV (horizontal flip)
        flipped_frame = cv2.flip(frame, 1)

        # Unmap the buffer (as soon as possible)
        buf.unmap(map_info)

        # Create a new Gst.Buffer from the flipped frame without copying the memory
        new_buffer = Gst.Buffer.new_wrapped(flipped_frame.tobytes())

        # # Copy timestamps for proper sync
        # new_buffer.pts = buf.pts
        # new_buffer.duration = buf.duration

        # Push buffer to appsrc
        print("DIDA")
        return self.appsrc.emit("push-buffer", new_buffer)
        #return None

    
    def run(self):
        # Start the pipelines
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Create GLib MainLoop
        self.loop = GLib.MainLoop()
        
        try:
            # Run the main loop
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up
            self.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()