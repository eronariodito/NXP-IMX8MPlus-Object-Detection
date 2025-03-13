#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import os

# Set environment variable to enable dot file dumps
# You can also set this outside the script using:
# export GST_DEBUG_DUMP_DOT_DIR=/path/to/output/directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["GST_DEBUG_DUMP_DOT_DIR"] = dir_path + "/dotfiles"

# Initialize GStreamer


# Define the pipeline as a string
pipeline_description = "v4l2src device=/dev/video2 ! videoconvert ! autovideosink"
pipeline_description = "v4l2src name=cam_src device=/dev/video2 num-buffers=-1 ! video/x-raw,width=1920,height=1080 ! tee name=t t. ! queue name=thread-nn max-size-buffers=2 leaky=2 ! imxvideoconvert_g2d ! video/x-raw,width=300,height=300,format=RGBA ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_filter framework=tensorflow-lite model=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/ssdlite_mobilenet_v2_coco_quant_uint8_float32_no_postprocess.tflite custom=Delegate:External,ExtDelegateLib:libvx_delegate.so ! tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/coco_labels_list.txt option3=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/box_priors.txt option4=1920:1080 option5=300:300 ! videoconvert ! mix. t. ! queue name=thread-img max-size-buffers=2 leaky=2 ! videoconvert ! imxcompositor_g2d name=mix sink_0::zorder=2 sink_1::zorder=1 latency=40000000 min-upstream-latency=40000000 ! textoverlay name=text_overlay text='DIDAAAAAA' valignment=bottom halignment=center font-desc='Arial, 36' ! fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=false signal-fps-measurements=true"
#pipeline_description = "v4l2src name=cam_src device=/dev/video2 num-buffers=-1 ! video/x-raw,width=1280,height=720,framerate=30/1 ! tee name=t t. ! queue name=thread-nn max-size-buffers=2 leaky=2 ! imxvideoconvert_g2d ! video/x-raw,width=416,height=416,format=RGBA ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:int16,add:-128 ! tensor_transform mode=typecast option=int8 ! tensor_filter framework=tensorflow-lite model=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/yolov4-tiny_416_quant.tflite custom=Delegate:External,ExtDelegateLib:libvx_delegate.so ! tensor_filter framework=python3 model=/home/root/nxp-nnstreamer-examples/detection/../detection/postprocess_yolov4_tiny.py custom=Height:416,Width:416,Threshold:0.4 ! tensor_decoder mode=bounding_boxes option1=yolov5 option2=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/coco-labels-2014_2017.txt option4=1280:720 option5=416:416 ! videoconvert ! mix. t. ! queue name=thread-img max-size-buffers=2 leaky=2 ! videoconvert ! imxcompositor_g2d name=mix sink_0::zorder=2 sink_1::zorder=1 latency=60000000 min-upstream-latency=60000000 ! autovideosink"
#pipeline_description = "v4l2src device=/dev/video2 ! video/x-raw,width=640,height=480,framerate=30/1 ! tee name=t t. ! queue max-size-buffers=2 leaky=2 ! imxvideoconvert_g2d ! video/x-raw,width=300,height=300,format=RGBA ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_filter framework=tensorflow-lite model=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/ssdlite_mobilenet_v2_coco_quant_uint8_float32_no_postprocess.tflite custom=Delegate:External,ExtDelegateLib:libvx_delegate.so ! tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=/home/root/nxp-nnstreamer-examples/detection/../downloads/models/detection/coco_labels_list.txt option3=0:1:2:3,50 option4=640:480 option5=300:300 ! mix. t. ! queue max-size-buffers=2 ! imxcompositor_g2d name=mix latency=30000000 min-upstream-latency=30000000 sink_0::zorder=2 sink_1::zorder=1 ! waylandsink"

class GstreamerApp:
    def __init__(self):
        self.pipeline = Gst.init(None)

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        new_text = f"FPS: {fps:.2f}\nDroprate: {droprate:.2f}\nAvg FPS: {avgfps:.2f}"
        self.pipeline.get_by_name("text_overlay").set_property("text", new_text)
        #print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def on_message(self, bus, message):
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            # Dump dot file on error
            Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline-error")
            self.loop.quit()
        elif msg_type == Gst.MessageType.EOS:
            print("End of stream")
            # Dump dot file at end of stream
            Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline-eos")
            self.loop.quit()
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            # Only interested in pipeline state changes
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                state_names = {
                    Gst.State.NULL: "NULL",
                    Gst.State.READY: "READY",
                    Gst.State.PAUSED: "PAUSED",
                    Gst.State.PLAYING: "PLAYING"
                }
                print(f"Pipeline state changed from {state_names.get(old_state)} to {state_names.get(new_state)}")
                # Dump dot file for each state change
                Gst.debug_bin_to_dot_file(self.pipeline, 
                                         Gst.DebugGraphDetails.ALL, 
                                         f"pipeline-{state_names.get(old_state)}-to-{state_names.get(new_state)}")
        
        return True

    def run(self):
        # Create the pipeline using parse_launch
        try:
            self.pipeline = Gst.parse_launch(pipeline_description)

            textoverlay = self.pipeline.get_by_name("text_overlay")
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
            
            # Get the pipeline's bus
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            
            # Dump dot file when the pipeline is in NULL state (before playing)
            Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline-null")
            
            # Start playing
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("Unable to set the pipeline to the playing state.")
                sys.exit(1)
            
            # Dump dot file when the pipeline is in PLAYING state
            Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline-playing")
            
            # Create a GLib MainLoop
            self.loop = GLib.MainLoop()
            
            # Define callback for bus messages
            bus.connect("message", self.on_message)
            self.pipeline.get_by_name("display").connect("fps-measurements", self.on_fps_measurement)
        
            # Run the main loop
            try:
                print(f"Running pipeline: {pipeline_description}")
                print(f"Dot files will be saved to: {os.environ['GST_DEBUG_DUMP_DOT_DIR']}")
                print("Press Ctrl+C to stop")
                self.loop.run()
            except KeyboardInterrupt:
                print("Stopping pipeline")
                # Dump dot file when stopping
                Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline-stopping")
        
            # Clean up
            self.pipeline.set_state(Gst.State.NULL)
            # Final dot file after cleanup
            Gst.debug_bin_to_dot_file(self.__dict__pipeline, Gst.DebugGraphDetails.ALL, "pipeline-final")

        except GLib.Error as e:
            print(f"Failed to create pipeline: {e}")
            sys.exit(1)


if __name__ == "__main__":
    app=GstreamerApp()
    app.run()
