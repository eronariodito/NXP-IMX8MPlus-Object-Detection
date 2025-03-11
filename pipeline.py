#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys

# Initialize GStreamer
Gst.init(None)

# Define the pipeline as a string
pipeline_description = "videotestsrc ! videoconvert ! autovideosink"

try:
    # Create the pipeline using parse_launch
    pipeline = Gst.parse_launch(pipeline_description)
    
    # Start playing
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Unable to set the pipeline to the playing state.")
        sys.exit(1)
    
    # Create a GLib MainLoop to process GStreamer messages
    loop = GLib.MainLoop()
    
    # Add a watch on the bus for messages
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    # Define callback for bus messages
    def on_message(bus, message):
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        elif msg_type == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        
        return True
    
    # Connect the callback to the message signal
    bus.connect("message", on_message)
    
    # Run the main loop
    try:
        print(f"Running pipeline: {pipeline_description}")
        print("Press Ctrl+C to stop")
        loop.run()
    except KeyboardInterrupt:
        print("Stopping pipeline")
    
    # Clean up
    pipeline.set_state(Gst.State.NULL)

except GLib.Error as e:
    print(f"Failed to create pipeline: {e}")
    sys.exit(1)