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
import numba as nb


class GstOpenCVPipeline:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)

        self.conf = 0.25
        self.iou = 0.45
        self.image_width = 1280 
        self.image_height = 720

        # Define the pipeline as a string
        pipeline_str = (
            f'v4l2src device=/dev/video2 ! video/x-raw,width={self.image_width},height={self.image_height} ! imxvideoconvert_g2d ! video/x-raw,format=BGRA,width={self.image_width},height={self.image_height},framerate=30/1 ! '
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
            f'caps=video/x-raw,format=BGRA,width={self.image_width},height={self.image_height},framerate=30/1 ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            "imxvideoconvert_g2d ! fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=true signal-fps-measurements=true"
        )
        
        # Create the sink pipeline
        self.sink_pipeline = Gst.parse_launch(sink_pipeline_str)
        
        # Get the appsrc element
        self.appsrc = self.sink_pipeline.get_by_name("opencv_src")

        path = os.getcwd() + "/yolov8n_full_integer_quant_224.tflite"

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
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGRA2RGB)
        
        # tic_pre = time.time()
        input_data, pad = self.preprocess(frame)
        input_data = (input_data / self.in_scale + self.in_zero_point).astype(np.int8)
        # print("Preprocessing Time: ", time.time()-tic_pre, " s")
        
        # tic_inf = time.time()
        # set frame as input tensors
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # perform inference
        self.interpreter.invoke()
        # print("Inference Time: ", time.time()-tic_inf, " s")
        # tic_post = time.time()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        output = (output.astype(np.float32) - self.zero_point) * self.scale
        #print(time.time() - tic)


        # frame = self.letterbox(frame)
        self.postprocess(frame, output, pad)

        
        cv2.putText(frame, "DIDAAAAAAAAAAAAAa", (500, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        

        # Unmap the buffer (as soon as possible)
        buf.unmap(map_info)

        # Create a new Gst.Buffer from the flipped frame without copying the memory
        new_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        # print("Post-Processing Time: ", time.time()-tic_post, " s")


        #print(buf)

        # # Copy timestamps for proper sync
        # new_buffer.pts = buf.pts
        # new_buffer.duration = buf.duration
        
        #print("PROCESSING TIME")
        
        # # Push buffer to appsrc
        return self.appsrc.emit("push-buffer", new_buffer)

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

    def postprocess(self, frame, outputs, pad) :
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Adjust coordinates based on padding and scale to original image size
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(frame.shape)

        # Transform outputs to [x, y, w, h] format
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2  # x center to top-left x
        outputs[..., 1] -= outputs[..., 3] / 2  # y center to top-left y

        for out in outputs:
            # Get scores and apply confidence threshold
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)

            if len(indices) > 0:
                indices.flatten()
            else:
                return frame

            # Draw detections that survived NMS
            [self.draw_detections(frame, boxes[i], scores[i], class_ids[i]) for i in indices]

        return frame

    def draw_detections(self, frame, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            frame,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(frame, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
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
