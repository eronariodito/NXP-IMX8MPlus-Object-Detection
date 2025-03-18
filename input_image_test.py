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
        # Create an empty (height, width, 4) array filled with 255
       	# self.frame_rgbx = np.full((1080, 1920, 4), 255, dtype=np.uint8)
        # Initialize GStreamer
        Gst.init(None)

        self.conf = 0.25
        self.iou = 0.45
        self.image_width = 1280
        self.image_height = 720

        self.model_name = "yolo11n_full_integer_quant_224"

        path = os.getcwd() + "/" + self.model_name + ".tflite"

        delegate_lib = "/usr/lib/libvx_delegate.so"

        self.count = 0

        try:
            # Create the delegate
            delegate = litert.load_delegate(delegate_lib)
            
            # Initialize the interpreter with the delegate
            self.interpreter = litert.Interpreter(
                model_path=path,
                #experimental_delegates=[delegate]
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
        
                
    def on_new_sample(self):
        img = cv2.imread("bus.jpg", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        x, pad = self.preprocess(img)
        x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Filename
        filename = 'savedImageCPU_' + self.model_name + '.jpg'

        # set frame as input tensors
        self.interpreter.set_tensor(self.input_details[0]['index'], x)

        # perform inference
        
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        #tic = time.time()
        output = (output.astype(np.float32) - self.zero_point) * self.scale
        #print(time.time() - tic)

        # frame = self.letterbox(frame)
        self.postprocess(img, output, pad)
        
        cv2.putText(img, "DIDAAAAAAAAAAAAAa", (500, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        # # Using cv2.imwrite() method

        cv2.imwrite(filename, img)

        # cv2.imshow("image", img)

        # cv2.waitKey(0)

        # cv2.destroyAllWindows()

        return

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

            print(indices)

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
        return self.on_new_sample()
        

if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()
