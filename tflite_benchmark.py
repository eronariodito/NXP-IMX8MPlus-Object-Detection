#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import time
import ai_edge_litert.interpreter as litert
import os
import time
import yaml
import json
import glob
from tqdm import tqdm
from pathlib import Path

class TFLiteBenchmark:
    def __init__(self):
        self.conf_all = 0.001
        self.conf = 0.25
        self.iou = 0.7
        self.count = 0
        self.int8 = True

        self.model_name = "models/yolov8n_full_integer_quant_416.tflite"
        model_path = os.getcwd() + "/" + self.model_name

        self.device = "npu"

        if self.device == "npu":
            try:
                # Create the delegate
                delegate_lib = "/usr/lib/libvx_delegate.so"
                delegate = litert.load_delegate(delegate_lib)
                
                # Initialize the interpreter with the delegate
                self.interpreter = litert.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate]
                )
                self.interpreter.allocate_tensors()
                
                print("Successfully loaded NPU delegate")
            except Exception as e:
                print(f"Error loading delegate: {e}")
                print("Falling back to CPU execution")
                self.interpreter = litert.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
        else:
            self.interpreter = litert.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            print("Successfully loaded CPU")

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.scale, self.zero_point = self.output_details[0]["quantization"]
        self.in_scale, self.in_zero_point = self.input_details[0]["quantization"]

        if self.scale == 0:
            self.int8 = False

        with open("coco128.yaml", "r") as f:
            data = yaml.safe_load(f)

        self.classes = data.get("names", {})

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.dataset = os.getcwd() + '/datasets/coco'
        #elf.dataset = os.getcwd() + '/datasets/coco8'

        print("\n")
        folders = glob.glob(self.dataset + "/images/val*")

        if folders:
            self.val_folder = folders[0]  # Get the first match (or loop through if multiple exist)
            print(f"Found folder: {self.val_folder}")
        else:
            raise FileNotFoundError(f"No folder starting with 'val' found in {self.dataset}")
        
        folders = glob.glob(self.dataset + "/annotation*")

        if folders:
            ann_folder = folders[0]  # Get the first match (or loop through if multiple exist)
            print(f"Found annotation folder: {ann_folder}")
        else:
            print("No annotation folder found, skipping pycocoeval")
            return
        
        files = glob.glob(f"{ann_folder}/*val*.json")

        if files:
            self.ann_file = files[0]
            print(f"Found annotation json: {self.ann_file}")
            self.pycoco = True
        else:
            print("No annotation json found, skipping pycocoeval")
            self.pycoco = False
        print("\n")

        self.coco_mapping_80to91 = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 13, "13": 14, "14": 15, "15": 16, "16": 17, "17": 18, "18": 19, "19": 20, "20": 21, "21": 22, "22": 23, "23": 24, "24": 25, "25": 27, "26": 28, "27": 31, "28": 32, "29": 33, "30": 34, "31": 35, "32": 36, "33": 37, "34": 38, "35": 39, "36": 40, "37": 41, "38": 42, "39": 43, "40": 44, "41": 46, "42": 47, "43": 48, "44": 49, "45": 50, "46": 51, "47": 52, "48": 53, "49": 54, "50": 55, "51": 56, "52": 57, "53": 58, "54": 59, "55": 60, "56": 61, "57": 62, "58": 63, "59": 64, "60": 65, "61": 67, "62": 70, "63": 72, "64": 73, "65": 74, "66": 75, "67": 76, "68": 77, "69": 78, "70": 79, "71": 80, "72": 81, "73": 82, "74": 84, "75": 85, "76": 86, "77": 87, "78": 88, "79": 89, "80": 90}
        self.time_metrics = {"preprocess": 0, "inference": 0, "postprocess": 0, "total": 0}

        self.output_dir = self.val_dir()

        filename = os.path.basename(model_path)
        print(f"Running benchmark for {filename}")
        print(f"Input image size {self.input_details[0]['shape']} \n")

    def preprocess(self, frame):
        
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

    def postprocess(self, frame, outputs, pad, plot) :
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
            keep = scores > self.conf_all
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_all, self.iou)

            if plot:
                if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
                        indices.flatten()
                else:
                    return
            
                # Draw detections that survived NMS
                [self.draw_detections(frame, boxes[i], scores[i], class_ids[i]) for i in indices]

        return indices, boxes, scores, class_ids
    

    def postprocess_json(self, indices, boxes, scores, class_ids, image_id) :
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        coco_json = []
    
        if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
            indices.flatten()
        else:
            return 

        # Draw detections that survived NMS
        for i in indices:
            x1, y1, w, h = boxes[i]
            
            coco_json.append({
                    "image_id": int(image_id),
                    "category_id": self.coco_mapping_80to91[f"{int(class_ids[i]) + 1}"],
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[i])
                })

        return coco_json

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
        if score < self.conf:
            return
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        font_scale = min(frame.shape[0], frame.shape[1]) / 1000

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

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

        a = 1 - (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255

        if a < 0.5:
            font_color = (0, 0, 0)
        else:
            font_color = (255, 255, 255)

        # Draw the label text on the image
        cv2.putText(frame, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def val_dir(self):
        base_path = os.getcwd() + "/run/val"
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)  # Creates 'run/val' if missing

        # Find existing valX folders
        existing_folders = [d for d in os.listdir(base_dir) if d.startswith("val") and d[3:].isdigit()]
        
        # Extract numbers and determine the next available one
        existing_numbers = sorted([int(d[3:]) for d in existing_folders]) if existing_folders else []
        next_number = existing_numbers[-1] + 1 if existing_numbers else 1

        # Create new directory
        new_folder = base_dir / f"val{next_number}"
        new_folder.mkdir()
        
        return new_folder
    
    def run(self):
        detection_json = []
        i = 0
        plot = True

        image_path = Path(self.val_folder)
        files = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']]
        
        for idx,item in tqdm(enumerate(files), total=len(files), desc="Inference"):
            image_id = int(item.stem.lstrip("0"))

            img = cv2.imread(item, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

            preprocess_tic = time.time()
            x, pad = self.preprocess(img)
            if self.int8:
                x = (x / self.in_scale + self.in_zero_point).astype(np.int8)
            self.time_metrics["preprocess"] += time.time() - preprocess_tic

            inference_tic = time.time()

            # set frame as input tensors
            self.interpreter.set_tensor(self.input_details[0]['index'], x)

            # perform inference
            self.interpreter.invoke()

            self.time_metrics["inference"] += time.time() - inference_tic

            postprocess_tic = time.time()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
            if self.int8:
                output = (output.astype(np.float32) - self.zero_point) * self.scale
            indices, boxes, scores, class_ids = self.postprocess(img, output, pad, plot)
            self.time_metrics["postprocess"] += time.time() - postprocess_tic
            self.time_metrics["total"] += time.time() - preprocess_tic  
            result = self.postprocess_json(indices, boxes, scores, class_ids, image_id)    

            if result is not None:
                detection_json.extend(result)

            if i < 10:
                plot = True
                # Filename
                filename = Path(self.output_dir) / f'output{i}_{image_id}.jpg'
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, img)
            else:
                plot = False
            i+=1

        with open(Path(self.output_dir) / "prediction.json", "w") as json_file:
            json.dump(detection_json, json_file, indent=4)

        average_time = {key: (value / len(files)) * 1000 for key, value in self.time_metrics.items()}

        with open(Path(self.output_dir) / "time.json", "w") as json_file:
                    json.dump(self.time_metrics, json_file, indent=4)
                    
        print(f"Average preprocessing time: {average_time['preprocess']:.2f} ms")
        print(f"Average inference time: {average_time['inference']:.2f} ms")
        print(f"Average postprocessing time: {average_time['postprocess']:.2f} ms")
        print(f"Average total time: {average_time['total']:.2f} ms")

        print(f"Output saved to {Path(self.output_dir)}")

        return detection_json

    def validate_pycoco(self, preds_file):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        print("\n")

        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(preds_file)

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return

if __name__ == "__main__":
    benchmark = TFLiteBenchmark()

    preds_file = benchmark.run()
    if benchmark.pycoco:
        benchmark.validate_pycoco(preds_file)


