import cv2

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
        [draw_detections(self, frame, boxes[i], scores[i], class_ids[i]) for i in indices]

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
    cv2.putText(frame, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)