import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("4.mp4")  # For Video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
output_video = cv2.VideoWriter("4result.mp4", fourcc, fps, (width, height))
model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []
prev_positions = {}
track_history = defaultdict(lambda: [])

while True:
    success, img = cap.read()

    result = model.track(img, stream=True)

    for r in result:
        count = 0
        boxes = r.boxes

        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate speed
            currentClass = classNames[int(box.cls[0])]
            if currentClass in prev_positions:
                prev_x1, prev_y1, _, _ = prev_positions[currentClass]
                speed = 0.4*math.sqrt((x1 - prev_x1)**2 + (y1 - prev_y1)**2)
                cv2.putText(img, f'Speed: {speed:.2f}', (max(0, x1), max(35, y1 + 20)),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # Update previous positions
            prev_positions[currentClass] = (x1, y1, x2, y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            currentClass = classNames[int(box.cls[0])]

            if currentClass == "car" or currentClass == "bicycle" or currentClass == "motorbike" or \
                    currentClass == "aeroplane" or currentClass == "bus" or currentClass == "train" or \
                    currentClass == "truck" or currentClass == "boat" and conf > 0.5:
                count += 1
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                    scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=5)

        cv2.putText(img,'The count: ' + str(count), (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 2)

    cv2.imshow("Image", img)
    output_video.write(img)
    cv2.waitKey(1)
