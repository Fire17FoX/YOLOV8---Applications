## TASK IV:

Vehicle Counting and Speed Estimation: Use YOLOv8 object detection to exclusively
identify vehicles, count the number of vehicles present within that video frame, and
implement speed estimation for the detected vehicles in the video.
Save the source video, inference code, inference video and the documentation in a zip
file.


# Models:
- The pre-trained model used is yolov8l.pt.

# Program:
- The Python program Car-counter.py was created to detect vehicles, vehicle count, and speed of the vehicle.
- First, all objects were detected in each frame using the object detection model i.e. yolov8l.pt.
- Then bounding boxes, speed, and bounding boxes were applied only to the 8 vehicle classes available on the yolov8l.pt.
- Code inspiration: https://www.youtube.com/watch?v=WgPbbWmnXJ8
- speed was calculated using the tracking and change in the position of the bounding boxes.

# Challenges Faced: 
- Since the videos are from different distances from the road the calculated speed is not accurate.
