## TASK II:

Image Classification: Train a YOLOV8 object detection model for face detection using
the dataset provided in the link below.
Then train a Image classification model which takes the input as a crop from the object
detection model(face crop) to classify emotions in categories [‘Happy’, ‘Sad’, ‘Neutral’]
and display the class name on the face class bounding box on 4 videos.

# Dataset:
- Face detection: https://www.kaggle.com/datasets/deepakat002/face-mask-detection-yolov5.
- For emotion classification cropped images from face detection of nomask class were used.
- Emotions were manually annotated from the cropped images.

# Models:
- The pre-trained model used is yolov8m.pt was used to train the face detection.
- The pre-trained model used is yolov8m-cls.pt was used to train the Emotion classification.

# Challenges Faced: 
- From the cropped images from face detection very low number of sad faces were present i.e sad class is underrepresented.
