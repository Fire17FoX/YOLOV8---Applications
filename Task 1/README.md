## TASK I:

Object Detection: Implement YOLOv8 object detection model to identify and locate
specifically person class(No other class should be displayed on the screen in the
output) in images.
Use a pre-trained model, fine-tune it on a custom dataset containing images of your
choice, with at least 4 types of augmentation(Include the augmentation snippet in the
source code) applied on the original images.
Train the model on the generated dataset.
Document your data preparation, model selection, and evaluation process.
Inference the model over 4 videos and save the output in a zip file with the dataset,
weight file, source video, inference code and the documentation.
Provide insights into the challenges you faced and the solutions you implemented.

Custom dataset reference: https://universe.roboflow.com/tank-detect/person-dataset-kzsop

Used Roboflow projects to pre-process and add Augmentations to the dataset.

# PREPROCESSING: 
- Auto-Orient: Applied,
- Resize: Stretch to 640x640.

# AUGMENTATION:
- Flip: Horizontal, Vertical,
- 90° Rotate: Clockwise & Counter-Clockwise,
- Grayscale: Apply to 100% of images,
- Mosaic: Applied.

# Dataset split: 
- Train 90%
- Valid 10%.

The pre-trained model used is yolov8n.pt 

# Challenges Faced: 
- Local GPU was not sufficient to train.
# Solutions:
- Used Google Collab to train the model then exported the weight file to the local machine.
