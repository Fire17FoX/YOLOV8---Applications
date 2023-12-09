import cv2
import ultralytics
from ultralytics import YOLO
import math
import cvzone

model1 = YOLO('yolov8_face.pt')
model2 = YOLO('yolov8_emotion_classification.pt')


#results = model1.track(source='3.mp4', \
#save=True, show=False, project='./result', conf=0.5, classes=1)
cap = cv2.VideoCapture("1.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
output_video = cv2.VideoWriter("1result.mp4", fourcc, fps, (width, height))
while True:
        success, img = cap.read()
        result1=model1(img,stream=True)
        for r in result1:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cropped_image = img[y1:y2, x1:x2]
                
                res=model2(source=cropped_image)
                emo=res[0].names[res[0].probs.top1]
                #print(res)
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                #currentClass = classNames[cls]
                cvzone.putTextRect(img, f'{emo}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=1)
                cv2.imshow("Image", img)
                output_video.write(img)
                cv2.imshow("crop", cropped_image)