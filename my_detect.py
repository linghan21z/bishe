from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('F://bishe_DetectTrack//ultralytics-main//runs//detect//train2//weights//best.pt')  # load a custom model

# Predict with the model
results = model(source='./myVideoPic/crossing.jpg', stream=True, imgsz=320, conf=0.5)
# return a generator of Results objects
# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='resultVis.jpg')  # save to disk



