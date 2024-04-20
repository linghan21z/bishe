from ultralytics import YOLO

if __name__ == '__main__':
    # Load an official or custom model
    # model = YOLO('F://bishe_DetectTrack//ultralytics-main//runs//detect//train2//weights//best.pt')  # load a custom model
    model = YOLO('yolov8n.pt')  # Load an official Detect model
    # model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
    # model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
    # model = YOLO('path/to/best.pt')  # Load a custom trained model

    results = model.track(source='./myVideoPic/bus.jpg', stream=True, show=True)
    # Perform tracking with the model
    # results = model.track(source="./myVideoPic/MyVideo.mp4", show=True, stream=True, save=True)
    # Tracking with default tracker, 用这一句可以运行

    # results = model.track(source="0", show=True, stream=True) # generator of Results objects

    # results = model.track(source="./mp4_1.mp4", show=True)  # Tracking with ByteTrack tracker 不知道为什么stram加上就不能显示mp4了
    # Tracking with ByteTrack tracker
    # results = model.track(source="./mp4_1.mp4", show=True, stream=True, tracker="bytetrack.yaml")

    # Process results generator
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     result.show()  # display to screen
    #     result.save(filename='resultmp41.mp4')  # save to disk



# 带跟踪功能的流式 for 循环
# import cv2
# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')
#
# # Open the video file
# video_path = "path/to/video.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True)
#
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()






# from PIL import Image  # 这个方法可以存下来，也可以看图显示，但不知道PIL是什么
# model = YOLO('yolov8n.pt')
# results = model('bus.jpg')  # results list
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image