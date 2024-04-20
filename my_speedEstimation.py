from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
from ultralytics.utils.files import increment_path
from pathlib import Path
import cv2

model = YOLO("yolov8n.pt")
names = model.model.names
source = "los_angeles.mp4"
# source = "MyVideo.mp4"
cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Video setup
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

# Output setup
save_dir = increment_path(Path("ultralytics_rc_output") / "expSpeed")
save_dir.mkdir(parents=True, exist_ok=True)

# Video writer
# video_writer = cv2.VideoWriter("speed_estimation.avi",
#                                cv2.VideoWriter_fourcc(*'mp4v'),
#                                fps,
#                                (w, h))
video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

# line_pts = [(0, 360), (1280, 360)]
line_pts = [(400, 600), (1200, 600)]  #过线


# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # tracks = model.track(im0, persist=True, show=False)
    tracks = model.track(im0, persist=True, show=False, tracker="bytetrack.yaml")

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()