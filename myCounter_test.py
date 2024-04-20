from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from ultralytics.utils.files import increment_path
from pathlib import Path

model = YOLO("yolov8n.pt")
source = "los_angeles.mp4"
# source = "MyVideo.mp4"
cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
model.to("cuda")   #my

# Define region points
region_points = [(400, 500), (1200, 500), (1200, 700), (400, 700)]
# region_points = [(0, 0), (1080, 500)]
# region_points = [(400, 600), (1200, 600)]  #过线计数
# classes_to_count = [0, 2]  # person and car classes for count

# Video setup
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

# Output setup
save_dir = increment_path(Path("ultralytics_rc_output") / "expCount")
save_dir.mkdir(parents=True, exist_ok=True)

# Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi",
#                        cv2.VideoWriter_fourcc(*'mp4v'),
#                        fps,
#                        (w, h))
video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))


# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, tracker="bytetrack.yaml")
    # tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)  #如果要类别计数的话

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()