from ultralytics import YOLO
import torch
if __name__ == '__main__':
    torch.cuda.empty_cache()  # 在GPU上进行内存管理：可以尝试清理不再使用的内存，以便释放GPU内存
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('F://bishe_DetectTrack//ultralytics-main//runs//detect//train2//weights//best.pt')  # load a custom model

    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # validation_results = model.val(data='coco8.yaml',
    metrics = model.val(data='VisDrone.yaml',
                                   imgsz=320,
                                   batch=4,
                                   conf=0.25,
                                   iou=0.6,
                                   device='0',
                                   workers=0)
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

