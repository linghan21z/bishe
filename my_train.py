from ultralytics import YOLO
import torch

if __name__ == '__main__':
    torch.cuda.empty_cache()  # 在GPU上进行内存管理：可以尝试清理不再使用的内存，以便释放GPU内存
    # 加载模型 # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML，start training from scratch
    # model = YOLO('yolov8n.pt')  # Start training from a pretrained *.pt model
    # Build a new model from YAML, transfer pretrained weights to it and start training
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # Train the model
    # results = model.train(data='VisDrone.yaml', epochs=1, imgsz=640, batch=16, device='0')
    results = model.train(data='VisDrone.yaml', epochs=1, imgsz=320, batch=4, device='0', workers=0)

    # Customize validation settings
    # validation_results = model.val(data='coco8.yaml',
    # validation_results = model.val(data='VisDrone.yaml',
    #                                epochs=3,
    #                                imgsz=640,
    #                                batch=16,
    #                                # conf=0.25,
    #                                # iou=0.6,
    #                                # device='0'
    #                                 )

