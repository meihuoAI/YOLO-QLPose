import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose-CGFPN.yaml').load('weights/yolov8n-pose.pt')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='grinding-dataset.yaml',
                task='pose',
                mode='train',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=10,
                workers=8,
                project='runs/train',
                name='v8n-CGFPN',
                )