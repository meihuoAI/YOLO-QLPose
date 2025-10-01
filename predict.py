import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/base-exp1/weights/best.pt')
    model.predict(source = '/data1/home_data/chendi/datasets/grinding/images/val' ,
                  task = 'pose', 
                  mode = 'predict', 
                  imgsz = 640,
                  project = 'runs/predict/',
                  name = 'base',
                  save = True,
                  save_txt = True  # 生成检测锚框的txt文件，用于显示正确检测、误检、漏检的锚框
                  )
