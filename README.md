# YOLO-QLPose: Real-time Keypoint Detection for Grinding Wheel Grooves

**YOLO-QLPose** is a lightweight, real-time deep learning model built on **YOLOv8** for high-precision measurement of chamfered grinding wheel grooves.  
It introduces **MAF (Multi-modal Attention Fusion) module**, **KPSEF (KeyPoint Shared Enhancement Fusion) network**, and a novel **QLIoU loss function**, significantly improving the accuracy and robustness of keypoint localization in complex industrial environments.

---

## 🚀 Highlights

- 📊 **Improved Accuracy**: Outperforms YOLOv8n-Pose with higher Precision, Recall, and mAP50 on the Grinding-Pose dataset.  
- ⚡ **Real-time Performance**: Evaluates a grinding wheel in ~11s, compared to 6 minutes with GGPD.  
- 🪶 **Lightweight Design**: Only ~2.3M parameters, suitable for deployment on resource-constrained devices.  
- 🧩 **Industrial Applicability**: Tailored for **high-end manufacturing** requiring real-time, high-precision measurement.  

---

## 📖 Documentation

### Installation

```bash
git clone https://github.com/yourusername/YOLO-QLPose.git
cd YOLO-QLPose
pip install -r requirements.txt
```

Requirements:
- Python >= 3.8  
- PyTorch >= 1.12  
- CUDA 11.0+ (if GPU training)  

---

### Usage

#### Training
```bash
yolo train model=yolo-qlpose.yaml data=grinding-pose.yaml epochs=200 imgsz=640
```

#### Validation
```bash
yolo val model=weights/best.pt data=grinding-pose.yaml
```

#### Inference
```bash
yolo predict model=weights/best.pt source='test_images/' save=True
```

Python interface:
```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("test_images/sample.bmp")
results.show()
```

---

## 📊 Results

### Ablation Study (Grinding-Pose Dataset)

| Model         | P (%) | R (%) | mAP50 (%) | F1   |
|---------------|-------|-------|-----------|------|
| YOLOv8n-Pose  | 95.2  | 94.6  | 94.3      | 94.9 |
| **YOLO-QLPose** | **98.2** | **99.3** | **97.0** | **98.7** |

YOLO-QLPose shows significant improvements with lower parameter count (~2.3M) compared to YOLOv8n-Pose (~3.5M).  

---

## 📂 Dataset

We introduce the **Grinding-Pose dataset**:  
- 📷 14,400 high-resolution contour images (4096×3000).  
- 🏷️ Keypoints annotated using LabelMe, including line-fitting and arc-fitting points.  
- 🔧 Captured with a custom grinding wheel dressing machine using telecentric optics.  

---

## 🏗️ Model Architecture

YOLO-QLPose builds on YOLOv8n with the following improvements:

- **MAF Module**: Multi-modal attention fusion for edge-sensitive feature aggregation.  
- **KPSEF Network**: Shared DEConv detection head combining object and keypoint detection.  
- **QLIoU Loss**: Combines Quality Focal Loss (QFL) and PIoU for robust bounding box regression.  

---

## 📜 Citation

If you use this work in your research, please cite:

```
@article{YOLOQLPose2025,
  title={YOLO-QLPose: A Deep Learning-based Approach for Real-time Measurement of Key Dimensions of Chamfered Grinding Wheel Grooves},
  author={Your Name et al.},
  journal={Journal/Conference},
  year={2025}
}
```

---

## 🤝 Acknowledgements

- Built upon [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
- Dataset collected with custom grinding wheel dressing machine.  
