# Real-Time Traffic Detection and Counting 🚗📊

This project implements a real-time vehicle detection, tracking, and counting system using **YOLOv8**, **Faster R-CNN**, and **Deep SORT**. It is designed to assess traffic flow from video footage in urban scenes.

---

## 📦 Project Structure

```
traffic-vision/
├── data/ # Images and label files (YOLO format)
│ ├── images/
│ ├── labels/
│ ├── train.coco.json # Converted COCO annotations for FRCNN
│ └── val.coco.json
├── results/
│ └── final_counts.json # Vehicle count summary from DeepSORT
├── src/
│ ├── demo_yolo_deepsort.py
│ ├── train_frcnn.py
│ ├── convert_yolo_to_coco.py
│ ├── make_subset.py
├── tools/
│ ├── fix_labels.py
│ └── fix_neg_labels.py
├── weights_clean/
│ └── yolov8n-fresh/ # Cleaned YOLOv8 weights after label fixes
├── traffic1.mp4
├── ua.yaml # YOLO dataset config
├── README.md
└── requirements.txt
```

---

## 🚀 Features

- YOLOv8 object detection trained on UA-DETRAC
- COCO-formatted annotations for Faster R-CNN (Torchvision)
- DeepSORT multi-object tracking
- Real-time per-class vehicle counting
- Compact label and ID cleanup tools
- Final output to `results/final_counts.json`

---

## 🧪 Evaluation

### YOLOv8 (Validation Metrics)
- **Precision, Recall, mAP50, mAP50-95** via `yolo val`
- Validated using clean annotations

### DeepSORT + YOLOv8 (Tracking)
- Real-time tracking performance at ~10–15 FPS on RTX 3060
- Counts stored and displayed live per class

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
Contents of requirements.txt:

ultralytics==8.3.137
torch>=2.0
torchvision
opencv-python
deep_sort_realtime
```

## ▶️ Run the Real-Time Tracker

```
python src/demo_yolo_deepsort.py --source traffic1.mp4
```

Or use your webcam:

```
python src/demo_yolo_deepsort.py --source 0
```

## 📊 Output Example
- Detected vehicle types: car, bus, truck, motorcycle
- Count displayed in real time (top-left corner)
- JSON summary saved to:
```
results/final_counts.json
```

## 📄 License
MIT License — free to use, modify, and share.

---