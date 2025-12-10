Here is the **complete, clean, and professional README.md in English**, ready for GitHub:

# Intelligent System for Detecting the Freshness of Agricultural Products

### Computer Vision & Artificial Intelligence – D.R. Congo

##  **Project Description**

This project focuses on designing and developing an **intelligent system capable of detecting the freshness of agricultural products** (safou, tomatoes, eggplants, oranges, mangoes) using advanced **computer vision** and **deep learning** techniques.

The system aims to support farmers, traders, and consumers in the **Democratic Republic of Congo (DRC)**, where product quality is strongly affected by storage, transportation, and market conditions.

The complete pipeline includes:

* building a custom dataset of fresh and rotten agricultural products,
* image annotation,
* training a detection/classification model (YOLOv8),
* evaluating performances,
* deploying a user-friendly prediction system.

##  **Objectives**

* Automatically classify agricultural products as **fresh** or **rotten**.
* Provide a reliable, practical tool for real-world markets in DRC.
* Reduce post-harvest losses.
* Support digital agriculture and quality control.

##  **Project Structure**

```txt
fruit_freshness_detection/
│
├── datasets/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── annotations/          # Annotation files (.xml or YOLO .txt)
├── models/               # Trained models (.pt)
│   └── yolov8_fresh.pt
│
├── scripts/
│   ├── train.py          # YOLO training
│   ├── detect.py         # Detection on images/videos
│   ├── evaluate.py       # Metrics & evaluation
│   └── export.py         # Export to ONNX/TFLite
│
├── results/
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   └── detection_samples/
│
├── app/                  # Optional API or interface
│   ├── main.py
│   └── requirements.txt
│
├── README.md
└── requirements.txt
```

##  **Technologies Used**

* **YOLOv11 (Ultralytics)**
* **Python 3.10+**
* **PyTorch / TensorFlow**
* **OpenCV**
* **Roboflow** (dataset management & annotation)
* **Jupyter Notebook**
* **FastAPI / Flask / Streamlit** (optional interface)

##  **Dataset Preparation**

The dataset includes images collected from:

* real markets in DRC,
* manually captured mobile phone images,
* selected and cleaned online sources,
* extracted frames from videos.

###  Annotation

Images were annotated using:

* **Roboflow Annotate**, or
* **Label Studio**

Classes used:

```
fresh
rotten
```

The dataset was exported in YOLOv8 format.

##  **Model Training**

Example YOLOv8 training command:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

Training outputs include:

* learning curves,
* confusion matrix,
* precision/recall curves,
* `best.pt` model weights.

##  **Running Detection**

Run prediction on a single image:

```bash
yolo detect predict model=models/yolov8_fresh.pt source=images/test.jpg
```

Run on webcam or video:

```bash
yolo detect predict model=models/yolov8_fresh.pt source=0
```

##  **Model Evaluation**

Evaluation includes:

* mAP50 and mAP50-95
* Precision
* Recall
* F1-score
* Inference speed

All evaluation results are saved in the `results/` directory.

## **User Interface (Optional)**

A simple interface can be deployed using:

* **Gradio**
* **Streamlit**
* **FastAPI**

Example usage in Python:

```python
from ultralytics import YOLO

model = YOLO("models/yolov8_fresh.pt")
results = model("image.jpg")
results.show()

##  **Impact for the DRC**

This system helps address key challenges:

* improve food quality and safety,
* reduce waste from rotten products,
* support farmers and sellers in local markets,
* promote low-cost digital agriculture solutions,
* provide a foundation for smart agriculture in DRC.



##  **Future Improvements**

* Multi-class detection (fresh, intermediate, rotten).
* Mobile application (Android) deployment.
* Product-specific models (e.g., mango freshness detector).
* Integration with IoT cameras for real-time monitoring.
* Vision Transformers (ViT) experiments.



##  **Author**

**Kutuka F. Xavier**
Master in Intelligent Systems & Multimedia
Hanoi, Vietnam




