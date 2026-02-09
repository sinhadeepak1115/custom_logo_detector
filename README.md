# Custom Logo Detection with YOLOv5

This project implements a custom logo detection system using YOLOv5, following the tutorial approach for training on custom datasets.

## Project Structure

```
custom_logo/
├── data/                          # Dataset directory
│   ├── Label_Studio_Output/      # Original annotations
│   │   ├── images/               # Original images
│   │   ├── result.json           # COCO format annotations
│   │   └── yolo_annotations/     # Converted YOLO annotations
│   └── suzuki_logo_detection/  # Split dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── yolov5/                        # YOLOv5 repository (cloned)
├── scripts/                       # Utility scripts
│   ├── convert_coco_to_yolo.py
│   ├── split_dataset.py
│   ├── visualize_annotations.py
│   ├── prepare_dataset.py
│   ├── run_inference.py
│   ├── process_video.py
│   └── analyze_results.py
├── data_config.yaml              # Dataset configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch (GPU support)

If you have a CUDA-capable GPU:

```bash
# Uninstall CPU-only versions first
pip uninstall torch torchvision torchaudio

# Install GPU version (adjust CUDA version as needed)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Verify GPU availability:
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

### 4. Setup YOLOv5

```bash
chmod +x setup_yolov5.sh
./setup_yolov5.sh
```

Or manually:
```bash
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

## Dataset Preparation

### Step 1: Annotate Images with Label Studio

1. Install Label Studio:
```bash
pip install label-studio
label-studio start
```

2. Create a new project and import your images
3. Annotate images with bounding boxes and labels
4. Export annotations in COCO JSON format

### Step 2: Convert COCO to YOLO Format

```bash
python convert_coco_to_yolo.py \
    --coco_json data/Label_Studio_Output/result.json \
    --images_dir data/Label_Studio_Output/images \
    --output_dir data/Label_Studio_Output/yolo_annotations
```

### Step 3: Split Dataset

```bash
python split_dataset.py \
    --base_path data/Label_Studio_Output \
    --output_base data/suzuki_logo_detection \
    --test_size 0.2 \
    --val_size 0.5
```

### Or Use the Combined Script

```bash
python prepare_dataset.py \
    --coco_json data/Label_Studio_Output/result.json \
    --images_dir data/Label_Studio_Output/images \
    --output_base data
```

### Step 4: Update Data Configuration

Copy `data_config.yaml` to `yolov5/data/custom_logo.yaml` and update paths:

```yaml
nc: 2
names: ['Suzuki_Logo', 'Suzuki_Text']
train: data/suzuki_logo_detection/images/train
val: data/suzuki_logo_detection/images/val
test: data/suzuki_logo_detection/images/test
```

## Training

### CPU Training

```bash
python yolov5/train.py \
    --img 640 \
    --cfg yolov5s.yaml \
    --hyp hyp.scratch-low.yaml \
    --batch 16 \
    --epochs 200 \
    --data custom_logo.yaml \
    --weights yolov5s.pt \
    --workers 24 \
    --name yolo_logo_detection
```

### GPU Training

```bash
python yolov5/train.py \
    --img 640 \
    --cfg yolov5s.yaml \
    --hyp hyp.scratch-low.yaml \
    --batch 16 \
    --epochs 200 \
    --data custom_logo.yaml \
    --weights yolov5s.pt \
    --workers 1 \
    --name yolo_logo_detection
```

**Note:** Adjust `--batch` based on your GPU memory. For 4GB GPUs, use batch size 16 or lower.

Training outputs will be saved to: `yolov5/runs/train/yolo_logo_detection/`

Best weights: `yolov5/runs/train/yolo_logo_detection/weights/best.pt`

## Inference

### Python API

```python
import torch

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       'yolov5/runs/train/yolo_logo_detection/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')
results.show()
print(results.pandas().xyxy[0])
```

### Command Line

```bash
python run_inference.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --method cli \
    --save_txt
```

Or directly:
```bash
python yolov5/detect.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --name yolo_logo_detection \
    --save-txt
```

## Video Processing

### Download and Process Video

```bash
python process_video.py \
    --youtube_url "https://www.youtube.com/watch?v=zc3JYvvmXxw" \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --max_frames 200 \
    --conf 0.3
```

### Analyze Results

```bash
python analyze_results.py \
    --output_folder yolov5/runs/detect/yolo_frame_detection \
    --label_map "0:Suzuki_Logo,1:Suzuki_Text" \
    --save_path coverage_plot.png
```

## Visualization

View annotated images:

```bash
python visualize_annotations.py \
    --coco_json data/Label_Studio_Output/result.json \
    --images_dir data/Label_Studio_Output/images \
    --img_id 0 \
    --save_path example_annotation.png
```

## Tips for Better Results

1. **More Training Data**: YOLOv5 recommends >=1500 images per class for production use
2. **Data Augmentation**: Use higher augmentation settings (`hyp.scratch-med.yaml` or `hyp.scratch-high.yaml`) if you have GPU resources
3. **Model Architecture**: Try larger models (`yolov5m.yaml`, `yolov5l.yaml`, `yolov5x.yaml`) for better accuracy
4. **Training Parameters**: Increase epochs and batch size if possible
5. **Anchor Boxes**: YOLOv5 automatically learns anchor boxes, but you can tune them manually

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `yolov5n.yaml` (nano)

### No GPU Detected
- Verify CUDA installation: `nvcc --version`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with correct CUDA version

### Import Errors
- Make sure virtual environment is activated
- Install missing dependencies: `pip install -r requirements.txt`
- For YOLOv5: `pip install -r yolov5/requirements.txt`

## References

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [Label Studio](https://labelstud.io/)
- [COCO Format](https://cocodataset.org/#format-data)
- [YOLO Format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## License

This project follows the same license as YOLOv5 (AGPL-3.0).
