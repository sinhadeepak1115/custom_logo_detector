# Custom Logo Detection with YOLOv8

This project implements a custom logo detection system using YOLOv8, following the tutorial approach for training on custom datasets.

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
├── runs/                          # Training and detection results
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

### 4. Setup YOLOv8

```bash
chmod +x setup_yolov5.sh
./setup_yolov5.sh
```

Or manually:
```bash
pip install ultralytics
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

Update `data_config.yaml` with correct paths:

```yaml
nc: 2
names: ['Suzuki_Logo', 'Suzuki_Text']
train: data/suzuki_logo_detection/images/train
val: data/suzuki_logo_detection/images/val
test: data/suzuki_logo_detection/images/test
```

## Training

### Using the Training Script

```bash
python train_model.py \
    --data data_config.yaml \
    --imgsz 640 \
    --model yolov8n.pt \
    --batch 16 \
    --epochs 200 \
    --workers 8 \
    --name yolo_logo_detection
```

### Model Options

YOLOv8 offers different model sizes:
- `yolov8n.pt` - Nano (smallest, fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (largest, most accurate)

**Note:** Adjust `--batch` based on your GPU memory. For 4GB GPUs, use batch size 16 or lower.

Training outputs will be saved to: `runs/train/yolo_logo_detection/`

Best weights: `runs/train/yolo_logo_detection/weights/best.pt`

## Inference

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/train/yolo_logo_detection/weights/best.pt')

# Run inference
results = model.predict('path/to/image.jpg', conf=0.3)

# Access results
result = results[0]
boxes = result.boxes
for box in boxes:
    print(f"Class: {result.names[int(box.cls[0])]}, Confidence: {box.conf[0]:.2f}")
```

### Command Line

```bash
python run_inference.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --method cli \
    --save_txt
```

Or using YOLOv8 CLI directly:
```bash
yolo predict \
    model=runs/train/yolo_logo_detection/weights/best.pt \
    source=data/suzuki_logo_detection/images/test/ \
    conf=0.3 \
    save_txt=True
```

## Video Processing

### Download and Process Video

```bash
python process_video.py \
    --youtube_url "https://www.youtube.com/watch?v=zc3JYvvmXxw" \
    --weights runs/train/yolo_logo_detection/weights/best.pt \
    --max_frames 200 \
    --conf 0.3
```

### Analyze Results

```bash
python analyze_results.py \
    --output_folder runs/detect/yolo_frame_detection \
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

1. **More Training Data**: YOLOv8 recommends >=1500 images per class for production use
2. **Data Augmentation**: YOLOv8 includes built-in augmentation - adjust augmentation settings in training arguments if needed
3. **Model Architecture**: Try larger models (`yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`) for better accuracy
4. **Training Parameters**: Increase epochs and batch size if possible
5. **Image Size**: Larger image sizes (e.g., 1280) can improve accuracy but require more memory

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `yolov8n.pt` (nano)
- Reduce image size: `--imgsz 416` instead of 640

### No GPU Detected
- Verify CUDA installation: `nvcc --version`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with correct CUDA version

### Import Errors
- Make sure virtual environment is activated
- Install missing dependencies: `pip install -r requirements.txt`
- For YOLOv8: `pip install ultralytics`

## References

- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Label Studio](https://labelstud.io/)
- [COCO Format](https://cocodataset.org/#format-data)
- [YOLO Format](https://docs.ultralytics.com/datasets/)

## License

This project follows the same license as YOLOv8 (AGPL-3.0).
