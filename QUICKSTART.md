# Quick Start Guide

## Prerequisites

1. Python 3.8+
2. Virtual environment (recommended)
3. CUDA-capable GPU (optional but recommended)

## Setup (5 minutes)

### Linux/Mac

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup YOLOv8
pip install ultralytics

# 4. Install PyTorch with GPU support (if you have CUDA)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Windows (PowerShell)

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup YOLOv8
pip install ultralytics

# 4. Install PyTorch with GPU support (if you have CUDA)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Windows (Command Prompt / CMD)

```cmd
REM 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat

REM 2. Install dependencies
pip install -r requirements.txt

REM 3. Setup YOLOv8
pip install ultralytics

REM 4. Install PyTorch with GPU support (if you have CUDA)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Process Your Dataset (2 minutes)

If you have Label Studio output:

**Linux/Mac:**
```bash
python quick_start.py --label_studio_dir project-5-at-2026-02-06-06-49-4f3e5bf6
```

**Windows:**
```cmd
python quick_start.py --label_studio_dir project-5-at-2026-02-06-06-49-4f3e5bf6
```

Or manually:

**Linux/Mac:**
```bash
# 1. Convert COCO to YOLO format
python convert_coco_to_yolo.py \
    --coco_json project-5-at-2026-02-06-06-49-4f3e5bf6/result.json \
    --images_dir project-5-at-2026-02-06-06-49-4f3e5bf6/images \
    --output_dir data/Label_Studio_Output/yolo_annotations

# 2. Split dataset
python split_dataset.py \
    --base_path data/Label_Studio_Output \
    --output_base data/suzuki_logo_detection
```

**Windows (PowerShell):**
```powershell
# 1. Convert COCO to YOLO format
python convert_coco_to_yolo.py `
    --coco_json project-5-at-2026-02-06-06-49-4f3e5bf6/result.json `
    --images_dir project-5-at-2026-02-06-06-49-4f3e5bf6/images `
    --output_dir data/Label_Studio_Output/yolo_annotations

# 2. Split dataset
python split_dataset.py `
    --base_path data/Label_Studio_Output `
    --output_base data/suzuki_logo_detection
```

**Windows (CMD):**
```cmd
REM 1. Convert COCO to YOLO format
python convert_coco_to_yolo.py ^
    --coco_json project-5-at-2026-02-06-06-49-4f3e5bf6/result.json ^
    --images_dir project-5-at-2026-02-06-06-49-4f3e5bf6/images ^
    --output_dir data/Label_Studio_Output/yolo_annotations

REM 2. Split dataset
python split_dataset.py ^
    --base_path data/Label_Studio_Output ^
    --output_base data/suzuki_logo_detection
```

## Update Configuration

Edit `data_config.yaml`:

```yaml
nc: 2
names: ['Suzuki_Logo', 'Suzuki_Text']
train: data/suzuki_logo_detection/images/train
val: data/suzuki_logo_detection/images/val
test: data/suzuki_logo_detection/images/test
```

## Train Model (30 minutes - several hours)

**Linux/Mac:**
```bash
python train_model.py \
    --data data_config.yaml \
    --model yolov8n.pt \
    --batch 16 \
    --epochs 200 \
    --imgsz 640 \
    --name my_logo_detection
```

**Windows (PowerShell):**
```powershell
python train_model.py `
    --data data_config.yaml `
    --model yolov8n.pt `
    --batch 16 `
    --epochs 200 `
    --imgsz 640 `
    --name my_logo_detection
```

**Windows (CMD):**
```cmd
python train_model.py ^
    --data data_config.yaml ^
    --model yolov8n.pt ^
    --batch 16 ^
    --epochs 200 ^
    --imgsz 640 ^
    --name my_logo_detection
```

**Or use single line (all platforms):**
```bash
python train_model.py --data data_config.yaml --model yolov8n.pt --batch 16 --epochs 200 --imgsz 640 --name my_logo_detection
```

**Model Options:**
- `yolov8n.pt` - Nano (smallest, fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

Or use YOLOv8 Python API directly:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data='data_config.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    name='yolo_logo_detection'
)
```

## Run Inference

**Linux/Mac:**
```bash
python run_inference.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --save_txt
```

**Windows (PowerShell):**
```powershell
python run_inference.py `
    --source data/suzuki_logo_detection/images/test/ `
    --weights runs/train/yolo_logo_detection/weights/best.pt `
    --conf 0.3 `
    --save_txt
```

**Windows (CMD):**
```cmd
python run_inference.py ^
    --source data/suzuki_logo_detection/images/test/ ^
    --weights runs/train/yolo_logo_detection/weights/best.pt ^
    --conf 0.3 ^
    --save_txt
```

**Or use single line (all platforms):**
```bash
python run_inference.py --source data/suzuki_logo_detection/images/test/ --weights runs/train/yolo_logo_detection/weights/best.pt --conf 0.3 --save_txt
```

Or use YOLOv8 Python API:

```python
from ultralytics import YOLO

model = YOLO('runs/train/yolo_logo_detection/weights/best.pt')
results = model.predict('path/to/image.jpg', conf=0.3, save=True)
```

## Process Video

**Linux/Mac:**
```bash
python process_video.py \
    --youtube_url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --weights runs/train/yolo_logo_detection/weights/best.pt \
    --max_frames 200 \
    --conf 0.3
```

**Windows (PowerShell):**
```powershell
python process_video.py `
    --youtube_url "https://www.youtube.com/watch?v=VIDEO_ID" `
    --weights runs/train/yolo_logo_detection/weights/best.pt `
    --max_frames 200 `
    --conf 0.3
```

**Windows (CMD):**
```cmd
python process_video.py ^
    --youtube_url "https://www.youtube.com/watch?v=VIDEO_ID" ^
    --weights runs/train/yolo_logo_detection/weights/best.pt ^
    --max_frames 200 ^
    --conf 0.3
```

**Or use single line (all platforms):**
```bash
python process_video.py --youtube_url "https://www.youtube.com/watch?v=VIDEO_ID" --weights runs/train/yolo_logo_detection/weights/best.pt --max_frames 200 --conf 0.3
```

## Analyze Results

**Linux/Mac:**
```bash
python analyze_results.py \
    --output_folder runs/detect/yolo_frame_detection \
    --label_map "0:Suzuki_Logo,1:Suzuki_Text"
```

**Windows (PowerShell):**
```powershell
python analyze_results.py `
    --output_folder runs/detect/yolo_frame_detection `
    --label_map "0:Suzuki_Logo,1:Suzuki_Text"
```

**Windows (CMD):**
```cmd
python analyze_results.py ^
    --output_folder runs/detect/yolo_frame_detection ^
    --label_map "0:Suzuki_Logo,1:Suzuki_Text"
```

**Or use single line (all platforms):**
```bash
python analyze_results.py --output_folder runs/detect/yolo_frame_detection --label_map "0:Suzuki_Logo,1:Suzuki_Text"
```

## Common Issues

### Out of Memory
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--model yolov8n.pt`
- Reduce image size: `--imgsz 416` instead of 640

### No GPU Detected
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with correct CUDA version
- **Windows**: Make sure you have NVIDIA drivers installed and CUDA toolkit

### Import Errors
- Activate virtual environment
  - **Linux/Mac**: `source venv/bin/activate`
  - **Windows PowerShell**: `.\venv\Scripts\Activate.ps1`
  - **Windows CMD**: `venv\Scripts\activate.bat`
- Install: `pip install -r requirements.txt`
- Install YOLOv8: `pip install ultralytics`

### Windows-Specific Issues

**PowerShell Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path Issues:**
- Use forward slashes `/` in Python paths (works on Windows too)
- Or use raw strings: `r"C:\path\to\file"` in Python code

**CUDA Installation on Windows:**
- Download CUDA toolkit from NVIDIA website
- Install PyTorch with matching CUDA version:
  ```cmd
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## File Structure

```
custom_logo/
├── convert_coco_to_yolo.py    # Convert annotations
├── split_dataset.py            # Split train/val/test
├── visualize_annotations.py   # View annotations
├── train_model.py             # Train wrapper
├── run_inference.py           # Run detection
├── process_video.py           # Video processing
├── analyze_results.py         # Analyze results
├── quick_start.py             # Quick setup
├── data_config.yaml           # Dataset config
└── README.md                  # Full documentation
```

## Next Steps

1. **More Data**: Collect >=1500 images per class for production
2. **Better Model**: Try `yolov8m.pt` or `yolov8l.pt` for better accuracy
3. **Larger Image Size**: Try `--imgsz 1280` for better detection of small objects
4. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and epochs
5. **Data Augmentation**: YOLOv8 includes built-in augmentation - adjust in training arguments if needed

For detailed documentation, see [README.md](README.md)
