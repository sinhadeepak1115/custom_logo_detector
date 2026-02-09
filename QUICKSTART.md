# Quick Start Guide

## Prerequisites

1. Python 3.8+
2. Virtual environment (recommended)
3. CUDA-capable GPU (optional but recommended)

## Setup (5 minutes)

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup YOLOv5
./setup_yolov5.sh
# Or manually:
# git clone https://github.com/ultralytics/yolov5
# pip install -r yolov5/requirements.txt

# 4. Install PyTorch with GPU support (if you have CUDA)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Process Your Dataset (2 minutes)

If you have Label Studio output:

```bash
python quick_start.py --label_studio_dir project-5-at-2026-02-06-06-49-4f3e5bf6
```

Or manually:

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

## Update Configuration

Edit `yolov5/data/custom_logo.yaml`:

```yaml
nc: 2
names: ['Suzuki_Logo', 'Suzuki_Text']
train: data/suzuki_logo_detection/images/train
val: data/suzuki_logo_detection/images/val
test: data/suzuki_logo_detection/images/test
```

## Train Model (30 minutes - several hours)

```bash
python train_model.py \
    --data custom_logo.yaml \
    --batch 16 \
    --epochs 200 \
    --name my_logo_detection
```

Or use YOLOv5 directly:

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

## Run Inference

```bash
python run_inference.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3
```

## Process Video

```bash
python process_video.py \
    --youtube_url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --max_frames 200
```

## Analyze Results

```bash
python analyze_results.py \
    --output_folder yolov5/runs/detect/yolo_frame_detection \
    --label_map "0:Suzuki_Logo,1:Suzuki_Text"
```

## Common Issues

### Out of Memory
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--cfg yolov5n.yaml`

### No GPU Detected
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with correct CUDA version

### Import Errors
- Activate virtual environment
- Install: `pip install -r requirements.txt`
- Install YOLOv5: `pip install -r yolov5/requirements.txt`

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
2. **Better Model**: Try `yolov5m.yaml` or `yolov5l.yaml`
3. **More Augmentation**: Use `hyp.scratch-med.yaml` or `hyp.scratch-high.yaml`
4. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes

For detailed documentation, see [README.md](README.md)
