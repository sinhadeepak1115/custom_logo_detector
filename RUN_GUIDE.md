# How to Run the Code - Step by Step Guide

## Prerequisites Check

Before starting, make sure you have:
- Python 3.8 or higher
- Git installed
- (Optional) CUDA-capable GPU for faster training

## Step 1: Setup Environment (First Time Only)

### 1.1 Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/deepak/Projects/dave/custom_logo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 1.2 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install -r requirements.txt
```

### 1.3 Setup YOLOv5

```bash
# Run setup script
./setup_yolov5.sh

# Or manually:
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

### 1.4 Install PyTorch with GPU Support (Optional but Recommended)

```bash
# First, uninstall CPU-only versions
pip uninstall torch torchvision torchaudio

# Install GPU version (adjust CUDA version as needed)
# For CUDA 11.6:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# For CUDA 11.8:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Step 2: Prepare Your Dataset

### 2.1 Quick Start (If you have Label Studio output)

```bash
python quick_start.py --label_studio_dir project-5-at-2026-02-06-06-49-4f3e5bf6
```

This will:
- Convert COCO annotations to YOLO format
- Split dataset into train/val/test sets
- Create the required directory structure

### 2.2 Manual Process (Alternative)

If you prefer to run steps individually:

```bash
# Step 2a: Convert COCO to YOLO format
python convert_coco_to_yolo.py \
    --coco_json project-5-at-2026-02-06-06-49-4f3e5bf6/result.json \
    --images_dir project-5-at-2026-02-06-06-49-4f3e5bf6/images \
    --output_dir data/Label_Studio_Output/yolo_annotations

# Step 2b: Split dataset
python split_dataset.py \
    --base_path data/Label_Studio_Output \
    --output_base data/suzuki_logo_detection \
    --test_size 0.2 \
    --val_size 0.5
```

### 2.3 Update Data Configuration

Edit `yolov5/data/custom_logo.yaml` (or create it if it doesn't exist):

```bash
# Copy the template
cp data_config.yaml yolov5/data/custom_logo.yaml
```

Then edit `yolov5/data/custom_logo.yaml`:

```yaml
nc: 2
names: ['Suzuki_Logo', 'Suzuki_Text']
train: data/suzuki_logo_detection/images/train
val: data/suzuki_logo_detection/images/val
test: data/suzuki_logo_detection/images/test
```

**Important:** Make sure the paths are correct relative to where you'll run the training command.

## Step 3: Train the Model

### 3.1 Using the Training Wrapper (Easier)

```bash
python train_model.py \
    --data custom_logo.yaml \
    --batch 16 \
    --epochs 200 \
    --name suzuki_logo_detection
```

### 3.2 Using YOLOv5 Directly

```bash
cd yolov5

python train.py \
    --img 640 \
    --cfg yolov5s.yaml \
    --hyp hyp.scratch-low.yaml \
    --batch 16 \
    --epochs 200 \
    --data data/custom_logo.yaml \
    --weights yolov5s.pt \
    --workers 1 \
    --name suzuki_logo_detection

cd ..
```

**Training Notes:**
- Training time depends on your hardware (CPU: hours, GPU: 30min-2hrs)
- Monitor training progress in `yolov5/runs/train/suzuki_logo_detection/`
- Best weights saved to: `yolov5/runs/train/suzuki_logo_detection/weights/best.pt`

### 3.3 Adjust Training Parameters (If Needed)

**For smaller GPU memory:**
```bash
python train_model.py --data custom_logo.yaml --batch 8 --epochs 200
```

**For CPU training:**
```bash
python train_model.py --data custom_logo.yaml --batch 4 --epochs 50 --device cpu
```

**For better accuracy (requires more GPU memory):**
```bash
python train_model.py --data custom_logo.yaml --cfg yolov5m.yaml --batch 8 --epochs 300
```

## Step 4: Run Inference

### 4.1 Test on Single Image

```bash
python run_inference.py \
    --source path/to/your/image.jpg \
    --weights yolov5/runs/train/suzuki_logo_detection/weights/best.pt \
    --conf 0.3 \
    --method python
```

### 4.2 Test on Test Dataset

```bash
python run_inference.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights yolov5/runs/train/suzuki_logo_detection/weights/best.pt \
    --conf 0.3 \
    --method cli \
    --save_txt
```

Results will be saved to: `yolov5/runs/detect/yolo_detection/`

### 4.3 Using Python API Directly

```python
import torch

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       'yolov5/runs/train/suzuki_logo_detection/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results.show()

# Get results as DataFrame
print(results.pandas().xyxy[0])
```

## Step 5: Process Video (Optional)

### 5.1 Download and Process YouTube Video

```bash
python process_video.py \
    --youtube_url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --weights yolov5/runs/train/suzuki_logo_detection/weights/best.pt \
    --max_frames 200 \
    --conf 0.3
```

### 5.2 Process Local Video File

```bash
python process_video.py \
    --video_file data/videos/my_video.mp4 \
    --weights yolov5/runs/train/suzuki_logo_detection/weights/best.pt \
    --max_frames 200 \
    --skip_download
```

### 5.3 Analyze Video Results

```bash
python analyze_results.py \
    --output_folder yolov5/runs/detect/yolo_frame_detection \
    --label_map "0:Suzuki_Logo,1:Suzuki_Text" \
    --save_path suzuki_coverage.png
```

## Step 6: Visualize Annotations

### 6.1 View Annotated Images

```bash
python visualize_annotations.py \
    --coco_json project-5-at-2026-02-06-06-49-4f3e5bf6/result.json \
    --images_dir project-5-at-2026-02-06-06-49-4f3e5bf6/images \
    --img_id 0 \
    --save_path example_annotation.png
```

## Complete Workflow Example

Here's a complete example from start to finish:

```bash
# 1. Setup (one time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./setup_yolov5.sh

# 2. Prepare dataset
python quick_start.py --label_studio_dir project-5-at-2026-02-06-06-49-4f3e5bf6

# 3. Update yolov5/data/custom_logo.yaml with correct paths

# 4. Train model
python train_model.py --data custom_logo.yaml --batch 16 --epochs 200

# 5. Run inference
python run_inference.py \
    --source data/suzuki_logo_detection/images/test/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python train_model.py --data custom_logo.yaml --batch 4
```

### Issue: "Module not found"
**Solution:** Activate virtual environment and install dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "YOLOv5 directory not found"
**Solution:** Run setup script
```bash
./setup_yolov5.sh
```

### Issue: "Data YAML not found"
**Solution:** Make sure `yolov5/data/custom_logo.yaml` exists with correct paths

### Issue: "No images found"
**Solution:** Check that image paths in YAML are correct relative to where you run the command

## Quick Reference

| Task | Command |
|------|---------|
| Setup environment | `./setup_yolov5.sh` |
| Prepare dataset | `python quick_start.py --label_studio_dir project-5-at-2026-02-06-06-49-4f3e5bf6` |
| Train model | `python train_model.py --data custom_logo.yaml` |
| Run inference | `python run_inference.py --source <path> --weights <weights>` |
| Process video | `python process_video.py --youtube_url <url> --weights <weights>` |
| Analyze results | `python analyze_results.py --output_folder <folder>` |

## Need Help?

- Check `README.md` for detailed documentation
- Check `QUICKSTART.md` for quick reference
- Review error messages - they usually indicate what's missing
