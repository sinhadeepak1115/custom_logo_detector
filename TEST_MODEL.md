# How to Test Your Trained Model

After training completes, you'll find the best model weights at:
`yolov5/runs/train/yolo_logo_detection/weights/best.pt`

## Method 1: Test on Test Dataset (Recommended)

### Using Python Script

```bash
python run_inference.py \
    --source data/images/test/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --method cli \
    --save_txt
```

This will:
- Run inference on all test images
- Save annotated images with bounding boxes
- Save text files with detection coordinates
- Results saved to: `yolov5/runs/detect/yolo_detection/`

### Using YOLOv5 Directly

```bash
python yolov5/detect.py \
    --source data/images/test/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --name suzuki_test_results \
    --save-txt
```

## Method 2: Test on Single Image

### Using Python API

Create a test script `test_single_image.py`:

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       'yolov5/runs/train/yolo_logo_detection/weights/best.pt')

# Set confidence threshold
model.conf = 0.3

# Test on a single image
image_path = 'data/images/test/your_image.jpg'
results = model(image_path)

# Display results
results.show()

# Get results as DataFrame
df = results.pandas().xyxy[0]
print("\nDetections:")
print(df)

# Show image with matplotlib
img = Image.open(image_path)
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title('Detection Results')
plt.show()
```

Run it:
```bash
python test_single_image.py
```

### Using Command Line

```bash
python run_inference.py \
    --source data/images/test/your_image.jpg \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3 \
    --method python
```

## Method 3: Test on Custom Images

Place your test images in a folder and run:

```bash
python run_inference.py \
    --source path/to/your/test/images/ \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --conf 0.3
```

## Method 4: Interactive Testing Script

Create `interactive_test.py`:

```python
import torch
import os
from pathlib import Path

# Load model
weights_path = 'yolov5/runs/train/yolo_logo_detection/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
model.conf = 0.3  # Confidence threshold

print("Model loaded successfully!")
print(f"Confidence threshold: {model.conf}")

# Test on multiple images
test_dir = Path('data/images/test')
test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

print(f"\nFound {len(test_images)} test images")

for img_path in test_images[:5]:  # Test first 5 images
    print(f"\nTesting: {img_path.name}")
    results = model(str(img_path))
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    if len(detections) > 0:
        print(f"  Found {len(detections)} detection(s):")
        for idx, det in detections.iterrows():
            print(f"    - Confidence: {det['confidence']:.2f}")
            print(f"      BBox: ({det['xmin']:.0f}, {det['ymin']:.0f}) to ({det['xmax']:.0f}, {det['ymax']:.0f})")
    else:
        print("  No detections found")

# Save all results
print("\nSaving results for all test images...")
results = model([str(img) for img in test_images])
results.save('test_results/')
print("Results saved to test_results/")
```

Run it:
```bash
python interactive_test.py
```

## Method 5: Evaluate Model Performance

Create `evaluate_model.py`:

```python
import torch
from pathlib import Path
import pandas as pd

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       'yolov5/runs/train/yolo_logo_detection/weights/best.pt')
model.conf = 0.3

# Get test images
test_dir = Path('data/images/test')
test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

print(f"Evaluating on {len(test_images)} test images...")

all_results = []
for img_path in test_images:
    results = model(str(img_path))
    detections = results.pandas().xyxy[0]
    
    all_results.append({
        'image': img_path.name,
        'num_detections': len(detections),
        'avg_confidence': detections['confidence'].mean() if len(detections) > 0 else 0
    })

# Create summary
df = pd.DataFrame(all_results)
print("\nEvaluation Summary:")
print(f"Total images: {len(df)}")
print(f"Images with detections: {(df['num_detections'] > 0).sum()}")
print(f"Average detections per image: {df['num_detections'].mean():.2f}")
print(f"Average confidence: {df[df['num_detections'] > 0]['avg_confidence'].mean():.2f}")

# Show images with most detections
print("\nTop 5 images by detection count:")
print(df.nlargest(5, 'num_detections')[['image', 'num_detections', 'avg_confidence']])
```

## Method 6: Test on Video

```bash
python process_video.py \
    --video_file path/to/your/video.mp4 \
    --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt \
    --max_frames 100 \
    --conf 0.3
```

## Understanding Results

### Output Files

After running inference, you'll find:

1. **Annotated Images**: `yolov5/runs/detect/yolo_detection/`
   - Images with bounding boxes drawn
   - Labels showing class and confidence

2. **Text Annotations**: `yolov5/runs/detect/yolo_detection/labels/`
   - YOLO format: `class_id x_center y_center width height confidence`
   - One `.txt` file per image

### Reading Detection Results

```python
import pandas as pd

# Load results from a single image
results = model('path/to/image.jpg')
df = results.pandas().xyxy[0]

# Columns: xmin, ymin, xmax, ymax, confidence, class, name
print(df)
```

### Adjusting Confidence Threshold

Lower threshold = more detections (but may include false positives)
Higher threshold = fewer detections (but more confident)

```python
model.conf = 0.25  # Lower threshold (more detections)
model.conf = 0.5   # Higher threshold (fewer but more confident)
```

## Quick Test Commands

```bash
# Test on single image
python yolov5/detect.py --source data/images/test/image.jpg --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt --conf 0.3

# Test on all test images
python yolov5/detect.py --source data/images/test/ --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt --conf 0.3 --save-txt

# Test with different confidence
python yolov5/detect.py --source data/images/test/ --weights yolov5/runs/train/yolo_logo_detection/weights/best.pt --conf 0.5
```

## Troubleshooting

**No detections found?**
- Lower confidence threshold: `--conf 0.2`
- Check if images are similar to training data
- Verify model weights path is correct

**Too many false positives?**
- Increase confidence threshold: `--conf 0.5`
- Model may need more training data
- Try training for more epochs

**Results not saving?**
- Check output directory exists
- Verify write permissions
- Check disk space
