#!/bin/bash
# Setup script for YOLOv5

echo "Setting up YOLOv5..."

# Clone YOLOv5 repository if it doesn't exist
if [ ! -d "yolov5" ]; then
    echo "Cloning YOLOv5 repository..."
    git clone https://github.com/ultralytics/yolov5
else
    echo "YOLOv5 directory already exists"
fi

# Install requirements
if [ -d "yolov5" ]; then
    echo "Installing YOLOv5 requirements..."
    pip install -r yolov5/requirements.txt
else
    echo "Error: YOLOv5 directory not found"
    exit 1
fi

# Copy data configuration file
if [ -f "data_config.yaml" ]; then
    echo "Copying data configuration file..."
    mkdir -p yolov5/data
    cp data_config.yaml yolov5/data/custom_logo.yaml
    echo "Data configuration copied to yolov5/data/custom_logo.yaml"
    echo ""
    echo "IMPORTANT: Update the train/val/test paths in yolov5/data/custom_logo.yaml"
    echo "  train: data/suzuki_logo_detection/images/train"
    echo "  val: data/suzuki_logo_detection/images/val"
    echo "  test: data/suzuki_logo_detection/images/test"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Prepare your dataset (convert COCO to YOLO format, split dataset)"
echo "2. Update yolov5/data/custom_logo.yaml with correct paths"
echo "3. Train your model: python yolov5/train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch 16 --epochs 200 --data custom_logo.yaml --weights yolov5s.pt --workers 1 --name yolo_logo_detection"
