#!/bin/bash
# Setup script for YOLOv8

echo "Setting up YOLOv8..."

# Install ultralytics (YOLOv8)
echo "Installing ultralytics (YOLOv8)..."
pip install ultralytics

# Copy data configuration file
if [ -f "data_config.yaml" ]; then
    echo "Data configuration file found: data_config.yaml"
    echo ""
    echo "IMPORTANT: Update the train/val/test paths in data_config.yaml"
    echo "  train: data/suzuki_logo_detection/images/train"
    echo "  val: data/suzuki_logo_detection/images/val"
    echo "  test: data/suzuki_logo_detection/images/test"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Prepare your dataset (convert COCO to YOLO format, split dataset)"
echo "2. Update data_config.yaml with correct paths"
echo "3. Train your model: python train_model.py --data data_config.yaml --epochs 200 --imgsz 640 --batch 16 --model yolov8n.pt"
