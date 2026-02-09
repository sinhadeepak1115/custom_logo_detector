"""
Training wrapper script for YOLOv8.
This script provides an easier interface for training custom models.
"""

import os
import argparse
from ultralytics import YOLO


def train_yolov8(
    data_yaml,
    imgsz=640,
    model='yolov8n.pt',
    batch=16,
    epochs=200,
    workers=8,
    name='yolo_logo_detection',
    device='',
    project='runs/train'
):
    """
    Train YOLOv8 model.
    
    Args:
        data_yaml: Path to data configuration YAML file
        imgsz: Image size for training
        model: Model file (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        batch: Batch size
        epochs: Number of training epochs
        workers: Number of worker threads
        name: Experiment name
        device: Device to use ('' for auto, 'cpu', '0' for GPU 0, etc.)
        project: Project directory
    """
    # Check if data YAML exists
    if not os.path.exists(data_yaml):
        print(f"Error: Data YAML not found: {data_yaml}")
        return False

    # Initialize model
    print(f"Loading model: {model}")
    yolo_model = YOLO(model)

    # Prepare training arguments
    train_args = {
        'data': data_yaml,
        'imgsz': imgsz,
        'batch': batch,
        'epochs': epochs,
        'workers': workers,
        'name': name,
        'project': project,
    }

    if device:
        train_args['device'] = device

    print("=" * 60)
    print("Starting YOLOv8 Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data: {data_yaml}")
    print(f"  Model: {model}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Epochs: {epochs}")
    print(f"  Workers: {workers}")
    print(f"  Device: {device or 'auto'}")
    print(f"  Experiment name: {name}")
    print("=" * 60)
    print()

    try:
        # Train the model
        results = yolo_model.train(**train_args)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {project}/{name}/")
        print(f"Best weights: {project}/{name}/weights/best.pt")
        print(f"Last weights: {project}/{name}/weights/last.pt")
        return True
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--data', type=str, default='data_config.yaml',
                        help='Data configuration YAML file (default: data_config.yaml)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model file: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (default: yolov8n.pt)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads (default: 8)')
    parser.add_argument('--name', type=str, default='yolo_logo_detection',
                        help='Experiment name (default: yolo_logo_detection)')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (default: auto, use "cpu" or "0" for GPU 0)')
    
    args = parser.parse_args()
    
    train_yolov8(
        data_yaml=args.data,
        imgsz=args.imgsz,
        model=args.model,
        batch=args.batch,
        epochs=args.epochs,
        workers=args.workers,
        name=args.name,
        device=args.device
    )
