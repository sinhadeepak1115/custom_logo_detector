"""
Training wrapper script for YOLOv5.
This script provides an easier interface for training custom models.
"""

import os
import sys
import subprocess
import argparse


def train_yolov5(
    data_yaml,
    img_size=640,
    cfg='yolov5s.yaml',
    hyp='hyp.scratch-low.yaml',
    batch=16,
    epochs=200,
    weights='yolov5s.pt',
    workers=5,
    name='yolo_logo_detection',
    device='',
    project='yolov5/runs/train'
):
    """
    Train YOLOv5 model.
    
    Args:
        data_yaml: Path to data configuration YAML file
        img_size: Image size for training
        cfg: Model configuration file
        hyp: Hyperparameter file
        batch: Batch size
        epochs: Number of training epochs
        weights: Initial weights file
        workers: Number of worker threads
        name: Experiment name
        device: Device to use ('' for auto, 'cpu', '0' for GPU 0, etc.)
        project: Project directory
    """
    yolov5_dir = "yolov5"
    if not os.path.exists(yolov5_dir):
        print(f"Error: YOLOv5 directory not found: {yolov5_dir}")
        print("Please run setup first: ./setup_yolov5.sh")
        return False

    train_script = os.path.join(yolov5_dir, "train.py")
    if not os.path.exists(train_script):
        print(f"Error: train.py not found: {train_script}")
        return False

    # Check if data YAML exists
    if not os.path.exists(data_yaml):
        # Try relative to yolov5/data
        data_yaml_alt = os.path.join(yolov5_dir, "data", data_yaml)
        if os.path.exists(data_yaml_alt):
            data_yaml = data_yaml_alt
        else:
            print(f"Error: Data YAML not found: {data_yaml}")
            print(f"Also checked: {data_yaml_alt}")
            return False

    # Build command
    cmd = [
        sys.executable,
        train_script,
        "--img", str(img_size),
        "--cfg", cfg,
        "--hyp", hyp,
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", data_yaml,
        "--weights", weights,
        "--workers", str(workers),
        "--name", name,
        "--project", project,
    ]

    if device:
        cmd.extend(["--device", device])

    print("=" * 60)
    print("Starting YOLOv5 Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data: {data_yaml}")
    print(f"  Model: {cfg}")
    print(f"  Hyperparameters: {hyp}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch}")
    print(f"  Epochs: {epochs}")
    print(f"  Workers: {workers}")
    print(f"  Device: {device or 'auto'}")
    print(f"  Experiment name: {name}")
    print(f"\nCommand: {' '.join(cmd)}")
    print("=" * 60)
    print()

    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {project}/{name}/")
        print(f"Best weights: {project}/{name}/weights/best.pt")
        print(f"Last weights: {project}/{name}/weights/last.pt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv5 model')
    parser.add_argument('--data', type=str, default='custom_logo.yaml',
                        help='Data configuration YAML file')
    parser.add_argument('--img', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml',
                        help='Model configuration file (default: yolov5s.yaml)')
    parser.add_argument('--hyp', type=str, default='hyp.scratch-low.yaml',
                        help='Hyperparameter file (default: hyp.scratch-low.yaml)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--weights', type=str, default='yolov5s.pt',
                        help='Initial weights file (default: yolov5s.pt)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker threads (default: 1)')
    parser.add_argument('--name', type=str, default='yolo_logo_detection',
                        help='Experiment name (default: yolo_logo_detection)')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (default: auto, use "cpu" or "0" for GPU 0)')
    
    args = parser.parse_args()
    
    train_yolov5(
        data_yaml=args.data,
        img_size=args.img,
        cfg=args.cfg,
        hyp=args.hyp,
        batch=args.batch,
        epochs=args.epochs,
        weights=args.weights,
        workers=args.workers,
        name=args.name,
        device=args.device
    )
