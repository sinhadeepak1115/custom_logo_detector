"""
Run YOLO inference on images or video.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys


def run_inference_python(weights_path, image_path, conf_threshold=0.3, save_path=None):
    """
    Run inference using Python API.
    
    Args:
        weights_path: Path to trained model weights
        image_path: Path to image file or directory
        conf_threshold: Confidence threshold for detections
        save_path: Optional path to save results
    """
    # Load the custom trained model weights
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
        model.conf = conf_threshold
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure YOLOv5 is installed: pip install yolov5")
        return None

    # Run inference
    results = model(image_path)

    # Print the results
    if isinstance(image_path, str) and os.path.isfile(image_path):
        # Single image
        df = results.pandas().xyxy[0]
        print("\nDetection Results:")
        print(df)
        
        # Display image with detections
        if save_path:
            results.save(save_path)
            print(f"\nSaved results to: {save_path}")
        else:
            # Show results
            results.show()
            
        return df
    else:
        # Multiple images or directory
        results.save(save_path or 'inference_results')
        print(f"\nSaved results to: {save_path or 'inference_results'}")
        return results


def run_inference_cli(source, weights_path, conf_threshold=0.3, output_name="yolo_detection", save_txt=True):
    """
    Run inference using command line interface.
    
    Args:
        source: Source path (image, directory, or video)
        weights_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        output_name: Name for output directory
        save_txt: Whether to save text annotations
    """
    yolov5_dir = "yolov5"
    if not os.path.exists(yolov5_dir):
        print(f"YOLOv5 directory not found: {yolov5_dir}")
        print("Please clone YOLOv5 repository first:")
        print("  git clone https://github.com/ultralytics/yolov5")
        return False

    detect_script = os.path.join(yolov5_dir, "detect.py")
    if not os.path.exists(detect_script):
        print(f"detect.py not found: {detect_script}")
        return False

    # Build command
    cmd = [
        sys.executable,
        detect_script,
        "--source", source,
        "--weights", weights_path,
        "--conf", str(conf_threshold),
        "--name", output_name,
    ]
    
    if save_txt:
        cmd.append("--save-txt")

    print(f"Running inference: {' '.join(cmd)}")
    try:
        import subprocess
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print(f"\nInference complete! Results saved to: yolov5/runs/detect/{output_name}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLO inference')
    parser.add_argument('--source', type=str, required=True,
                        help='Source path (image, directory, or video)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--output_name', type=str, default='yolo_detection',
                        help='Name for output directory')
    parser.add_argument('--method', type=str, choices=['python', 'cli'], default='cli',
                        help='Method to use: python API or CLI')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save text annotations (CLI only)')
    
    args = parser.parse_args()
    
    if args.method == 'python':
        run_inference_python(args.weights, args.source, args.conf)
    else:
        run_inference_cli(args.source, args.weights, args.conf, args.output_name, args.save_txt)
