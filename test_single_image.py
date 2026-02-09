"""
Test the trained model on a single image.
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

def test_image(image_path, weights_path='yolov5/runs/train/yolo_logo_detection/weights/best.pt', conf=0.3):
    """
    Test trained model on a single image.
    
    Args:
        image_path: Path to test image
        weights_path: Path to trained model weights
        conf: Confidence threshold
    """
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        print("Please train the model first or check the path.")
        return None
    
    # Load the trained model
    print(f"Loading model from {weights_path}...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
    model.conf = conf
    
    print(f"Confidence threshold: {conf}")
    print(f"Testing on: {image_path}")
    
    # Run inference
    results = model(image_path)
    
    # Get results as DataFrame
    df = results.pandas().xyxy[0]
    
    print(f"\nFound {len(df)} detection(s):")
    if len(df) > 0:
        for idx, det in df.iterrows():
            print(f"  Detection {idx + 1}:")
            print(f"    Class: {det['name']}")
            print(f"    Confidence: {det['confidence']:.2%}")
            print(f"    Bounding Box: ({det['xmin']:.0f}, {det['ymin']:.0f}) to ({det['xmax']:.0f}, {det['ymax']:.0f})")
    else:
        print("  No detections found (try lowering confidence threshold)")
    
    # Display results
    results.show()
    
    return results, df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model on a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--weights', type=str, 
                        default='yolov5/runs/train/yolo_logo_detection/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    test_image(args.image, args.weights, args.conf)
