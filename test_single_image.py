"""
Test the trained model on a single image.
"""

import pandas as pd
from ultralytics import YOLO
import sys
import os

def test_image(image_path, weights_path='runs/train/yolo_logo_detection/weights/best.pt', conf=0.3):
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
    model = YOLO(weights_path)
    
    print(f"Confidence threshold: {conf}")
    print(f"Testing on: {image_path}")
    
    # Run inference
    results = model.predict(image_path, conf=conf)
    result = results[0]
    boxes = result.boxes
    
    # Convert to DataFrame
    if len(boxes) > 0:
        detections = []
        for box in boxes:
            detections.append({
                'xmin': float(box.xyxy[0][0]),
                'ymin': float(box.xyxy[0][1]),
                'xmax': float(box.xyxy[0][2]),
                'ymax': float(box.xyxy[0][3]),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
                'name': result.names[int(box.cls[0])]
            })
        df = pd.DataFrame(detections)
    else:
        df = pd.DataFrame()
    
    print(f"\nFound {len(df)} detection(s):")
    if len(df) > 0:
        for idx, det in df.iterrows():
            print(f"  Detection {idx + 1}:")
            print(f"    Class: {det['name']}")
            print(f"    Confidence: {det['confidence']:.2%}")
            print(f"    Bounding Box: ({det['xmin']:.0f}, {det['ymin']:.0f}) to ({det['xmax']:.0f}, {det['ymax']:.0f})")
    else:
        print("  No detections found (try lowering confidence threshold)")
    
    # Save and display results
    result.save()
    print(f"\nResults saved to: runs/detect/predict/")
    
    return results, df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model on a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--weights', type=str, 
                        default='runs/train/yolo_logo_detection/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    test_image(args.image, args.weights, args.conf)
