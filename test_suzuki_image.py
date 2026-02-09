"""
Test if an image contains a Suzuki logo.
"""

import torch
from pathlib import Path
import sys

def test_suzuki_detection(image_path, weights_path='yolov5/runs/train/yolo_logo_detection12/weights/best.pt', conf=0.3):
    """
    Test if image contains Suzuki logo.
    
    Args:
        image_path: Path to image file
        weights_path: Path to trained model weights
        conf: Confidence threshold
    """
    # Check if files exist
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    if not Path(weights_path).exists():
        print(f"Error: Model weights not found: {weights_path}")
        return False
    
    print(f"Loading model from {weights_path}...")
    print(f"Testing image: {image_path}")
    print(f"Confidence threshold: {conf}")
    print("-" * 60)
    
    try:
        # Load the trained model
        model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
        model.conf = conf
        
        # Run inference
        results = model(image_path)
        
        # Get results as DataFrame
        df = results.pandas().xyxy[0]
        
        print(f"\nResults:")
        print(f"  Found {len(df)} detection(s)")
        
        if len(df) > 0:
            print("\n  Detections:")
            for idx, det in df.iterrows():
                print(f"    Detection {idx + 1}:")
                print(f"      Class: {det['name']}")
                print(f"      Confidence: {det['confidence']:.2%}")
                print(f"      Bounding Box: ({det['xmin']:.0f}, {det['ymin']:.0f}) to ({det['xmax']:.0f}, {det['ymax']:.0f})")
            
            print(f"\n✅ SUZUKI LOGO DETECTED!")
            print(f"   Confidence: {df['confidence'].max():.2%}")
            
            # Save annotated image
            output_path = 'suzuki_detection_result.jpg'
            results.save(output_path)
            print(f"\n  Annotated image saved to: {output_path}")
            
            return True
        else:
            print("\n❌ No Suzuki logo detected")
            print("   Try lowering confidence threshold (e.g., --conf 0.2)")
            return False
            
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test if image contains Suzuki logo')
    parser.add_argument('--image', type=str, default='image(2).png',
                        help='Path to test image')
    parser.add_argument('--weights', type=str,
                        default='yolov5/runs/train/yolo_logo_detection12/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    
    args = parser.parse_args()
    
    test_suzuki_detection(args.image, args.weights, args.conf)
