"""
Run YOLO inference on images or video.
"""

import os
import pandas as pd
from ultralytics import YOLO


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
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure ultralytics is installed: pip install ultralytics")
        return None

    # Run inference
    results = model.predict(image_path, conf=conf_threshold, save=save_path is not None)

    # Print the results
    if isinstance(image_path, str) and os.path.isfile(image_path):
        # Single image
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) > 0:
            # Convert to DataFrame
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
            print("\nDetection Results:")
            print(df)
        else:
            print("\nNo detections found")
            df = pd.DataFrame()
        
        # Save results if requested
        if save_path:
            result.save(save_path)
            print(f"\nSaved results to: {save_path}")
        else:
            # Show results (save to default location)
            result.save()
            print(f"\nResults saved to: runs/detect/predict/")
            
        return df
    else:
        # Multiple images or directory
        output_dir = save_path or 'inference_results'
        print(f"\nProcessed {len(results)} image(s)")
        print(f"Results saved to: runs/detect/predict/")
        return results


def run_inference_cli(source, weights_path, conf_threshold=0.3, output_name="yolo_detection", save_txt=True):
    """
    Run inference using command line interface (YOLOv8 CLI).
    
    Args:
        source: Source path (image, directory, or video)
        weights_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        output_name: Name for output directory
        save_txt: Whether to save text annotations
    """
    try:
        model = YOLO(weights_path)
        
        # Run prediction
        results = model.predict(
            source=source,
            conf=conf_threshold,
            save=True,
            save_txt=save_txt,
            name=output_name
        )
        
        print(f"\nInference complete! Results saved to: runs/detect/{output_name}/")
        return True
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
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
                        help='Save text annotations')
    
    args = parser.parse_args()
    
    if args.method == 'python':
        run_inference_python(args.weights, args.source, args.conf)
    else:
        run_inference_cli(args.source, args.weights, args.conf, args.output_name, args.save_txt)
