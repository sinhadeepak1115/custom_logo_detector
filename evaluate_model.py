"""
Evaluate trained model performance on test dataset.
"""

import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import os

def evaluate_model(weights_path='runs/train/yolo_logo_detection/weights/best.pt',
                   test_dir='data/images/test',
                   conf=0.3):
    """
    Evaluate model on test dataset.
    
    Args:
        weights_path: Path to trained model weights
        test_dir: Directory containing test images
        conf: Confidence threshold
    """
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        print("Please train the model first or check the path.")
        return None
    
    # Load model
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    # Get test images
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return None
    
    test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.jpeg')) + \
                  list(test_path.glob('*.png')) + list(test_path.glob('*.JPG'))
    
    if len(test_images) == 0:
        print(f"No images found in {test_dir}")
        return None
    
    print(f"\nEvaluating on {len(test_images)} test images...")
    print(f"Confidence threshold: {conf}")
    
    all_results = []
    images_with_detections = 0
    
    for img_path in test_images:
        results = model.predict(str(img_path), conf=conf)
        result = results[0]
        boxes = result.boxes
        
        # Convert to DataFrame
        if len(boxes) > 0:
            detections = []
            for box in boxes:
                detections.append({
                    'confidence': float(box.conf[0])
                })
            df = pd.DataFrame(detections)
        else:
            df = pd.DataFrame()
        
        num_detections = len(df)
        avg_confidence = df['confidence'].mean() if num_detections > 0 else 0
        
        if num_detections > 0:
            images_with_detections += 1
        
        all_results.append({
            'image': img_path.name,
            'num_detections': num_detections,
            'avg_confidence': avg_confidence,
            'max_confidence': df['confidence'].max() if num_detections > 0 else 0
        })
    
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total test images: {len(df)}")
    print(f"Images with detections: {images_with_detections} ({images_with_detections/len(df)*100:.1f}%)")
    print(f"Images without detections: {len(df) - images_with_detections} ({(len(df) - images_with_detections)/len(df)*100:.1f}%)")
    
    if images_with_detections > 0:
        print(f"\nDetection Statistics:")
        print(f"  Average detections per image: {df['num_detections'].mean():.2f}")
        print(f"  Max detections in single image: {df['num_detections'].max()}")
        print(f"  Average confidence: {df[df['num_detections'] > 0]['avg_confidence'].mean():.2%}")
        print(f"  Max confidence: {df['max_confidence'].max():.2%}")
        print(f"  Min confidence: {df[df['num_detections'] > 0]['max_confidence'].min():.2%}")
    
    # Show top images
    if images_with_detections > 0:
        print("\n" + "-" * 60)
        print("Top 5 images by detection count:")
        print("-" * 60)
        top5 = df.nlargest(5, 'num_detections')[['image', 'num_detections', 'avg_confidence']]
        for idx, row in top5.iterrows():
            print(f"  {row['image']}: {int(row['num_detections'])} detections, avg confidence: {row['avg_confidence']:.2%}")
    
    # Show images with highest confidence
    if images_with_detections > 0:
        print("\n" + "-" * 60)
        print("Top 5 images by confidence:")
        print("-" * 60)
        top_conf = df[df['num_detections'] > 0].nlargest(5, 'max_confidence')[['image', 'num_detections', 'max_confidence']]
        for idx, row in top_conf.iterrows():
            print(f"  {row['image']}: {row['max_confidence']:.2%} confidence, {int(row['num_detections'])} detection(s)")
    
    # Save results to CSV
    output_file = 'evaluation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model on test dataset')
    parser.add_argument('--weights', type=str,
                        default='runs/train/yolo_logo_detection/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--test_dir', type=str, default='data/images/test',
                        help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    evaluate_model(args.weights, args.test_dir, args.conf)
