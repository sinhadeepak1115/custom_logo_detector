"""
Simpler batch testing script that directly uses YOLOv8.
Faster than calling test_suzuki_image.py for each image.
"""

import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import argparse
from collections import defaultdict

def batch_test_images_direct(directory, weights_path='runs/train/yolo_logo_detection12/weights/best.pt', conf=0.3):
    """
    Test all images in a directory directly using YOLOv8.
    
    Args:
        directory: Directory containing images
        weights_path: Path to model weights
        conf: Confidence threshold
    """
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    images = []
    for ext in image_extensions:
        images.extend(directory.glob(f'*{ext}'))
    
    images = sorted(images)
    
    if len(images) == 0:
        print(f"No images found in {directory}")
        return
    
    print("=" * 70)
    print(f"Batch Testing Suzuki Logo Detection")
    print("=" * 70)
    print(f"Directory: {directory}")
    print(f"Found {len(images)} image(s)")
    print(f"Model: {weights_path}")
    print(f"Confidence threshold: {conf}")
    print("=" * 70)
    print()
    
    # Load model once
    print("Loading model...")
    try:
        model = YOLO(weights_path)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    results = {
        'total': len(images),
        'detected': 0,
        'not_detected': 0,
        'errors': 0,
        'details': []
    }
    
    # Process each image
    for idx, image_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Testing: {image_path.name}")
        
        try:
            # Run inference
            results_yolo = model.predict(str(image_path), conf=conf)
            result = results_yolo[0]
            boxes = result.boxes
            
            # Convert to DataFrame
            if len(boxes) > 0:
                detections = []
                for box in boxes:
                    detections.append({
                        'confidence': float(box.conf[0]),
                        'class': int(box.cls[0]),
                        'name': result.names[int(box.cls[0])]
                    })
                df = pd.DataFrame(detections)
            else:
                df = pd.DataFrame()
            
            if len(df) > 0:
                results['detected'] += 1
                max_conf = df['confidence'].max()
                print(f"  ✅ Suzuki logo detected! (confidence: {max_conf:.2%})")
                
                # Show all detections
                for i, det in df.iterrows():
                    print(f"     - Detection {i+1}: {det['confidence']:.2%} confidence ({det['name']})")
                
                results['details'].append({
                    'image': image_path.name,
                    'detected': True,
                    'confidence': max_conf,
                    'num_detections': len(df)
                })
            else:
                results['not_detected'] += 1
                print(f"  ❌ No Suzuki logo detected")
                results['details'].append({
                    'image': image_path.name,
                    'detected': False,
                    'confidence': 0,
                    'num_detections': 0
                })
                
        except Exception as e:
            results['errors'] += 1
            print(f"  ⚠️  Error: {e}")
            results['details'].append({
                'image': image_path.name,
                'detected': False,
                'error': str(e)
            })
        
        print()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images tested: {results['total']}")
    print(f"✅ Suzuki logo detected: {results['detected']} ({results['detected']/results['total']*100:.1f}%)")
    print(f"❌ No detection: {results['not_detected']} ({results['not_detected']/results['total']*100:.1f}%)")
    print(f"⚠️  Errors: {results['errors']} ({results['errors']/results['total']*100:.1f}%)")
    print("=" * 70)
    
    # List images with detections
    if results['detected'] > 0:
        print("\nImages with Suzuki logo detected:")
        detected_images = [d for d in results['details'] if d.get('detected', False)]
        detected_images.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for detail in detected_images:
            conf = detail.get('confidence', 0)
            num = detail.get('num_detections', 0)
            print(f"  ✅ {detail['image']} - {conf:.2%} confidence ({num} detection(s))")
    
    # List images without detections
    if results['not_detected'] > 0:
        print("\nImages without Suzuki logo:")
        for detail in results['details']:
            if not detail.get('detected', False) and 'error' not in detail:
                print(f"  ❌ {detail['image']}")
    
    # Save annotated images
    print("\nSaving annotated images...")
    try:
        all_results = model.predict([str(img) for img in images], conf=conf, save=True, name='batch_test_results')
        print(f"Annotated images saved to: runs/detect/batch_test_results/")
    except Exception as e:
        print(f"Warning: Could not save annotated images: {e}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch test multiple images for Suzuki logo detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all images in current directory
  python test_batch_simple.py --dir .

  # Test images in specific directory
  python test_batch_simple.py --dir images/

  # Test with different confidence threshold
  python test_batch_simple.py --dir images/ --conf 0.5

  # Test specific image file
  python test_batch_simple.py --dir . --image test1.jpg
        """
    )
    
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing images to test (default: current directory)')
    parser.add_argument('--image', type=str, default=None,
                        help='Test a single specific image file')
    parser.add_argument('--weights', type=str,
                        default='runs/train/yolo_logo_detection12/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    if args.image:
        # Test single image
        image_path = Path(args.image)
        if image_path.exists():
            batch_test_images_direct(image_path.parent, args.weights, args.conf)
        else:
            print(f"Error: Image not found: {args.image}")
    else:
        # Test all images in directory
        batch_test_images_direct(args.dir, args.weights, args.conf)
