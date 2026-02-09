"""
Batch test multiple images for Suzuki logo detection.
Runs test_suzuki_image.py on all images in a directory.
"""

import subprocess
import sys
from pathlib import Path
import argparse
from collections import defaultdict

def find_images(directory):
    """Find all image files in directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    
    images = []
    for ext in image_extensions:
        images.extend(directory.glob(f'*{ext}'))
    
    return sorted(images)


def test_single_image(image_path, weights_path='runs/train/yolo_logo_detection12/weights/best.pt', conf=0.3):
    """Test a single image using test_suzuki_image.py."""
    try:
        result = subprocess.run(
            [sys.executable, 'test_suzuki_image.py', 
             '--image', str(image_path),
             '--weights', weights_path,
             '--conf', str(conf)],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per image
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout after 60 seconds"
    except Exception as e:
        return False, "", str(e)


def batch_test_images(directory, weights_path='runs/train/yolo_logo_detection12/weights/best.pt', conf=0.3):
    """
    Test all images in a directory.
    
    Args:
        directory: Directory containing images
        weights_path: Path to model weights
        conf: Confidence threshold
    """
    images = find_images(directory)
    
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
    
    results = {
        'total': len(images),
        'detected': 0,
        'not_detected': 0,
        'errors': 0,
        'details': []
    }
    
    for idx, image_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Testing: {image_path.name}")
        
        success, stdout, stderr = test_single_image(image_path, weights_path, conf)
        
        # Check output for detection
        detected = False
        if success:
            if "SUZUKI LOGO DETECTED" in stdout or "detection" in stdout.lower():
                # Try to parse confidence from output
                detected = True
                results['detected'] += 1
                print(f"  ✅ Suzuki logo detected")
            else:
                results['not_detected'] += 1
                print(f"  ❌ No Suzuki logo detected")
        else:
            results['errors'] += 1
            print(f"  ⚠️  Error processing image")
            if stderr:
                print(f"     Error: {stderr[:100]}")
        
        results['details'].append({
            'image': image_path.name,
            'detected': detected,
            'success': success
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
        for detail in results['details']:
            if detail['detected']:
                print(f"  ✅ {detail['image']}")
    
    # List images without detections
    if results['not_detected'] > 0:
        print("\nImages without Suzuki logo:")
        for detail in results['details']:
            if not detail['detected'] and detail['success']:
                print(f"  ❌ {detail['image']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch test multiple images for Suzuki logo detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all images in current directory
  python test_batch_images.py --dir .

  # Test images in specific directory
  python test_batch_images.py --dir /path/to/images

  # Test with different confidence threshold
  python test_batch_images.py --dir images/ --conf 0.5
        """
    )
    
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing images to test (default: current directory)')
    parser.add_argument('--weights', type=str,
                        default='runs/train/yolo_logo_detection12/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    batch_test_images(args.dir, args.weights, args.conf)
