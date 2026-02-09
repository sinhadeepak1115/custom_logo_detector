"""
Example workflow demonstrating the complete pipeline.
This script shows how to use all the components together.
"""

import os
import sys

# Example 1: Process existing Label Studio output
def example_process_label_studio():
    """Example: Process Label Studio output"""
    print("Example 1: Processing Label Studio Output")
    print("-" * 60)
    
    from quick_start import quick_start
    
    # Process the existing Label Studio output
    label_studio_dir = "project-5-at-2026-02-06-06-49-4f3e5bf6"
    quick_start(label_studio_dir, output_base='data')


# Example 2: Visualize annotations
def example_visualize():
    """Example: Visualize annotations"""
    print("\nExample 2: Visualizing Annotations")
    print("-" * 60)
    
    from visualize_annotations import visualize_annotation
    
    coco_json = "project-5-at-2026-02-06-06-49-4f3e5bf6/result.json"
    images_dir = "project-5-at-2026-02-06-06-49-4f3e5bf6/images"
    
    if os.path.exists(coco_json):
        visualize_annotation(coco_json, images_dir, img_id=0, save_path="example_annotation.png")


# Example 3: Run inference
def example_inference():
    """Example: Run inference on test images"""
    print("\nExample 3: Running Inference")
    print("-" * 60)
    
    weights_path = "yolov5/runs/train/yolo_logo_detection/weights/best.pt"
    test_images = "data/suzuki_logo_detection/images/test"
    
    if os.path.exists(weights_path) and os.path.exists(test_images):
        from run_inference import run_inference_cli
        run_inference_cli(
            source=test_images,
            weights_path=weights_path,
            conf_threshold=0.3,
            output_name="example_inference",
            save_txt=True
        )
    else:
        print(f"Weights not found: {weights_path}")
        print("Train a model first using train_model.py")


# Example 4: Process video
def example_video_processing():
    """Example: Process video and analyze results"""
    print("\nExample 4: Video Processing")
    print("-" * 60)
    
    weights_path = "yolov5/runs/train/yolo_logo_detection/weights/best.pt"
    
    if os.path.exists(weights_path):
        from process_video import download_youtube_video, extract_frames, run_inference
        from analyze_results import analyze_frame_coverage, plot_coverage
        
        # Download video (optional)
        youtube_url = "https://www.youtube.com/watch?v=zc3JYvvmXxw"
        save_path = "data/videos"
        download_name = "formula1_test"
        
        # Extract frames
        video_file = os.path.join(save_path, f"{download_name}.mp4")
        frames_dir = os.path.join(save_path, "frames")
        
        if os.path.exists(video_file):
            extract_frames(video_file, frames_dir, max_frames=200)
            
            # Run inference
            run_inference(frames_dir, weights_path, conf_threshold=0.3, output_name="video_detection")
            
            # Analyze results
            output_folder = "yolov5/runs/detect/video_detection"
            if os.path.exists(output_folder):
                label_map = {0: 'Suzuki_Logo', 1: 'Suzuki_Text'}
                frame_coverage = analyze_frame_coverage(output_folder, label_map)
                plot_coverage(frame_coverage, save_path="video_coverage.png")
    else:
        print(f"Weights not found: {weights_path}")
        print("Train a model first using train_model.py")


if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv5 Custom Logo Detection - Example Workflow")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "process":
            example_process_label_studio()
        elif example_name == "visualize":
            example_visualize()
        elif example_name == "inference":
            example_inference()
        elif example_name == "video":
            example_video_processing()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: process, visualize, inference, video")
    else:
        print("\nAvailable examples:")
        print("  1. python example_workflow.py process    - Process Label Studio output")
        print("  2. python example_workflow.py visualize  - Visualize annotations")
        print("  3. python example_workflow.py inference  - Run inference")
        print("  4. python example_workflow.py video      - Process video")
        print("\nOr run individual scripts:")
        print("  - python quick_start.py")
        print("  - python train_model.py")
        print("  - python run_inference.py")
