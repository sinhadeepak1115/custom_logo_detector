"""
Process video: download from YouTube, extract frames, and run inference.
"""

import cv2
import os
import subprocess
import sys


def download_youtube_video(youtube_url, save_path, download_name):
    """
    Download a video from YouTube.
    
    Args:
        youtube_url: YouTube video URL
        save_path: Directory to save video
        download_name: Name for downloaded file
    """
    # Create output path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Try using yt-dlp (newer) or youtube-dl (older)
    try:
        import yt_dlp as youtube_dl
    except ImportError:
        try:
            import youtube_dl
        except ImportError:
            print("Error: yt-dlp or youtube-dl not installed.")
            print("Install with: pip install yt-dlp")
            return False

    # Execute the download
    ydl_opts = {
        'outtmpl': f'{save_path}/{download_name}.%(ext)s',
    }
    
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"Downloaded video to: {save_path}/{download_name}")
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def extract_frames(source_file, output_path, max_frames=None):
    """
    Extract frames from video file.
    
    Args:
        source_file: Path to video file
        output_path: Directory to save frames
        max_frames: Maximum number of frames to extract (None for all)
    """
    if not os.path.exists(source_file):
        print(f"Video file not found: {source_file}")
        return False

    # Define the output path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Execute slice to frames
    vidcap = cv2.VideoCapture(source_file)
    success, image = vidcap.read()
    count = 0
    
    while success:
        if max_frames and count >= max_frames:
            break
            
        cv2.imwrite(os.path.join(output_path, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        
        if count % 50 == 0:
            print(f'Extracted {count} frames...')
        count += 1

    vidcap.release()
    print(f"Extracted {count} frames to: {output_path}")
    return True


def run_inference(frames_dir, weights_path, conf_threshold=0.3, output_name="yolo_frame_detection"):
    """
    Run YOLO inference on extracted frames.
    
    Args:
        frames_dir: Directory containing frames
        weights_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        output_name: Name for output directory
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
        "--source", frames_dir,
        "--weights", weights_path,
        "--conf", str(conf_threshold),
        "--name", output_name,
        "--save-txt"
    ]

    print(f"Running inference: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print(f"Inference complete! Results saved to: yolov5/runs/detect/{output_name}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video: download, extract frames, run inference')
    parser.add_argument('--youtube_url', type=str, default=None,
                        help='YouTube video URL to download')
    parser.add_argument('--video_file', type=str, default=None,
                        help='Path to local video file')
    parser.add_argument('--save_path', type=str, default='data/videos',
                        help='Directory to save video and frames')
    parser.add_argument('--download_name', type=str, default='formula1_test',
                        help='Name for downloaded video file')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum number of frames to extract')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--output_name', type=str, default='yolo_frame_detection',
                        help='Name for output directory')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download step (use existing video)')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip frame extraction step (use existing frames)')
    
    args = parser.parse_args()
    
    # Download video if URL provided and not skipping
    video_file = args.video_file
    if not args.skip_download and args.youtube_url:
        if download_youtube_video(args.youtube_url, args.save_path, args.download_name):
            video_file = os.path.join(args.save_path, f"{args.download_name}.mp4")
    elif not video_file:
        video_file = os.path.join(args.save_path, f"{args.download_name}.mp4")
    
    # Extract frames if not skipping
    frames_dir = os.path.join(args.save_path, "frames")
    if not args.skip_extract and video_file and os.path.exists(video_file):
        extract_frames(video_file, frames_dir, max_frames=args.max_frames)
    
    # Run inference
    if os.path.exists(frames_dir):
        run_inference(frames_dir, args.weights, args.conf, args.output_name)
    else:
        print(f"Frames directory not found: {frames_dir}")
