"""
Analyze inference results: compute frame coverage and visualize branding presence.
"""

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_frame_coverage(output_folder, label_map=None):
    """
    Analyze frame coverage from YOLO inference results.
    
    Args:
        output_folder: Directory containing inference results (images and labels)
        label_map: Dictionary mapping class IDs to label names
        
    Returns:
        DataFrame with frame coverage data
    """
    # Get list of frame inference outputs
    frames = [f for f in os.listdir(output_folder) 
              if os.path.isfile(os.path.join(output_folder, f)) and f.endswith('.jpg')]
    frames.sort()

    if label_map is None:
        # Try to infer from first annotation file
        label_map = {}
        labels_dir = os.path.join(output_folder, 'labels')
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            if label_files:
                # Read first file to get class IDs
                first_file = os.path.join(labels_dir, label_files[0])
                with open(first_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if class_id not in label_map:
                            label_map[class_id] = f'Class_{class_id}'

    # Iterate through frames and compute logo presence
    frame_coverage = pd.DataFrame()
    
    for frame in frames:
        # Obtain the frame number
        frame_number = int(frame.replace('frame', '').replace('.jpg', ''))

        # Create path to the frame annotation text file (if it exists)
        annotation_file = os.path.join(output_folder, 'labels', frame.replace('.jpg', '.txt'))

        # Check if there are annotations for this frame
        if os.path.isfile(annotation_file):
            # Read the bbox data if it exists
            annotations = pd.read_csv(annotation_file, sep=' ', header=None)
            annotations.columns = ['label', 'x_centre', 'y_centre', 'width', 'height']

            # Obtain the image size
            img_path = os.path.join(output_folder, frame)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_height, img_width, img_depth = img.shape
            image_size = img_height * img_width

            # Obtain the bbox size
            annotations['abs_width'] = round(annotations['width'] * img_width, 3)
            annotations['abs_height'] = round(annotations['height'] * img_height, 3)
            annotations['size'] = round(annotations['abs_width'] * annotations['abs_height'], 3)

            # Obtain total area by label
            coverage_df = annotations[['label', 'size']].groupby('label').sum() / image_size * 100
            coverage_df = coverage_df.reset_index()
            coverage_df['frame'] = frame_number

            # Add data to master df
            frame_coverage = pd.concat([frame_coverage, coverage_df], axis=0, ignore_index=True)

    # Update the mapping for label
    if label_map:
        frame_coverage['label'] = frame_coverage['label'].map(label_map)

    return frame_coverage


def plot_coverage(frame_coverage, save_path=None):
    """
    Plot frame coverage over time.
    
    Args:
        frame_coverage: DataFrame with frame coverage data
        save_path: Optional path to save the plot
    """
    if len(frame_coverage) == 0:
        print("No coverage data to plot")
        return

    plt.figure(figsize=(12, 6))
    
    # Get unique labels
    unique_labels = frame_coverage['label'].unique()
    
    for lbl in unique_labels:
        # Cut to the relevant label
        frame_coverage_cut = frame_coverage[frame_coverage['label'] == lbl]

        # Create a master df with all frame numbers
        max_frame = frame_coverage['frame'].max()
        frame_count = list(range(0, int(max_frame) + 1))
        master_frame_df = pd.DataFrame(columns=['frame'], data=frame_count)

        # Join on the annotation coverage
        master_frame_df = pd.merge(
            master_frame_df, frame_coverage_cut[['frame', 'size']],
            on='frame', how='left'
        ).fillna(0)

        plt.plot(master_frame_df['frame'], master_frame_df['size'], label=lbl, linewidth=2)

    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Proportion of Frame (%)', fontsize=12)
    plt.title('Frame Coverage by Label', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze inference results and plot coverage')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Directory containing inference results')
    parser.add_argument('--label_map', type=str, default=None,
                        help='Comma-separated label mappings (e.g., "0:Logo,1:Text")')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the coverage plot')
    
    args = parser.parse_args()
    
    # Parse label map if provided
    label_map = None
    if args.label_map:
        label_map = {}
        for mapping in args.label_map.split(','):
            key, value = mapping.split(':')
            label_map[int(key.strip())] = value.strip()
    
    # Analyze coverage
    frame_coverage = analyze_frame_coverage(args.output_folder, label_map)
    
    if len(frame_coverage) > 0:
        print(f"\nAnalyzed {len(frame_coverage)} detections across frames")
        print("\nSummary statistics:")
        print(frame_coverage.groupby('label')['size'].describe())
        
        # Plot coverage
        plot_coverage(frame_coverage, args.save_path)
    else:
        print("No detections found in the results")
