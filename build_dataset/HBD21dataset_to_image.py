"""

Extract frames from videos for annotation purposes.

- Extract frames at a specified frame rate (e.g., 1 frame per second) to create a dataset for Human-Object Interaction (HOI) annotation.
- Save extracted frames in a structured output directory.
"""

import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, fps=1,prefix=None):
    """
    Extract frames from video at specified frame rate for HOI annotation.
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        fps (int): Frames per second to extract (default: 1)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Normalize path and check if file exists
    video_path = os.path.normpath(video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Get video name without extension for prefix
    if prefix is None:
        prefix = Path(video_path).stem
    
    print(f"Opening video: {video_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video FPS: {video_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting {fps} frame(s) per second...")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            # Generate filename with timestamp
            filename = f"{prefix}_frame_{saved_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"\nExtraction complete!")
    print(f"Total frames extracted: {saved_count}")
    print(f"Frames saved to: {output_dir}")


def batch_extract_frames(video_dir, output_base_dir, fps=1):
    """
    Extract frames from multiple videos in a directory.
    
    Args:
        video_dir (str): Directory containing video files
        output_base_dir (str): Base directory for output folders
        fps (int): Frames per second to extract
    """
    video_extensions = '.mp4'
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
        video_files.extend(Path(video_dir).glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        
        extract_frames(str(video_path), output_base_dir, fps)


if __name__ == "__main__":

    base_path = "extracted_frames"

    video_dir= "Train\sabotage_violence"
    output_base_dir = os.path.join(base_path, "violence")
    batch_extract_frames(video_dir, output_base_dir, fps=0.5)

    ## fps=0.5 means 1 frame every 2 seconds, adjust as needed for annotation workload and video content.