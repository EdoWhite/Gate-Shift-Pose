import argparse
import pandas as pd
import random

"""
Given a train.txt file in the format "video_folder frames class_id", it produces
a similar file corresponding to a downsampled dataset. The new file respects the original classes distribution.
"""

def downsample_dataset(input_file, output_file, ratio):
    # Load the dataset
    data = pd.read_csv(input_file, sep=" ", header=None, names=["video_folder", "total_frames", "class_id"])
    
    # Group by class to ensure proportional downsampling
    grouped = data.groupby("class_id")
    
    downsampled_data = []
    for class_id, group in grouped:
        # Calculate the number of samples to retain for each class
        downsample_size = int(len(group) * ratio)
        
        # Sample without replacement
        downsampled_group = group.sample(n=downsample_size, random_state=42)
        downsampled_data.append(downsampled_group)
    
    # Concatenate the downsampled groups and save to the output file
    downsampled_dataset = pd.concat(downsampled_data).sort_index()
    downsampled_dataset.to_csv(output_file, sep=" ", header=False, index=False)
    print(f"Downsampled dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample the Diving48 dataset.")
    parser.add_argument("input_file", type=str, help="Path to the input dataset file.")
    parser.add_argument("output_file", type=str, help="Path to save the downsampled dataset file.")
    parser.add_argument("ratio", type=float, help="Downsampling ratio (e.g., 0.5 for half size).")
    args = parser.parse_args()
    
    downsample_dataset(args.input_file, args.output_file, args.ratio) 