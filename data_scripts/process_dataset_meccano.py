import os
import csv
import shutil
import argparse

"""
MECCANO Dataset Arrangement Script

This script arranges the frames of the MECCANO dataset into new folders, where each folder corresponds to an action.
The MECCANO dataset consists of videos with associated action labels in a CSV file. The frames of each video are stored
in the 'MECCANO/frames/<video_id>/<frame_id>.jpg' directory.

The CSV file with the actions label comes in the following format:
    video_id, action_id, action_name, start_frame, end_frame

Each row of the CSV file represents an action in a video, where:
    - video_id: ID of the video.
    - action_id: ID of the action.
    - action_name: Name of the action.
    - start_frame: Start frame of the action.
    - end_frame: End frame of the action.

The script will create new directories for each action, and it will copy the corresponding frames from MECCANO to the 
respective action directory. The script will also generate a labels file in the format:
    video_folder num_frames label

Usage:
    python process_dataset_meccano.py

Input:
    - 'meccano dataset dir': MECCANO dataset directory on the disk
    - 'output dir': Directory where to place the processed data.

Output:
    - 'frames': Directory containing the frames arranged into new folders for each action.
    - 'train_videofolder.txt': Labels file containing information about train data.
    - 'val_videofolder.txt': Labels file containing information about val data.
    - 'test_videofolder.txt': Labels file containing information about test data.

Please note:
    - The script ensures that the folder names are unique by appending a counter if a folder with the same name exists.
    - The video IDs in the new directory names will have the same length with leading zeros for better sorting.

Author:
    Edoardo Bianchi
"""

def ensure_unique_folder_name(folder_path):
    # Add a counter to the folder name to make it unique
    folder_name = os.path.basename(folder_path)
    counter = 1
    while os.path.exists(folder_path):
        folder_path = os.path.join(os.path.dirname(folder_path), f"{folder_name}_{counter}")
        counter += 1
    return folder_path

def process_action(video_id, action_id, action_name, start_frame, end_frame, meccano_frames_dir, output_frames_dir):
    # Create a new folder for each action
    action_dir_name = f"{video_id}_{action_id}"
    action_dir = os.path.join(output_frames_dir, action_dir_name)
    action_dir = ensure_unique_folder_name(action_dir)
    os.makedirs(action_dir, exist_ok=True)

    # Copy frames from MECCANO to the action folder. Offset is used to name all the frames from 00001.jpg to <num_frames>.jpg
    start_frame_num = int(start_frame.split('.')[0])
    end_frame_num = int(end_frame.split('.')[0])
    frame_num_offset = 1

    for frame_num in range(start_frame_num, end_frame_num + 1):
        frame_name = f'{frame_num:05}.jpg'
        src_path = os.path.join(meccano_frames_dir, video_id, frame_name)
        dst_path = os.path.join(action_dir, f'{frame_num_offset:05}.jpg')
        shutil.copyfile(src_path, dst_path)
        frame_num_offset += 1

    return action_dir_name, end_frame_num - start_frame_num + 1, action_id

def arrange_meccano_dataset(meccano_labels_file, meccano_frames_dir, output_frames_dir, output_labels_file):
    # Create output directories if they don't exist
    os.makedirs(output_frames_dir, exist_ok=True)

    # Initialize action information list
    actions = []

    # Read MECCANO labels from CSV file
    with open(meccano_labels_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            video_id, action_id, action_name, start_frame, end_frame = row
            actions.append((video_id, action_id, action_name, start_frame, end_frame, meccano_frames_dir, output_frames_dir))

    results = []
    for action in actions:
        result = process_action(*action)
        results.append(result)

    # Write action information to the labels file
    with open(output_labels_file, 'w') as file:
        for video_id, num_frames, label in results:
            label_line = f'{video_id} {num_frames} {label}\n'
            file.write(label_line)

    print('MECCANO dataset arrangement and labels file generation completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MECCANO Dataset Arrangement Script')
    #parser.add_argument('--meccano_labels_dir', help='Path to the MECCANO labels CSV files', default='../MECCANO')
    #parser.add_argument('--meccano_frames_dir', help='Path to the MECCANO frames directory', default='../MECCANO/frames')
    #parser.add_argument('--output_frames_dir', help='Path to the output frames directory', default='./meccanoTest/frames')
    #parser.add_argument('--output_labels_file', help='Path to the output labels file', default='./meccanoTest/meccano_val_labels.txt')

    parser.add_argument('--meccano_dataset_dir', help='Path to the MECCANO labels CSV files', default='../MECCANO')
    parser.add_argument('--output_dir', help='Path to the output directory')

    args = parser.parse_args()

    meccano_dataset_dir = args.meccano_dataset_dir
    meccano_frames_dir = os.path.join(args.meccano_dataset_dir, 'frames')

    output_dir = args.output_dir
    output_frames_dir = os.path.join(args.output_dir, 'frames')

    train_csv = os.path.join(meccano_dataset_dir, 'MECCANO_train_actions.csv')
    val_csv = os.path.join(meccano_dataset_dir, 'MECCANO_val_actions.csv')
    test_csv = os.path.join(meccano_dataset_dir, 'MECCANO_test_actions.csv')

    train_txt = os.path.join(output_dir, 'train_videofolder.txt')
    val_txt = os.path.join(output_dir, 'val_videofolder.txt')
    test_txt = os.path.join(output_dir, 'test_videofolder.txt')

    print("ARRANGING TRAIN FRAMES")
    arrange_meccano_dataset(train_csv, meccano_frames_dir, output_frames_dir, train_txt)
    print("ARRANGING VALIDATION FRAMES")
    arrange_meccano_dataset(val_csv, meccano_frames_dir, output_frames_dir, val_txt)
    print("ARRANGING TEST FRAMES")
    arrange_meccano_dataset(test_csv, meccano_frames_dir, output_frames_dir, test_txt)
