import os
import csv
import shutil
from multiprocessing import Pool

meccano_labels_file = '../MECCANO/MECCANO_val_actions.csv'
meccano_frames_dir = '../MECCANO/frames'
output_frames_dir = './meccanoTest/frames'
output_labels_file = './meccanoTest/meccano_val_labels.txt'

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
    action_dir = os.path.join(output_frames_dir, video_id)
    action_dir = ensure_unique_folder_name(action_dir)
    os.makedirs(action_dir, exist_ok=True)

    # Copy frames from MECCANO to the action folder
    start_frame_num = int(start_frame.split('.')[0])
    end_frame_num = int(end_frame.split('.')[0])

    for frame_num in range(start_frame_num, end_frame_num + 1):
        frame_name = f'{frame_num:05}.jpg'
        src_path = os.path.join(meccano_frames_dir, video_id, frame_name)
        dst_path = os.path.join(action_dir, frame_name)
        shutil.copyfile(src_path, dst_path)

    return video_id, end_frame_num - start_frame_num + 1, action_id

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

    # Process actions in parallel
    with Pool() as pool:
        results = pool.starmap(process_action, actions)

    # Write action information to the labels file
    with open(output_labels_file, 'w') as file:
        for video_id, num_frames, label in results:
            label_line = f'{video_id} {num_frames} {label}\n'
            file.write(label_line)

    print('MECCANO dataset arrangement and labels file generation completed.')


arrange_meccano_dataset(meccano_labels_file, meccano_frames_dir, output_frames_dir, output_labels_file)
