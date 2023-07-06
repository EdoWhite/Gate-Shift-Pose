import os
import csv
import shutil
import argparse


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

    # Copy frames from MECCANO to the output folder. Offset is used to name all the frames from 00001.jpg to <num_frames>.jpg
    start_frame_num = int(start_frame.split('.')[0])
    end_frame_num = int(end_frame.split('.')[0])
    frame_num_offset = 1

    for frame_num in range(start_frame_num, end_frame_num + 1):
        frame_name = f'{frame_num:05}.jpg'
        src_path = os.path.join(meccano_frames_dir, video_id, frame_name)
        dst_path = os.path.join(action_dir, f'{frame_num_offset:05}.jpg')
        shutil.copyfile(src_path, dst_path)
        frame_num_offset += 1

    return os.path.basename(action_dir), end_frame_num - start_frame_num + 1, action_id

def arrange_meccano_dataset(meccano_labels_file, meccano_frames_dir, output_frames_dir, output_labels_file=None):
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
    if output_labels_file != None:
        with open(output_labels_file, 'w') as file:
            for video_id, num_frames, label in results:
                label_line = f'{video_id} {num_frames} {label}\n'
                file.write(label_line)

    print('MECCANO dataset arrangement and labels file generation completed.')

def copy_subfolders_to_parent_folder(directory):
    train_folder = os.path.join(directory, 'Train')
    val_folder = os.path.join(directory, 'Val')
    test_folder = os.path.join(directory, 'Test')
    depth_frames_folder = directory

    # Create the 'Depth_frames' directory if it doesn't exist
    if not os.path.exists(depth_frames_folder):
        os.makedirs(depth_frames_folder)

    # Copy subfolders to 'Depth_frames' directory
    for folder in [train_folder, val_folder, test_folder]:
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            destination_path = os.path.join(depth_frames_folder, subfolder)
            shutil.copytree(subfolder_path, destination_path)

    print("Subfolders have been copied to the 'Depth_frames' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MECCANO Dataset Arrangement Script')
    #parser.add_argument('--meccano_labels_dir', help='Path to the MECCANO labels CSV files', default='../MECCANO')
    #parser.add_argument('--meccano_frames_dir', help='Path to the MECCANO frames directory', default='../MECCANO/frames')
    #parser.add_argument('--output_frames_dir', help='Path to the output frames directory', default='./meccanoTest/frames')
    #parser.add_argument('--output_labels_file', help='Path to the output labels file', default='./meccanoTest/meccano_val_labels.txt')

    parser.add_argument('--meccano_dataset_dir', help='Path to the MECCANO labels CSV files', default='../MECCANO')
    parser.add_argument('--output_dir', help='Path to the output directory')
    parser.add_argument('--process_only_depth', default=False, action="store_true", help='Process only depth frames')

    args = parser.parse_args()

    meccano_dataset_dir = args.meccano_dataset_dir
    output_dir = args.output_dir
    
    train_csv = os.path.join(meccano_dataset_dir, 'MECCANO_train_actions.csv')
    val_csv = os.path.join(meccano_dataset_dir, 'MECCANO_val_actions.csv')
    test_csv = os.path.join(meccano_dataset_dir, 'MECCANO_test_actions.csv')

    if args.process_only_depth == False:
        # ARRANGE RGB FRAMES
        meccano_frames_dir = os.path.join(meccano_dataset_dir, 'frames')
        output_frames_dir = os.path.join(output_dir, 'frames')

        train_txt = os.path.join(output_dir, 'train_videofolder.txt')
        val_txt = os.path.join(output_dir, 'val_videofolder.txt')
        test_txt = os.path.join(output_dir, 'test_videofolder.txt')

        print("ARRANGING TRAIN RGB FRAMES")
        arrange_meccano_dataset(train_csv, meccano_frames_dir, output_frames_dir, train_txt)
        print("ARRANGING VALIDATION RGB FRAMES")
        arrange_meccano_dataset(val_csv, meccano_frames_dir, output_frames_dir, val_txt)
        print("ARRANGING TEST RGB FRAMES")
        arrange_meccano_dataset(test_csv, meccano_frames_dir, output_frames_dir, test_txt)

    # ARRANGE DEPTH FRAMES
    meccano_depth_frames_dir = os.path.join(meccano_dataset_dir, 'Depth_frames')
    output_depth_frames_dir = os.path.join(output_dir, 'depth_frames')

    copy_subfolders_to_parent_folder(meccano_depth_frames_dir)

    print("ARRANGING TRAIN DEPTH FRAMES")
    arrange_meccano_dataset(train_csv, meccano_depth_frames_dir, output_depth_frames_dir)
    print("ARRANGING VALIDATION DEPTH FRAMES")
    arrange_meccano_dataset(val_csv, meccano_depth_frames_dir, output_depth_frames_dir)
    print("ARRANGING TEST DEPTH FRAMES")
    arrange_meccano_dataset(test_csv, meccano_depth_frames_dir, output_depth_frames_dir)
