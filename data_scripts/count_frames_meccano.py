import os
import csv
import shutil
import argparse


def process_action(video_id, action_id, action_name, start_frame, end_frame, meccano_frames_dir):
    # Copy frames from MECCANO to the output folder. Offset is used to name all the frames from 00001.jpg to <num_frames>.jpg
    start_frame_num = int(start_frame.split('.')[0])
    end_frame_num = int(end_frame.split('.')[0])

    return end_frame_num - start_frame_num + 1

def arrange_meccano_dataset(meccano_labels_file, meccano_frames_dir):
    # Initialize action information list
    actions = []

    # Read MECCANO labels from CSV file
    with open(meccano_labels_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            video_id, action_id, action_name, start_frame, end_frame = row
            actions.append((video_id, action_id, action_name, start_frame, end_frame, meccano_frames_dir))

    results = []
    for action in actions:
        result = process_action(*action)
        results.append(result)

    rst = sum(results)

    print('Num Frames: ' + str(rst) + ' for ' + str(meccano_labels_file))
    return rst

def count_from_txt(meccano_labels_file, meccano_frames_dir):
    # Initialize action information list
    frames = []

    # Read MECCANO labels from CSV file
    with open(meccano_labels_file, 'r') as file:
        for line in file:
            video_id, num_frames, action_id = line.split(sep=' ')
            frames.append(int(num_frames))

    rst = sum(frames)

    print('Num Frames: ' + str(rst) + ' for ' + str(meccano_labels_file))
    return rst



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MECCANO Dataset Arrangement Script')

    parser.add_argument('--meccano_dataset_dir', help='Path to the MECCANO labels CSV files', default='../MECCANO')
    parser.add_argument('--meccano_txt_dir', help='Path to the MECCANO labels TXT files', default='./dataset/meccano')

    args = parser.parse_args()

    meccano_dataset_dir = args.meccano_dataset_dir
    meccano_txt_dir = args.meccano_txt_dir
    
    train_csv = os.path.join(meccano_dataset_dir, 'MECCANO_train_actions.csv')
    val_csv = os.path.join(meccano_dataset_dir, 'MECCANO_val_actions.csv')
    test_csv = os.path.join(meccano_dataset_dir, 'MECCANO_test_actions.csv')

    train_txt = os.path.join(meccano_txt_dir, 'train_videofolder.txt')
    val_txt = os.path.join(meccano_txt_dir, 'val_videofolder.txt')
    test_txt = os.path.join(meccano_txt_dir, 'test_videofolder.txt')

    # ARRANGE RGB FRAMES
    meccano_frames_dir = os.path.join(meccano_dataset_dir, 'frames')

    print("COUNTING TRAIN RGB FRAMES")
    cnt = arrange_meccano_dataset(train_csv, meccano_frames_dir)
    cnt_txt = count_from_txt(train_txt, meccano_txt_dir)
    print("COUNTING VALIDATION RGB FRAMES")
    cnt += arrange_meccano_dataset(val_csv, meccano_frames_dir)
    cnt_txt += count_from_txt(val_txt, meccano_txt_dir)
    print("COUNTING TEST RGB FRAMES")
    cnt += arrange_meccano_dataset(test_csv, meccano_frames_dir)
    cnt_txt += count_from_txt(test_txt, meccano_txt_dir)

    print('TOTAL NUM CSV: ' + str(cnt))
    print('TOTAL NUM TXT: ' + str(cnt_txt))
