import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import sys
import h5py
import io
import gulpio2
from ultralytics import YOLO
import cv2

class VideoRecord(object):
    def __init__(self, row, multilabel):
        self._data = row
        self._multilabel = multilabel

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def label_verb(self):
        if self._multilabel:
            return int(self._data[3])
        else:
            return 0

    @property
    def label_noun(self):
        if self._multilabel:
            return int(self._data[4])
        else:
            return 0

    @property
    def start_frame(self):
        if self._multilabel:
            return int(self._data[5])
        else:
            return 0
        
class MeccanoVideoRecord(object):
    def __init__(self, row, multilabel):
        self._data = row
        self._multilabel = multilabel

    @property
    def path(self):
        return self._data[0]
    
    @property
    def num_frames(self):

        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[1])

    @property
    def label_name(self):
        return int(self._data[2])


    @property
    def start_frame(self):
        return self._data[3]

    @property
    def end_frame(self):
        return self._data[4]

"""
All datasets that represent a map from keys to data samples should subclass it. 
All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key. 
Subclasses could also optionally overwrite __len__(), which is expected to return the size of 
the dataset by many Sampler implementations and the default options of DataLoader.
"""

class VideoDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, num_clips=1,
                 load_from_video=False, frame_interval=5,
                 sparse_sampling=True, multilabel=False, dense_sample=False, from_hdf5=False, from_gulp=False, mode="train", 
                 random_shuffling=False,):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_clips = num_clips
        self.load_from_video = load_from_video
        self.frame_interval = frame_interval
        self.sparse_sampling = sparse_sampling
        self.multilabel = multilabel
        self.dense_sample = dense_sample
        self.from_hdf5 = from_hdf5
        self.from_gulp = from_gulp
        self.h5_file = None
        if self.from_gulp and self.multilabel:
            self.root_path = self.root_path.replace("frames", "frames_gulp")
        self.mode = mode
        self.random_shuffling = random_shuffling

        self._parse_list()


    def _load_image(self, directory, idx):
        try:
            return [
                Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]


    def _load_from_video(self, path, frame_ids):
        return 0


    def _parse_list(self):
        #self.list_file is the dataset file
        # check the frame number is large >3:
        
        # usualy it is [video_id, num_frames, class_idx]
        if self.from_gulp:
            self.list_file = self.list_file.replace(".txt", "_gulp.txt")

        if "kinetics" in self.root_path:
            tmp = [x.strip().split(',') for x in open(self.list_file)]

        elif "MECCANO__" in self.root_path:
            print("Using MECCANO dataset...")
            tmp = [x.strip().split(',') for x in open(self.list_file)]

        else:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]

        tmp = [item for item in tmp if int(item[1])>=3]

        if "MECCANO__" in self.root_path:
             print("Using MeccanoVideoRecord...")
             self.video_list = [MeccanoVideoRecord(item, self.multilabel) for item in tmp]
        else:
            self.video_list = [VideoRecord(item, self.multilabel) for item in tmp]

        if self.from_gulp:
            mode = "train" if "val" not in self.list_file else "val"
            self.gulp = gulpio2.GulpDirectory(os.path.join(self.root_path, mode))
            len_gulp = 0
            for dict in self.gulp.all_meta_dicts:
                len_gulp += len(dict)
            assert len_gulp == len(self.video_list), f"No. of samples is different {self.list_file}({len(self.video_list)}) | {os.path.join(self.root_path, mode)}({len_gulp})"
            
        print('video number:%d'%(len(self.video_list)))


    def _sample_indices(self, record=None, num_frames=None):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        
        # self.dense_sample is false
        else:

            num_frames = record.num_frames if record is not None else num_frames

            average_duration = num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) + 1
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames, size=self.num_segments)) + 1
            else:
                tick = num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
                # offsets = np.zeros((self.num_segments,))
                # offsets = np.concatenate(
                #     [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        return offsets
    

    def _sample_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1


    def _sample_indices_video(self, record):

        """

        :param record: VideoRecord
        :return: list
        """
        num_frames = record.num_frames

        if not self.sparse_sampling:
            max_frame_ind = num_frames - (self.frame_interval * (self.num_segments-1)) - 1
            if max_frame_ind > 0:
                start_frame_ind = randint(max_frame_ind, size=1)[0]
                offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
                # print(offsets)
            else:
                average_duration = num_frames // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                      size=self.num_segments)
                elif num_frames > self.num_segments:
                    offsets = np.sort(randint(num_frames, size=self.num_segments))
                else:
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        else:
            average_duration = num_frames // self.num_segments
            if average_duration > 0:
                offsets = list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments))
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames, size=self.num_segments))
            else:
                # offsets = np.zeros((self.num_segments,))
                offsets = np.concatenate(
                    [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
            return offsets + 1
        return np.array(offsets) + 1


    def _get_val_indices(self, record=None, num_frames=None):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            num_frames = record.num_frames if record is not None else num_frames
            # if num_frames > self.num_segments:
            #     tick = num_frames / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            # else:
            #     tick = num_frames / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            tick = num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
                # offsets = np.zeros((self.num_segments,))
                # offsets = np.concatenate(
                #     [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        return offsets


    def _get_val_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1


    def _get_val_indices_video(self, record):
        num_frames = record.num_frames

        if not self.sparse_sampling:
            max_frame_ind = (num_frames-1) // 2 - ((self.frame_interval) * (self.num_segments // 2 - 1)) - 2
            if max_frame_ind > 0:
                start_frame_ind = max_frame_ind
                offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
            else:
                if num_frames > self.num_segments:
                    tick = num_frames / float(self.num_segments)
                    offsets = [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                else:
                    # offsets = np.zeros((self.num_segments,))
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        else:
            if num_frames > self.num_segments:
                tick = num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            offsets = offsets + 1
            offsets = list(offsets)
        return np.array(offsets) + 1


    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        num_frames = record.num_frames

        tick = num_frames / float(self.num_segments)

        if self.num_clips == 1:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

        elif self.num_clips == 2:
                offsets = [np.array([int(tick * x) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        elif self.num_clips == 10:
            offsets_clips = []
            for k in range(10):
                average_duration = num_frames // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                size=self.num_segments)
                elif num_frames > self.num_segments:
                    offsets = np.sort(randint(num_frames, size=self.num_segments))
                else:
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
                offsets_clips.append(offsets+1)
            offsets = offsets_clips
        return offsets
    

    def _get_test_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
        offsets = []
        for start_idx in start_list.tolist():
            offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1


    def _get_test_indices_video(self, record):
            num_frames = record.num_frames
            if not self.sparse_sampling:
                num_frames = num_frames - 1
                if self.num_clips == 1:
                    max_frame_ind = num_frames // 2 - ((self.frame_interval) * (self.num_segments // 2 - 1)) - 1
                    if max_frame_ind > 0:
                        start_frame_ind = max_frame_ind
                        offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
                        # offsets = np.array(offsets) + 1
                    else:
                        if num_frames > self.num_segments:
                            tick = num_frames / float(self.num_segments)
                            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                        else:
                            offsets = np.concatenate(
                                [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)],
                                axis=-1)
                    offsets = np.array(offsets) + 1
                else:
                    max_frame_ind = num_frames - (self.frame_interval * (self.num_segments-1)) - 1
                    if max_frame_ind > 0:
                        start_inds = np.linspace(1, max_frame_ind, self.num_clips)
                        offsets = []
                        for start_ind in start_inds:
                            offsets.append([int(start_ind) + (self.frame_interval * x) for x in range(self.num_segments)])
                    else:
                        if num_frames > self.num_segments:
                            tick = num_frames / float(self.num_segments)
                            offsets = [np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])] * self.num_clips
                        else:
                            offsets = [np.concatenate(
                                [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)],
                                axis=-1)] * self.num_clips
                # offsets = offsets + 1

            else:
                tick = num_frames / float(self.num_segments)

                if self.num_clips == 1:
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

                elif self.num_clips == 2:
                        offsets = [np.array([int(tick * x) for x in range(self.num_segments)]) + 1,
                                   np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        # print(offsets)
            return offsets


    def __getitem__(self, index):
        """ 
        n_segments is the num of frames to consider. We sample n_segments frames from the num_frames. Then we load the sampled frames.

        RECORD: Y7QZcr24ye0_00815 115 17
        SEGMENT_INDICES: 24 48 89
        (record.path, p): nOlRwoxsDJ0_00574 24
        (record.path, p): nOlRwoxsDJ0_00574 48
        (record.path, p): nOlRwoxsDJ0_00574 89
        """
        record = self.video_list[index]
        # check this is a legit video folder
        # while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
        #     print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
        #     index = np.random.randint(len(self.video_list))
        #     record = self.video_list[index]
        if self.load_from_video: 
            if not self.test_mode:
                segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
            else:
                segment_indices = self._get_test_indices_video(record)
            return self.get_video(record, segment_indices)
        
        elif self.from_hdf5: 
            if not self.sparse_sampling:
                if not self.test_mode:
                    segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
                else:
                    segment_indices = self._get_test_indices_video(record)
            else:
                if not self.test_mode:
                    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                else:
                    segment_indices = self._get_test_indices(record)
            if self.random_shuffling:
                segment_indices = np.random.permutation(segment_indices)
            return self.get_from_hdf5(record, segment_indices)
        
        elif self.from_gulp:
            if not self.sparse_sampling:
                if not self.test_mode:
                    segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
                else:
                    segment_indices = self._get_test_indices_video(record)
            else:
                if not self.test_mode:
                    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                else:
                    segment_indices = self._get_test_indices(record)
            return self.get_from_gulp(record, segment_indices)
        
        else:
            if not self.sparse_sampling:
                if not self.test_mode:
                    segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
                else:
                    segment_indices = self._get_test_indices_video(record)
            else:
                # EXECUTE FROM HERE because the other conditions are not met
                if not self.test_mode:
                    # training and val
                    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                    #print("RECORD: " + record.path + " " + str(record.num_frames) + " " + str(record.label))
                    #print("SEGMENT_INDICES: " + " ".join(str(idx) for idx in segment_indices))
                else:
                    # test
                    segment_indices = self._get_test_indices(record)

            if self.random_shuffling:
                segment_indices = np.random.permutation(segment_indices)

            item = self.get(record, segment_indices)
            return item


    def get(self, record, indices):
        # num_clips = 1 by default, in training = 1, in test can be modified (on the repo is = 1)
        if self.num_clips > 1:
            process_data_final = []
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)

            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
                
            return process_data_final, label

        else:
            images = list()
            if self.multilabel:
                indices = indices + record.start_frame
            for seg_ind in indices:
                p = int(seg_ind)
                # (record.path, p): nOlRwoxsDJ0_00574 24
                # (record.path, p): nOlRwoxsDJ0_00574 48
                # (record.path, p): nOlRwoxsDJ0_00574 89
                #print("(record.path, p): " + str(record.path) + str(p))
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
            process_data, label = self.transform((images, record.label))

            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
                
            return process_data, label
            

    def get_from_hdf5(self, record, indices):
        if self.num_clips > 1:
            process_data_final = []
            hdf5_video_key = record.path
            if "something" in self.root_path.lower():
                single_h5 = os.path.join(
                    self.root_path, "Something-Something-v2-frames.h5"
                )
                if os.path.isfile(single_h5):
                    if self.h5_file is None:
                        self.h5_file = h5py.File(single_h5, "r")
                    video_binary = self.h5_file[hdf5_video_key]
                else:
                    video_binary = h5py.File(os.path.join(self.root_path, "seq_h5_30fps", record.path+".h5"))[
                        hdf5_video_key
                    ]
            else:
                single_h5 = os.path.join(self.root_path, self.mode, record.path + ".h5")
                video_binary = h5py.File(single_h5, "r")[record.path.split("/")[-1]]
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    try:
                        seg_imgs = [Image.open(io.BytesIO(video_binary[p])).convert("RGB")]
                    except:
                        seg_imgs = [Image.open(io.BytesIO(video_binary[p-1])).convert("RGB")]
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data_final, label

        else:
            images = list()
            if self.multilabel:
                indices = indices + record.start_frame
            hdf5_video_key = record.path
            if "something" in self.root_path.lower():
                single_h5 = os.path.join(
                    self.root_path, "Something-Something-v2-frames.h5"
                )
                if os.path.isfile(single_h5):
                    if self.h5_file is None:
                        self.h5_file = h5py.File(single_h5, "r")
                    video_binary = self.h5_file[hdf5_video_key]
                else:
                    video_binary = h5py.File(os.path.join(self.root_path, "seq_h5_30fps", record.path+".h5"))[
                        hdf5_video_key
                    ]
            else:
                single_h5 = os.path.join(self.root_path, self.mode, record.path + ".h5")
                video_binary = h5py.File(single_h5, "r")[record.path.split("/")[-1]]
            indices = self._sample_indices(None, len(video_binary)-1) if self.random_shift else self._get_val_indices(None, len(video_binary)-1)
            for seg_ind in indices:
                p = int(seg_ind)
                try:
                    seg_imgs = [Image.open(io.BytesIO(video_binary[p])).convert("RGB")]
                except:
                    print(record.path, p, len(video_binary))
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
            process_data, label = self.transform((images, record.label))
            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data, label
            
    def get_from_gulp(self, record, indices):
        if self.num_clips > 1:
            process_data_final = []
            video_id = record.path
            video_data = self.gulp[video_id][0]
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    seg_imgs = Image.fromarray(video_data[p])
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data_final, label

        else:
            images = list()
            video_id = record.path
            video_data = self.gulp[video_id][0]
            
            for seg_ind in indices:
                p = int(seg_ind)
                try:
                    seg_imgs = [Image.fromarray(video_data[p])]
                except:
                    print(record.path, p, len(video_data))
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
            # print(len(images), images[0].shape, type(images[0]))
            process_data, label = self.transform((images, record.label))
            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data, label


    def get_video(self, record, indices):
        # print(indices)
        if self.num_clips > 1:
            process_data_final = []
            # try:
            with open(os.path.join(self.root_path, record.path), 'rb') as f:
                vr = VideoReader(f)
                for k in range(self.num_clips):
                    images = vr.get_batch(indices[k]).asnumpy()
                    images = [Image.fromarray(images[i]).convert('RGB') for i in range(self.num_segments)]

                    process_data, label = self.transform((images, record.label))
                    process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            return process_data_final, label
            # except:
            #     print('Error loading {}'.format(os.path.join(self.root_path, record.path)))

        else:
            # try:
            with open(os.path.join(self.root_path, record.path), 'rb') as f:
                vr = VideoReader(f)
                images = vr.get_batch(indices).asnumpy()
                # print(images.shape)
            # print(indices)
            images = [Image.fromarray(images[i]).convert('RGB') for i in range(self.num_segments)]
            # print(len(images), images[0].size)
            process_data, label = self.transform((images, record.label))
            # print('read success', process_data.size())
            return process_data, label
            # except:
            #     print('Error loading {}'.format(os.path.join(self.root_path, record.path)))

    def __len__(self):
        return len(self.video_list)

# Load the images and compute the poses
class VideoDatasetPoses(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, num_clips=1,
                 load_from_video=False, frame_interval=5,
                 sparse_sampling=True, multilabel=False, dense_sample=False, mode="train", 
                 random_shuffling=False,):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_clips = num_clips
        self.load_from_video = load_from_video
        self.frame_interval = frame_interval
        self.sparse_sampling = sparse_sampling
        self.multilabel = multilabel
        self.dense_sample = dense_sample
        self.mode = mode
        self.random_shuffling = random_shuffling
        self.pose_model = YOLO('/home/clusterusers/edbianchi/POSE/yolov8m-pose.pt')

        self._parse_list()

    """
    def _load_image_with_pose(self, directory, idx):
        try:
            # Load RGB frame
            rgb_frames = [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]

            # Convert to numpy for YOLO processing
            frame_np = np.array(rgb_frames[0])

            # Perform inference using YOLO pose model
            result = self.pose_model(rgb_frames)[0]  # Assuming single frame inference
            keypoints = result.keypoints

            if keypoints is not None:
                # Generate pose heatmaps
                heatmaps = self.generate_pose_heatmaps(keypoints, frame_np.shape[:2])
                pose_frames = [Image.fromarray((heatmap * 255).astype(np.uint8)).convert('L') for heatmap in heatmaps]

                # Combine RGB frames with pose heatmaps (concatenate channels)
                return rgb_frames + pose_frames
            else:
                return rgb_frames  # No keypoints detected
            
        except Exception as e:
            print(f"Error loading image or pose: {e}")
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
    """
    
    def _load_image(self, directory, idx):
        try:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB').resize((640,360))]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]


    def generate_pose_heatmap(self, keypoints, img_size=(360, 640), sigma=10):
        """
        Generates a single heatmap from a set of keypoints using a Gaussian distribution.
        :param keypoints: Array of keypoints (x, y) coordinates
        :param img_size: Size of the image (height, width)
        :param sigma: Standard deviation of the Gaussian distribution
        :return: 2D heatmap of shape (img_size[0], img_size[1])
        """
        heatmap = np.zeros(img_size, dtype=np.float32)

        # Define the size of the Gaussian kernel
        size = int(6 * sigma + 1)
        x_range = np.arange(0, size, 1, float)
        y_range = np.arange(0, size, 1, float)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        gaussian_kernel = np.exp(-((x_grid - size // 2) ** 2 + (y_grid - size // 2) ** 2) / (2 * sigma ** 2))

        # Iterate over keypoints (ignoring zero-valued keypoints)
        for i in range(keypoints.shape[1]):  # keypoints.shape[1] is 17 (number of keypoints)
            x, y = int(keypoints[0, i, 0].item()), int(keypoints[0, i, 1].item())  # Access the x, y of each keypoint

            # Ensure valid keypoints (non-zero)
            if x > 0 and y > 0:
                # Define the bounding box for the Gaussian to be placed
                ul_x = max(0, x - size // 2)
                ul_y = max(0, y - size // 2)
                br_x = min(img_size[1], x + size // 2 + 1)
                br_y = min(img_size[0], y + size // 2 + 1)

                # Compute the region of the heatmap that the Gaussian will affect
                g_x_ul = max(0, size // 2 - x)
                g_y_ul = max(0, size // 2 - y)
                g_x_br = min(size, img_size[1] - ul_x)
                g_y_br = min(size, img_size[0] - ul_y)

                # Add the Gaussian to the heatmap
                heatmap[ul_y:br_y, ul_x:br_x] += gaussian_kernel[g_y_ul:g_y_br, g_x_ul:g_x_br]

        return heatmap

    def generate_blank_heatmap(self, img_size=(360, 640)):
        # Generate a blank heatmap for frames with no detected keypoints
        return np.zeros(img_size, dtype=np.float32)

    def append_heatmap_to_image(self, img, heatmap):
        img_np = np.array(img)  # Convert PIL image to NumPy array
        heatmap = np.expand_dims(heatmap, axis=-1)  # Add channel dimension to heatmap

        # Make sure that the heatmap has the same spatial dimensions as the image
        if heatmap.shape[:2] != img_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

        print("try to execute line 813 Dataset.py")
        combined = np.concatenate((img_np, heatmap), axis=-1)  # Append heatmap as an additional channel
        print("executed line 813 Dataset.py") 

        # Ensure that the combined array is in a valid format for PIL
        combined_img = Image.fromarray(combined.astype(np.uint8), mode='RGBA')  # Convert back to PIL image, assuming 4 channels

        return combined_img

    def _parse_list(self):
        #self.list_file is the dataset file
        # check the frame number is large >3:
        
        # usualy it is [video_id, num_frames, class_idx]
        if "kinetics" in self.root_path:
            tmp = [x.strip().split(',') for x in open(self.list_file)]

        elif "MECCANO__" in self.root_path:
            print("Using MECCANO dataset...")
            tmp = [x.strip().split(',') for x in open(self.list_file)]

        else:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]

        tmp = [item for item in tmp if int(item[1])>=3]

        if "MECCANO__" in self.root_path:
             print("Using MeccanoVideoRecord...")
             self.video_list = [MeccanoVideoRecord(item, self.multilabel) for item in tmp]
        else:
            self.video_list = [VideoRecord(item, self.multilabel) for item in tmp]
            
        print('video number:%d'%(len(self.video_list)))


    def _sample_indices(self, record=None, num_frames=None):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        
        # self.dense_sample is false
        else:

            num_frames = record.num_frames if record is not None else num_frames

            average_duration = num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) + 1
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames, size=self.num_segments)) + 1
            else:
                tick = num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
                # offsets = np.zeros((self.num_segments,))
                # offsets = np.concatenate(
                #     [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        return offsets
    

    def _sample_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1


    def _sample_indices_video(self, record):

        """

        :param record: VideoRecord
        :return: list
        """
        num_frames = record.num_frames

        if not self.sparse_sampling:
            max_frame_ind = num_frames - (self.frame_interval * (self.num_segments-1)) - 1
            if max_frame_ind > 0:
                start_frame_ind = randint(max_frame_ind, size=1)[0]
                offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
                # print(offsets)
            else:
                average_duration = num_frames // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                      size=self.num_segments)
                elif num_frames > self.num_segments:
                    offsets = np.sort(randint(num_frames, size=self.num_segments))
                else:
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        else:
            average_duration = num_frames // self.num_segments
            if average_duration > 0:
                offsets = list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments))
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames, size=self.num_segments))
            else:
                # offsets = np.zeros((self.num_segments,))
                offsets = np.concatenate(
                    [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
            return offsets + 1
        return np.array(offsets) + 1


    def _get_val_indices(self, record=None, num_frames=None):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            num_frames = record.num_frames if record is not None else num_frames
            # if num_frames > self.num_segments:
            #     tick = num_frames / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            # else:
            #     tick = num_frames / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            tick = num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
                # offsets = np.zeros((self.num_segments,))
                # offsets = np.concatenate(
                #     [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        return offsets


    def _get_val_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1


    def _get_val_indices_video(self, record):
        num_frames = record.num_frames

        if not self.sparse_sampling:
            max_frame_ind = (num_frames-1) // 2 - ((self.frame_interval) * (self.num_segments // 2 - 1)) - 2
            if max_frame_ind > 0:
                start_frame_ind = max_frame_ind
                offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
            else:
                if num_frames > self.num_segments:
                    tick = num_frames / float(self.num_segments)
                    offsets = [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                else:
                    # offsets = np.zeros((self.num_segments,))
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        else:
            if num_frames > self.num_segments:
                tick = num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            offsets = offsets + 1
            offsets = list(offsets)
        return np.array(offsets) + 1


    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        num_frames = record.num_frames

        tick = num_frames / float(self.num_segments)

        if self.num_clips == 1:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

        elif self.num_clips == 2:
                offsets = [np.array([int(tick * x) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        elif self.num_clips == 10:
            offsets_clips = []
            for k in range(10):
                average_duration = num_frames // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                size=self.num_segments)
                elif num_frames > self.num_segments:
                    offsets = np.sort(randint(num_frames, size=self.num_segments))
                else:
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
                offsets_clips.append(offsets+1)
            offsets = offsets_clips
        return offsets
    

    def _get_test_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
        offsets = []
        for start_idx in start_list.tolist():
            offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1


    def _get_test_indices_video(self, record):
            num_frames = record.num_frames
            if not self.sparse_sampling:
                num_frames = num_frames - 1
                if self.num_clips == 1:
                    max_frame_ind = num_frames // 2 - ((self.frame_interval) * (self.num_segments // 2 - 1)) - 1
                    if max_frame_ind > 0:
                        start_frame_ind = max_frame_ind
                        offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
                        # offsets = np.array(offsets) + 1
                    else:
                        if num_frames > self.num_segments:
                            tick = num_frames / float(self.num_segments)
                            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                        else:
                            offsets = np.concatenate(
                                [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)],
                                axis=-1)
                    offsets = np.array(offsets) + 1
                else:
                    max_frame_ind = num_frames - (self.frame_interval * (self.num_segments-1)) - 1
                    if max_frame_ind > 0:
                        start_inds = np.linspace(1, max_frame_ind, self.num_clips)
                        offsets = []
                        for start_ind in start_inds:
                            offsets.append([int(start_ind) + (self.frame_interval * x) for x in range(self.num_segments)])
                    else:
                        if num_frames > self.num_segments:
                            tick = num_frames / float(self.num_segments)
                            offsets = [np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])] * self.num_clips
                        else:
                            offsets = [np.concatenate(
                                [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)],
                                axis=-1)] * self.num_clips
                # offsets = offsets + 1

            else:
                tick = num_frames / float(self.num_segments)

                if self.num_clips == 1:
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

                elif self.num_clips == 2:
                        offsets = [np.array([int(tick * x) for x in range(self.num_segments)]) + 1,
                                   np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        # print(offsets)
            return offsets


    def __getitem__(self, index):
        """ 
        n_segments is the num of frames to consider. We sample n_segments frames from the num_frames. Then we load the sampled frames.

        RECORD: Y7QZcr24ye0_00815 115 17
        SEGMENT_INDICES: 24 48 89
        (record.path, p): nOlRwoxsDJ0_00574 24
        (record.path, p): nOlRwoxsDJ0_00574 48
        (record.path, p): nOlRwoxsDJ0_00574 89
        """
        record = self.video_list[index]
        # check this is a legit video folder
        # while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
        #     print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
        #     index = np.random.randint(len(self.video_list))
        #     record = self.video_list[index]

        if not self.sparse_sampling:
            if not self.test_mode:
                segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
            else:
                segment_indices = self._get_test_indices_video(record)
        else:
            # EXECUTE FROM HERE because the other conditions are not met
            if not self.test_mode:
                # training and val
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                #print("RECORD: " + record.path + " " + str(record.num_frames) + " " + str(record.label))
                #print("SEGMENT_INDICES: " + " ".join(str(idx) for idx in segment_indices))
            else:
                # test
                segment_indices = self._get_test_indices(record)

        if self.random_shuffling:
            segment_indices = np.random.permutation(segment_indices)

        item = self.get(record, segment_indices)
        return item


    def get(self, record, indices):
        # num_clips = 1 by default, in training = 1, in test can be modified (on the repo is = 1)
        if self.num_clips > 1:
            process_data_final = []
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)

            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
                
            return process_data_final, label

        else:
            keypoints = None
            images = list()
            pose_heatmaps = list()
            
            if self.multilabel:
                indices = indices + record.start_frame
            
            for seg_ind in indices:
                p = int(seg_ind)
                # seg_imgs contains more than one image
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)

                # Run pose detection and generate heatmaps
                for img in seg_imgs:
                    #print("\n")
                    #print("Computing Poses")
                    #print("image size: {}".format(img.size))
                    result = self.pose_model.predict(img, conf=0.5, half=True)
                    #print("\n")
                    for res in result:
                        keypoints = res.keypoints

                    if keypoints is not None and keypoints.shape[1] > 0:
                        heatmap = self.generate_pose_heatmap(keypoints[0].xy) # Get only the poses of the first person detected
                        pose_heatmaps.append(heatmap)
                    else:
                        pose_heatmaps.append(self.generate_blank_heatmap())

                if p < record.num_frames:
                    p += 1

            # Ensure combined_images is valid
            if len(images) == len(pose_heatmaps):
                combined_images = [self.append_heatmap_to_image(img, heatmap) for img, heatmap in zip(images, pose_heatmaps)]
            else:
                print(f"Mismatch in images and heatmaps length: {len(images)} vs {len(pose_heatmaps)}")
                return None  # Handle the error more gracefully here

            # Ensure combined_images is valid before applying the transform
            if combined_images is None or len(combined_images) == 0:
                print("No valid combined images")
                return None

            print(f"Applying transform on combined images of length {len(combined_images)}")
            
            # Apply transformations to combined images
            process_data, label = self.transform((combined_images, record.label))

            if process_data is None:
                raise ValueError("The transformed data is None")

            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
                
            return process_data, label

    def __len__(self):
        return len(self.video_list)