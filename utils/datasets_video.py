import os
import torch
import torchvision
import torchvision.datasets as datasets


#ROOT_DATASET = './dataset'
# ADDED SUPPORT FOR 3 DATASET SPLITS
# ADDED SUPPORT TO FIND DATA IN DIFFEREN DISK LOCATIONS

def return_something_v1():
    root_data = 'something-v1/20bn-something-something-v1'
    filename_imglist_train = 'something-v1/train_videofolder.txt'
    filename_imglist_val = 'something-v1/val_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_something_v2():
    root_data = 'something-v2/20bn-something-something-v2'
    filename_imglist_train = 'something-v2/train_videofolder.txt'
    filename_imglist_val = 'something-v2/val_videofolder.txt'
    prefix = '{:06d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix
    
def return_diving48():
    root_data = 'Diving48/frames'
    filename_imglist_train = 'Diving48/train_videofolder.txt'
    filename_imglist_val = 'Diving48/val_videofolder.txt'
    filename_imglist_test = 'Diving48/val_videofolder.txt'
    prefix = 'frames{:06d}.jpg'

    return filename_imglist_train, filename_imglist_val, filename_imglist_test, root_data, prefix
    
def return_kinetics400():
    root_data = 'kinetics400'
    filename_imglist_train = 'kinetics400/train.txt'
    filename_imglist_val = 'kinetics400/val.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_meccano():
    root_data = 'meccano/frames'
    filename_imglist_train = 'meccano/train_videofolder.txt'
    filename_imglist_val = 'meccano/val_videofolder.txt'
    filename_imglist_test = 'meccano/test_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, filename_imglist_test, root_data, prefix

def return_meccano_inference():
    root_data = 'sample'
    filename_imglist_test = 'sample/inference_videofolder.txt'
    prefix = '{:05d}.jpg'

    return '', '', filename_imglist_test, root_data, prefix

def return_dataset(dataset, path):
    dict_single = {'something-v1': return_something_v1, 'something-v2': return_something_v2, 
                   'diving48':return_diving48, 'kinetics400': return_kinetics400, 'meccano': return_meccano, 'meccano-inference': return_meccano_inference}
    
    if dataset in dict_single:
            #file_imglist_train, file_imglist_val, filename_imglist_test, root_data, prefix = dict_single[dataset](split)
            file_imglist_train, file_imglist_val, file_imglist_test, root_data, prefix = dict_single[dataset]()
    else:
        raise ValueError('Unknown dataset '+dataset)
    
    file_imglist_train = os.path.join(path, file_imglist_train)
    file_imglist_val = os.path.join(path, file_imglist_val)
    file_imglist_test = os.path.join(path, file_imglist_test)
    root_data = os.path.join(path, root_data)

    return file_imglist_train, file_imglist_val, file_imglist_test, root_data, prefix
