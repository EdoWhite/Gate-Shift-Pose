import argparse
import time
import torchvision
from torch.cuda import amp
import numpy as np
from scipy import stats as st
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
from utils.dataset import VideoDataset
from models import VideoModel
from utils.transforms import *
from ops import ConsensusModule
from utils import datasets_video
import pdb
from torch.nn import functional as F
import sys
import pickle as pkl
import os


# options
parser = argparse.ArgumentParser(
    description="GSF testing on the full test set")
parser.add_argument('--dataset', type=str, choices=['something-v1', 'something-v2', 'diving48', 'kinetics400', 'meccano', 'meccano-inference'])

parser.add_argument('--dataset_path_rgb', type=str, default='./dataset')
parser.add_argument('--dataset_path_depth', type=str, default='./dataset_depth')

parser.add_argument('--rgb_models', type=str)
parser.add_argument('--depth_models', type=str)

parser.add_argument('--weight_rgb', type=float, default=0.5)

parser.add_argument('--test_crops', type=int, default=3)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=0)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--num_clips',type=int, default=2,help='Number of clips sampled from a video')
parser.add_argument('--softmax', type=int, default=0)
parser.add_argument('--gsf', default=True, action="store_true")
parser.add_argument('--gsf_ch_ratio', default=100, type=float)
parser.add_argument('--with_amp', default=True, action="store_true")
parser.add_argument('--frame_interval', type=int, default=5)

args = parser.parse_args()

_, _, args.test_list, args.root_path_rgb, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path_rgb)
_, _, args.test_list, args.root_path_depth, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path_depth)

print(args.test_list)

# MECCANO dataset
if args.dataset == 'meccano-inference':
    num_class = 61
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"

else:
    raise ValueError('Unknown dataset '+args.dataset)


# READING MODELS LIST
rgb_models = []
depth_models = []

with open(args.rgb_models, 'r') as file:
    for line in file:
        model, backbone, num_segments = line.split(" ")
        rgb_models.append((model, backbone, int(num_segments)))

with open(args.depth_models, 'r') as file:
    for line in file:
        model, backbone, num_segments = line.split(" ")
        depth_models.append((model, backbone, int(num_segments)))


# CREATE NETS and DATALOADERS
net_rgb_list = []
net_depth_list = []

data_loader_rgb_list = []
data_loader_depth_list = []

for rgb_model, depth_model in zip(rgb_models, depth_models):
    # RGB
    net_rgb = VideoModel(num_class=num_class, num_segments=rgb_model[2], base_model=rgb_model[1],
                    consensus_type=args.crop_fusion_type, gsf=args.gsf, gsf_ch_ratio = args.gsf_ch_ratio)
    
    checkpoint_rgb = torch.load(rgb_model[0])
    base_dict_rgb = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint_rgb['model_state_dict'].items())}
    net_rgb.load_state_dict(base_dict_rgb, strict=True)
    net_rgb_list.append(net_rgb)

    # DEPTH
    net_depth = VideoModel(num_class=num_class, num_segments=depth_model[2], base_model=depth_model[1],
                    consensus_type=args.crop_fusion_type, gsf=args.gsf, gsf_ch_ratio = args.gsf_ch_ratio)
    
    checkpoint_depth = torch.load(depth_model[0])
    base_dict_depth = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint_depth['model_state_dict'].items())}
    net_depth.load_state_dict(base_dict_depth, strict=True)
    net_depth_list.append(net_depth)

    # set input_size and scale_size
    if args.input_size == 0:
        input_size_rgb = net_rgb.input_size
        scale_size_rgb = net_rgb.scale_size
        input_size_depth = net_depth.input_size
        scale_size_depth = net_depth.scale_size
    else:
        input_size_rgb = args.input_size
        scale_size_rgb = net_rgb.scale_size
        input_size_depth = args.input_size
        scale_size_depth = net_depth.scale_size

    # set cropping strategy
    if args.test_crops == 1:
        cropping_rgb = torchvision.transforms.Compose([
            GroupScale(scale_size_rgb),
            GroupCenterCrop(input_size_rgb)])
        cropping_depth = torchvision.transforms.Compose([
            GroupScale(scale_size_depth),
            GroupCenterCrop(input_size_depth)])
        
    elif args.test_crops == 10:
        cropping_rgb = torchvision.transforms.Compose([
            GroupOverSample(input_size_rgb, scale_size_rgb)])
        cropping_depth = torchvision.transforms.Compose([
            GroupOverSample(input_size_depth, input_size_depth)])
        
    elif args.test_crops == 3:  # do not flip, so only 3 crops
        cropping_rgb = torchvision.transforms.Compose([
            GroupFullResSample(input_size_rgb, net_rgb.scale_size, flip=False)])
        cropping_depth = torchvision.transforms.Compose([
            GroupFullResSample(input_size_depth, net_depth.scale_size, flip=False)])
        
    elif args.test_crops == 5:
        cropping_rgb = torchvision.transforms.Compose([
            GroupFiveCrops(input_size_rgb, scale_size_rgb)])
        cropping_depth = torchvision.transforms.Compose([
            GroupFiveCrops(input_size_depth, input_size_depth)])
    else:
        raise ValueError("Unsupported number of test crops: {}".format(args.test_crops))
    
    # set dataloadrs
    data_loader_rgb = torch.utils.data.DataLoader(
            VideoDataset(
                args.root_path_rgb,
                args.test_list,
                num_segments=rgb_model[2],
                image_tmpl=args.rgb_prefix + rgb_read_format,
                test_mode=True,
                transform=torchvision.transforms.Compose([
                    cropping_rgb,
                    Stack(roll=(rgb_model[1] in ['bninception', 'inceptionv3'])),
                    ToTorchFormatTensor(),
                    GroupNormalize(net_rgb.input_mean, net_rgb.input_std,
                                    div=(rgb_model[1] not in ['bninception', 'inceptionv3']))
                ]),
                num_clips=args.num_clips,
                mode="val"
            ),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    data_loader_rgb_list.append(data_loader_rgb)

    data_loader_depth = torch.utils.data.DataLoader(
            VideoDataset(
                args.root_path_depth,
                args.test_list,
                num_segments=depth_model[2],
                image_tmpl=args.rgb_prefix + rgb_read_format,
                test_mode=True,
                transform=torchvision.transforms.Compose([
                    cropping_depth,
                    Stack(roll=(depth_model[1] in ['bninception', 'inceptionv3'])),
                    ToTorchFormatTensor(),
                    GroupNormalize(net_depth.input_mean, net_depth.input_std,
                                    div=(depth_model[1] not in ['bninception', 'inceptionv3']))
                ]),
                num_clips=args.num_clips,
                mode="val"
            ),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    data_loader_depth_list.append(data_loader_depth)
    
# RGB
net_rgb_list = [net.cuda() for net in net_rgb_list]
# DEPTH
net_depth_list = [net.cuda() for net in net_depth_list]

for net_rgb, net_depth in zip(net_rgb_list, net_depth_list):
    net_rgb.eval()
    net_depth.eval()

def eval_video(video_data, model):
    i, data, _ = video_data

    length = 3
    if args.num_clips > 1:
        input_var = data.view(-1, length, data.size(3), data.size(4)).cuda()
    else:
        input_var = data.view(-1, length, data.size(2), data.size(3)).cuda()
    with amp.autocast(enabled=args.with_amp):
        rst = model(input_var, with_amp=args.with_amp, idx=i)
        
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        rst = F.softmax(rst)

    rst = rst.reshape(-1, 1, num_class)
    rst = torch.mean(rst, dim=0, keepdim=False).data.cpu().numpy()

    return i, rst, _

total_num_rgb = len(data_loader_rgb_list[0].dataset) # all have the same len

proc_start_time = time.time()

total_scores = []
total_avg_scores = []
total_gmean_scores = []

total_logits_rgb = []
total_logits_depth = []

weight_depth = 1 - args.weight_rgb   
cnt = 1     

with torch.no_grad():
    for data_gen_rgb, data_gen_depth, net_rgb, net_depth in zip(data_loader_rgb_list, data_loader_depth_list, net_rgb_list, net_depth_list):
        data_gen_rgb = enumerate(data_gen_rgb)
        data_gen_depth = enumerate(data_gen_depth)
        partial_rgb_score = []
        partial_depth_score = []
        partial_avg_scores = []
        partial_gmean_scores = []

        partial_logits_rgb = []
        partial_logits_depth = []

        for (j, (data_rgb, label_rgb)), (k, (data_depth, label_depth)) in zip(data_gen_rgb, data_gen_depth):

            cnt_time = time.time() - proc_start_time

            rst_rgb = eval_video((j, data_rgb, label_rgb), net_rgb)
            rst_depth = eval_video((k, data_depth, label_depth), net_depth)

            # SAVE SCORES
            partial_logits_rgb.append(rst_rgb[1])
            partial_logits_depth.append(rst_depth[1])
            #

            rst_avg = (args.weight_rgb * rst_rgb[1] + weight_depth * rst_depth[1]) / (args.weight_rgb + weight_depth)

            rst_gmean = st.gmean(np.stack([rst_rgb[1], rst_depth[1]]), axis=0)

            partial_rgb_score.append(rst_rgb[1])
            partial_depth_score.append(rst_depth[1])
            partial_avg_scores.append(rst_avg)
            partial_gmean_scores.append(rst_gmean)
            

            print('video {} done, total {}/{}, average {:.3f} sec/video'.format(j, j+1,
                                                                            total_num_rgb,
                                                                            float(cnt_time) / (j+1)))
            
        total_avg_scores.append(partial_avg_scores)
        total_gmean_scores.append(partial_gmean_scores)
        total_scores.append(partial_rgb_score)
        total_scores.append(partial_depth_score)
        
        # ALL TOGETHER
        total_logits_rgb.append(partial_logits_rgb)
        total_logits_depth.append(partial_logits_depth)


print("TOTAL SCORES:") #(4, 20, 1, 61) --> (4, 20, 61)
print(np.squeeze(np.array(total_scores)).shape)
print('\n')
print(total_scores)
print('\n')
print(np.squeeze(np.array(total_scores)))
print("###############################################\n")

print("TOTAL SCORES PAIRS:") #(2, 20, 1, 61) --> (2, 20, 61)
print(np.squeeze(np.array(total_avg_scores)).shape)
print("###############################################\n")


avg_scores = np.squeeze(np.mean(np.array(total_avg_scores), axis=0))
ensemble_scores = np.squeeze(np.mean(np.array(total_scores), axis=0))
ensemble_scores_gmean = np.squeeze(st.gmean(np.array(total_scores), axis=0))
gmean_scores = np.squeeze(st.gmean(np.array(total_gmean_scores), axis=0))

hard_preds_avg = np.argmax(np.squeeze(np.array(total_avg_scores)), axis=-1)
video_pred_hv_avg = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=hard_preds_avg)

hard_preds = np.argmax(np.squeeze(np.array(total_scores)), axis=-1)
video_pred_hv = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=hard_preds)

hard_preds_gmean = np.argmax(np.squeeze(np.array(total_gmean_scores)), axis=-1)
video_pred_hv_gmean = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=hard_preds_gmean)


print("TOTAL AVG SCORES PAIRS:") #(20, 61)
print(avg_scores.shape)
print("###############################################\n")

print("TOTAL AVG SCORES:") #(20, 61)
print(ensemble_scores.shape)
print("###############################################\n")

print("TOTAL GMEAN SCORES:") #(20, 61)
print(gmean_scores.shape)
print("###############################################\n")


video_pred_avg = [np.argmax(x) for x in avg_scores]
video_pred = [np.argmax(x) for x in ensemble_scores]
video_pred_gmean = [np.argmax(x) for x in gmean_scores]
video_pred_ens_gmean = [np.argmax(x) for x in ensemble_scores_gmean]

print("video preds avg:")
print(video_pred_avg)
print("\n")

print("video preds:")
print(video_pred)
print("\n")

print("video preds gmean:")
print(video_pred_gmean)
print("\n")

print("video preds gmean:")
print(video_pred_ens_gmean)
print("\n")

print("video preds HV avg:")
print(video_pred_hv_avg)
print("\n")
print("video preds HV:")
print(video_pred_hv)
print("\n")
print("video preds HV gmean:")
print(video_pred_hv_gmean)
print("\n")

print('-----Evaluation of {} and {} is finished------'.format(args.rgb_models, args.rgb_models))

