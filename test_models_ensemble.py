import argparse
import time
import torchvision
from torch.cuda import amp
import numpy as np
import torch.nn.parallel
import torch.optim
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.dataset import VideoDataset
from models import VideoModel
from utils.transforms import *
from ops import ConsensusModule
from utils import datasets_video
import pdb
from torch.nn import functional as F
import sys
import pickle as pkl


# options
parser = argparse.ArgumentParser(
    description="GSF testing on the full test set")
parser.add_argument('--dataset', type=str, choices=['something-v1', 'something-v2', 'diving48', 'kinetics400', 'meccano'])

parser.add_argument('--dataset_path_rgb', type=str, default='./dataset')
parser.add_argument('--dataset_path_depth', type=str, default='./dataset_depth')

parser.add_argument('--rgb_models', type=str)
parser.add_argument('--depth_models', type=str)
parser.add_argument('--weight_rgb', type=float, default=0.5)
#parser.add_argument('--hard_voting', default=False, action="store_true")
parser.add_argument('--save_scores', default=False, action="store_true")
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=0)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--num_clips',type=int, default=1,help='Number of clips sampled from a video')
parser.add_argument('--softmax', type=int, default=0)
parser.add_argument('--gsf', default=False, action="store_true")
parser.add_argument('--gsf_ch_ratio', default=100, type=float)
parser.add_argument('--with_amp', default=False, action="store_true")
parser.add_argument('--frame_interval', type=int, default=5)

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


args.train_list, args.val_list, args.test_list, args.root_path_rgb, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path_rgb)
args.train_list, args.val_list, args.test_list, args.root_path_depth, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path_depth)

print(args.train_list, args.val_list, args.test_list)

# MECCANO dataset
if args.dataset == 'meccano':
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


# EVAL MULTIPLE MODELS on same loader!
def eval_ensemble_video(video_data, models):
    i, data, label = video_data
    ensembled_result = np.zeros((len(models), 1, num_class))

    length = 3
    if args.num_clips > 1:
        input_var = data.view(-1, length, data.size(3), data.size(4)).cuda()
    else:
        input_var = data.view(-1, length, data.size(2), data.size(3)).cuda()

    if args.hard_voting == False:
        with amp.autocast(enabled=args.with_amp):
            for model in models:
                rst = model(input_var, with_amp=args.with_amp, idx=i, target=label)
                if args.softmax == 1:
                    # take the softmax to normalize the output to probability
                    rst = F.softmax(rst)

                rst = rst.reshape(-1, 1, num_class)
                ensembled_result += rst

        ensembled_result /= len(models)
        ensembled_result = ensembled_result.data.cpu().numpy()
        label = label[0]

    else:
        ensemble_predictions = []
        with amp.autocast(enabled=args.with_amp):
            for model in models:
                rst = model(input_var, with_amp=args.with_amp, idx=i, target=label)
                if args.softmax == 1:
                    # take the softmax to normalize the output to probability
                    rst = F.softmax(rst)

                rst = rst.reshape(-1, 1, num_class)
                ensemble_predictions.append(rst)

        ensemble_predictions = torch.cat(ensemble_predictions, dim=1)  # Shape: (batch_size, num_models, num_class)
        ensemble_predictions = torch.argmax(ensemble_predictions, dim=2)  # Shape: (batch_size, num_models)
        ensembled_result = torch.mode(ensemble_predictions, dim=1).values  # Shape: (batch_size,)

    return i, ensembled_result, label

def eval_video(video_data, model):
    i, data, label = video_data

    length = 3
    if args.num_clips > 1:
        input_var = data.view(-1, length, data.size(3), data.size(4)).cuda()
    else:
        input_var = data.view(-1, length, data.size(2), data.size(3)).cuda()
    with amp.autocast(enabled=args.with_amp):
        rst = model(input_var, with_amp=args.with_amp, idx=i, target=label)
        
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        rst = F.softmax(rst)

    rst = rst.reshape(-1, 1, num_class)
    rst = torch.mean(rst, dim=0, keepdim=False).data.cpu().numpy()
    label = label[0]

    return i, rst, label

#data_gen_rgb_list = [enumerate(data_loader_rgb) for data_loader_rgb in data_loader_rgb_list]
#data_gen_depth_list = [enumerate(data_loader_depth) for data_loader_depth in data_loader_depth_list]
total_num_rgb = len(data_loader_rgb_list[0].dataset) # all have the same len

proc_start_time = time.time()

#ensemble_scores = np.zeros((total_num_rgb, num_class))
#total_scores = np.zeros((total_num_rgb, num_class))
ensemble_scores = []
total_scores = []

num_preds = 0
video_labels = np.zeros(total_num_rgb)
top1 = AverageMeter()
top5 = AverageMeter()

weight_depth = 1 - args.weight_rgb        

# fix that ensemble_scores and total_scores reset each time new models are tested!
with torch.no_grad():
    for data_gen_rgb, data_gen_depth, net_rgb, net_depth in zip(data_loader_rgb_list, data_loader_depth_list, net_rgb_list, net_depth_list):
        data_gen_rgb = enumerate(data_gen_rgb)
        data_gen_depth = enumerate(data_gen_depth)
        partial_rgb_score = []
        partial_depth_score = []
        for (j, (data_rgb, label_rgb)), (k, (data_depth, label_depth)) in zip(data_gen_rgb, data_gen_depth):

            rst_rgb = eval_video((j, data_rgb, label_rgb), net_rgb)
            rst_depth = eval_video((k, data_depth, label_depth), net_depth)

            rst_avg = (rst_rgb[1] + rst_depth[1]) / 2.0

            video_labels[j] = rst_rgb[2]

            cnt_time = time.time() - proc_start_time
            #ensemble_scores[j] = ensemble_scores[j] + rst_avg
            ensemble_scores.append(rst_avg)
            num_preds += 1.0

            temp = rst_rgb[1] + rst_depth[1]
            #total_scores[j] = total_scores[j] + temp
            #total_scores.append(temp)

            partial_rgb_score.append(rst_rgb[1])
            partial_depth_score.append(rst_depth[1])

            prec1, prec5 = accuracy(torch.from_numpy(rst_rgb[1]).cuda(), label_rgb.cuda(), topk=(1, 5))
            top1.update(prec1, 1)
            top5.update(prec5, 1)

            prec1, prec5 = accuracy(torch.from_numpy(rst_depth[1]).cuda(), label_depth.cuda(), topk=(1, 5))
            top1.update(prec1, 1)
            top5.update(prec5, 1)


            print('video {} done, total {}/{}, average {:.3f} sec/video, moving Acc@1 {:.3f} Acc@5 {:.3f}'.format(j, j+1,
                                                                            total_num_rgb,
                                                                            float(cnt_time) / (j+1), top1.avg, top5.avg))
        total_scores.append(partial_rgb_score)
        total_scores.append(partial_depth_score)


total_avg_scores = np.mean(np.array(total_scores), axis=0)
ensemble_scores = np.mean(np.array(ensemble_scores), axis=0)

video_pred = [np.argmax(x) for x in total_avg_scores]
print("video labels:")
print(video_labels)
print("video preds:")
print(video_pred)


print('-----Evaluation of {} and {} is finished------'.format(args.rgb_models, args.rgb_models))

# Compute the overall accuracy
acc = accuracy_score(video_labels, video_pred)
print('Overall Accuracy SKlearn {:.02f}%'.format(acc * 100))
print('Overall Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(top1.avg, top5.avg))

#total1, total5 = accuracy(torch.from_numpy(total_avg_scores).cuda(), torch.from_numpy(video_labels), topk=(1, 5))

accuracy_1 = Accuracy(task="multiclass", num_classes=61, top_k=1).cuda()
accuracy_5 = Accuracy(task="multiclass", num_classes=61, top_k=5).cuda()

total1 = accuracy_1(torch.from_numpy(total_avg_scores).cuda(), torch.from_numpy(video_labels).cuda())
total5 = accuracy_5(torch.from_numpy(total_avg_scores).cuda(), torch.from_numpy(video_labels).cuda())

print('Total Average Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(total1, total5))


if args.save_scores:
    save_name = args.checkpoint_rgb[:-8] + args.checkpoint_depth[:-8] + '_clips_' + str(args.num_clips) + '_crops_' + str(args.test_crops) + '.pkl'
    np.savez(save_name, scores=total_avg_scores, labels=video_labels, predictions=np.array(video_pred), cf=cf)

