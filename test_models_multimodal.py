import argparse
import time
import torchvision
from torch.cuda import amp
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
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
    description="GSF testing on the full validation set")
parser.add_argument('--dataset', type=str, choices=['something-v1', 'something-v2', 'diving48', 'kinetics400', 'meccano'])

parser.add_argument('--dataset_path_rgb', type=str, default='./dataset')
parser.add_argument('--dataset_path_depth', type=str, default='./dataset_depth')

parser.add_argument('--checkpoint_rgb', type=str)
parser.add_argument('--checkpoint_depth', type=str)

parser.add_argument('--weight_rgb', type=str)

parser.add_argument('--test_segments_rgb', type=int, default=8)
parser.add_argument('--test_segments_depth', type=int, default=32)

#parser.add_argument('--split', type=str, default="val")
parser.add_argument('--arch', type=str, default="bninception")
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


args.train_list, args.val_list, args.test_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path_rgb)
args.train_list, args.val_list, args.test_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path_depth)

print(args.train_list, args.val_list, args.test_list)

# MECCANO dataset
if args.dataset == 'meccano':
    num_class = 61
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"

else:
    raise ValueError('Unknown dataset '+args.dataset)

# RGB
net_rgb = VideoModel(num_class=num_class, num_segments=args.test_segments_rgb, base_model=args.arch,
                 consensus_type=args.crop_fusion_type, gsf=args.gsf, gsf_ch_ratio = args.gsf_ch_ratio)
checkpoint_rgb = torch.load(args.checkpoint_rgb)
base_dict_rgb = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint_rgb['model_state_dict'].items())}
net_rgb.load_state_dict(base_dict_rgb, strict=True)


# DEPTH
net_depth = VideoModel(num_class=num_class, num_segments=args.test_segments_depth, base_model=args.arch,
                 consensus_type=args.crop_fusion_type, gsf=args.gsf, gsf_ch_ratio = args.gsf_ch_ratio)
checkpoint_depth = torch.load(args.checkpoint_depth)
base_dict_depth = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint_depth['model_state_dict'].items())}
net_depth.load_state_dict(base_dict_depth, strict=True)

# RGB
if args.input_size == 0:
    input_size_rgb = net_rgb.input_size
    scale_size_rgb = net_rgb.scale_size
else:
    input_size_rgb = args.input_size
    scale_size_rgb = net_rgb.scale_size

# DEPTH
if args.input_size == 0:
    input_size_depth = net_depth.input_size
    scale_size_depth = net_depth.scale_size
else:
    input_size_depth = args.input_size
    scale_size_depth = net_depth.scale_size


# RGB
if args.test_crops == 1:
    cropping_rgb = torchvision.transforms.Compose([
        GroupScale(scale_size_rgb),
        GroupCenterCrop(input_size_rgb),
    ])
elif args.test_crops == 10:
    cropping_rgb = torchvision.transforms.Compose([
        GroupOverSample(input_size_rgb, scale_size_rgb)
    ])
elif args.test_crops == 3:  # do not flip, so only 3 crops
    cropping_rgb = torchvision.transforms.Compose([
        GroupFullResSample(input_size_rgb, net_rgb.scale_size, flip=False)
    ])
elif args.test_crops == 5:
    cropping_rgb = torchvision.transforms.Compose([
        GroupFiveCrops(input_size_rgb, scale_size_rgb)
    ])
else:
    raise ValueError("Unsupported number of test crops: {}".format(args.test_crops))

# DEPTH
if args.test_crops == 1:
    cropping_depth = torchvision.transforms.Compose([
        GroupScale(scale_size_depth),
        GroupCenterCrop(input_size_depth),
    ])
elif args.test_crops == 10:
    cropping_depth = torchvision.transforms.Compose([
        GroupOverSample(input_size_depth, input_size_depth)
    ])
elif args.test_crops == 3:  # do not flip, so only 3 crops
    cropping_depth = torchvision.transforms.Compose([
        GroupFullResSample(input_size_depth, net_depth.scale_size, flip=False)
    ])
elif args.test_crops == 5:
    cropping_depth = torchvision.transforms.Compose([
        GroupFiveCrops(input_size_depth, input_size_depth)
    ])
else:
    raise ValueError("Unsupported number of test crops: {}".format(args.test_crops))




# RGB
data_loader_rgb = torch.utils.data.DataLoader(VideoDataset(args.root_path, args.test_list, num_segments=args.test_segments_rgb,
                                                       image_tmpl=args.rgb_prefix+rgb_read_format, test_mode=True,
                                                       transform=torchvision.transforms.Compose([cropping_rgb,
                                                                                                 Stack(roll=(args.arch in ['bninception','inceptionv3'])),
                                                                                                 ToTorchFormatTensor(),
                                                                                                 GroupNormalize(net_rgb.input_mean, net_rgb.input_std, div=(args.arch not in ['bninception', 'inceptionv3'])),]),
                                                       num_clips=args.num_clips, mode="val"),
                                          batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

# DEPTH
data_loader_depth = torch.utils.data.DataLoader(VideoDataset(args.root_path, args.test_list, num_segments=args.test_segments_depth,
                                                       image_tmpl=args.rgb_prefix+rgb_read_format, test_mode=True,
                                                       transform=torchvision.transforms.Compose([cropping_depth,
                                                                                                 Stack(roll=(args.arch in ['bninception','inceptionv3'])),
                                                                                                 ToTorchFormatTensor(),
                                                                                                 GroupNormalize(net_depth.input_mean, net_depth.input_std, div=(args.arch not in ['bninception', 'inceptionv3'])),]),
                                                       num_clips=args.num_clips, mode="val"),
                                          batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

# RGB
net_rgb = net_rgb.cuda()
net_rgb.eval()

# DEPTH
net_depth = net_depth.cuda()
net_depth.eval()


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


# RGB
data_gen_rgb = enumerate(data_loader_rgb)
total_num_rgb = len(data_loader_rgb.dataset)

# DEPTH
data_gen_depth = enumerate(data_loader_depth)
total_num_depth = len(data_loader_depth.dataset)

print('total_num_rgb: ' + str(total_num_rgb) + ' ' + 'total_num_depth: ' + str(total_num_depth))

proc_start_time = time.time()

# total_num_rgb = total_num_depth
output_scores = np.zeros((total_num_rgb, num_class))
video_labels = np.zeros(total_num_rgb)
top1 = AverageMeter()
top5 = AverageMeter()

weight_depth = 1 - args.weight_rgb

with torch.no_grad():
    for (i, (data_rgb, label_rgb)), (j, (data_depth, label_depth)) in zip(data_gen_rgb, data_gen_depth):
        rst_rgb = eval_video((i, data_rgb, label_rgb), net_rgb)
        rst_depth = eval_video((j, data_depth, label_depth), net_depth)

        #rst_avg = (rst_rgb[1] + rst_depth[1]) / 2.0

        rst_avg = (args.weight_rgb * rst_rgb[1] + weight_depth * rst_depth[1]) / (args.weight_rgb + weight_depth)

        video_labels[i] = rst_rgb[2]

        cnt_time = time.time() - proc_start_time
        output_scores[i] = rst_avg

        prec1, prec5 = accuracy(torch.from_numpy(rst_avg).cuda(), label_rgb.cuda(), topk=(1, 5))
        top1.update(prec1, 1)
        top5.update(prec5, 1)
        print('video {} done, total {}/{}, average {:.3f} sec/video, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                        total_num_rgb,
                                                                        float(cnt_time) / (i+1), top1.avg, top5.avg))

video_pred = [np.argmax(x) for x in output_scores]
cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit / cls_cnt
print('-----Evaluation of {} and {} is finished------'.format(args.checkpoint_rgb, args.checkpoint_depth))
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(top1.avg, top5.avg))


if args.save_scores:
    save_name = args.weights[:-8] + '_clips_' + str(args.num_clips) + '_crops_' + str(args.test_crops) + '.pkl'
    np.savez(save_name, scores=output_scores, labels=video_labels, predictions=np.array(video_pred), cf=cf)
