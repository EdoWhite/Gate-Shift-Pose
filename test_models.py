import argparse
import time
import torchvision
from torch.cuda import amp
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.dataset import VideoDataset, VideoDatasetPoses, VideoDatasetPosesFast
from models import VideoModel, VideoModelLateFusion, VideoModelLateFusionFast, VideoModelLateFusionAttention
from utils.transforms import *
from ops import ConsensusModule
from utils import datasets_video
import pdb
from torch.nn import functional as F
import sys
import pickle as pkl
from torchmetrics.classification import BinaryAccuracy


# options
parser = argparse.ArgumentParser(
    description="GSF testing on the full validation set")
parser.add_argument('--dataset', type=str, choices=['something-v1', 'something-v2', 'diving48', 'kinetics400', 'meccano', 'FRFS'])
parser.add_argument('--dataset_path', type=str, default='./dataset')
parser.add_argument('--weights', type=str)

parser.add_argument('--early_fusion_poses', default=False, action="store_true")
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--rand_augment_4ch', default=False, action="store_true")
parser.add_argument('--rand_augment', default=False, action="store_true")
parser.add_argument('--late_fusion_poses', default=False, action="store_true")
parser.add_argument('--late_fusion_poses_fast', default=False, action="store_true")
parser.add_argument('--late_fusion_poses_attention', default=False, action="store_true")

#parser.add_argument('--split', type=str, default="val")
parser.add_argument('--arch', type=str, default="bninception")
parser.add_argument('--save_scores', default=False, action="store_true")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=0)
parser.add_argument('--consensus_type', type=str, default='avg')
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

#args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.split)
args.train_list, args.val_list, args.test_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path) #args.split

print(args.train_list, args.val_list, args.test_list)

if args.dataset == 'something-v1':
    num_class = 174
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"

elif args.dataset == 'something-v2':
    num_class = 174
    args.rgb_prefix = ''
    rgb_read_format = "{:06d}.jpg"

elif args.dataset == 'diving48':
    num_class = 48
    args.rgb_prefix = 'frames'
    rgb_read_format = "{:05d}.jpg"

elif args.dataset == 'kinetics400':
    num_class = 400
    args.rgb_prefix = 'img_{:05d}.jpg'
    rgb_read_format = ''

# MECCANO dataset
elif args.dataset == 'meccano':
    num_class = 61
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"
    
# FRFS dataset
elif args.dataset == 'FRFS':
    num_class = 2
    args.rgb_prefix = ''
    args.pose_prefix = ''
    rgb_read_format = "{:05d}.jpg"
    args.prec25 = 2

else:
    raise ValueError('Unknown dataset '+args.dataset)

# Load the base model
if args.late_fusion_poses:
    net = VideoModelLateFusion(num_class=num_class, num_segments=args.num_segments,
                    base_model=args.arch, consensus_type=args.consensus_type,
                    gsf=args.gsf, gsf_ch_ratio=args.gsf_ch_ratio, num_channels=args.num_channels)
    
elif args.late_fusion_poses_fast:
    net = VideoModelLateFusionFast(num_class=num_class, num_segments=args.num_segments,
                    base_model=args.arch, consensus_type=args.consensus_type,
                    gsf=args.gsf, gsf_ch_ratio=args.gsf_ch_ratio, num_channels=args.num_channels)
    
elif args.late_fusion_poses_attention:
    net = VideoModelLateFusionAttention(num_class=num_class, num_segments=args.num_segments,
            base_model=args.arch, consensus_type=args.consensus_type,
            gsf=args.gsf, gsf_ch_ratio=args.gsf_ch_ratio, num_channels=args.num_channels)
    
else:
    net = VideoModel(num_class=num_class, num_segments=args.num_segments,
                    base_model=args.arch, consensus_type=args.consensus_type,
                    gsf=args.gsf, gsf_ch_ratio=args.gsf_ch_ratio, num_channels=args.num_channels)

checkpoint = torch.load(args.weights)

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['model_state_dict'].items())}
net.load_state_dict(base_dict, strict=True)

if args.input_size == 0:
    input_size = net.input_size
    scale_size = net.scale_size
else:
    input_size = args.input_size
    scale_size = net.scale_size
    
print("\n")
print("Test Clips: {}".format(args.num_clips))
print("Test Crops: {}".format(args.test_crops))
print("Test Segments: {}".format(args.num_segments))
print("\n")

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, scale_size)
    ])
elif args.test_crops == 3:  # do not flip, so only 3 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 5:
    cropping = torchvision.transforms.Compose([
        GroupFiveCrops(input_size, scale_size)
    ])
else:
    raise ValueError("Unsupported number of test crops: {}".format(args.test_crops))

# Load the dataset
if args.early_fusion_poses:
    print("Uses Early-Fusion Poses")

    data_loader = torch.utils.data.DataLoader(
        VideoDatasetPoses(args.root_path, args.val_list, num_segments=args.num_segments,
                    image_tmpl=args.rgb_prefix+rgb_read_format, test_mode=True, pose_prefix=args.pose_prefix,
                    transform=torchvision.transforms.Compose([cropping,
                                                            Stack(roll=(args.arch in ['bninception','inceptionv3'])),
                                                            ToTorchFormatTensor(),
                                                            GroupNormalize(net.input_mean, net.input_std, div=(args.arch not in ['bninception', 'inceptionv3'])),]), 
                    num_clips=args.num_clips, mode="val"),
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    
elif args.late_fusion_poses_fast or args.late_fusion_poses_attention:
    print("Uses Late-Fusion Poses Fast or Late-Fusion Poses Attention")

    data_loader = torch.utils.data.DataLoader(
        VideoDatasetPosesFast(args.root_path, args.val_list, num_segments=args.num_segments,
                    image_tmpl=args.rgb_prefix+rgb_read_format, test_mode=True, pose_prefix=args.pose_prefix,
                    transform=torchvision.transforms.Compose([cropping,
                                                            Stack(roll=(args.arch in ['bninception','inceptionv3'])),
                                                            ToTorchFormatTensor(),
                                                            GroupNormalize(net.input_mean, net.input_std, div=(args.arch not in ['bninception', 'inceptionv3'])),]), 
                    num_clips=args.num_clips, mode="val"),
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    
else:
    data_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.test_list, num_segments=args.test_segments,
                                                        image_tmpl=args.rgb_prefix+rgb_read_format, test_mode=True,
                                                        transform=torchvision.transforms.Compose([cropping,
                                                                                                    Stack(roll=(args.arch in ['bninception','inceptionv3'])),
                                                                                                    ToTorchFormatTensor(),
                                                                                                    GroupNormalize(net.input_mean, net.input_std, div=(args.arch not in ['bninception', 'inceptionv3'])),]),
                                                        num_clips=args.num_clips, mode="val"),
                                            batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

net = net.cuda()
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)

def eval_video(video_data):
    i, data, label = video_data

    if args.early_fusion_poses:
        length = 4
    else:
        length = 3
        
    if args.num_clips > 1:
        input_var = data.view(-1, length, data.size(3), data.size(4)).cuda()
    else:
        input_var = data.view(-1, length, data.size(2), data.size(3)).cuda()
        
    with amp.autocast(enabled=args.with_amp):
        rst = net(input_var, with_amp=args.with_amp, idx=i, target=label)
        
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        rst = F.softmax(rst)

    rst = rst.reshape(-1, 1, num_class)
    rst = torch.mean(rst, dim=0, keepdim=False).data.cpu().numpy()
    label = label[0]

    return i, rst, label


proc_start_time = time.time()

output_scores = np.zeros((total_num, num_class))
video_labels = np.zeros(total_num)
top1 = AverageMeter()
top1_tm = AverageMeter()
top5 = AverageMeter()

bin_acc = BinaryAccuracy()

with torch.no_grad():
    for i, (data, label) in data_gen:
        rst = eval_video((i, data, label))
        video_labels[i] = rst[2]
        cnt_time = time.time() - proc_start_time
        output_scores[i] = rst[1]
        prec1, prec5 = accuracy(torch.from_numpy(rst[1]).cuda(), label.cuda(), topk=(1, 2))
        top1.update(prec1, 1)
        top5.update(prec5, 1)
        
        acc = bin_acc(torch.from_numpy(rst[1]).cuda(), label.cuda())
        top1_tm.update(acc.item(), 1)
        
        print('video {} done, total {}/{}, average {:.3f} sec/video, moving Acc@1 {:.3f} Acc@1_tm {:.3f}'.format(i, i+1,
                                                                        total_num,
                                                                        float(cnt_time) / (i+1), top1.avg, top1_tm.avg))

video_pred = [np.argmax(x) for x in output_scores]
cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit / cls_cnt
print('-----Evaluation of {} is finished------'.format(args.weights))
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Acc@1 {:.02f}% Acc@1_tm {:.02f}%'.format(top1.avg, top1_tm.avg))

acc = accuracy_score(video_labels, video_pred)
print('Class Accuracy SKlearn {:.02f}%'.format(acc * 100))


if args.save_scores:
    save_name = args.weights[:-8] + '_clips_' + str(args.num_clips) + '_crops_' + str(args.test_crops) + '.pkl'
    np.savez(save_name, scores=output_scores, labels=video_labels, predictions=np.array(video_pred), cf=cf)
