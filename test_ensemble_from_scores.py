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
parser = argparse.ArgumentParser(description="GSF testing with saved logits")

parser.add_argument('--rgb_models', type=str)
parser.add_argument('--depth_models', type=str)

parser.add_argument('--test_labels_path', type=str)

parser.add_argument('--weight_rgb', type=float, default=0.5)

"""
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
"""
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

def accuracy_top5_hardvoting(true_labels, scores):
    """Computes the Top-5 Accuracy for Ensembles in an Hard Voting approach"""
    top5_predictions = np.argsort(scores, axis=-1)[:, :, -5:]

    true_labels_expanded = np.expand_dims(true_labels, axis=0)
    top5_correct = np.any(top5_predictions == true_labels_expanded[..., np.newaxis], axis=-1)
    top5_accuracy = np.mean(top5_correct)

    return top5_accuracy

def read_labels(file_path):
    last_columns = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            last_column = int(columns[-1])
            last_columns.append(last_column)
    return last_columns


# READING SCORES LIST
with open(args.rgb_models, 'r') as file:
    rgb_scores_paths = file.read().splitlines()

with open(args.depth_models, 'r') as file:
    depth_scores_paths = file.read().splitlines()

# Load the saved softmax scores
rgb_scores_list = [np.squeeze(np.load(path)) for path in rgb_scores_paths]
depth_scores_list = [np.squeeze(np.load(path)) for path in depth_scores_paths]

video_labels = np.array(read_labels(args.test_labels_path))

total_scores = []
total_avg_scores = []
total_gmean_scores = []

top1 = AverageMeter()
top5 = AverageMeter()

weight_depth = 1 - args.weight_rgb      

for score_rgb, score_depth in zip(rgb_scores_list, depth_scores_list):
    partial_rgb_score = []
    partial_depth_score = []
    partial_avg_scores = []
    partial_gmean_scores = []

    cnt = 1 

    for rst_rgb, rst_depth in zip(score_rgb, score_depth):
        
        rst_avg = (args.weight_rgb * rst_rgb + weight_depth * rst_depth) / (args.weight_rgb + weight_depth)

        rst_gmean = st.gmean(np.stack([rst_rgb, rst_depth]), axis=0)

        partial_rgb_score.append(rst_rgb)
        partial_depth_score.append(rst_depth)
        partial_avg_scores.append(rst_avg)
        partial_gmean_scores.append(rst_gmean)

        #prec1, prec5 = accuracy(torch.from_numpy(rst_avg).cuda(), torch.from_numpy(video_labels).cuda(), topk=(1, 5))
        #top1.update(prec1, 1)
        #top5.update(prec5, 1)
        

        #print('video {} done, total {}/{}, average {:.3f} sec/video, moving Acc@1 {:.3f} Acc@5 {:.3f}'.format(cnt, cnt+1,0,float(0) / (cnt+1), top1.avg, top5.avg))
        cnt += 1
        
    total_avg_scores.append(partial_avg_scores)
    total_gmean_scores.append(partial_gmean_scores)
    total_scores.append(partial_rgb_score)
    total_scores.append(partial_depth_score)


print("TOTAL SCORES:") #(4, 20, 1, 61) --> (4, 20, 61)
print(np.squeeze(np.array(total_scores)).shape)
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


print("video labels:")
print(video_labels.shape)
print("\n")

print("video preds avg:")
print(len(video_pred_avg))
print("\n")

print("video preds:")
print(len(video_pred))
print("\n")

print("video preds gmean:")
print(len(video_pred_gmean))
print("\n")

print("video preds gmean:")
print(len(video_pred_ens_gmean))
print("\n")

print("video preds HV avg:")
print(video_pred_hv_avg.shape)
print("\n")
print("video preds HV:")
print(video_pred_hv.shape)
print("\n")
print("video preds HV gmean:")
print(video_pred_hv_gmean.shape)
print("\n")

print('-----Evaluation of {} and {} is finished------'.format(args.rgb_models, args.rgb_models))

# Compute the overall accuracy averaging each pair of models
acc_1_avg = accuracy_score(video_labels, video_pred_avg)
acc_5_avg = top_k_accuracy_score(video_labels, avg_scores, k=5, labels=[x for x in range(61)])

# Compute the overall accuracy averaging each model independently
acc_1 = accuracy_score(video_labels, video_pred)
acc_5 = top_k_accuracy_score(video_labels, ensemble_scores, k=5, labels=[x for x in range(61)])

# Compute the overall accuracy with gmean over each pair of models
acc_1_gmean = accuracy_score(video_labels, video_pred_gmean)
acc_5_gmean = top_k_accuracy_score(video_labels, gmean_scores, k=5, labels=[x for x in range(61)])

# # Compute the overall accuracy with gmean over each model independently
acc_1_ens_gmean = accuracy_score(video_labels, video_pred_ens_gmean)
acc_5_ens_gmean = top_k_accuracy_score(video_labels, ensemble_scores_gmean, k=5, labels=[x for x in range(61)])

# Compute the overall accuracy Hard Voting pairs of models
acc_1_hv_avg = accuracy_score(video_labels, video_pred_hv_avg)
acc_5_hv_avg = accuracy_top5_hardvoting(video_labels, np.squeeze(np.array(total_avg_scores)))

# Compute the overall accuracy Hard Voting each model independently
acc_1_hv = accuracy_score(video_labels, video_pred_hv)
acc_5_hv = accuracy_top5_hardvoting(video_labels, np.squeeze(np.array(total_scores)))

# Compute the overall accuracy Hard Voting pairs of models with gmean
acc_1_hv_gmean = accuracy_score(video_labels, video_pred_hv_gmean)
acc_5_hv_gmean = accuracy_top5_hardvoting(video_labels, np.squeeze(np.array(total_gmean_scores)))



print('Overall Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1 * 100, acc_5 * 100))

print('Overall gmean Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_ens_gmean * 100, acc_5_ens_gmean * 100))

print('Overall Pairs Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_avg * 100, acc_5_avg * 100))

print('Overall Pairs (gmean) Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_gmean * 100, acc_5_gmean * 100))

print('Overall HV Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_hv * 100, acc_5_hv * 100))

print('Overall HV Pairs Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_hv_avg * 100, acc_5_hv_avg * 100))

print('Overall HV Pairs (gmean) Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_hv_gmean * 100, acc_5_hv_gmean * 100))

