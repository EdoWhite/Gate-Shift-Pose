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

parser = argparse.ArgumentParser(description="GSF testing with saved logits")

parser.add_argument('--rgb_models', type=str)
parser.add_argument('--depth_models', type=str)
parser.add_argument('--test_labels', type=str)
parser.add_argument('--weight_rgb', type=float, default=0.5)
args = parser.parse_args()

def accuracy_top5_hardvoting(true_labels, scores):
    """Computes the Top-5 Accuracy for Ensembles in an Hard Voting approach"""
    top5_predictions = np.argsort(scores, axis=-1)[:, :, -5:]

    true_labels_expanded = np.expand_dims(true_labels, axis=0)
    top5_correct = np.any(top5_predictions == true_labels_expanded[..., np.newaxis], axis=-1)
    top5_accuracy = np.mean(top5_correct)

    return top5_accuracy

# READING SCORES LIST
with open(args.rgb_models, 'r') as file:
    rgb_scores_paths = file.read().splitlines()

with open(args.depth_models, 'r') as file:
    depth_scores_paths = file.read().splitlines()

# Load the saved softmax scores
rgb_scores_list = [np.load(path) for path in rgb_scores_paths]
depth_scores_list = [np.load(path) for path in depth_scores_paths]

video_labels = np.load(args.test_labels)

total_scores = []
total_avg_scores = []
total_gmean_scores = []

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

        cnt += 1
        
    total_avg_scores.append(partial_avg_scores)
    total_gmean_scores.append(partial_gmean_scores)
    total_scores.append(partial_rgb_score)
    total_scores.append(partial_depth_score)


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


video_pred_avg = [np.argmax(x) for x in avg_scores]
video_pred = [np.argmax(x) for x in ensemble_scores]
video_pred_gmean = [np.argmax(x) for x in gmean_scores]
video_pred_ens_gmean = [np.argmax(x) for x in ensemble_scores_gmean]

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

