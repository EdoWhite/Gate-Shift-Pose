import argparse
import numpy as np
from scipy import stats as st
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import itertools

parser = argparse.ArgumentParser(description="GSF testing with saved logits")

parser.add_argument('--rgb_models', type=str)
parser.add_argument('--depth_models', type=str)
parser.add_argument('--test_labels', type=str)
parser.add_argument('--weight_rgb', type=float, default=0.5)
args = parser.parse_args()

def format_percentage(value):
    return "{:.2f}%".format(value * 100)

def accuracy_top5_hardvoting(true_labels, scores):
    """Computes the Top-5 Accuracy for Ensembles in an Hard Voting approach"""
    top5_predictions = np.argsort(scores, axis=-1)[:, :, -5:]

    true_labels_expanded = np.expand_dims(true_labels, axis=0)
    top5_correct = np.any(top5_predictions == true_labels_expanded[..., np.newaxis], axis=-1)
    top5_accuracy = np.mean(top5_correct)

    return top5_accuracy

def evaluate_combination(rgb_scores_list, depth_scores_list, rgb_paths_list, depth_paths_list):
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

    """
    print("TOTAL SCORES:") #(4, 20, 1, 61) --> (4, 20, 61)
    print(np.squeeze(np.array(total_scores)).shape)
    print("###############################################\n")

    print("TOTAL SCORES PAIRS:") #(2, 20, 1, 61) --> (2, 20, 61)
    print(np.squeeze(np.array(total_avg_scores)).shape)
    print("###############################################\n")
    """
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

    """
    print("TOTAL AVG SCORES PAIRS:") #(20, 61)
    print(avg_scores.shape)
    print("###############################################\n")

    print("TOTAL AVG SCORES:") #(20, 61)
    print(ensemble_scores.shape)
    print("###############################################\n")

    print("TOTAL GMEAN SCORES:") #(20, 61)
    print(gmean_scores.shape)
    print("###############################################\n")
    """

    video_pred_avg = [np.argmax(x) for x in avg_scores]
    video_pred = [np.argmax(x) for x in ensemble_scores]
    video_pred_gmean = [np.argmax(x) for x in gmean_scores]
    video_pred_ens_gmean = [np.argmax(x) for x in ensemble_scores_gmean]

    """
    print("video labels:")
    print(video_labels.shape)
    print("\n")

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
    """

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

    print("COMBO:")
    print('Overall Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1 * 100, acc_5 * 100))
    print('Overall gmean Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_ens_gmean * 100, acc_5_ens_gmean * 100))
    print('Overall Pairs Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_avg * 100, acc_5_avg * 100))
    print('Overall Pairs (gmean) Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_gmean * 100, acc_5_gmean * 100))
    print('Overall HV Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_hv * 100, acc_5_hv * 100))
    print('Overall HV Pairs Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_hv_avg * 100, acc_5_hv_avg * 100))
    print('Overall HV Pairs (gmean) Acc@1 {:.02f}% Acc@5 {:.02f}%'.format(acc_1_hv_gmean * 100, acc_5_hv_gmean * 100))
    print("\n")

    return {
        'depth weightsw': weight_depth,
        'rgb_paths': rgb_paths_list,
        'depth_paths': depth_paths_list,
        'acc_1_avg': format_percentage(acc_1_avg),
        'acc_5_avg': format_percentage(acc_5_avg),
        'acc_1': format_percentage(acc_1),
        'acc_5': format_percentage(acc_5),
        'acc_1_gmean': format_percentage(acc_1_gmean),
        'acc_5_gmean': format_percentage(acc_5_gmean),
        'acc_1_ens_gmean': format_percentage(acc_1_ens_gmean),
        'acc_5_ens_gmean': format_percentage(acc_5_ens_gmean),
        'acc_1_hv_avg': format_percentage(acc_1_hv_avg),
        'acc_5_hv_avg': format_percentage(acc_5_hv_avg),
        'acc_1_hv': format_percentage(acc_1_hv),
        'acc_5_hv': format_percentage(acc_5_hv),
        'acc_1_hv_gmean': format_percentage(acc_1_hv_gmean),
        'acc_5_hv_gmean': format_percentage(acc_5_hv_gmean),
    }

def get_max_accuracy_dict(dicts_list):
    def get_accuracy_value(dictionary):
        return max(
            dictionary["acc_1_avg"],
            dictionary["acc_1"],
            dictionary["acc_1_gmean"],
            dictionary["acc_1_ens_gmean"],
            dictionary["acc_1_hv_avg"],
            dictionary["acc_1_hv"],
            dictionary["acc_1_hv_gmean"]
        )

    max_dict = max(dicts_list, key=get_accuracy_value)
    return max_dict

# READING SCORES LIST
with open(args.rgb_models, 'r') as file:
    rgb_scores_paths = file.read().splitlines()

with open(args.depth_models, 'r') as file:
    depth_scores_paths = file.read().splitlines()

# Load the saved softmax scores
rgb_scores_list = [{"path":path, "score":np.load(path)} for path in rgb_scores_paths]
depth_scores_list =  [{"path":path, "score":np.load(path)} for path in depth_scores_paths]
video_labels = np.load(args.test_labels)

# Generate all models combinations
num_elements = len(rgb_scores_list) + len(depth_scores_list)
combinations_lengths = range(3, 11)
#combinations_lengths = range(6, 7)

# Create list for each lenght of combinations
all_combinations_rgb = []
all_combinations_depth = []
for length in combinations_lengths:
    # Creiamo le combinazioni di lunghezza "length" tra rgb_scores_list e depth_scores_list
    print("lenght: {}".format(length))
    combinations_rgb = list(itertools.combinations(rgb_scores_list, length))
    combinations_depth = list(itertools.combinations(depth_scores_list, length))

    all_combinations_rgb.extend(combinations_rgb)
    all_combinations_depth.extend(combinations_depth)

# Evaluate all the combinations
eval = []
for i, (combination_rgb, combination_depth) in enumerate(zip(all_combinations_rgb, all_combinations_depth)):
    """
    print(f"Combination {i + 1}:")
    num_models = len(combination) // 2
    print(num_models)
    rgb_models = combination[:num_models]
    depth_models = combination[num_models:]

    rgb_paths = [rgb_scores_paths[rgb_scores_list.index(model)] for model in rgb_models]
    depth_paths = [depth_scores_paths[depth_scores_list.index(model)] for model in depth_models]

    res = evaluate_combination(rgb_models, depth_models, rgb_paths, depth_paths)
    eval.append(res)

    print(f"RGB Models Paths: {rgb_paths}")
    print(f"Depth Models Paths: {depth_paths}")
    print("\n")
    """
    print("###########")
    print(f"Combination {i + 1}:")
    rgb_models = [model["score"] for model in combination_rgb]
    depth_models = [model["score"] for model in combination_depth]

    rgb_models_names = [model["path"] for model in combination_rgb]
    depth_models_names = [model["path"] for model in combination_depth]
    print(rgb_models_names)
    print("\n")
    print(depth_models_names)
    print("###########")
    res = evaluate_combination(rgb_models, depth_models, rgb_models_names, depth_models_names)
    eval.append(res)

result_dict = get_max_accuracy_dict(eval)

for key, value in result_dict.items():
    print(f"{key}: {value}")


