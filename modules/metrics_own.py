import numpy as np
import pandas as pd
from numpy import vstack
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix, precision_recall_fscore_support, average_precision_score

def class_accuracy(y_true, y_pred):
    accuracies = []
    for i in range(y_true.shape[1]):  # 对于每个标签
        accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))  # 对每个标签计算准确率
    return accuracies
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions, result_dir=None, kfold=None, save_csv=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label >= threshold_optimal] = 1
    this_class_label[this_class_label < threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1 - np.count_nonzero(np.array(bag_labels).astype(int) - bag_predictions.astype(int)) / len(bag_labels)

    # Save metrics to CSV
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': fscore,
        'ROC AUC': auc_value
    }

    metrics_df = pd.DataFrame([metrics_dict])
    # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    # if not os.path.exists(folder_dir):
    #     os.makedirs(folder_dir)
    if save_csv:
        if not os.path.exists(result_dir):
            if kfold:
                result_dir = f'{result_dir}/metrics/'
            os.makedirs(result_dir)

        if kfold:
            metrics_filename = f'{result_dir}/fold{kfold}.csv'
        else:
            metrics_filename = f'{result_dir}/metrics.csv'

        metrics_df.to_csv(metrics_filename, index=False)
    return accuracy, auc_value, precision, recall, fscore


def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def Accuracy(y_true, y_pred):
    """Jaccard-based Accuracy for multi-label classification."""
    count = 0
    for i in range(y_true.shape[0]):
        intersection = sum(np.logical_and(y_true[i], y_pred[i]))  # 交集
        union = sum(np.logical_or(y_true[i], y_pred[i]))  # 并集
        if union == 0:
            continue  # 避免 0/0
        count += intersection / union
    return count / y_true.shape[0]

def Precision(y_true, y_pred):
    """Per-sample Precision for multi-label classification."""
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue  # 避免 0/0
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]

def Recall(y_true, y_pred):
    """Per-sample Recall for multi-label classification."""
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue  # 避免 0/0
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]

def F1Measure(y_true, y_pred):
    """Per-sample F1 Measure for multi-label classification."""
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue  # 避免 0/0
        intersection = sum(np.logical_and(y_true[i], y_pred[i]))
        total = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * intersection) / total
    return count / y_true.shape[0]

def cal_metrics(true_labels, predicted_labels, pred_scores, result_dir=None, kfold=None, save_csv=False,
                three_label=False, verbose=False):
    '''Calculate metrics for classification tasks
    return accuracy, precision, recall, f1, macro_roc_auc_ovr, pr_auc'''
    # # Compute ROC curve and AUC
    # fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    # roc_auc, optim_thresh = roc_threshold(true_labels, pred_scores)

    # predicted_labels = (pred_scores > optim_thresh).astype(int)
    # accuracy = accuracy_score(true_labels, predicted_labels)
    accuracy = Accuracy(true_labels, predicted_labels)
    accuracy_dict = class_accuracy(true_labels, predicted_labels)
    precision = Precision(true_labels, predicted_labels)
    recall = Recall(true_labels, predicted_labels)
    # recall1 = recall_score(true_labels, predicted_labels, average='samples')
    f1 = F1Measure(true_labels, predicted_labels)

    pred_scores = vstack(pred_scores)
    # print(pred_scores, pred_scores.shape)
    try:
        macro_roc_auc_ovr = roc_auc_score(true_labels, pred_scores, multi_class="ovr", average="macro", )
        pr_auc = average_precision_score(true_labels, pred_scores, average="macro")
        # calculate roc auc for each category
        rocauc_per_class = []
        for class_index in range(pred_scores.shape[1]):
            roc_auc = roc_auc_score(true_labels[:, class_index], pred_scores[:, class_index])
            rocauc_per_class.append(roc_auc)
        if verbose:
            for class_index, roc_auc in enumerate(rocauc_per_class):
                print(f"ROC AUC of Class {class_index:.4f} is: {roc_auc:.4f}")
    except ValueError:
        macro_roc_auc_ovr = 0
        pr_auc = 0
        rocauc_per_class = [0, 0, 0, 0]

    # Save metrics to CSV
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': macro_roc_auc_ovr,
        'PR AUC': pr_auc
    }
    metrics_dict.update({f'Class {i} Accuracy': accuracy_dict[i] for i in range(len(accuracy_dict))})
    metrics_dict.update({f'{item} rocauc': rocauc_per_class[item] for item in range(pred_scores.shape[1])})
    metrics_df = pd.DataFrame([metrics_dict])
    if save_csv:
        if not os.path.exists(result_dir):
            if kfold:
                result_dir = f'{result_dir}/metrics/'
            os.makedirs(result_dir)

        if kfold:
            metrics_filename = f'{result_dir}/fold{kfold}.csv'
        else:
            metrics_filename = f'{result_dir}/metrics2.csv'

        metrics_df.to_csv(metrics_filename, index=False)

    return accuracy, precision, recall, f1, macro_roc_auc_ovr, pr_auc