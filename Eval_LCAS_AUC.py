import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"


T_TOLERANCE = 3.0
GRAD_THRES = -0.10


def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 1
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 1
    return tp / (tp + fn)


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(predictions, ground_truth, total_samples, time_window=T_TOLERANCE):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for prediction in predictions:
        matched = False
        for truth in ground_truth:
            diff_t = abs(prediction - truth)
            if diff_t <= time_window:
                # Found a match within the time window
                true_positives += 1
                matched = True
                break

        if not matched:
            # Prediction did not match any ground-truth event within the time window
            false_positives += 1

    # Calculation for TN and FN would depend on your specific scenario
    for truth in ground_truth:
        matched = False
        for prediction in predictions:
            diff_t = abs(prediction - truth)
            if diff_t <= time_window:
                # Found a match within the time window
                # it is NOT a FN
                matched = True
                break

        if not matched:
            false_negatives += 1

    true_negatives = total_samples - true_positives - false_positives - false_negatives

    # Calculate Precision, Recall, F1-Score using TP, FP, TN, FN per sample
    #precision = calculate_precision(true_positives, false_positives)
    #recall = calculate_recall(true_positives, false_negatives)
    #F1 = calculate_f1_score(precision, recall)
    # return precision, recall, F1
    return true_positives, false_positives, false_negatives, true_negatives


def calculate_auc(TP, FP, TN, FN):
    total_positive_samples = TP + FN
    total_negative_samples = FP + TN

    tpr = 0  # True Positive Rate
    auc = 0  # Area Under the Curve
    old_fpr = 0
    old_tpr = 0

    for i in range(TP + FP):
        if i < TP:  # True positives
            tpr += 1 / total_positive_samples
            auc += (tpr - old_tpr) * (old_fpr + (1 / total_negative_samples - old_fpr) / 2)
        else:  # False positives
            fpr = old_fpr + 1 / total_negative_samples
            auc += (tpr - old_tpr) * (old_fpr + (fpr - old_fpr) / 2)
            old_fpr = fpr
            old_tpr = tpr

    return auc


def calculate_pr_auc(precision, recall):
    pr_auc = 0
    old_recall = 0
    for r, p in zip(recall, precision):
        pr_auc += (r - old_recall) * p
        old_recall = r
    return pr_auc



if __name__ == "__main__":

    with open('LCAS_value.pkl', 'rb') as fp:
        LCAS_value = pickle.load(fp)
        print('Vales loaded successfully.')

    Annotations_path = "./Annotations.xlsx"
    src_fd = r"/Users/zhuangzhuangdai/Downloads/dataset_Annotations"
    sheets_to_load = [item for item in os.listdir(src_fd) if not item.startswith('.')]

    anno_dict = pd.read_excel(Annotations_path, sheet_name=sheets_to_load)
    #print(anno_dict.keys())

    start_t = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    Precision, Recall = [], []

    #grads = np.linspace(-0.25, -0.02, num=200)
    grads = [-0.13]
    results = []

    for grad in grads:

        for sample in sheets_to_load:
            #print("Studying sample: \n", sample)
            # print(anno_dict[sample]['timestamp'][18], anno_dict[sample]['timestamp'][19])

            gt_TP = []
            for i in range(0, int(len(anno_dict[sample]['timestamp']) / 2)):
                gt_TP.append(anno_dict[sample]['timestamp'][2 * i])
            # print(gt_TP)

            # Get start_t by getting first sample
            _, start_t = next(zip(LCAS_value[sample][0], LCAS_value[sample][1]))
            # print(f"Start Time in Sec: {start_t}")

            timestamps, values = [], []
            for msg, t, in zip(LCAS_value[sample][0], LCAS_value[sample][1]):
                # print(msg, t - start_t)
                timestamps.append(t - start_t)
                values.append(msg)

            # get outstanding gradient of LCAS values
            time_diff = np.asarray(timestamps, dtype=float)
            # print(time_diff, values)
            # print(len(time_diff), len(values))
            gradient = np.gradient(values, time_diff, edge_order=1)

            # Accuracy, Precision, F1 calculator
            tp, fp, fn, tn = calculate_metrics(time_diff[gradient <= grad], gt_TP, 50)

            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, fn)
            #print(precision, recall)
            Precision.append(precision)
            Recall.append(recall)

            TP += tp
            FP += fp
            FN += fn
            TN += tn

        # AUC
        auc = calculate_auc(TP, FP, FN, TN)

        pr_auc = calculate_pr_auc(sorted(Precision), sorted(Recall))

        print(auc, pr_auc)

