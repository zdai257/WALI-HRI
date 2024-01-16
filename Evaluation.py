import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt


T_TOLERANCE = 3.0
GRAD_THRES = -0.10


def calculate_precision(tp, fp):
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    return tp / (tp + fn)


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(predictions, ground_truth, time_window=T_TOLERANCE):
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

    # Calculate Precision, Recall, F1-Score using TP, FP, TN, FN per sample
    #precision = calculate_precision(true_positives, false_positives)
    #recall = calculate_recall(true_positives, false_negatives)
    #F1 = calculate_f1_score(precision, recall)
    # return precision, recall, F1
    return true_positives, false_positives, false_negatives


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
    TP, FP, FN = 0, 0, 0

    grads = np.linspace(-0.25, -0.02, num=200)
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
            tp, fp, fn = calculate_metrics(time_diff[gradient <= grad], gt_TP)
            TP += tp
            FP += fp
            FN += fn

            # break

        # Final stats
        precision = calculate_precision(TP, FP)
        recall = calculate_recall(TP, FN)
        F1 = calculate_f1_score(precision, recall)

        #print("Precision = {0}, Recall = {1}, F1 score = {2}".format(precision, recall, F1))

        results.append((precision, recall, F1))

    #print(results)

    ### PLOT ###
    font = {'family': 'serif', 'serif': ['Times New Roman']}

    plt.rc('font', **font)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))

    # Set font sizes for all texts and axis labels
    plt.rcParams.update({'font.size': 15})
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ax1.plot(grads, [x[0] for x in results], linestyle='-', linewidth=3, color='blue', label='Precision')
    ax1.plot(grads, [x[1] for x in results], linestyle='-', linewidth=3, color='grey', label='Recall')
    ax1.set_xlabel('Gradient Threshold', fontsize=19)
    ax1.set_ylabel('Precision & Recall', fontsize=19)

    # Create a secondary y-axis for F1-Score on the right side
    ax2 = ax1.twinx()
    ax2.tick_params(axis='both', which='major', labelsize=15)

    ax2.plot(grads, [x[2] for x in results], linestyle='-', linewidth=3, color='red', label='F1-Score')
    ax2.set_ylabel('F1 Score', fontsize=19)

    ax1.set_title('Prediction Performance as a function of Engagement Decrease Gradient', fontsize=21)

    # set axis limits
    ax1.set_xlim(-0.265, 0.005)
    ax1.set_ylim(-0.01, 0.53)
    ax2.set_ylim(-0.01, 0.33)

    # Combine legends from both y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='best', fontsize=18)
    #ax1.grid(False)
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig('./imgs/LCAS_Precision_Recall2.pdf')

    plt.show()
