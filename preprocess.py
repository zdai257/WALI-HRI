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


# Data loader class #
class Preprocessor(object):
    def __init__(self):
        pass


if __name__ == "__main__":

    Annotations_path = "./Annotations.xlsx"

    src_fd = r"/Users/zhuangzhuangdai/Downloads/dataset_Annotations"
    #sheets_to_load = [item for item in os.listdir(src_fd) if not item.startswith('.')]
    sheets_to_load = ['GBSIOT_07B', 'GBSIOT_07D']

    anno_dict = pd.read_excel(Annotations_path, sheet_name=sheets_to_load)
    #print(anno_dict.keys())

    pupil_lst = ["gaze_positions.csv", "fixations.csv", "blinks.csv", "head_pose_tracker_poses.csv"]

    start_t = 0
    TP, FP, FN = 0, 0, 0
    results = []

    for sample in sheets_to_load:
        print("\nStudying sample: ", sample)

        Gaze = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[0]))
        Fixa = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[1]))
        Blin = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[2]))
        Head = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[3]))

        gt_TP = []
        for i in range(0, int(len(anno_dict[sample]['timestamp']) / 2)):
            gt_TP.append(anno_dict[sample]['timestamp'][2 * i])
        # print(gt_TP)

        # Get start_t by getting first sample
        #_, start_t = next(zip(LCAS_value[sample][0], LCAS_value[sample][1]))
        # print(f"Start Time in Sec: {start_t}")

        for t, idx, conf, posX, posY in zip(Gaze['gaze_timestamp'], Gaze['world_index'], Gaze['confidence'],
                                            Gaze['norm_pos_x'], Gaze['norm_pos_y']):
            print(t, conf)




    # break

    # Final stats


    #print(results)

