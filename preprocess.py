import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
import json
import datetime
from datetime import timedelta
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
    def __init__(self, sheets_to_load=None, src_dir=r"/Users/zhuangzhuangdai/Downloads/dataset_Annotations"):
        if sheets_to_load is not None:
            assert isinstance(sheets_to_load, list)
            self.sheets_to_load = sheets_to_load
        else:
            self.sheets_to_load = [item for item in os.listdir(src_dir) if not item.startswith('.')]

        self.freq = '100ms'
        self.src_dir = src_dir
        # Annotations
        self.anno_dict = pd.read_excel("./Annotations.xlsx", sheet_name=sheets_to_load)
        # list of Pupil features of interest
        self.pupil_lst = ["fixations.csv", "blinks.csv", "head_pose_tracker_poses.csv"]  # "gaze_positions.csv"

        self.data = {}
        for samp in self.sheets_to_load:
            self.data[samp] = self.resample_data(samp)

    def resample_data(self, sample):

        with open(join(self.src_dir, sample, 'Pupil', "info.player.json")) as f:
            Info = json.load(f)
            start_time = Info['start_time_system_s']
            dt_start = datetime.datetime.fromtimestamp(start_time)

        for modality in self.pupil_lst:
            # TODO: need a common start_t=0 to 300 as shared indexed new_timestamp
            if modality == 'head_pose_tracker_poses.csv':
                Head = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # Resample timestamps to the desired frequency
                Head['new_timestamp'] = dt_start + pd.to_timedelta(Head['timestamp'], unit='S')
                # Set 'timestamp' column as the index for resampling
                Head.set_index('new_timestamp', inplace=True)
                resampled_Head = Head.resample(self.freq).mean()
                resampled_Head.fillna(method='ffill', inplace=True)

            elif modality == 'blinks.csv':
                Blin = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # create columns of start/end timestamps
                Blin['s_timestamp'] = dt_start + pd.to_timedelta(Blin['start_timestamp'], unit='S')
                Blin['e_timestamp'] = Blin['s_timestamp'] + pd.to_timedelta(Blin['duration'], unit='S')

                resampled_Blin = pd.DataFrame()
                # Iterate through intervals to expand and add feature information
                for idx, row in Blin.iterrows():
                    # print(row['id'])
                    interval = pd.date_range(start=row['s_timestamp'], end=row['e_timestamp'], freq=self.freq)
                    interval_df = pd.DataFrame({'new_timestamp': interval})
                    interval_df['blink_conf'] = row['confidence']
                    resampled_Blin = pd.concat([resampled_Blin, interval_df], ignore_index=True)

                resampled_Blin.set_index('new_timestamp', inplace=True)
                #resampled_Blin.fillna(method='ffill', inplace=True)

            elif modality == 'fixations.csv':
                Fixa = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # create columns of start/end timestamps
                Fixa['s_timestamp'] = dt_start + pd.to_timedelta(Fixa['start_timestamp'], unit='S')
                Fixa['e_timestamp'] = Fixa['s_timestamp'] + pd.to_timedelta(Fixa['duration'], unit='ms')

                resampled_Fixa = pd.DataFrame()
                # Iterate through intervals to expand and add feature information
                for idx, row in Fixa.iterrows():
                    # print(row['id'])
                    interval = pd.date_range(start=row['s_timestamp'], end=row['e_timestamp'], freq=self.freq)
                    interval_df = pd.DataFrame({'new_timestamp': interval})
                    interval_df['norm_pos_x'] = row['norm_pos_x']
                    interval_df['norm_pos_y'] = row['norm_pos_y']
                    resampled_Fixa = pd.concat([resampled_Fixa, interval_df], ignore_index=True)

                resampled_Fixa.set_index('new_timestamp', inplace=True)

            else:
                raise TypeError("No such Pupil Feature Modality")

        #print(resampled_Blin, resampled_Head)
        resampled_data = resampled_Fixa.merge(resampled_Blin, how='left', left_index=True, right_index=True)
        return resampled_data


if __name__ == "__main__":

    prep = Preprocessor(sheets_to_load=['GBSIOT_07B', 'GBSIOT_07D'])

    print(prep.data)
    prep.data['GBSIOT_07B'].to_csv('./test.csv')
    exit()

    for sample in prep.sheets_to_load:
        print("\nStudying sample: ", sample)

        src_fd = prep.src_dir
        pupil_lst = prep.pupil_lst

        with open(join(src_fd, sample, 'Pupil', "info.player.json")) as f:
            Info = json.load(f)
            start_time = Info['start_time_system_s']
            print(datetime.datetime.fromtimestamp(start_time))
        Gaze = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[0]))
        Fixa = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[1]))
        Blin = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[2]))
        Head = pd.read_csv(join(src_fd, sample, 'Pupil', pupil_lst[3]))

        #print(pd.to_timedelta(Head['timestamp'], unit='S'))
        Head_ts = pd.to_timedelta(Head['timestamp'], unit='S')
        Head['timestamp'] = Head_ts
        #print(Head.resample('100ms', on='timestamp').transform('mean'))

        Gaze_ts = pd.to_timedelta(Gaze['gaze_timestamp'], unit='S')
        Gaze['timestamp'] = Gaze_ts
        #print(Gaze.resample('100ms', on='timestamp').transform('mean'))

        # Convert the 'timestamp' column to a datetime object if it's not already in datetime format
        Head['new_timestamp'] = datetime.datetime.fromtimestamp(start_time) + pd.to_timedelta(Head['timestamp'], unit='S')
        # Set 'timestamp' column as the index for resampling
        Head.set_index('new_timestamp', inplace=True)

        resampled_Head = Head.resample('100ms').mean()

        resampled_Head.fillna(method='ffill', inplace=True)

        # Interval-modalities
        Blin['s_timestamp'] = datetime.datetime.fromtimestamp(start_time) + pd.to_timedelta(Blin['start_timestamp'], unit='S')
        Blin['e_timestamp'] = Blin['s_timestamp'] + pd.to_timedelta(Blin['duration'], unit='ms')

        resampled_Blin = pd.DataFrame()
        # Iterate through intervals to expand and add feature information
        for idx, row in Blin.iterrows():
            #print(row['id'])
            interval = pd.date_range(start=row['s_timestamp'], end=row['e_timestamp'], freq='100ms')
            interval_df = pd.DataFrame({'new_timestamp': interval})
            interval_df['blink_conf'] = row['confidence']
            resampled_Blin = pd.concat([resampled_Blin, interval_df], ignore_index=True)

        resampled_Blin.set_index('new_timestamp', inplace=True)
        print(resampled_Blin)

        exit()

        for t, idx, conf, posX, posY in zip(Gaze['gaze_timestamp'], Gaze['world_index'], Gaze['confidence'],
                                            Gaze['norm_pos_x'], Gaze['norm_pos_y']):
            print(t, conf)




    # break

    # Final stats


    #print(results)

