import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
import json
import datetime
from datetime import timedelta
from functools import reduce
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
    def __init__(self, sheets_to_load=None, src_dir=None,
                 pupil_lst=["fixations.csv", "blinks.csv", "head_pose_tracker_poses.csv"]):
        if sheets_to_load is not None:
            assert isinstance(sheets_to_load, list)
            self.sheets_to_load = sheets_to_load
        else:
            self.sheets_to_load = [item for item in os.listdir(src_dir) if not item.startswith('.')]

        self.freq = '200ms'
        self.BLINK_CONF_THRES = 0.3
        self.FIXATION_CONF_THRES = 0.7
        self.src_dir = src_dir
        # Annotations
        self.anno_dict = pd.read_excel("./Annotations.xlsx", sheet_name=sheets_to_load)
        # list of Pupil features of interest
        self.pupil_lst = pupil_lst  # "gaze_positions.csv"

        self.data = {}
        for samp in self.sheets_to_load:
            self.data[samp] = self.resample_data(samp)

    def resample_data(self, sample):

        with open(join(self.src_dir, sample, 'Pupil', "info.player.json")) as f:
            Info = json.load(f)
            start_time = Info['start_time_system_s']
            dt_start = datetime.datetime.fromtimestamp(start_time)

        # create the index with the start and end times you want
        t_index = pd.DatetimeIndex(pd.date_range(start=dt_start, end=dt_start + datetime.timedelta(seconds=300),
                                                 freq=self.freq), name='new_timestamp')

        for modality in self.pupil_lst:
            # TODO: need a common start_t=0 to 300 as shared indexed new_timestamp
            if modality == 'head_pose_tracker_poses.csv':

                Head = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # Resample timestamps to the desired frequency
                Head['new_timestamp'] = dt_start + pd.to_timedelta(Head['timestamp'], unit='S')

                # RESAMPLE
                resampled_Head = Head.resample(self.freq, on='new_timestamp').mean().reindex(t_index, method='nearest', limit=2)
                # Fill NaN values in the DataFrame using 'ffill'
                resampled_Head.fillna(method='ffill', inplace=True)
                #print(resampled_Head.head())

            elif modality == 'blinks.csv':
                Blin = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # create columns of start/end timestamps
                Blin['s_timestamp'] = dt_start + pd.to_timedelta(Blin['start_timestamp'], unit='S')
                Blin['e_timestamp'] = Blin['s_timestamp'] + pd.to_timedelta(Blin['duration'], unit='S')

                resampled_Blin = pd.DataFrame()
                # Iterate through intervals to expand and add feature information
                interval = t_index
                interval_df2 = pd.DataFrame({'new_timestamp': interval, 'blink_conf': np.nan})
                for idx, row in Blin.iterrows():
                    for index, rerow in interval_df2.iterrows():

                        if rerow['new_timestamp'] >= row['s_timestamp'] and rerow['new_timestamp'] <= row['e_timestamp']\
                                and row['confidence'] > self.BLINK_CONF_THRES:
                            #print(interval_df2.loc[index, 'blink_conf'], row['confidence'])
                            interval_df2.loc[index, 'blink_conf'] = row['confidence']
                            #break

                resampled_Blin = pd.concat([resampled_Blin, interval_df2], ignore_index=True)

                #print(resampled_Blin)
                resampled_Blin.set_index('new_timestamp', inplace=True)
                #resampled_Blin.fillna(method='ffill', inplace=True)


            elif modality == 'fixations.csv':

                Fixa = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # create columns of start/end timestamps
                Fixa['s_timestamp'] = dt_start + pd.to_timedelta(Fixa['start_timestamp'], unit='S')
                Fixa['e_timestamp'] = Fixa['s_timestamp'] + pd.to_timedelta(Fixa['duration'], unit='ms')

                resampled_Fixa = pd.DataFrame()
                # Iterate through intervals to expand and add feature information
                interval = t_index
                interval_df = pd.DataFrame({'new_timestamp': interval, 'fixa_conf': np.nan,
                                            'norm_pos_x': np.nan, 'norm_pos_y': np.nan})
                for idx, row in Fixa.iterrows():
                    for index, rerow in interval_df.iterrows():

                        if rerow['new_timestamp'] >= row['s_timestamp'] and rerow['new_timestamp'] <= row['e_timestamp'] \
                                and row['confidence'] > self.FIXATION_CONF_THRES:
                            interval_df.loc[index, 'fixa_conf'] = row['confidence']
                            interval_df.loc[index, 'norm_pos_x'] = row['norm_pos_x']
                            interval_df.loc[index, 'norm_pos_y'] = row['norm_pos_y']
                            #break

                resampled_Fixa = pd.concat([resampled_Fixa, interval_df], ignore_index=True)

                resampled_Fixa.set_index('new_timestamp', inplace=True)
                #resampled_Fixa.fillna(method='ffill', inplace=True)

            elif modality == 'surface_events.csv':
                Surf = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # create columns of start/end timestamps
                Surf['world_timestamp'] = dt_start + pd.to_timedelta(Surf['world_timestamp'], unit='S')

                resampled_Surf = pd.DataFrame()
                # Iterate through intervals to expand and add feature information
                interval = t_index
                interval_df3 = pd.DataFrame({'new_timestamp': interval, 'in_surface': np.nan})
                for idx, row in Surf.iterrows():
                    if idx % 2 == 0:
                        row_s = row
                        continue
                    else:
                        row_e = row
                    for index, rerow in interval_df3.iterrows():
                        if rerow['new_timestamp'] >= row_s['world_timestamp'] and rerow['new_timestamp'] <= row_e['world_timestamp']:

                            interval_df3.loc[index, 'in_surface'] = 1
                            #break

                resampled_Surf = pd.concat([resampled_Surf, interval_df3], ignore_index=True)

                resampled_Surf.set_index('new_timestamp', inplace=True)

            elif modality == 'annotations.csv':
                Anno = pd.read_csv(join(self.src_dir, sample, 'Pupil', modality))
                # create columns of start/end timestamps
                Anno['timestamp'] = dt_start + pd.to_timedelta(Anno['timestamp'], unit='S')

                resampled_Anno = pd.DataFrame()
                # Iterate through intervals to expand and add feature information
                interval = t_index
                interval_df4 = pd.DataFrame({'new_timestamp': interval, 'annotation': np.nan})
                for idx, row in Anno.iterrows():
                    if idx % 2 == 0:
                        row_s = row
                        continue
                    else:
                        row_e = row
                    for index, rerow in interval_df4.iterrows():
                        if rerow['new_timestamp'] >= row_s['timestamp'] and rerow['new_timestamp'] <= row_e['timestamp']:
                            interval_df4.loc[index, 'annotation'] = 1
                            #break

                resampled_Anno = pd.concat([resampled_Anno, interval_df4], ignore_index=True)

                resampled_Anno.set_index('new_timestamp', inplace=True)

            else:
                raise TypeError("No such Pupil Feature Modality")

        #print(resampled_Blin)
        #print(len(resampled_Fixa), len(resampled_Head))
        #resampled_data = resampled_Head.merge(resampled_Fixa, how='left', left_index=True, right_index=True)
        #resampled_data = resampled_data.merge(resampled_Blin, how='left', left_index=True, right_index=True)
        # Use "reduce" function
        resampled_data = resampled_Head.merge(resampled_Fixa, on='new_timestamp').merge(resampled_Blin, on='new_timestamp')

        return resampled_data


if __name__ == "__main__":

    OUTPUT_TYPE = "csv"  # "csv" or "pkl"
    data_dir = r"/Users/zhuangzhuangdai/Downloads/dataset_Annotations"

    if OUTPUT_TYPE == "csv":
        # output one sample .csv for example
        prep = Preprocessor(sheets_to_load=['GBSIOT_07B', 'GBSIOT_07D'], src_dir=data_dir)

        for k, v in prep.data.items():
            csv_dir = "./data_csvs"
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            prep.data[k].to_csv(join(csv_dir, k + ".csv"))

    elif OUTPUT_TYPE == "pkl":
        sheet_lst = [item for item in os.listdir(data_dir) if not item.startswith(".")]

        prep = Preprocessor(sheets_to_load=sheet_lst, src_dir=data_dir)

        with open('data_pkl.pkl', 'wb') as handle:
            pickle.dump(prep.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise TypeError("Output format not supported!")

    exit()
