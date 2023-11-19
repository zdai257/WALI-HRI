import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
import datetime


def process(bag_file):
    pass


if __name__ == "__main__":

    with open('LCAS_value.pkl', 'rb') as fp:
        LCAS_value = pickle.load(fp)
        print('Vales loaded successfully.')


    Annotations_path = "./Annotations.xlsx"
    src_fd = r"/Users/zhuangzhuangdai/Downloads/dataset_Annotations"
    sheets_to_load = [item for item in os.listdir(src_fd) if not item.startswith('.')]

    anno_dict = pd.read_excel(Annotations_path, sheet_name=sheets_to_load)
    print(anno_dict.keys())

    start_t = 0

    for sample in sheets_to_load:
        print(sample)
        print(anno_dict[sample]['timestamp'][18], anno_dict[sample]['timestamp'][19])

        # TODO: get outstanding gradient of LCAS values
        # Get start_t by getting first sample - 0.6s
        for msg, t, in zip(LCAS_value[sample][0], LCAS_value[sample][1]):
            start_t = t - 0.6
            print(f"Start Time in Sec: {start_t}")
            break

        for msg, t, in zip(LCAS_value[sample][0], LCAS_value[sample][1]):
            print(msg, t - start_t)

        break
