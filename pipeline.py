import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from solution import get_predict

NUM_TRAIN_FILES = 128
NUM_TEST_FILES = 56
NUM_PUBLIC_TEST_FILES = 28


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='public_test', type=str)
    args = parser.parse_args()
    mode = args.mode

    assert mode in ['train', 'test', 'public_test']
    name = 'train' if mode == 'train' else 'test'
    if mode == 'train':
        NUM_FILES = NUM_TRAIN_FILES
    if mode == 'test':
        NUM_FILES = NUM_TEST_FILES
    if mode == 'public_test':
        NUM_FILES = NUM_PUBLIC_TEST_FILES

    data_dir = 'data_by_days'
    y_trues = []
    y_preds = []

    for i in range(NUM_FILES):
        df = pd.read_feather(f'{data_dir}/{name}_{i}.feather')
        if len(df) > 0:
            X_columns = [column for column in df.columns if column != 'target']
            y_trues.append(df['target'].values)
            y_preds.append(get_predict(df[X_columns]))

    return r2_score(np.concatenate(y_trues), np.concatenate(y_preds))


if __name__ == '__main__':
    print(main())
