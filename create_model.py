import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def main():
    data_dir = 'data_by_days'
    NUM_TRAIN_FILES = 71

    df = pd.concat([pd.read_feather(f'{data_dir}/train_{i}.feather')for i in range(NUM_TRAIN_FILES)])
    df.reset_index(drop=True, inplace=True)

    y_true = df['target']

    model = LinearRegression(fit_intercept=False)
    SANHOK_columns = [column for column in df.columns if column.startswith('SANHOK')]
    model.fit(df[SANHOK_columns], y_true)
    y_pred = model.predict(df[SANHOK_columns])
    print('r2_score = ', r2_score(y_true, y_pred))

    np.save('coef.npy', model.coef_)


if __name__ == '__main__':
    main()
