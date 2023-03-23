import numpy as np

def get_predict(df):
    lr_coef = np.load('coef.npy')
    SANHOK_columns = [column for column in df.columns if column.startswith('SANHOK')]
    y_pred = np.matmul(df[SANHOK_columns], lr_coef)
    return y_pred
