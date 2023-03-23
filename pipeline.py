import pandas as pd
from sklearn.metrics import r2_score

from solution import get_predict


def main():
    df = pd.read_feather('test_competition.feather')
    X_columns = [column for column in df.columns if column != 'target']
    X = df[X_columns]
    y_true = df['target']
    y_pred = get_predict(X)
    score = r2_score(y_true, y_pred)
    return score


if __name__ == '__main__':
    print(main())
