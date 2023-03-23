import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_feather('train_competition.feather')
y_true = df['target']

model = LinearRegression(fit_intercept=False)
SANHOK_columns = [column for column in df.columns if column.startswith('SANHOK')]
model.fit(df[SANHOK_columns], y_true)
y_pred = model.predict(df[SANHOK_columns])
print('r2_score = ', r2_score(y_true, y_pred))

np.save('coef.npy', model.coef_)
