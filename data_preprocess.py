import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('features_todo2.csv')

# 1. 移除不必要欄位
df.drop(['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'dischtime'], axis=1, inplace=True)

# 2. 將 gender 轉為 0 / 1
df['gender'] = (df['gender'] == 'M').astype(int)

# 3. 用中位數填補缺失值
df.fillna(df.median(), inplace=True)

# 4. 正規化數值欄位（排除 gender 和 label）
num_cols = df.drop(columns=['label']).select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [col for col in num_cols if col != 'gender']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#  5. 輸出成新的 CSV
df.to_csv('features_todo3.csv', index=False)
