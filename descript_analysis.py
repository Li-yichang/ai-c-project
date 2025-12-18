import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("features_todo3.csv")

# 基本摘要
print("\n Basic Summary:")
print(df.describe().T)

# 類別分布
print("\n Gender 分布:")
print(df['gender'].value_counts())

print("\n Mortality 標籤分布:")
print(df['label'].value_counts(normalize=True))  

# 存活 vs 死亡組的平均特徵比較
grouped_mean = df.groupby('label').mean().T
print("\n 死亡與存活組平均差異:")
print(grouped_mean)

# 可選擇要看的幾個重點變數
focus_features = ['age', 'bmi', 'heart_rate_mean', 'creatinine_mean', 'bun_mean']

# Boxplot 比較死亡與存活組特徵
for feat in focus_features:
    if feat in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='label', y=feat, data=df)
        plt.title(f'{feat} vs. In-hospital Mortality')
        plt.xlabel('Mortality (0=Survive, 1=Death)')
        plt.ylabel(feat)
        plt.tight_layout()
        plt.show()

# 與標籤的相關係數
correlation = df.corr(numeric_only=True)['label'].drop('label')  
correlation = correlation.sort_values(ascending=False)

plt.figure(figsize=(6, len(correlation) * 0.25))
sns.heatmap(correlation.to_frame(), annot=True, cmap='coolwarm')
plt.title('Correlation with In-hospital Mortality')
plt.tight_layout()
plt.show()
