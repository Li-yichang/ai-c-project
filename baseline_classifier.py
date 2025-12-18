import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('features_todo3.csv')

# 分離特徵與標籤
X = df.drop(columns=['label'])
y = df['label']

# 切分（80% train, 20% test）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_prob_lr = logreg.predict_proba(X_test)[:,1]

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# 定義評估函數
def evaluate(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'\n==== {model_name} ====')
    print(f'Accuracy: {acc:.3f}')
    print(f'ROC AUC: {auc:.3f}')
    print(f'Precision: {prec:.3f}')
    print(f'Recall: {rec:.3f}')
    print(f'F1 Score: {f1:.3f}')
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

evaluate(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression')

evaluate(y_test, y_pred_rf, y_prob_rf, 'Random Forest')
plt.show()