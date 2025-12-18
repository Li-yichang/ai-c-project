import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_csv('features_todo3.csv')
X = df.drop(columns=['label'])
y = df['label']

# 切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg.fit(X_train, y_train)
y_prob_lr = logreg.predict_proba(X_test)[:, 1]
y_pred_lr = (y_prob_lr >= 0.3).astype(int)  

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf >= 0.3).astype(int)  

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight= (y == 0).sum() / (y == 1).sum(), random_state=42)
xgb_clf.fit(X_train, y_train)
y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb >= 0.3).astype(int) 

def evaluate(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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
    plt.show(block=False)
    plt.pause(1)


evaluate(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression')
evaluate(y_test, y_pred_rf, y_prob_rf, 'Random Forest')
evaluate(y_test, y_pred_xgb, y_prob_xgb, 'XGBoost')

# Cross-validation 
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print(f"\nLogistic Regression 5-fold CV ROC AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
plt.show()