# -*- coding: utf-8 -*-
"""nfl-v8-XGB-836-ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15TjCWwIT3gReVB1T0dwtw6IRLCauoonE
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb

# カテゴリ変数を数値化に変換するエンコーダ
from sklearn.preprocessing import LabelEncoder, StandardScaler
# ランダムフォレストによる分類器
from sklearn.ensemble import RandomForestClassifier
# 層化K分割交差検証を行うクラス
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# ROC　AUC　スコアを計算する評価指標
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.stats import randint, uniform

PATH = '/content/drive/My Drive/GCI/NFL/'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')

# 前処理　preprocessing
columns_to_drop = ['Id', 'School']
train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])
test = test.drop(columns=[col for col in columns_to_drop if col in test.columns])

cols_to_fill = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                'Broad_Jump', 'Agility_3cone', 'Shuttle']
# train の平均で train/test 両方を補完
for col in cols_to_fill:
  mean_value = train[col].mean()
  train[col] = train[col].fillna(mean_value)
  test[col] = test[col].fillna(mean_value)


# カテゴリデータをラベルエンコーディング
label_encoders = {}
for c in ['Player_Type', 'Position_Type', 'Position']:
  label_encoders[c] = LabelEncoder()
  train[c] = label_encoders[c].fit_transform(train[c].astype(str))
  test[c] = label_encoders[c].transform(test[c].astype(str))

# BMI
train['BMI'] = round(train['Weight'] / train['Height'] ** 2, 2)
test['BMI'] = round(test['Weight'] / test['Height'] ** 2, 2)

#速度
train['Speed_Score'] = round((train['Weight'] * 200) / (train['Sprint_40yd'] ** 4), 2)
test['Speed_Score'] = round((test['Weight'] * 200) / (test['Sprint_40yd'] ** 4), 2)

train['Speed_pct'] = train.groupby(['Position','Year'])['Speed_Score'] \
                         .rank(pct=True)
test ['Speed_pct'] = test.groupby(['Position','Year'])['Speed_Score'] \
                         .rank(pct=True)

train.head()

# 特徴量と目的変数に分ける
X = train.drop(columns=['Drafted'])
y = train['Drafted']

# 高速版XGBoostパラメータ
xgb_param_grid_fast = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1.0]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost最適化
xgb_model = xgb.XGBClassifier(
    random_state=2025,
    eval_metric='auc',
    use_label_encoder=False
)

xgb_grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid_fast,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X, y)

print(f"XGBoost最適パラメータ: {xgb_grid.best_params_}")
print(f"XGBoost最適CV AUC: {round(xgb_grid.best_score_, 4)}")

model = xgb_grid.best_estimator_

auc_scores = []
test_pred_proba_list = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # モデル学習
    model.fit(X_train, y_train)

    # バリデーション予測
    y_valid_pred_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_pred_proba)
    auc_scores.append(auc)

    # テストデータ予測
    test_pred_proba = model.predict_proba(test)[:, 1]
    test_pred_proba_list.append(test_pred_proba)

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
test_pred_mean = np.mean(test_pred_proba_list, axis=0)

result = {
    'mean_auc': mean_auc,
    'std_auc': std_auc,
    'test_predictions': test_pred_mean,
    'cv_scores': auc_scores
}

print(f"Best AUC: {round(result['mean_auc'], 4)}")

best_test_predictions = result['test_predictions']

# 特徴量とその重要度をDataFrameにまとめる
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()

# ファイル作成
submission = pd.read_csv(PATH + 'sample_submission.csv')
submission['Drafted'] = best_test_predictions
submission.to_csv(PATH + 'v6_XGB.csv', index=False)