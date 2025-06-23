# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb

# カテゴリ変数を数値化に変換するエンコーダ
from sklearn.preprocessing import LabelEncoder
# ランダムフォレストによる分類器
from sklearn.ensemble import RandomForestClassifier
# 層化K分割交差検証を行うクラス
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# ROC　AUC　スコアを計算する評価指標
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

PATH = '/content/drive/My Drive/GCI/NFL/'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')

# 前処理　preprocessing
# 欠損補完、エンコーディングを行います
# 注意！ここでは「school」を削除しましたがDraftedに関わりかも
# IdとDrafted関係がないと予想される、削除する

# School frequency
school_freq = train["School"].value_counts(normalize=True)
train["school_freq"] = train["School"].map(school_freq)
test["school_freq"]  = test["School"].map(school_freq).fillna(0)

# Check if 'Id' and 'School' columns exist before dropping
columns_to_drop = ['Id', 'Year', 'School']
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
# for c in ['Player_Type', 'Position_Type', 'Position']:
for c in ['Player_Type', 'Position_Type', 'Position']:
  label_encoders[c] = LabelEncoder()
  train[c] = label_encoders[c].fit_transform(train[c].astype(str))
  test[c] = label_encoders[c].transform(test[c].astype(str))

# BMI
train['BMI'] = round(train['Weight'] / train['Height'] ** 2, 2)
test['BMI'] = round(test['Weight'] / test['Height'] ** 2, 2)

#爆発力
# train['Explosiveness'] = round(train['Vertical_Jump'] + train['Broad_Jump'], 2)
# test['Explosiveness'] = round(test['Vertical_Jump'] + test['Broad_Jump'], 2)

#速度
train['Speed_Score'] = round((train['Weight'] * 200) / (train['Sprint_40yd'] ** 4), 2)
test['Speed_Score'] = round((test['Weight'] * 200) / (test['Sprint_40yd'] ** 4), 2)

# train.head()

# 特徴量と目的変数に分ける
X = train.drop(columns=['Drafted'])
y = train['Drafted']

# ===== XGBoost設定 =====
print("=== XGBoost ハイパーパラメータ最適化 ===")

# XGBoost用パラメータグリッド（16特徴量に最適化）
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],  # L1正則化
    'reg_lambda': [1, 1.5]  # L2正則化
}

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
    param_grid=xgb_param_grid_fast,  # xgb_param_grid に変更すると詳細版
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X, y)

print(f"XGBoost最適パラメータ: {xgb_grid.best_params_}")
print(f"XGBoost最適CV AUC: {round(xgb_grid.best_score_, 4)}")

# ===== LightGBM設定 =====
print(f"\n=== LightGBM ハイパーパラメータ最適化 ===")

# LightGBM用パラメータグリッド
lgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, -1],  # -1は制限なし
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [15, 31, 63],  # 2^max_depth - 1 が目安
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}

# 高速版LightGBMパラメータ
lgb_param_grid_fast = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.2],
    'num_leaves': [15, 31]
}

# LightGBM最適化
lgb_model = lgb.LGBMClassifier(
    random_state=2025,
    verbose=-1  # ログを抑制
)

lgb_grid = GridSearchCV(
    estimator=lgb_model,
    param_grid=lgb_param_grid_fast,  # lgb_param_grid に変更すると詳細版
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

lgb_grid.fit(X, y)

print(f"LightGBM最適パラメータ: {lgb_grid.best_params_}")
print(f"LightGBM最適CV AUC: {round(lgb_grid.best_score_, 4)}")

# ===== 両モデルの詳細評価 =====
print(f"\n=== 詳細評価（5-fold CV）===")

models = {
    'XGBoost': xgb_grid.best_estimator_,
    'LightGBM': lgb_grid.best_estimator_
}

results = {}

for model_name, model in models.items():
    print(f"\n{model_name} 評価中...")

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

    results[model_name] = {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'test_predictions': test_pred_mean,
        'cv_scores': auc_scores
    }

    print(f"{model_name} Mean AUC: {round(mean_auc, 4)} (±{round(std_auc, 4)})")
    print(f"{model_name} CV scores: {[round(score, 4) for score in auc_scores]}")

# ===== 結果比較 =====
print(f"\n=== 最終比較 ===")
best_model = max(results.keys(), key=lambda x: results[x]['mean_auc'])
print(f"Best Model: {best_model}")
print(f"Best AUC: {round(results[best_model]['mean_auc'], 4)}")

print(f"\n全結果:")
for model_name in results:
    mean_auc = results[model_name]['mean_auc']
    std_auc = results[model_name]['std_auc']
    print(f"{model_name:12}: {round(mean_auc, 4)} (±{round(std_auc, 4)})")

# 最高性能モデルのテスト予測を保存
best_test_predictions = results[best_model]['test_predictions']

# ファイル作成
submission = pd.read_csv(PATH + 'sample_submission.csv')
submission['Drafted'] = best_test_predictions
submission.to_csv(PATH + 'v6_XGB.csv', index=False)