# HR データ深度価値発掘分析システム
# Phase 3: 離職予測、報酬最適化、組織効能全方位分析
# Google Colab専用設計

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# グラフスタイル設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("🚀 HR データ深度価値発掘分析システム")
print("="*80)
print("📋 分析モジュール: A.離職予測とコスト最適化 | B.報酬最適化分析 | C.組織効能向上")
print("="*80)

# データ読み込み
"""
# 実際のデータを読み込む
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('data.csv')
"""

# サンプルデータ生成（実際のデータがある場合は、この部分をコメントアウトしてください）
def create_comprehensive_hr_data():
    """包括的 HR データセット作成"""
    np.random.seed(42)
    n = 1470
    
    departments = ['Sales', 'Research & Development', 'Human Resources', 'Marketing', 'Finance']
    job_roles = ['Sales Executive', 'Research Scientist', 'HR Specialist', 'Manager', 
                'Marketing Specialist', 'Analyst', 'Director', 'Technician']
    education_fields = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Business']
    
    data = {
        'Age': np.random.normal(37, 10, n).astype(int),
        'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
        'Department': np.random.choice(departments, n),
        'JobRole': np.random.choice(job_roles, n),
        'JobLevel': np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'MonthlyIncome': np.random.normal(6500, 2500, n).astype(int),
        'PerformanceRating': np.random.choice([1, 2, 3, 4], n, p=[0.05, 0.15, 0.65, 0.15]),
        'JobSatisfaction': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.5, 0.2]),
        'EnvironmentSatisfaction': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.5, 0.2]),
        'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.5, 0.2]),
        'RelationshipSatisfaction': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.5, 0.2]),
        'StressRating': np.random.normal(2.7, 1.2, n),
        'OverTime': np.random.choice([0, 1], n, p=[0.72, 0.28]),
        'RemoteWork': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'FlexibleWork': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'TotalWorkingYears': np.random.exponential(10, n).astype(int),
        'YearsAtCompany': np.random.exponential(7, n).astype(int),
        'YearsInCurrentRole': np.random.exponential(4, n).astype(int),
        'YearsSinceLastPromotion': np.random.exponential(2, n).astype(int),
        'TrainingTimesLastYear': np.random.poisson(2.8, n),
        'MonthlyAchievement': np.random.normal(3.2, 0.8, n),
        'PerformanceIndex': np.random.normal(3.1, 0.6, n),
        'DistanceFromHome': np.random.exponential(9, n).astype(int),
        'Education': np.random.choice([1, 2, 3, 4, 5], n),
        'EducationField': np.random.choice(education_fields, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n),
        'NumCompaniesWorked': np.random.poisson(2.7, n),
        'StockOptionLevel': np.random.choice([0, 1, 2, 3], n),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n)
    }
    
    df = pd.DataFrame(data)
    
    # ロジック関連を作成し、データをより現実的にする
    # 低満足度 + 高ストレス = 高離職確率
    attrition_prob = np.random.random(n)
    
    # 満足度要因
    low_satisfaction = (df['JobSatisfaction'] <= 2)
    attrition_prob[low_satisfaction] += 0.3
    
    # ストレス要因
    high_stress = (df['StressRating'] > df['StressRating'].quantile(0.8))
    attrition_prob[high_stress] += 0.25
    
    # 報酬要因
    low_income = (df['MonthlyIncome'] < df['MonthlyIncome'].quantile(0.3))
    attrition_prob[low_income] += 0.2
    
    # 残業要因
    overtime = (df['OverTime'] == 1)
    attrition_prob[overtime] += 0.2
    
    # 離職ラベルを更新
    df['Attrition'] = np.where(attrition_prob > 0.7, 'Yes', 'No')
    
    return df

# サンプルデータを使用
df = create_comprehensive_hr_data()
print(f"✅ データ読み込み完了: {df.shape[0]} 行, {df.shape[1]} 列")

class HRValueMiner:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.results = {}
        
        print(f"📊 データ概観:")
        print(f"   離職率: {(self.df['Attrition'] == 'Yes').mean():.1%}")
        print(f"   部門数: {self.df['Department'].nunique()}")
        print(f"   職種数: {self.df['JobRole'].nunique()}")
    
    # =================== A. 離職予測とコスト最適化 ===================
    
    def build_attrition_risk_model(self):
        """離職リスクスコアモデル構築"""
        print("\n" + "="*60)
        print("🎯 A1. 離職リスクスコアモデル構築")
        print("="*60)
        
        # データ前処理
        X = self.df.copy()
        y = (X['Attrition'] == 'Yes').astype(int)
        X = X.drop(['Attrition'], axis=1)
        
        # カテゴリカル変数のエンコード
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 複数モデルの訓練
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        model_results = {}
        
        for name, model in models.items():
            # モデル訓練
            model.fit(X_train, y_train)
            
            # 予測
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 評価
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            model_results[name] = {
                'model': model,
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"📈 {name}:")
            print(f"   AUC: {auc_score:.3f}")
            print(f"   CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # 最適モデルの選択
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
        best_model = model_results[best_model_name]['model']
        
        print(f"\n🏆 最適モデル: {best_model_name}")
        
        # リスクスコア計算
        risk_scores = best_model.predict_proba(X)[:, 1]
        self.df['AttritionRiskScore'] = risk_scores
        
        # リスクレベル分類
        self.df['RiskLevel'] = pd.cut(risk_scores, 
                                     bins=[0, 0.3, 0.6, 0.8, 1.0],
                                     labels=['低リスク', '中リスク', '高リスク', '極高リスク'])
        
        # リスク分布統計
        risk_distribution = self.df['RiskLevel'].value_counts()
        print(f"\n📊 リスクレベル分布:")
        for level, count in risk_distribution.items():
            percentage = count / len(self.df) * 100
            print(f"   {level}: {count}人 ({percentage:.1f}%)")
        
        # 特徴重要度（ランダムフォレストの場合）
        if best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 TOP10 離職予測重要要因:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"   {i}. {row['feature']}: {row['importance']:.3f}")
        
        # 可視化
        plt.figure(figsize=(15, 5))
        
        # リスク分布
        plt.subplot(1, 3, 1)
        risk_distribution.plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('従業員離職リスクレベル分布')
        plt.ylabel('従業員数')
        plt.xticks(rotation=45)
        
        # ROC曲線
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {model_results[best_model_name]["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC曲線')
        plt.legend()
        
        # リスクスコア分布
        plt.subplot(1, 3, 3)
        plt.hist(risk_scores, bins=30, alpha=0.7, color='orange')
        plt.xlabel('離職リスクスコア')
        plt.ylabel('従業員数')
        plt.title('リスクスコア分布')
        
        plt.tight_layout()
        plt.show()
        
        self.results['attrition_model'] = {
            'best_model': best_model,
            'model_name': best_model_name,
            'feature_importance': feature_importance if best_model_name == 'Random Forest' else None,
            'risk_distribution': risk_distribution
        }
        
        return best_model, risk_scores
    
    def calculate_replacement_costs(self):
        """各部門の人材代替コスト計算"""
        print("\n" + "="*60)
        print("💰 A2. 部門人材代替コスト分析")
        print("="*60)
        
        # 基本コスト仮定
        RECRUITMENT_COST_RATIO = 0.3  # 採用コストは年収の30%
        TRAINING_COST_RATIO = 0.2     # 研修コストは年収の20%
        PRODUCTIVITY_LOSS_RATIO = 0.25 # 生産性損失は年収の25%
        
        # 部門別計算
        dept_analysis = self.df.groupby('Department').agg({
            'MonthlyIncome': ['mean', 'count'],
            'Attrition': lambda x: (x == 'Yes').sum(),
            'AttritionRiskScore': 'mean' if 'AttritionRiskScore' in self.df.columns else lambda x: 0
        }).round(2)
        
        dept_analysis.columns = ['平均月給', '従業員総数', '実際離職人数', '平均リスクスコア']
        
        # 代替コスト計算
        dept_analysis['年収'] = dept_analysis['平均月給'] * 12
        dept_analysis['単人代替コスト'] = dept_analysis['年収'] * (RECRUITMENT_COST_RATIO + TRAINING_COST_RATIO + PRODUCTIVITY_LOSS_RATIO)
        dept_analysis['年度離職コスト'] = dept_analysis['単人代替コスト'] * dept_analysis['実際離職人数']
        dept_analysis['離職率'] = dept_analysis['実際離職人数'] / dept_analysis['従業員総数']
        
        # 将来リスク予測
        if 'AttritionRiskScore' in self.df.columns:
            dept_analysis['予測離職人数'] = (dept_analysis['従業員総数'] * dept_analysis['平均リスクスコア']).round(0)
            dept_analysis['予測年度コスト'] = dept_analysis['単人代替コスト'] * dept_analysis['予測離職人数']
        
        print(f"📊 各部門人材代替コスト分析:")
        print(dept_analysis[['従業員総数', '離職率', '単人代替コスト', '年度離職コスト']].to_string())
        
        # 全体コスト統計
        total_current_cost = dept_analysis['年度離職コスト'].sum()
        total_predicted_cost = dept_analysis['予測年度コスト'].sum() if 'AttritionRiskScore' in self.df.columns else 0
        
        print(f"\n💸 コスト総計:")
        print(f"   現在年度総離職コスト: ${total_current_cost:,.0f}")
        if total_predicted_cost > 0:
            print(f"   予測年度総離職コスト: ${total_predicted_cost:,.0f}")
            print(f"   コスト変化: ${total_predicted_cost - total_current_cost:+,.0f}")
        
        # 可視化
        plt.figure(figsize=(15, 10))
        
        # 各部門離職コスト
        plt.subplot(2, 2, 1)
        dept_analysis['年度離職コスト'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('各部門年度離職コスト')
        plt.ylabel('コスト ($)')
        plt.xticks(rotation=45)
        
        # 離職率比較
        plt.subplot(2, 2, 2)
        dept_analysis['離職率'].plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('各部門離職率')
        plt.ylabel('離職率')
        plt.xticks(rotation=45)
        
        # コスト構成円グラフ
        plt.subplot(2, 2, 3)
        cost_components = {
            '採用コスト': RECRUITMENT_COST_RATIO,
            '研修コスト': TRAINING_COST_RATIO,
            '生産性損失': PRODUCTIVITY_LOSS_RATIO
        }
        plt.pie(cost_components.values(), labels=cost_components.keys(), autopct='%1.1f%%')
        plt.title('代替コスト構成')
        
        # 部門従業員数
        plt.subplot(2, 2, 4)
        dept_analysis['従業員総数'].plot(kind='bar', color='lightgreen', alpha=0.8)
        plt.title('各部門従業員数')
        plt.ylabel('従業員数')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        self.results['replacement_costs'] = dept_analysis
        
        return dept_analysis
    
    def identify_hidden_flight_risk(self):
        """隠れ離職従業員識別"""
        print("\n" + "="*60)
        print("👻 A3. 隠れ離職従業員識別")
        print("="*60)
        
        # 隠れ離職条件定義：低満足度だが未離職
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_cols if col in self.df.columns]
        
        if len(available_satisfaction) == 0:
            print("❌ 満足度データ不足")
            return None
        
        # 総合満足度計算
        self.df['OverallSatisfaction'] = self.df[available_satisfaction].mean(axis=1)
        
        # 隠れ離職条件
        conditions = {
            '低満足度': self.df['OverallSatisfaction'] <= 2.0,
            '在職状態': self.df['Attrition'] == 'No',
            '高ストレス': self.df['StressRating'] > self.df['StressRating'].quantile(0.7) if 'StressRating' in self.df.columns else False
        }
        
        # 基本隠れ離職群：低満足度 + 在職
        hidden_flight_basic = self.df[conditions['低満足度'] & conditions['在職状態']]
        
        # 高リスク隠れ離職：基本条件 + 高ストレス
        if 'StressRating' in self.df.columns:
            hidden_flight_high_risk = self.df[
                conditions['低満足度'] & 
                conditions['在職状態'] & 
                conditions['高ストレス']
            ]
        else:
            hidden_flight_high_risk = hidden_flight_basic
        
        print(f"📊 隠れ離職従業員識別結果:")
        print(f"   基本隠れ離職群: {len(hidden_flight_basic)}人 ({len(hidden_flight_basic)/len(self.df)*100:.1f}%)")
        print(f"   高リスク隠れ離職群: {len(hidden_flight_high_risk)}人 ({len(hidden_flight_high_risk)/len(self.df)*100:.1f}%)")
        
        # 隠れ離職群特徴分析
        if len(hidden_flight_basic) > 0:
            print(f"\n🔍 隠れ離職群特徴分析:")
            
            # 部門分布
            dept_distribution = hidden_flight_basic['Department'].value_counts()
            print(f"   部門分布: {dict(dept_distribution.head(3))}")
            
            # 職種分布
            role_distribution = hidden_flight_basic['JobRole'].value_counts()
            print(f"   職種分布: {dict(role_distribution.head(3))}")
            
            # 重要数値特徴
            key_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
            available_features = [f for f in key_features if f in self.df.columns]
            
            if available_features:
                print(f"\n📈 重要特徴比較 (隠れ離職 vs 正常従業員):")
                normal_employees = self.df[
                    (self.df['OverallSatisfaction'] > 2.5) & 
                    (self.df['Attrition'] == 'No')
                ]
                
                for feature in available_features:
                    hidden_mean = hidden_flight_basic[feature].mean()
                    normal_mean = normal_employees[feature].mean()
                    diff = hidden_mean - normal_mean
                    
                    print(f"   {feature}: {hidden_mean:.1f} vs {normal_mean:.1f} (差異: {diff:+.1f})")
        
        # リスクスコアがある場合、リスク分布分析
        if 'AttritionRiskScore' in self.df.columns and len(hidden_flight_basic) > 0:
            avg_risk_hidden = hidden_flight_basic['AttritionRiskScore'].mean()
            avg_risk_normal = self.df[self.df['Attrition'] == 'No']['AttritionRiskScore'].mean()
            
            print(f"\n⚠️ リスクスコア比較:")
            print(f"   隠れ離職群平均リスク: {avg_risk_hidden:.3f}")
            print(f"   正常従業員平均リスク: {avg_risk_normal:.3f}")
            print(f"   リスク差異: {avg_risk_hidden - avg_risk_normal:+.3f}")
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 満足度分布比較
        axes[0, 0].hist(self.df[self.df['Attrition'] == 'No']['OverallSatisfaction'], 
                       alpha=0.7, label='正常従業員', bins=20, color='green')
        axes[0, 0].hist(hidden_flight_basic['OverallSatisfaction'], 
                       alpha=0.7, label='隠れ離職', bins=20, color='red')
        axes[0, 0].set_title('満足度分布比較')
        axes[0, 0].set_xlabel('総合満足度')
        axes[0, 0].legend()
        
        # 部門分布
        if len(hidden_flight_basic) > 0:
            dept_dist = hidden_flight_basic['Department'].value_counts()
            axes[0, 1].bar(range(len(dept_dist)), dept_dist.values, color='orange', alpha=0.8)
            axes[0, 1].set_title('隠れ離職従業員部門分布')
            axes[0, 1].set_xticks(range(len(dept_dist)))
            axes[0, 1].set_xticklabels(dept_dist.index, rotation=45)
        
        # リスクスコア分布（ある場合）
        if 'AttritionRiskScore' in self.df.columns:
            axes[1, 0].hist(self.df[self.df['Attrition'] == 'No']['AttritionRiskScore'], 
                           alpha=0.7, label='正常従業員', bins=20, color='blue')
            if len(hidden_flight_basic) > 0:
                axes[1, 0].hist(hidden_flight_basic['AttritionRiskScore'], 
                               alpha=0.7, label='隠れ離職', bins=20, color='red')
            axes[1, 0].set_title('リスクスコア分布比較')
            axes[1, 0].set_xlabel('離職リスクスコア')
            axes[1, 0].legend()
        
        # 年齢分布比較
        if 'Age' in self.df.columns:
            axes[1, 1].hist(self.df[self.df['Attrition'] == 'No']['Age'], 
                           alpha=0.7, label='正常従業員', bins=20, color='green')
            if len(hidden_flight_basic) > 0:
                axes[1, 1].hist(hidden_flight_basic['Age'], 
                               alpha=0.7, label='隠れ離職', bins=20, color='red')
            axes[1, 1].set_title('年齢分布比較')
            axes[1, 1].set_xlabel('年齢')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 介入提案生成
        print(f"\n💡 介入提案:")
        if len(hidden_flight_basic) > 0:
            recommendations = [
                f"即座に{len(hidden_flight_high_risk)}名の高リスク隠れ離職従業員に注目",
                "満足度向上専門行動を展開、職場環境とワークライフバランスに重点",
                "定期的コミュニケーション機構を構築、従業員の真の想いを理解",
                "職種調整やキャリア発展機会を検討"
            ]
            
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        self.results['hidden_flight_risk'] = {
            'basic_count': len(hidden_flight_basic),
            'high_risk_count': len(hidden_flight_high_risk),
            'basic_group': hidden_flight_basic,
            'high_risk_group': hidden_flight_high_risk
        }
        
        return hidden_flight_basic, hidden_flight_high_risk
    
    # =================== B. 報酬最適化と公平性分析 ===================
    
    def analyze_compensation_equity(self):
        """同職種報酬格差分析"""
        print("\n" + "="*60)
        print("⚖️ B1. 同職種報酬公平性分析")
        print("="*60)
        
        # 職種別報酬分布分析
        job_salary_stats = self.df.groupby('JobRole')['MonthlyIncome'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(0)
        
        job_salary_stats.columns = ['従業員数', '平均値', '中央値', '標準偏差', '最小値', '最大値']
        job_salary_stats['変動係数'] = (job_salary_stats['標準偏差'] / job_salary_stats['平均値']).round(3)
        job_salary_stats['報酬範囲'] = job_salary_stats['最大値'] - job_salary_stats['最小値']
        
        # 十分な従業員数の職種に絞って分析
        significant_roles = job_salary_stats[job_salary_stats['従業員数'] >= 10]
        
        print(f"📊 主要職種報酬統計 (従業員数≥10):")
        print(significant_roles[['従業員数', '平均値', '中央値', '変動係数']].to_string())
        
        # 報酬格差が大きい職種を識別
        high_variance_roles = significant_roles[significant_roles['変動係数'] > 0.3]
        
        if len(high_variance_roles) > 0:
            print(f"\n⚠️ 報酬格差が大きい職種 (変動係数>0.3):")
            for role in high_variance_roles.index:
                cv = high_variance_roles.loc[role, '変動係数']
                range_val = high_variance_roles.loc[role, '報酬範囲']
                print(f"   {role}: 変動係数{cv:.3f}, 報酬範囲${range_val:,.0f}")
        
        # 性別報酬公平性分析
        if 'Gender' in self.df.columns:
            print(f"\n👥 性別報酬公平性分析:")
            
            gender_salary = self.df.groupby(['JobRole', 'Gender'])['MonthlyIncome'].mean().unstack()
            if gender_salary.shape[1] == 2:  # 男女両性別があることを確認
                gender_salary['報酬差異'] = gender_salary.iloc[:, 0] - gender_salary.iloc[:, 1]
                gender_salary['差異百分比'] = (gender_salary['報酬差異'] / gender_salary.mean(axis=1) * 100).round(1)
                
                # 差異が大きい職種を見つける
                significant_gaps = gender_salary[abs(gender_salary['差異百分比']) > 10]
                
                if len(significant_gaps) > 0:
                    print(f"   {len(significant_gaps)}職種で顕著な性別報酬差異(>10%)を発見:")
                    for role in significant_gaps.index:
                        gap = significant_gaps.loc[role, '差異百分比']
                        print(f"   {role}: {gap:+.1f}%")
        
        # 学歴と報酬関係
        if 'Education' in self.df.columns:
            edu_salary = self.df.groupby('Education')['MonthlyIncome'].mean().sort_index()
            print(f"\n🎓 学歴と報酬関係:")
            for edu_level, salary in edu_salary.items():
                print(f"   学歴レベル{edu_level}: ${salary:,.0f}")
        
        # 可視化
        plt.figure(figsize=(18, 12))
        
        # 職種報酬分布箱ひげ図
        plt.subplot(2, 3, 1)
        roles_to_plot = significant_roles.head(6).index
        salary_data = [self.df[self.df['JobRole'] == role]['MonthlyIncome'] for role in roles_to_plot]
        plt.boxplot(salary_data, labels=roles_to_plot)
        plt.title('主要職種報酬分布')
        plt.ylabel('月給 ($)')
        plt.xticks(rotation=45)
        
        # 報酬変動係数
        plt.subplot(2, 3, 2)
        significant_roles['変動係数'].plot(kind='bar', color='orange', alpha=0.8)
        plt.title('職種報酬変動係数')
        plt.ylabel('変動係数')
        plt.xticks(rotation=45)
        
        # 性別報酬比較（データがある場合）
        if 'Gender' in self.df.columns:
            plt.subplot(2, 3, 3)
            self.df.boxplot(column='MonthlyIncome', by='Gender', ax=plt.gca())
            plt.title('性別報酬分布比較')
            plt.suptitle('')
        
        # 学歴報酬関係
        if 'Education' in self.df.columns:
            plt.subplot(2, 3, 4)
            self.df.boxplot(column='MonthlyIncome', by='Education', ax=plt.gca())
            plt.title('学歴と報酬関係')
            plt.suptitle('')
        
        # 報酬範囲分析
        plt.subplot(2, 3, 5)
        significant_roles['報酬範囲'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('職種報酬範囲')
        plt.ylabel('報酬範囲 ($)')
        plt.xticks(rotation=45)
        
        # 全体報酬分布
        plt.subplot(2, 3, 6)
        plt.hist(self.df['MonthlyIncome'], bins=30, alpha=0.7, color='skyblue')
        plt.title('全体報酬分布')
        plt.xlabel('月給 ($)')
        plt.ylabel('従業員数')
        
        plt.tight_layout()
        plt.show()
        
        self.results['compensation_equity'] = {
            'job_salary_stats': job_salary_stats,
            'high_variance_roles': high_variance_roles,
            'gender_salary': gender_salary if 'Gender' in self.df.columns else None
        }
        
        return job_salary_stats, high_variance_roles
    
    def evaluate_performance_compensation_alignment(self):
        """業績と報酬マッチング度評価"""
        print("\n" + "="*60)
        print("🎯 B2. 業績と報酬マッチング度評価")
        print("="*60)
        
        # 業績と報酬相関性
        performance_cols = ['PerformanceRating', 'PerformanceIndex', 'MonthlyAchievement']
        available_perf = [col for col in performance_cols if col in self.df.columns]
        
        if len(available_perf) == 0:
            print("❌ 業績データ不足")
            return None
        
        print(f"📊 業績と報酬相関性分析:")
        correlations = {}
        
        for perf_col in available_perf:
            corr = self.df[perf_col].corr(self.df['MonthlyIncome'])
            correlations[perf_col] = corr
            print(f"   {perf_col} と報酬相関係数: {corr:.3f}")
        
        # 主要業績指標で深度分析
        main_perf_col = max(correlations, key=correlations.get)
        print(f"\n🎯 {main_perf_col}を主要業績指標として深度分析")
        
        # 業績グループ作成
        perf_groups = pd.qcut(self.df[main_perf_col], q=4, labels=['低業績', '中下業績', '中上業績', '高業績'])
        self.df['PerformanceGroup'] = perf_groups
        
        # 各業績グループ報酬統計
        perf_salary_stats = self.df.groupby('PerformanceGroup')['MonthlyIncome'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(0)
        
        print(f"\n📈 各業績グループ報酬統計:")
        print(perf_salary_stats.to_string())
        
        # 報酬不一致状況識別
        # 高業績低報酬
        high_perf_threshold = self.df[main_perf_col].quantile(0.8)
        low_salary_threshold = self.df['MonthlyIncome'].quantile(0.3)
        
        high_perf_low_pay = self.df[
            (self.df[main_perf_col] >= high_perf_threshold) & 
            (self.df['MonthlyIncome'] <= low_salary_threshold)
        ]
        
        # 低業績高報酬
        low_perf_threshold = self.df[main_perf_col].quantile(0.2)
        high_salary_threshold = self.df['MonthlyIncome'].quantile(0.8)
        
        low_perf_high_pay = self.df[
            (self.df[main_perf_col] <= low_perf_threshold) & 
            (self.df['MonthlyIncome'] >= high_salary_threshold)
        ]
        
        print(f"\n⚠️ 報酬不一致状況:")
        print(f"   高業績低報酬: {len(high_perf_low_pay)}人 ({len(high_perf_low_pay)/len(self.df)*100:.1f}%)")
        print(f"   低業績高報酬: {len(low_perf_high_pay)}人 ({len(low_perf_high_pay)/len(self.df)*100:.1f}%)")
        
        # 報酬公平性指数計算
        expected_salary = self.df.groupby('PerformanceGroup')['MonthlyIncome'].transform('mean')
        actual_salary = self.df['MonthlyIncome']
        fairness_index = 1 - abs(actual_salary - expected_salary) / expected_salary
        self.df['SalaryFairnessIndex'] = fairness_index
        
        avg_fairness = fairness_index.mean()
        print(f"\n📊 報酬公平性指数: {avg_fairness:.3f} (1.0が完全公平)")
        
        # 不一致従業員の離職リスク
        if len(high_perf_low_pay) > 0:
            high_perf_low_pay_attrition = (high_perf_low_pay['Attrition'] == 'Yes').mean()
            print(f"   高業績低報酬従業員離職率: {high_perf_low_pay_attrition:.1%}")
        
        if len(low_perf_high_pay) > 0:
            low_perf_high_pay_attrition = (low_perf_high_pay['Attrition'] == 'Yes').mean()
            print(f"   低業績高報酬従業員離職率: {low_perf_high_pay_attrition:.1%}")
        
        # 可視化
        plt.figure(figsize=(15, 10))
        
        # 業績vs報酬散布図
        plt.subplot(2, 3, 1)
        colors = ['red' if x == 'Yes' else 'blue' for x in self.df['Attrition']]
        plt.scatter(self.df[main_perf_col], self.df['MonthlyIncome'], c=colors, alpha=0.6)
        plt.xlabel(main_perf_col)
        plt.ylabel('月給 ($)')
        plt.title(f'{main_perf_col} vs 報酬 (赤=離職)')
        
        # 各業績グループ報酬分布
        plt.subplot(2, 3, 2)
        self.df.boxplot(column='MonthlyIncome', by='PerformanceGroup', ax=plt.gca())
        plt.title('各業績グループ報酬分布')
        plt.suptitle('')
        
        # 報酬公平性指数分布
        plt.subplot(2, 3, 3)
        plt.hist(fairness_index, bins=30, alpha=0.7, color='green')
        plt.xlabel('報酬公平性指数')
        plt.ylabel('従業員数')
        plt.title('報酬公平性指数分布')
        
        # 不一致状況可視化
        plt.subplot(2, 3, 4)
        mismatch_data = [len(high_perf_low_pay), len(low_perf_high_pay)]
        mismatch_labels = ['高業績低報酬', '低業績高報酬']
        plt.bar(mismatch_labels, mismatch_data, color=['orange', 'red'], alpha=0.8)
        plt.title('報酬不一致従業員数')
        plt.ylabel('従業員数')
        
        # 業績グループ報酬平均値
        plt.subplot(2, 3, 5)
        perf_salary_stats['mean'].plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('各業績グループ平均報酬')
        plt.ylabel('平均月給 ($)')
        plt.xticks(rotation=45)
        
        # 相関性ヒートマップ
        plt.subplot(2, 3, 6)
        corr_data = self.df[available_perf + ['MonthlyIncome']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
        plt.title('業績指標と報酬相関性')
        
        plt.tight_layout()
        plt.show()
        
        self.results['performance_compensation'] = {
            'correlations': correlations,
            'high_perf_low_pay': high_perf_low_pay,
            'low_perf_high_pay': low_perf_high_pay,
            'fairness_index': avg_fairness
        }
        
        return correlations, high_perf_low_pay, low_perf_high_pay
    
    def market_competitiveness_analysis(self):
        """市場報酬競争力分析"""
        print("\n" + "="*60)
        print("🏢 B3. 市場報酬競争力分析")
        print("="*60)
        
        # 市場報酬データシミュレーション（実際の応用では真実の市場調査データを使用する必要がある）
        # 市場報酬が社内報酬より10-20%高いと仮定
        market_multiplier = {
            'Sales': 1.15,
            'Research & Development': 1.20,
            'Human Resources': 1.10,
            'Marketing': 1.18,
            'Finance': 1.22
        }
        
        print(f"📊 各部門市場競争力分析:")
        print(f"注: シミュレーション市場データに基づく、実際の応用には真実の市場調査データが必要")
        
        dept_competitiveness = []
        
        for dept in self.df['Department'].unique():
            dept_data = self.df[self.df['Department'] == dept]
            internal_avg = dept_data['MonthlyIncome'].mean()
            
            # 事前設定の市場倍数を使用
            multiplier = market_multiplier.get(dept, 1.15)
            market_avg = internal_avg * multiplier
            
            competitiveness_gap = (internal_avg - market_avg) / market_avg * 100
            
            dept_competitiveness.append({
                'Department': dept,
                'Internal_Avg': internal_avg,
                'Market_Avg': market_avg,
                'Gap_Percentage': competitiveness_gap,
                'Competitiveness': '競争力強' if competitiveness_gap > -5 else 
                                 '普通' if competitiveness_gap > -15 else '競争力弱'
            })
            
            print(f"   {dept}:")
            print(f"     社内平均: ${internal_avg:,.0f}")
            print(f"     市場平均: ${market_avg:,.0f}")
            print(f"     競争力差: {competitiveness_gap:+.1f}%")
        
        competitiveness_df = pd.DataFrame(dept_competitiveness)
        
        # 職種レベル競争力分析
        if 'JobLevel' in self.df.columns:
            print(f"\n📈 異なる職種レベル競争力分析:")
            level_competitiveness = []
            
            for level in sorted(self.df['JobLevel'].unique()):
                level_data = self.df[self.df['JobLevel'] == level]
                internal_avg = level_data['MonthlyIncome'].mean()
                
                # 高レベル職種の市場プレミアムはより高い
                market_multiplier_level = 1.1 + (level - 1) * 0.05
                market_avg = internal_avg * market_multiplier_level
                
                gap = (internal_avg - market_avg) / market_avg * 100
                
                level_competitiveness.append({
                    'Level': f'Level {level}',
                    'Internal_Avg': internal_avg,
                    'Market_Avg': market_avg,
                    'Gap': gap
                })
                
                print(f"   Level {level}: 社内${internal_avg:,.0f} vs 市場${market_avg:,.0f} ({gap:+.1f}%)")
        
        # 高リスク離職の報酬競争力
        if 'AttritionRiskScore' in self.df.columns:
            high_risk_employees = self.df[self.df['AttritionRiskScore'] > 0.7]
            
            if len(high_risk_employees) > 0:
                print(f"\n⚠️ 高離職リスク従業員報酬競争力:")
                
                for dept in high_risk_employees['Department'].unique():
                    dept_high_risk = high_risk_employees[high_risk_employees['Department'] == dept]
                    if len(dept_high_risk) > 0:
                        avg_salary = dept_high_risk['MonthlyIncome'].mean()
                        dept_market_avg = competitiveness_df[
                            competitiveness_df['Department'] == dept
                        ]['Market_Avg'].iloc[0]
                        
                        gap = (avg_salary - dept_market_avg) / dept_market_avg * 100
                        print(f"   {dept}: {len(dept_high_risk)}人, 平均報酬${avg_salary:,.0f} ({gap:+.1f}%)")
        
        # 可視化
        plt.figure(figsize=(15, 10))
        
        # 部門競争力比較
        plt.subplot(2, 3, 1)
        x_pos = range(len(competitiveness_df))
        plt.bar(x_pos, competitiveness_df['Internal_Avg'], alpha=0.7, label='社内平均', color='blue')
        plt.bar(x_pos, competitiveness_df['Market_Avg'], alpha=0.7, label='市場平均', color='red')
        plt.xlabel('部門')
        plt.ylabel('平均月給 ($)')
        plt.title('社内 vs 市場報酬比較')
        plt.xticks(x_pos, competitiveness_df['Department'], rotation=45)
        plt.legend()
        
        # 競争力差
        plt.subplot(2, 3, 2)
        colors = ['green' if x > -5 else 'orange' if x > -15 else 'red' 
                 for x in competitiveness_df['Gap_Percentage']]
        plt.bar(x_pos, competitiveness_df['Gap_Percentage'], color=colors, alpha=0.8)
        plt.xlabel('部門')
        plt.ylabel('競争力差 (%)')
        plt.title('各部門報酬競争力差')
        plt.xticks(x_pos, competitiveness_df['Department'], rotation=45)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 職種レベル競争力（データがある場合）
        if 'JobLevel' in self.df.columns:
            plt.subplot(2, 3, 3)
            level_df = pd.DataFrame(level_competitiveness)
            plt.plot(level_df['Level'], level_df['Internal_Avg'], 'o-', label='社内平均', linewidth=2)
            plt.plot(level_df['Level'], level_df['Market_Avg'], 's-', label='市場平均', linewidth=2)
            plt.xlabel('職種レベル')
            plt.ylabel('平均月給 ($)')
            plt.title('異なるレベル報酬競争力')
            plt.legend()
            plt.xticks(rotation=45)
        
        # 競争力分布円グラフ
        plt.subplot(2, 3, 4)
        competitiveness_counts = competitiveness_df['Competitiveness'].value_counts()
        plt.pie(competitiveness_counts.values, labels=competitiveness_counts.index, autopct='%1.1f%%')
        plt.title('部門競争力分布')
        
        # 報酬分布と市場基準線
        plt.subplot(2, 3, 5)
        plt.hist(self.df['MonthlyIncome'], bins=30, alpha=0.7, color='skyblue', label='社内報酬分布')
        
        # 市場基準線追加
        overall_market_avg = competitiveness_df['Market_Avg'].mean()
        plt.axvline(x=overall_market_avg, color='red', linestyle='--', linewidth=2, label=f'市場平均線')
        plt.xlabel('月給 ($)')
        plt.ylabel('従業員数')
        plt.title('報酬分布 vs 市場基準')
        plt.legend()
        
        # 高リスク従業員報酬分布
        plt.subplot(2, 3, 6)
        if 'AttritionRiskScore' in self.df.columns:
            high_risk = self.df[self.df['AttritionRiskScore'] > 0.7]
            low_risk = self.df[self.df['AttritionRiskScore'] <= 0.3]
            
            plt.hist(low_risk['MonthlyIncome'], alpha=0.7, label='低リスク従業員', bins=20, color='green')
            plt.hist(high_risk['MonthlyIncome'], alpha=0.7, label='高リスク従業員', bins=20, color='red')
            plt.xlabel('月給 ($)')
            plt.ylabel('従業員数')
            plt.title('異なるリスク従業員報酬分布')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 競争力向上提案
        print(f"\n💡 報酬競争力向上提案:")
        
        weak_depts = competitiveness_df[competitiveness_df['Gap_Percentage'] < -10]
        if len(weak_depts) > 0:
            print(f"   優先調整部門: {', '.join(weak_depts['Department'].tolist())}")
            
        total_adjustment_cost = 0
        for _, dept in weak_depts.iterrows():
            dept_employees = len(self.df[self.df['Department'] == dept['Department']])
            monthly_adjustment = abs(dept['Internal_Avg'] - dept['Market_Avg'])
            annual_cost = monthly_adjustment * dept_employees * 12
            total_adjustment_cost += annual_cost
            
            print(f"   {dept['Department']}: 調整必要${monthly_adjustment:,.0f}/月/人, 年コスト${annual_cost:,.0f}")
        
        if total_adjustment_cost > 0:
            print(f"   総調整コスト: ${total_adjustment_cost:,.0f}/年")
            
            # ROI計算
            if 'AttritionRiskScore' in self.df.columns:
                current_attrition_cost = self.results.get('replacement_costs', {}).get('年度離職コスト', pd.Series()).sum()
                if current_attrition_cost > 0:
                    roi = (current_attrition_cost * 0.3 - total_adjustment_cost) / total_adjustment_cost * 100
                    print(f"   期待ROI: {roi:+.1f}% (報酬調整により30%離職コスト削減と仮定)")
        
        self.results['market_competitiveness'] = competitiveness_df
        
        return competitiveness_df
    
    # =================== C. 組織効能向上 ===================
    
    def identify_high_performance_team_characteristics(self):
        """高業績チーム特徴識別"""
        print("\n" + "="*60)
        print("🏆 C1. 高業績チーム特徴識別")
        print("="*60)
        
        # 高業績チーム定義
        performance_cols = ['PerformanceRating', 'PerformanceIndex', 'MonthlyAchievement']
        available_perf = [col for col in performance_cols if col in self.df.columns]
        
        if len(available_perf) == 0:
            print("❌ 業績データ不足")
            return None
        
        # 総合業績スコア計算
        perf_data = self.df[available_perf].copy()
        # 業績指標標準化
        for col in available_perf:
            perf_data[col] = (perf_data[col] - perf_data[col].mean()) / perf_data[col].std()
        
        self.df['OverallPerformance'] = perf_data.mean(axis=1)
        
        # 部門別平均業績計算
        dept_performance = self.df.groupby('Department').agg({
            'OverallPerformance': 'mean',
            'MonthlyIncome': 'mean',
            'JobSatisfaction': 'mean',
            'Attrition': lambda x: (x == 'Yes').mean(),
            'OverTime': lambda x: (x == 1).mean() if self.df['OverTime'].dtype in [int, float] else (x == 'Yes').mean()
        }).round(3)
        
        dept_performance.columns = ['平均業績', '平均報酬', '平均満足度', '離職率', '残業比率']
        dept_performance = dept_performance.sort_values('平均業績', ascending=False)
        
        print(f"📊 各部門業績表現:")
        print(dept_performance.to_string())
        
        # 高業績部門識別
        high_perf_threshold = dept_performance['平均業績'].quantile(0.7)
        high_perf_depts = dept_performance[dept_performance['平均業績'] >= high_perf_threshold]
        
        print(f"\n🏆 高業績部門: {', '.join(high_perf_depts.index.tolist())}")
        
        # 高業績チーム特徴分析
        high_perf_employees = self.df[self.df['Department'].isin(high_perf_depts.index)]
        normal_perf_employees = self.df[~self.df['Department'].isin(high_perf_depts.index)]
        
        print(f"\n🔍 高業績チーム特徴分析:")
        
        # 勤務モード特徴
        work_mode_features = ['RemoteWork', 'FlexibleWork', 'OverTime']
        available_work_modes = [col for col in work_mode_features if col in self.df.columns]
        
        for feature in available_work_modes:
            if self.df[feature].dtype in [int, float]:
                high_perf_avg = high_perf_employees[feature].mean()
                normal_perf_avg = normal_perf_employees[feature].mean()
            else:
                high_perf_avg = (high_perf_employees[feature] == 'Yes').mean()
                normal_perf_avg = (normal_perf_employees[feature] == 'Yes').mean()
            
            diff = high_perf_avg - normal_perf_avg
            print(f"   {feature}: 高業績{high_perf_avg:.2%} vs 普通{normal_perf_avg:.2%} (差異: {diff:+.1%})")
        
        # 従業員発展特徴
        development_features = ['TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion']
        available_dev = [col for col in development_features if col in self.df.columns]
        
        if available_dev:
            print(f"\n📈 従業員発展特徴:")
            for feature in available_dev:
                high_perf_avg = high_perf_employees[feature].mean()
                normal_perf_avg = normal_perf_employees[feature].mean()
                diff = high_perf_avg - normal_perf_avg
                
                print(f"   {feature}: 高業績{high_perf_avg:.1f} vs 普通{normal_perf_avg:.1f} (差異: {diff:+.1f})")
        
        # 従業員満足度特徴
        satisfaction_features = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_features if col in self.df.columns]
        
        if available_satisfaction:
            print(f"\n😊 従業員満足度特徴:")
            for feature in available_satisfaction:
                high_perf_avg = high_perf_employees[feature].mean()
                normal_perf_avg = normal_perf_employees[feature].mean()
                diff = high_perf_avg - normal_perf_avg
                
                print(f"   {feature}: 高業績{high_perf_avg:.2f} vs 普通{normal_perf_avg:.2f} (差異: {diff:+.2f})")
        
        # 可視化
        plt.figure(figsize=(18, 12))
        
        # 部門業績レーダーチャート
        plt.subplot(2, 3, 1)
        dept_performance_top5 = dept_performance.head(5)
        categories = ['平均業績', '平均満足度', '平均報酬標準化']
        
        # レーダーチャート用の報酬データ標準化
        dept_performance_top5['報酬標準化'] = (dept_performance_top5['平均報酬'] - dept_performance_top5['平均報酬'].min()) / (dept_performance_top5['平均報酬'].max() - dept_performance_top5['平均報酬'].min())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, dept in enumerate(dept_performance_top5.index[:3]):  # 上位3部門のみ表示
            values = [
                dept_performance_top5.loc[dept, '平均業績'],
                dept_performance_top5.loc[dept, '平均満足度'] / 4,  # 0-1に標準化
                dept_performance_top5.loc[dept, '報酬標準化']
            ]
            values += values[:1]
            
            plt.subplot(2, 3, 1, projection='polar')
            plt.plot(angles, values, 'o-', linewidth=2, label=dept)
            plt.fill(angles, values, alpha=0.25)
        
        plt.xticks(angles[:-1], categories)
        plt.title('高業績部門特徴レーダーチャート')
        plt.legend()
        
        # 部門業績ランキング
        plt.subplot(2, 3, 2)
        dept_performance['平均業績'].plot(kind='bar', color='gold', alpha=0.8)
        plt.title('各部門業績ランキング')
        plt.ylabel('平均業績スコア')
        plt.xticks(rotation=45)
        
        # 業績と離職率関係
        plt.subplot(2, 3, 3)
        plt.scatter(dept_performance['平均業績'], dept_performance['離職率'], 
                   s=100, alpha=0.7, color='red')
        plt.xlabel('平均業績')
        plt.ylabel('離職率')
        plt.title('部門業績 vs 離職率')
        
        # 業績と満足度関係
        plt.subplot(2, 3, 4)
        plt.scatter(dept_performance['平均業績'], dept_performance['平均満足度'], 
                   s=100, alpha=0.7, color='blue')
        plt.xlabel('平均業績')
        plt.ylabel('平均満足度')
        plt.title('部門業績 vs 満足度')
        
        # 高業績チーム勤務モード比較
        plt.subplot(2, 3, 5)
        if available_work_modes:
            work_mode_comparison = []
            labels = []
            
            for feature in available_work_modes[:3]:  # 最初の3つのみ表示
                if self.df[feature].dtype in [int, float]:
                    high_perf_avg = high_perf_employees[feature].mean()
                    normal_perf_avg = normal_perf_employees[feature].mean()
                else:
                    high_perf_avg = (high_perf_employees[feature] == 'Yes').mean()
                    normal_perf_avg = (normal_perf_employees[feature] == 'Yes').mean()
                
                work_mode_comparison.extend([high_perf_avg, normal_perf_avg])
                labels.extend([f'{feature}\n(高業績)', f'{feature}\n(普通)'])
            
            colors = ['gold', 'lightblue'] * len(available_work_modes)
            plt.bar(range(len(work_mode_comparison)), work_mode_comparison, color=colors[:len(work_mode_comparison)])
            plt.title('勤務モード比較')
            plt.ylabel('比率')
            plt.xticks(range(len(work_mode_comparison)), labels, rotation=45)
        
        # 業績分布
        plt.subplot(2, 3, 6)
        plt.hist(high_perf_employees['OverallPerformance'], alpha=0.7, label='高業績部門', bins=20, color='gold')
        plt.hist(normal_perf_employees['OverallPerformance'], alpha=0.7, label='普通部門', bins=20, color='lightblue')
        plt.xlabel('総合業績スコア')
        plt.ylabel('従業員数')
        plt.title('業績分布比較')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 高業績チーム成功要素まとめ
        print(f"\n💡 高業績チーム成功要素:")
        
        success_factors = []
        
        # 勤務モード要素
        for feature in available_work_modes:
            if self.df[feature].dtype in [int, float]:
                high_perf_rate = high_perf_employees[feature].mean()
                normal_perf_rate = normal_perf_employees[feature].mean()
            else:
                high_perf_rate = (high_perf_employees[feature] == 'Yes').mean()
                normal_perf_rate = (normal_perf_employees[feature] == 'Yes').mean()
            
            if high_perf_rate > normal_perf_rate * 1.2:
                success_factors.append(f"より多く{feature}を採用")
            elif high_perf_rate < normal_perf_rate * 0.8:
                success_factors.append(f"より少なく{feature}を使用")
        
        # 満足度要素
        for feature in available_satisfaction:
            high_perf_avg = high_perf_employees[feature].mean()
            normal_perf_avg = normal_perf_employees[feature].mean()
            
            if high_perf_avg > normal_perf_avg + 0.3:
                success_factors.append(f"より高い{feature}")
        
        # 発展要素
        if 'TrainingTimesLastYear' in available_dev:
            high_perf_training = high_perf_employees['TrainingTimesLastYear'].mean()
            normal_perf_training = normal_perf_employees['TrainingTimesLastYear'].mean()
            
            if high_perf_training > normal_perf_training * 1.2:
                success_factors.append("より多くの研修投入")
        
        for i, factor in enumerate(success_factors, 1):
            print(f"   {i}. {factor}")
        
        if not success_factors:
            print("   現在のデータに基づいて顕著な差異要素は発見されませんでした")
        
        self.results['high_performance_teams'] = {
            'dept_performance': dept_performance,
            'high_perf_depts': high_perf_depts.index.tolist(),
            'success_factors': success_factors
        }
        
        return dept_performance, high_perf_depts
    
    def evaluate_work_mode_effectiveness(self):
        """勤務モード（リモート/フレキシブル）効果評価"""
        print("\n" + "="*60)
        print("🏠 C2. 勤務モード(リモート/フレキシブル)効果評価")
        print("="*60)
        
        work_modes = ['RemoteWork', 'FlexibleWork', 'OverTime']
        available_modes = [col for col in work_modes if col in self.df.columns]
        
        if len(available_modes) == 0:
            print("❌ 勤務モードデータ不足")
            return None
        
        mode_effectiveness = {}
        
        for mode in available_modes:
            print(f"\n📊 {mode} 効果分析:")
            
            # 異なるデータタイプ処理
            if self.df[mode].dtype in [int, float]:
                mode_yes = self.df[self.df[mode] == 1]
                mode_no = self.df[self.df[mode] == 0]
                yes_label, no_label = "はい", "いいえ"
            else:
                mode_yes = self.df[self.df[mode] == 'Yes']
                mode_no = self.df[self.df[mode] == 'No']
                yes_label, no_label = "Yes", "No"
            
            if len(mode_yes) == 0 or len(mode_no) == 0:
                print(f"   データ不足、{mode}分析をスキップ")
                continue
            
            # 効果指標比較
            metrics = {
                '従業員数': [len(mode_yes), len(mode_no)],
                '離職率': [
                    (mode_yes['Attrition'] == 'Yes').mean(),
                    (mode_no['Attrition'] == 'Yes').mean()
                ],
                '平均業績': [
                    mode_yes['OverallPerformance'].mean() if 'OverallPerformance' in self.df.columns else 0,
                    mode_no['OverallPerformance'].mean() if 'OverallPerformance' in self.df.columns else 0
                ],
                '仕事満足度': [
                    mode_yes['JobSatisfaction'].mean() if 'JobSatisfaction' in self.df.columns else 0,
                    mode_no['JobSatisfaction'].mean() if 'JobSatisfaction' in self.df.columns else 0
                ],
                'ストレスレベル': [
                    mode_yes['StressRating'].mean() if 'StressRating' in self.df.columns else 0,
                    mode_no['StressRating'].mean() if 'StressRating' in self.df.columns else 0
                ]
            }
            
            mode_analysis = {}
            
            for metric, (yes_val, no_val) in metrics.items():
                if yes_val != 0 or no_val != 0:  # 有効データがあることを確認
                    diff = yes_val - no_val
                    if metric == '離職率' or metric == 'ストレスレベル':
                        improvement = "改善" if diff < 0 else "悪化"
                    else:
                        improvement = "改善" if diff > 0 else "悪化"
                    
                    mode_analysis[metric] = {
                        'yes': yes_val,
                        'no': no_val,
                        'diff': diff,
                        'improvement': improvement
                    }
                    
                    if metric == '離職率':
                        print(f"   離職率: {yes_label} {yes_val:.1%} vs {no_label} {no_val:.1%} ({improvement})")
                    elif metric == '従業員数':
                        print(f"   採用比率: {yes_val}/{yes_val+no_val} ({yes_val/(yes_val+no_val):.1%})")
                    else:
                        print(f"   {metric}: {yes_label} {yes_val:.2f} vs {no_label} {no_val:.2f} ({improvement})")
            
            mode_effectiveness[mode] = mode_analysis
        
        # 勤務モード組み合わせ効果分析
        print(f"\n🔄 勤務モード組み合わせ効果分析:")
        
        # 勤務モード組み合わせ作成
        if len(available_modes) >= 2:
            mode1, mode2 = available_modes[0], available_modes[1]
            
            # データタイプ処理
            if self.df[mode1].dtype in [int, float]:
                mode1_condition = self.df[mode1] == 1
            else:
                mode1_condition = self.df[mode1] == 'Yes'
                
            if self.df[mode2].dtype in [int, float]:
                mode2_condition = self.df[mode2] == 1
            else:
                mode2_condition = self.df[mode2] == 'Yes'
            
            # 4つの組み合わせ
            combinations = {
                f'両方採用': mode1_condition & mode2_condition,
                f'{mode1}のみ': mode1_condition & ~mode2_condition,
                f'{mode2}のみ': ~mode1_condition & mode2_condition,
                f'両方不採用': ~mode1_condition & ~mode2_condition
            }
            
            combo_results = {}
            
            for combo_name, combo_mask in combinations.items():
                combo_data = self.df[combo_mask]
                
                if len(combo_data) > 10:  # 十分なサンプル数
                    combo_results[combo_name] = {
                        'count': len(combo_data),
                        'attrition_rate': (combo_data['Attrition'] == 'Yes').mean(),
                        'satisfaction': combo_data['JobSatisfaction'].mean() if 'JobSatisfaction' in self.df.columns else 0,
                        'performance': combo_data['OverallPerformance'].mean() if 'OverallPerformance' in self.df.columns else 0
                    }
                    
                    print(f"   {combo_name}: {len(combo_data)}人, 離職率{combo_results[combo_name]['attrition_rate']:.1%}")
        
        # 可視化
        plt.figure(figsize=(18, 12))
        
        plot_idx = 1
        
        for mode in available_modes:
            if mode in mode_effectiveness:
                # 離職率比較
                plt.subplot(3, len(available_modes), plot_idx)
                
                attrition_data = mode_effectiveness[mode].get('離職率', {})
                if attrition_data:
                    values = [attrition_data['yes'], attrition_data['no']]
                    labels = ['採用', '不採用']
                    colors = ['green' if attrition_data['improvement'] == '改善' else 'red', 'lightblue']
                    
                    plt.bar(labels, values, color=colors, alpha=0.8)
                    plt.title(f'{mode}\n離職率比較')
                    plt.ylabel('離職率')
                
                # 満足度比較
                plt.subplot(3, len(available_modes), plot_idx + len(available_modes))
                
                satisfaction_data = mode_effectiveness[mode].get('仕事満足度', {})
                if satisfaction_data:
                    values = [satisfaction_data['yes'], satisfaction_data['no']]
                    labels = ['採用', '不採用']
                    colors = ['green' if satisfaction_data['improvement'] == '改善' else 'red', 'lightblue']
                    
                    plt.bar(labels, values, color=colors, alpha=0.8)
                    plt.title(f'{mode}\n満足度比較')
                    plt.ylabel('満足度')
                
                # 業績比較
                plt.subplot(3, len(available_modes), plot_idx + 2*len(available_modes))
                
                performance_data = mode_effectiveness[mode].get('平均業績', {})
                if performance_data:
                    values = [performance_data['yes'], performance_data['no']]
                    labels = ['採用', '不採用']
                    colors = ['green' if performance_data['improvement'] == '改善' else 'red', 'lightblue']
                    
                    plt.bar(labels, values, color=colors, alpha=0.8)
                    plt.title(f'{mode}\n業績比較')
                    plt.ylabel('業績スコア')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        # 勤務モード推奨
        print(f"\n💡 勤務モード最適化提案:")
        
        recommendations = []
        
        for mode, analysis in mode_effectiveness.items():
            attrition_improvement = analysis.get('離職率', {}).get('improvement')
            satisfaction_improvement = analysis.get('仕事満足度', {}).get('improvement')
            performance_improvement = analysis.get('平均業績', {}).get('improvement')
            
            positive_effects = sum(1 for imp in [attrition_improvement, satisfaction_improvement, performance_improvement] 
                                 if imp == '改善')
            
            if positive_effects >= 2:
                recommendations.append(f"{mode}政策を推進、積極的効果を示している")
            elif positive_effects == 0:
                recommendations.append(f"{mode}政策を再評価、調整が必要かもしれません")
            else:
                recommendations.append(f"{mode}実施方式を最適化、利害バランスを取る")
        
        # 組み合わせ効果に基づく提案
        if 'combo_results' in locals() and combo_results:
            best_combo = min(combo_results.items(), key=lambda x: x[1]['attrition_rate'])
            recommendations.append(f"推奨勤務モード組み合わせ: {best_combo[0]} (離職率最低: {best_combo[1]['attrition_rate']:.1%})")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        self.results['work_mode_effectiveness'] = {
            'mode_analysis': mode_effectiveness,
            'recommendations': recommendations
        }
        
        return mode_effectiveness
    
    def analyze_training_roi(self):
        """研修ROI分析"""
        print("\n" + "="*60)
        print("📚 C3. 研修ROI分析")
        print("="*60)
        
        if 'TrainingTimesLastYear' not in self.df.columns:
            print("❌ 研修データ不足")
            return None
        
        # 研修投入コスト仮定
        TRAINING_COST_PER_SESSION = 500  # 1回の研修コスト$500
        
        # 研修グループ分け
        training_groups = pd.cut(
            self.df['TrainingTimesLastYear'],
            bins=[-1, 0, 2, 4, 20],
            labels=['研修なし', '少量研修(1-2回)', '適量研修(3-4回)', '大量研修(5回以上)']
        )
        
        self.df['TrainingGroup'] = training_groups
        
        # 各研修グループ効果分析
        training_analysis = self.df.groupby('TrainingGroup').agg({
            'TrainingTimesLastYear': 'mean',
            'Attrition': lambda x: (x == 'Yes').mean(),
            'OverallPerformance': 'mean' if 'OverallPerformance' in self.df.columns else lambda x: 0,
            'JobSatisfaction': 'mean' if 'JobSatisfaction' in self.df.columns else lambda x: 0,
            'MonthlyIncome': 'mean',
            'YearsSinceLastPromotion': 'mean' if 'YearsSinceLastPromotion' in self.df.columns else lambda x: 0,
            'MonthlyIncome': ['count', 'mean']
        }).round(3)
        
        # 列名を再整理
        training_stats = self.df.groupby('TrainingGroup').agg({
            'TrainingTimesLastYear': 'mean',
            'Attrition': lambda x: (x == 'Yes').mean(),
            'JobSatisfaction': 'mean',
            'MonthlyIncome': ['count', 'mean']
        }).round(3)
        
        # 多重インデックスを平坦化
        training_stats.columns = ['平均研修回数', '離職率', '平均満足度', '従業員数', '平均報酬']
        
        if 'OverallPerformance' in self.df.columns:
            perf_by_training = self.df.groupby('TrainingGroup')['OverallPerformance'].mean()
            training_stats['平均業績'] = perf_by_training
        
        print(f"📊 各研修グループ効果統計:")
        print(training_stats.to_string())
        
        # 研修ROI計算
        print(f"\n💰 研修ROI計算:")
        
        baseline_group = '研修なし'
        
        if baseline_group in training_stats.index:
            baseline_attrition = training_stats.loc[baseline_group, '離職率']
            baseline_performance = training_stats.loc[baseline_group, '平均業績'] if '平均業績' in training_stats.columns else 0
            baseline_satisfaction = training_stats.loc[baseline_group, '平均満足度']
            
            roi_analysis = {}
            
            for group in training_stats.index:
                if group != baseline_group:
                    group_data = training_stats.loc[group]
                    employees = group_data['従業員数']
                    avg_training = group_data['平均研修回数']
                    
                    # 研修コスト
                    training_cost = employees * avg_training * TRAINING_COST_PER_SESSION
                    
                    # 収益計算
                    # 1. 離職率低下による節約
                    attrition_reduction = baseline_attrition - group_data['離職率']
                    avg_salary = group_data['平均報酬'] * 12  # 年収
                    replacement_cost_saving = attrition_reduction * employees * avg_salary * 0.5  # 代替コストは年収の50%
                    
                    # 2. 業績向上による価値（業績向上1標準偏差の価値を年収の10%と仮定）
                    if '平均業績' in training_stats.columns:
                        performance_improvement = group_data['平均業績'] - baseline_performance
                        performance_value = performance_improvement * employees * avg_salary * 0.1
                    else:
                        performance_value = 0
                    
                    # 3. 満足度向上の間接価値（隠れコスト削減）
                    satisfaction_improvement = group_data['平均満足度'] - baseline_satisfaction
                    satisfaction_value = satisfaction_improvement * employees * 1000  # 1ポイント向上当たり$1000の価値
                    
                    # 総収益とROI
                    total_benefit = replacement_cost_saving + performance_value + satisfaction_value
                    roi = (total_benefit - training_cost) / training_cost * 100 if training_cost > 0 else 0
                    
                    roi_analysis[group] = {
                        'training_cost': training_cost,
                        'replacement_saving': replacement_cost_saving,
                        'performance_value': performance_value,
                        'satisfaction_value': satisfaction_value,
                        'total_benefit': total_benefit,
                        'roi': roi
                    }
                    
                    print(f"\n   {group}:")
                    print(f"     研修コスト: ${training_cost:,.0f}")
                    print(f"     離職節約: ${replacement_cost_saving:,.0f}")
                    print(f"     業績価値: ${performance_value:,.0f}")
                    print(f"     満足度価値: ${satisfaction_value:,.0f}")
                    print(f"     総収益: ${total_benefit:,.0f}")
                    print(f"     ROI: {roi:+.1f}%")
        
        # 研修効果の統計的有意性検定
        print(f"\n🔬 研修効果有意性検定:")
        
        no_training = self.df[self.df['TrainingTimesLastYear'] == 0]
        with_training = self.df[self.df['TrainingTimesLastYear'] > 0]
        
        if len(no_training) > 0 and len(with_training) > 0:
            # 離職率検定
            from scipy.stats import chi2_contingency
            
            contingency_table = pd.crosstab(
                self.df['TrainingTimesLastYear'] > 0,
                self.df['Attrition']
            )
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"   研修の離職率への影響: p-value = {p_value:.4f} ({'有意' if p_value < 0.05 else '有意でない'})")
            
            # 満足度検定
            if 'JobSatisfaction' in self.df.columns:
                from scipy.stats import ttest_ind
                
                t_stat, p_value_sat = ttest_ind(
                    with_training['JobSatisfaction'],
                    no_training['JobSatisfaction']
                )
                print(f"   研修の満足度への影響: p-value = {p_value_sat:.4f} ({'有意' if p_value_sat < 0.05 else '有意でない'})")
        
        # 可視化
        plt.figure(figsize=(18, 12))
        
        # 研修回数分布
        plt.subplot(2, 4, 1)
        self.df['TrainingTimesLastYear'].hist(bins=15, alpha=0.7, color='skyblue')
        plt.title('研修回数分布')
        plt.xlabel('年間研修回数')
        plt.ylabel('従業員数')
        
        # 各グループ離職率
        plt.subplot(2, 4, 2)
        training_stats['離職率'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('各研修グループ離職率')
        plt.ylabel('離職率')
        plt.xticks(rotation=45)
        
        # 各グループ満足度
        plt.subplot(2, 4, 3)
        training_stats['平均満足度'].plot(kind='bar', color='lightgreen', alpha=0.8)
        plt.title('各研修グループ満足度')
        plt.ylabel('平均満足度')
        plt.xticks(rotation=45)
        
        # ROI比較
        plt.subplot(2, 4, 4)
        if 'roi_analysis' in locals():
            roi_values = [analysis['roi'] for analysis in roi_analysis.values()]
            roi_labels = list(roi_analysis.keys())
            colors = ['green' if roi > 0 else 'red' for roi in roi_values]
            
            plt.bar(range(len(roi_values)), roi_values, color=colors, alpha=0.8)
            plt.title('各研修グループROI')
            plt.ylabel('ROI (%)')
            plt.xticks(range(len(roi_labels)), roi_labels, rotation=45)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 研修回数vs離職率散布図
        plt.subplot(2, 4, 5)
        colors = ['red' if x == 'Yes' else 'blue' for x in self.df['Attrition']]
        plt.scatter(self.df['TrainingTimesLastYear'], self.df['JobSatisfaction'], c=colors, alpha=0.6)
        plt.xlabel('年間研修回数')
        plt.ylabel('仕事満足度')
        plt.title('研修回数 vs 満足度')
        
        # 研修コスト収益分解
        plt.subplot(2, 4, 6)
        if 'roi_analysis' in locals() and roi_analysis:
            best_group = max(roi_analysis.keys(), key=lambda k: roi_analysis[k]['roi'])
            best_analysis = roi_analysis[best_group]
            
            benefit_components = [
                best_analysis['replacement_saving'],
                best_analysis['performance_value'],
                best_analysis['satisfaction_value']
            ]
            component_labels = ['離職節約', '業績価値', '満足度価値']
            
            plt.pie(benefit_components, labels=component_labels, autopct='%1.1f%%')
            plt.title(f'{best_group}\n収益構成')
        
        # 研修投入と産出関係
        plt.subplot(2, 4, 7)
        if 'roi_analysis' in locals():
            costs = [analysis['training_cost'] for analysis in roi_analysis.values()]
            benefits = [analysis['total_benefit'] for analysis in roi_analysis.values()]
            labels = list(roi_analysis.keys())
            
            plt.scatter(costs, benefits, s=100, alpha=0.7)
            
            for i, label in enumerate(labels):
                plt.annotate(label, (costs[i], benefits[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            # 損益分岐線追加
            max_cost = max(costs) if costs else 1
            plt.plot([0, max_cost], [0, max_cost], 'r--', alpha=0.5, label='損益分岐線')
            
            plt.xlabel('研修コスト ($)')
            plt.ylabel('総収益 ($)')
            plt.title('研修投入産出関係')
            plt.legend()
        
        # 研修頻度vs業績
        plt.subplot(2, 4, 8)
        if '平均業績' in training_stats.columns:
            training_stats['平均業績'].plot(kind='bar', color='gold', alpha=0.8)
            plt.title('各研修グループ業績')
            plt.ylabel('平均業績スコア')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 研修戦略提案
        print(f"\n💡 研修戦略最適化提案:")
        
        recommendations = []
        
        if 'roi_analysis' in locals():
            # ROI最高の研修グループを見つける
            best_roi_group = max(roi_analysis.keys(), key=lambda k: roi_analysis[k]['roi'])
            best_roi = roi_analysis[best_roi_group]['roi']
            
            if best_roi > 50:
                recommendations.append(f"{best_roi_group}モデルを重点推進、ROI{best_roi:.1f}%に達する")
            
            # ROIマイナスのグループ識別
            negative_roi_groups = [group for group, analysis in roi_analysis.items() if analysis['roi'] < 0]
            if negative_roi_groups:
                recommendations.append(f"{', '.join(negative_roi_groups)}の研修効果を再評価")
        
        # 最適研修回数に基づく提案
        optimal_training = training_stats.loc[training_stats['離職率'].idxmin(), '平均研修回数']
        recommendations.append(f"推奨年間研修回数: {optimal_training:.0f}回程度")
        
        # 異なるグループへの研修提案
        if len(no_training) > 0:
            no_training_attrition = (no_training['Attrition'] == 'Yes').mean()
            if no_training_attrition > 0.2:
                recommendations.append(f"研修未受講従業員を優先的に研修配置、現在離職率{no_training_attrition:.1%}")
        
        recommendations.append("研修効果追跡メカニズムを構築、定期的ROI評価")
        recommendations.append("職種特性に応じたカスタマイズ研修内容")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        self.results['training_roi'] = {
            'training_stats': training_stats,
            'roi_analysis': roi_analysis if 'roi_analysis' in locals() else {},
            'recommendations': recommendations
        }
        
        return training_stats
    
    # =================== 総合レポート生成 ===================
    
    def generate_comprehensive_report(self):
        """総合価値発掘レポート生成"""
        print("\n" + "="*80)
        print("📋 HR データ深度価値発掘総合レポート")
        print("="*80)
        
        # エグゼクティブサマリー
        print(f"\n🎯 エグゼクティブサマリー:")
        
        current_attrition = (self.df['Attrition'] == 'Yes').mean()
        total_employees = len(self.df)
        
        print(f"   データセット規模: {total_employees}名従業員")
        print(f"   現在離職率: {current_attrition:.1%}")
        
        # A部分まとめ
        if 'attrition_model' in self.results:
            high_risk_count = len(self.df[self.df['AttritionRiskScore'] > 0.7]) if 'AttritionRiskScore' in self.df.columns else 0
            print(f"   高リスク従業員: {high_risk_count}人 ({high_risk_count/total_employees:.1%})")
        
        if 'replacement_costs' in self.results:
            total_cost = self.results['replacement_costs']['年度離職コスト'].sum()
            print(f"   年度離職コスト: ${total_cost:,.0f}")
        
        if 'hidden_flight_risk' in self.results:
            hidden_count = self.results['hidden_flight_risk']['basic_count']
            print(f"   隠れ離職リスク: {hidden_count}人 ({hidden_count/total_employees:.1%})")
        
        # B部分まとめ
        if 'compensation_equity' in self.results:
            high_variance_roles = len(self.results['compensation_equity']['high_variance_roles'])
            print(f"   報酬格差大職種: {high_variance_roles}個")
        
        if 'performance_compensation' in self.results:
            fairness_index = self.results['performance_compensation']['fairness_index']
            print(f"   報酬公平性指数: {fairness_index:.3f}")
        
        # C部分まとめ
        if 'high_performance_teams' in self.results:
            high_perf_depts = len(self.results['high_performance_teams']['high_perf_depts'])
            print(f"   高業績部門数: {high_perf_depts}個")
        
        if 'training_roi' in self.results and self.results['training_roi']['roi_analysis']:
            best_roi = max(self.results['training_roi']['roi_analysis'].values(), key=lambda x: x['roi'])['roi']
            print(f"   最高研修ROI: {best_roi:.1f}%")
        
        # 重要発見
        print(f"\n🔍 重要発見:")
        
        key_findings = []
        
        # 離職予測発見
        if 'attrition_model' in self.results and self.results['attrition_model']['feature_importance'] is not None:
            top_factor = self.results['attrition_model']['feature_importance'].iloc[0]['feature']
            key_findings.append(f"離職の最大影響要因は{top_factor}")
        
        # コスト発見
        if 'replacement_costs' in self.results:
            highest_cost_dept = self.results['replacement_costs']['年度離職コスト'].idxmax()
            highest_cost = self.results['replacement_costs'].loc[highest_cost_dept, '年度離職コスト']
            key_findings.append(f"{highest_cost_dept}部門離職コスト最高(${highest_cost:,.0f})")
        
        # 報酬公平性発見
        if 'performance_compensation' in self.results:
            mismatch_high = len(self.results['performance_compensation']['high_perf_low_pay'])
            mismatch_low = len(self.results['performance_compensation']['low_perf_high_pay'])
            if mismatch_high > 0 or mismatch_low > 0:
                key_findings.append(f"{mismatch_high + mismatch_low}名従業員報酬業績不一致を発見")
        
        # 勤務モード発見
        if 'work_mode_effectiveness' in self.results:
            effective_modes = []
            for mode, analysis in self.results['work_mode_effectiveness']['mode_analysis'].items():
                if analysis.get('離職率', {}).get('improvement') == '改善':
                    effective_modes.append(mode)
            if effective_modes:
                key_findings.append(f"{', '.join(effective_modes)}が離職率低下に寄与")
        
        # 研修効果発見
        if 'training_roi' in self.results:
            positive_roi_groups = [group for group, analysis in self.results['training_roi'].get('roi_analysis', {}).items() 
                                 if analysis['roi'] > 0]
            if positive_roi_groups:
                key_findings.append(f"{len(positive_roi_groups)}研修グループが正ROIを示す")
        
        for i, finding in enumerate(key_findings, 1):
            print(f"   {i}. {finding}")
        
        # 行動提案優先順位
        print(f"\n🎯 行動提案 (優先順位順):")
        
        high_priority = []
        medium_priority = []
        low_priority = []
        
        # 高優先度：コストに直接影響する措置
        if 'hidden_flight_risk' in self.results:
            high_risk_count = self.results['hidden_flight_risk']['high_risk_count']
            if high_risk_count > 0:
                high_priority.append(f"{high_risk_count}名高リスク隠れ離職従業員に即座介入")
        
        if 'performance_compensation' in self.results:
            high_perf_low_pay = len(self.results['performance_compensation']['high_perf_low_pay'])
            if high_perf_low_pay > 0:
                high_priority.append(f"{high_perf_low_pay}名高業績低報酬従業員報酬調整")
        
        # 中優先度：システム的改善措置
        if 'market_competitiveness' in self.results:
            weak_depts = len(self.results['market_competitiveness'][self.results['market_competitiveness']['Gap_Percentage'] < -10])
            if weak_depts > 0:
                medium_priority.append(f"{weak_depts}部門の報酬競争力向上")
        
        if 'work_mode_effectiveness' in self.results:
            recommendations = self.results['work_mode_effectiveness'].get('recommendations', [])
            for rec in recommendations[:2]:  # 最初の2つのみ
                medium_priority.append(rec)
        
        # 低優先度：長期最適化措置
        if 'high_performance_teams' in self.results:
            success_factors = self.results['high_performance_teams'].get('success_factors', [])
            for factor in success_factors[:2]:  # 最初の2つのみ
                low_priority.append(f"全社で{factor}を推進")
        
        if 'training_roi' in self.results:
            training_recs = self.results['training_roi'].get('recommendations', [])
            for rec in training_recs[:1]:  # 1つのみ
                low_priority.append(rec)
        
        # 優先順位提案出力
        print(f"\n   🔴 高優先度 (即座実行):")
        for i, action in enumerate(high_priority, 1):
            print(f"      {i}. {action}")
        
        print(f"\n   🟡 中優先度 (3ヶ月以内):")
        for i, action in enumerate(medium_priority, 1):
            print(f"      {i}. {action}")
        
        print(f"\n   🟢 低優先度 (6ヶ月以内):")
        for i, action in enumerate(low_priority, 1):
            print(f"      {i}. {action}")
        
        # ROI予測
        print(f"\n💰 投資収益予測:")
        
        # 潜在節約計算
        if 'replacement_costs' in self.results:
            current_total_cost = self.results['replacement_costs']['年度離職コスト'].sum()
            
            # 措置効果仮定
            risk_reduction = 0.05  # 5%離職率低下
            cost_saving = current_total_cost * risk_reduction
            
            # 投資コスト推算
            investment_cost = 0
            
            # 報酬調整コスト
            if 'performance_compensation' in self.results:
                mismatch_employees = len(self.results['performance_compensation']['high_perf_low_pay'])
                avg_adjustment = 500  # 一人当たり月$500調整と仮定
                annual_adjustment_cost = mismatch_employees * avg_adjustment * 12
                investment_cost += annual_adjustment_cost
            
            # 研修投資
            if 'training_roi' in self.results:
                untrained_employees = len(self.df[self.df['TrainingTimesLastYear'] == 0])
                training_investment = untrained_employees * 2 * 500  # 一人2回研修、1回$500
                investment_cost += training_investment
            
            # 勤務モード改善コスト
            flexible_work_cost = total_employees * 100  # 一人$100のフレキシブルワーク支援
            investment_cost += flexible_work_cost
            
            # ROI計算
            net_benefit = cost_saving - investment_cost
            roi_percentage = (net_benefit / investment_cost * 100) if investment_cost > 0 else 0
            
            print(f"   予想離職コスト節約: ${cost_saving:,.0f}")
            print(f"   必要投資コスト: ${investment_cost:,.0f}")
            print(f"   純利益: ${net_benefit:,.0f}")
            print(f"   期待ROI: {roi_percentage:+.1f}%")
        
        # 実施スケジュール
        print(f"\n📅 実施スケジュール:")
        print(f"   第1ヶ月: 高リスク従業員介入、報酬公平性調整")
        print(f"   第2-3ヶ月: 勤務モード最適化、研修計画開始")
        print(f"   第4-6ヶ月: 効果評価、政策調整")
        print(f"   第7-12ヶ月: 持続最適化、経験総括")
        
        print(f"\n✅ 総合レポート生成完了！")
        print(f"🚀 定期的(四半期)指標再評価、動的戦略調整を推奨")
        
        return {
            'current_status': {
                'total_employees': total_employees,
                'attrition_rate': current_attrition,
                'high_risk_employees': high_risk_count if 'high_risk_count' in locals() else 0
            },
            'action_plan': {
                'high_priority': high_priority,
                'medium_priority': medium_priority,
                'low_priority': low_priority
            },
            'roi_projection': {
                'cost_saving': cost_saving if 'cost_saving' in locals() else 0,
                'investment_cost': investment_cost if 'investment_cost' in locals() else 0,
                'roi': roi_percentage if 'roi_percentage' in locals() else 0
            }
        }

# 完全価値発掘分析実行
print("🚀 HR データ深度価値発掘分析を開始...")

# 分析器作成
miner = HRValueMiner(df)

# Phase A: 離職予測とコスト最適化
print("\n📊 Phase A: 離職予測とコスト最適化")
attrition_model, risk_scores = miner.build_attrition_risk_model()
replacement_costs = miner.calculate_replacement_costs()
hidden_flight_basic, hidden_flight_high_risk = miner.identify_hidden_flight_risk()

# Phase B: 報酬最適化と公平性分析
print("\n💰 Phase B: 報酬最適化と公平性分析")
job_salary_stats, high_variance_roles = miner.analyze_compensation_equity()
perf_comp_correlations, high_perf_low_pay, low_perf_high_pay = miner.evaluate_performance_compensation_alignment()
market_competitiveness = miner.market_competitiveness_analysis()

# Phase C: 組織効能向上
print("\n🏆 Phase C: 組織効能向上")
dept_performance, high_perf_depts = miner.identify_high_performance_team_characteristics()
work_mode_effectiveness = miner.evaluate_work_mode_effectiveness()
training_stats = miner.analyze_training_roi()

# 総合レポート生成
comprehensive_report = miner.generate_comprehensive_report()

print("\n🎉 HR データ深度価値発掘分析完了！")
print("📋 すべての分析結果はminer.resultsに保存され、さらなる出力や深度分析が可能です。")