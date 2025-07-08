# =====================================
# POS業務データ分析 - トップデータサイエンティスト手法
# =====================================

# 1. 環境設定とライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントとグラフスタイルの設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

print("📊 POS業務データ分析システム")
print("=" * 50)
print("🔬 トップデータサイエンティスト手法を基に")
print("📈 目標：商業価値の発掘、業務意思決定の最適化")
print("=" * 50)

# 2. データ読み込みと初期チェック
class DataLoader:
    def __init__(self):
        self.data = {}
        self.data_quality = {}
    
    def load_all_data(self):
        """全データファイルを読み込む"""
        print("\n📥 データファイルを読み込み中...")
        
        # 各データファイルを読み込む
        try:
            # カスタマーサポート通話データ
            self.data['call'] = pd.read_csv('regi_call_data_transform .csv')
            self.data['call']['cdr_date'] = pd.to_datetime(self.data['call']['cdr_date'])
            print(f"✅ カスタマーサポート通話データ: {len(self.data['call'])} 行")
            
            # アカウント獲得データ
            self.data['account'] = pd.read_csv('regi_acc_get_data_transform .csv')
            self.data['account']['cdr_date'] = pd.to_datetime(self.data['account']['cdr_date'])
            print(f"✅ アカウント獲得データ: {len(self.data['account'])} 行")
            
            # マーケティングキャンペーンデータ
            self.data['campaign'] = pd.read_csv('cm_data .csv')
            self.data['campaign']['cdr_date'] = pd.to_datetime(self.data['campaign']['cdr_date'])
            print(f"✅ マーケティングキャンペーンデータ: {len(self.data['campaign'])} 行")
            
            # 検索トレンドデータ
            self.data['search'] = pd.read_csv('gt_service_name .csv')
            self.data['search']['week'] = pd.to_datetime(self.data['search']['week'])
            print(f"✅ 検索トレンドデータ: {len(self.data['search'])} 行")
            
            # カレンダーデータ
            self.data['calendar'] = pd.read_csv('calender_data .csv')
            self.data['calendar']['cdr_date'] = pd.to_datetime(self.data['calendar']['cdr_date'])
            print(f"✅ カレンダーデータ: {len(self.data['calendar'])} 行")
            
            return True
            
        except Exception as e:
            print(f"❌ データ読み込み失敗: {e}")
            return False
    
    def check_data_quality(self):
        """データ品質をチェック"""
        print("\n🔍 データ品質チェック...")
        
        for name, df in self.data.items():
            if df is not None:
                missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                date_range = None
                
                if name == 'search':
                    date_range = (df['week'].min(), df['week'].max())
                elif 'cdr_date' in df.columns:
                    date_range = (df['cdr_date'].min(), df['cdr_date'].max())
                
                self.data_quality[name] = {
                    'missing_rate': missing_rate,
                    'date_range': date_range,
                    'shape': df.shape
                }
                
                print(f"📊 {name}: 欠損率={missing_rate:.2%}, 時間範囲={date_range}")
        
        return self.data_quality

# 3. データ前処理と特徴量エンジニアリング
class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.master_data = None
    
    def create_master_dataset(self):
        """マスターデータセットを作成"""
        print("\n🔧 マスターデータセット作成中...")
        
        # カレンダーデータをベースとする
        master = self.data['calendar'].copy()
        
        # カスタマーサポート通話データをマージ
        master = master.merge(
            self.data['call'], 
            on='cdr_date', 
            how='left'
        )
        
        # アカウント獲得データをマージ
        master = master.merge(
            self.data['account'], 
            on='cdr_date', 
            how='left'
        )
        
        # マーケティングキャンペーンデータをマージ
        master = master.merge(
            self.data['campaign'], 
            on='cdr_date', 
            how='left'
        )
        
        # 検索データ（週次から日次に変換）を処理
        search_daily = self.convert_weekly_to_daily(self.data['search'])
        master = master.merge(
            search_daily, 
            on='cdr_date', 
            how='left'
        )
        
        # 欠損値を補完
        master['call_num'] = master['call_num'].fillna(0)
        master['acc_get_cnt'] = master['acc_get_cnt'].fillna(0)
        master['cm_flg'] = master['cm_flg'].fillna(0)
        master['search_cnt'] = master['search_cnt'].fillna(master['search_cnt'].mean())
        
        print(f"✅ マスターデータセット作成完了: {master.shape}")
        self.master_data = master
        return master
    
    def convert_weekly_to_daily(self, weekly_data):
        """週次データを日次データに変換"""
        daily_search = []
        
        for _, row in weekly_data.iterrows():
            week_start = row['week']
            search_cnt = row['search_cnt']
            
            # この週の各日に検索量を割り当て
            for i in range(7):
                daily_search.append({
                    'cdr_date': week_start + timedelta(days=i),
                    'search_cnt': search_cnt
                })
        
        return pd.DataFrame(daily_search)
    
    def create_features(self):
        """特徴量を作成"""
        print("\n⚙️ 特徴量作成中...")
        
        df = self.master_data.copy()
        
        # 時間特徴量
        df['year'] = df['cdr_date'].dt.year
        df['month'] = df['cdr_date'].dt.month
        df['day'] = df['cdr_date'].dt.day
        df['weekday'] = df['cdr_date'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # ラグ特徴量
        df['acc_get_cnt_lag1'] = df['acc_get_cnt'].shift(1)
        df['acc_get_cnt_lag3'] = df['acc_get_cnt'].shift(3)
        df['acc_get_cnt_lag7'] = df['acc_get_cnt'].shift(7)
        
        df['call_num_lag1'] = df['call_num'].shift(1)
        df['call_num_lag3'] = df['call_num'].shift(3)
        df['call_num_lag7'] = df['call_num'].shift(7)
        
        # 移動平均特徴量
        df['acc_get_cnt_ma7'] = df['acc_get_cnt'].rolling(window=7).mean()
        df['acc_get_cnt_ma30'] = df['acc_get_cnt'].rolling(window=30).mean()
        
        df['call_num_ma7'] = df['call_num'].rolling(window=7).mean()
        df['call_num_ma30'] = df['call_num'].rolling(window=30).mean()
        
        # マーケティングキャンペーン特徴量
        df['cm_flg_lag1'] = df['cm_flg'].shift(1)
        df['cm_flg_lag3'] = df['cm_flg'].shift(3)
        df['cm_flg_lag7'] = df['cm_flg'].shift(7)
        
        # 累積マーケティング効果
        df['cm_cumulative_7d'] = df['cm_flg'].rolling(window=7).sum()
        df['cm_cumulative_30d'] = df['cm_flg'].rolling(window=30).sum()
        
        print(f"✅ 特徴量作成完了: {df.shape[1]} 個の特徴量")
        self.master_data = df
        return df

# 4. 核心ビジネス分析クラス
class BusinessAnalyzer:
    def __init__(self, data):
        self.data = data
        self.insights = {}
    
    def analyze_marketing_effectiveness(self):
        """マーケティング効果分析"""
        print("\n🎯 マーケティング効果分析")
        print("=" * 40)
        
        # 1. 基本統計
        campaign_days = self.data[self.data['cm_flg'] == 1]
        no_campaign_days = self.data[self.data['cm_flg'] == 0]
        
        avg_acquisition_campaign = campaign_days['acc_get_cnt'].mean()
        avg_acquisition_no_campaign = no_campaign_days['acc_get_cnt'].mean()
        
        # 2. 統計的有意性検定
        t_stat, p_value = stats.ttest_ind(
            campaign_days['acc_get_cnt'].dropna(),
            no_campaign_days['acc_get_cnt'].dropna()
        )
        
        # 3. 効果の定量化
        lift = (avg_acquisition_campaign - avg_acquisition_no_campaign) / avg_acquisition_no_campaign
        
        insights = {
            'avg_acquisition_campaign': avg_acquisition_campaign,
            'avg_acquisition_no_campaign': avg_acquisition_no_campaign,
            'absolute_lift': avg_acquisition_campaign - avg_acquisition_no_campaign,
            'relative_lift': lift,
            'statistical_significance': p_value,
            'campaign_days': len(campaign_days),
            'total_days': len(self.data)
        }
        
        print(f"📊 マーケティング実施期間の日平均獲得数: {avg_acquisition_campaign:.1f}")
        print(f"📊 非実施期間の日平均獲得数: {avg_acquisition_no_campaign:.1f}")
        print(f"📈 絶対向上: {insights['absolute_lift']:.1f} 顧客/日")
        print(f"📈 相対向上: {lift:.1%}")
        print(f"🔬 統計的有意性: p={p_value:.4f}")
        
        # 4. 実施タイミング分析
        campaign_by_weekday = self.data[self.data['cm_flg'] == 1].groupby('weekday')['acc_get_cnt'].mean()
        
        print(f"\n📅 曜日別実施効果:")
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        for day, avg in campaign_by_weekday.items():
            print(f"   {weekday_names[day]}: {avg:.1f} 顧客/日")
        
        self.insights['marketing'] = insights
        return insights
    
    def analyze_customer_service_patterns(self):
        """カスタマーサービス需要パターン分析"""
        print("\n📞 カスタマーサービス需要パターン分析")
        print("=" * 40)
        
        # 1. 週次パターン
        weekly_pattern = self.data.groupby('weekday')['call_num'].mean()
        weekly_pattern_normalized = weekly_pattern / weekly_pattern.mean()
        
        # 2. 月次パターン
        monthly_pattern = self.data.groupby(self.data['cdr_date'].dt.day)['call_num'].mean()
        
        # 3. 祝日効果
        holiday_effect = self.data.groupby('holiday_flag')['call_num'].mean()
        before_holiday_effect = self.data.groupby('day_before_holiday_flag')['call_num'].mean()
        
        # 4. 季節性パターン
        seasonal_pattern = self.data.groupby('month')['call_num'].mean()
        
        insights = {
            'weekly_pattern': weekly_pattern_normalized.to_dict(),
            'holiday_effect': holiday_effect.to_dict(),
            'before_holiday_effect': before_holiday_effect.to_dict(),
            'seasonal_pattern': seasonal_pattern.to_dict(),
            'avg_daily_calls': self.data['call_num'].mean(),
            'peak_day': weekly_pattern.idxmax(),
            'low_day': weekly_pattern.idxmin()
        }
        
        print(f"📊 日平均カスタマーサービス電話数: {insights['avg_daily_calls']:.1f}")
        print(f"📈 ピーク日: {['月', '火', '水', '木', '金', '土', '日'][insights['peak_day']]}")
        print(f"📉 低需要日: {['月', '火', '水', '木', '金', '土', '日'][insights['low_day']]}")
        
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        print(f"\n📅 週次需要パターン:")
        for day, multiplier in weekly_pattern_normalized.items():
            print(f"   {weekday_names[day]}: {multiplier:.2f}x (平均比{(multiplier-1)*100:+.0f}%)")
        
        if True in insights['holiday_effect']:
            holiday_multiplier = insights['holiday_effect'][True] / insights['holiday_effect'][False]
            print(f"🎌 祝日効果: {holiday_multiplier:.2f}x")
        
        self.insights['customer_service'] = insights
        return insights
    
    def analyze_search_business_correlation(self):
        """検索トレンドとビジネス関連分析"""
        print("\n🔍 検索トレンドとビジネス関連分析")
        print("=" * 40)
        
        # 欠損値を除外
        valid_data = self.data.dropna(subset=['search_cnt', 'acc_get_cnt'])
        
        if len(valid_data) < 10:
            print("❌ 有効データが不足、関連分析実行不可")
            return {}
        
        # 1. 現在の相関性
        corr_current, p_current = pearsonr(valid_data['search_cnt'], valid_data['acc_get_cnt'])
        
        # 2. ラグ相関性分析
        lag_correlations = {}
        for lag in range(1, 8):  # 1-7日ラグ
            if len(valid_data) > lag:
                search_lag = valid_data['search_cnt'].shift(lag)
                mask = ~(search_lag.isna() | valid_data['acc_get_cnt'].isna())
                if mask.sum() > 10:
                    corr_lag, p_lag = pearsonr(search_lag[mask], valid_data['acc_get_cnt'][mask])
                    lag_correlations[lag] = {'correlation': corr_lag, 'p_value': p_lag}
        
        # 3. 最適ラグ期間を特定
        best_lag = 0
        best_corr = abs(corr_current)
        
        for lag, stats in lag_correlations.items():
            if abs(stats['correlation']) > best_corr:
                best_corr = abs(stats['correlation'])
                best_lag = lag
        
        insights = {
            'current_correlation': corr_current,
            'current_p_value': p_current,
            'lag_correlations': lag_correlations,
            'best_lag': best_lag,
            'best_correlation': best_corr,
            'search_mean': valid_data['search_cnt'].mean(),
            'search_std': valid_data['search_cnt'].std()
        }
        
        print(f"📊 現在の相関性: {corr_current:.3f} (p={p_current:.4f})")
        print(f"📈 最適ラグ期間: {best_lag} 日")
        print(f"📈 最適相関性: {best_corr:.3f}")
        
        if best_lag > 0:
            print(f"💡 洞察: 検索量がビジネス指標より {best_lag} 日先行")
        
        self.insights['search_correlation'] = insights
        return insights
    
    def build_prediction_models(self):
        """予測モデル構築"""
        print("\n🤖 予測モデル構築")
        print("=" * 40)
        
        # 特徴量と目的変数の準備
        feature_cols = [
            'dow', 'woy', 'wom', 'doy', 'is_weekend',
            'holiday_flag', 'day_before_holiday_flag',
            'cm_flg', 'cm_flg_lag1', 'cm_flg_lag3', 'cm_flg_lag7',
            'search_cnt', 'acc_get_cnt_lag1', 'acc_get_cnt_lag3', 'acc_get_cnt_lag7',
            'call_num_lag1', 'call_num_lag3', 'call_num_lag7'
        ]
        
        # 存在する特徴量をフィルタリング
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        # 1. カスタマーサービス需要予測モデル
        call_model_results = self.build_call_prediction_model(available_features)
        
        # 2. アカウント獲得予測モデル
        account_model_results = self.build_account_prediction_model(available_features)
        
        insights = {
            'call_prediction': call_model_results,
            'account_prediction': account_model_results,
            'features_used': available_features
        }
        
        self.insights['prediction_models'] = insights
        return insights
    
    def build_call_prediction_model(self, features):
        """カスタマーサービス需要予測モデル構築"""
        print("📞 カスタマーサービス需要予測モデル...")
        
        # データ準備
        model_data = self.data[features + ['call_num']].dropna()
        
        if len(model_data) < 50:
            print("❌ データ不足、モデル構築不可")
            return {}
        
        X = model_data[features]
        y = model_data['call_num']
        
        # 訓練・テストデータ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # モデル訓練
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 特徴量重要度
        feature_importance = dict(zip(features, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'feature_importance': top_features,
            'model_score': model.score(X_test, y_test),
            'sample_size': len(model_data)
        }
        
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.1f}%")
        print(f"   R²: {results['model_score']:.3f}")
        
        return results
    
    def build_account_prediction_model(self, features):
        """アカウント獲得予測モデル構築"""
        print("👥 アカウント獲得予測モデル...")
        
        # データ準備
        model_data = self.data[features + ['acc_get_cnt']].dropna()
        
        if len(model_data) < 50:
            print("❌ データ不足、モデル構築不可")
            return {}
        
        X = model_data[features]
        y = model_data['acc_get_cnt']
        
        # 訓練・テストデータ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # モデル訓練
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 特徴量重要度
        feature_importance = dict(zip(features, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'feature_importance': top_features,
            'model_score': model.score(X_test, y_test),
            'sample_size': len(model_data)
        }
        
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.1f}%")
        print(f"   R²: {results['model_score']:.3f}")
        
        return results

# 5. 可視化分析クラス
class BusinessVisualizer:
    def __init__(self, data, insights):
        self.data = data
        self.insights = insights
    
    def create_comprehensive_dashboard(self):
        """総合分析ダッシュボードを作成"""
        print("\n📊 総合分析ダッシュボード作成中...")
        
        # 大型チャート作成
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('POS Business Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. 時系列トレンド
        self.plot_time_series_trends(axes[0, 0])
        
        # 2. マーケティング効果分析
        self.plot_marketing_effectiveness(axes[0, 1])
        
        # 3. カスタマーサービス需要パターン
        self.plot_service_patterns(axes[0, 2])
        
        # 4. 検索トレンド関連
        self.plot_search_correlation(axes[1, 0])
        
        # 5. 予測モデル性能
        self.plot_model_performance(axes[1, 1])
        
        # 6. 主要指標概要
        self.plot_key_metrics_summary(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_time_series_trends(self, ax):
        """時系列トレンドを描画"""
        ax.set_title('Business Metrics Time Series', fontweight='bold')
        
        # 双軸作成
        ax2 = ax.twinx()
        
        # 左軸：アカウント獲得数
        line1 = ax.plot(self.data['cdr_date'], self.data['acc_get_cnt'], 
                       color='blue', label='Account Acquisition', linewidth=2)
        ax.set_ylabel('Account Acquisition', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 右軸：カスタマーサービス電話数
        line2 = ax2.plot(self.data['cdr_date'], self.data['call_num'], 
                        color='red', label='Service Calls', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Service Calls', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # マーケティング実施期間をマーク
        campaign_periods = self.data[self.data['cm_flg'] == 1]['cdr_date']
        for date in campaign_periods:
            ax.axvline(x=date, color='green', alpha=0.3, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        # 凡例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def plot_marketing_effectiveness(self, ax):
        """マーケティング効果分析を描画"""
        ax.set_title('Marketing Campaign Effectiveness', fontweight='bold')
        
        if 'marketing' in self.insights:
            insights = self.insights['marketing']
            
            categories = ['No Campaign', 'Campaign Days']
            values = [insights['avg_acquisition_no_campaign'], 
                     insights['avg_acquisition_campaign']]
            colors = ['lightblue', 'orange']
            
            bars = ax.bar(categories, values, color=colors)
            
            # 数値ラベル追加
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # 向上幅追加
            lift_text = f"Lift: {insights['relative_lift']:.1%}"
            ax.text(0.5, max(values) * 0.9, lift_text, 
                   transform=ax.transAxes, ha='center', 
                   fontsize=12, fontweight='bold', color='green')
            
            ax.set_ylabel('Avg Daily Acquisitions')
            ax.grid(True, alpha=0.3)
    
    def plot_service_patterns(self, ax):
        """カスタマーサービス需要パターンを描画"""
        ax.set_title('Customer Service Demand Patterns', fontweight='bold')
        
        if 'customer_service' in self.insights:
            insights = self.insights['customer_service']
            
            weekdays = ['月', '火', '水', '木', '金', '土', '日']
            pattern_values = [insights['weekly_pattern'].get(i, 1) for i in range(7)]
            
            bars = ax.bar(weekdays, pattern_values, 
                         color=['lightcoral' if x > 1.1 else 'lightblue' for x in pattern_values])
            
            # 基準線追加
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Average')
            
            # 数値ラベル追加
            for bar, value in zip(bars, pattern_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}x',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Demand Multiplier')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_search_correlation(self, ax):
        """検索トレンド関連を描画"""
        ax.set_title('Search Trend vs Business Metrics', fontweight='bold')
        
        # 有効データをフィルタリング
        valid_data = self.data.dropna(subset=['search_cnt', 'acc_get_cnt'])
        
        if len(valid_data) > 10:
            scatter = ax.scatter(valid_data['search_cnt'], valid_data['acc_get_cnt'], 
                               alpha=0.6, color='purple')
            
            # トレンドライン追加
            if len(valid_data) > 2:
                z = np.polyfit(valid_data['search_cnt'], valid_data['acc_get_cnt'], 1)
                p = np.poly1d(z)
                ax.plot(valid_data['search_cnt'], p(valid_data['search_cnt']), 
                       "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Search Volume')
            ax.set_ylabel('Account Acquisitions')
            
            # 相関情報追加
            if 'search_correlation' in self.insights:
                corr = self.insights['search_correlation']['current_correlation']
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.grid(True, alpha=0.3)
    
    def plot_model_performance(self, ax):
        """モデル性能を描画"""
        ax.set_title('Prediction Model Performance', fontweight='bold')
        
        if 'prediction_models' in self.insights:
            models = []
            accuracies = []
            
            if 'call_prediction' in self.insights['prediction_models']:
                call_model = self.insights['prediction_models']['call_prediction']
                if 'model_score' in call_model:
                    models.append('Service Calls')
                    accuracies.append(call_model['model_score'])
            
            if 'account_prediction' in self.insights['prediction_models']:
                account_model = self.insights['prediction_models']['account_prediction']
                if 'model_score' in account_model:
                    models.append('Account Acquisition')
                    accuracies.append(account_model['model_score'])
            
            if models:
                bars = ax.bar(models, accuracies, color=['skyblue', 'lightgreen'])
                
                # 数値ラベル追加
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontweight='bold')
                
                ax.set_ylabel('R² Score')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
    
    def plot_key_metrics_summary(self, ax):
        """主要指標概要を描画"""
        ax.set_title('Key Business Metrics Summary', fontweight='bold')
        
        # 主要指標計算
        metrics = {
            'Avg Daily Acquisitions': self.data['acc_get_cnt'].mean(),
            'Avg Daily Service Calls': self.data['call_num'].mean(),
            'Campaign Days': self.data['cm_flg'].sum(),
            'Total Days': len(self.data)
        }
        
        # 表形式で表示
        table_data = []
        for metric, value in metrics.items():
            if isinstance(value, float):
                table_data.append([metric, f"{value:.1f}"])
            else:
                table_data.append([metric, str(value)])
        
        # マーケティング向上情報追加
        if 'marketing' in self.insights:
            lift = self.insights['marketing']['relative_lift']
            table_data.append(['Marketing Lift', f"{lift:.1%}"])
        
        # 表作成
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 座標軸を隠す
        ax.axis('off')

# 6. 商業価値定量化クラス
class BusinessValueQuantifier:
    def __init__(self, insights):
        self.insights = insights
    
    def calculate_marketing_value(self):
        """マーケティング価値を算出"""
        print("\n💰 マーケティング価値定量化")
        print("=" * 40)
        
        if 'marketing' not in self.insights:
            print("❌ マーケティング分析データが不足")
            return {}
        
        marketing = self.insights['marketing']
        
        # 仮定パラメータ
        customer_ltv = 180000  # 顧客生涯価値（円）
        daily_campaign_cost = 50000  # 日次マーケティングコスト
        
        # 価値計算
        daily_incremental_customers = marketing['absolute_lift']
        daily_incremental_revenue = daily_incremental_customers * customer_ltv
        daily_roi = (daily_incremental_revenue - daily_campaign_cost) / daily_campaign_cost
        
        # 年間価値
        annual_campaign_cost = daily_campaign_cost * 365
        annual_incremental_revenue = daily_incremental_revenue * 365
        annual_net_value = annual_incremental_revenue - annual_campaign_cost
        
        value_metrics = {
            'daily_incremental_customers': daily_incremental_customers,
            'daily_incremental_revenue': daily_incremental_revenue,
            'daily_roi': daily_roi,
            'annual_campaign_cost': annual_campaign_cost,
            'annual_incremental_revenue': annual_incremental_revenue,
            'annual_net_value': annual_net_value
        }
        
        print(f"📊 日次増分顧客数: {daily_incremental_customers:.1f} 人")
        print(f"📊 日次増分収益: ¥{daily_incremental_revenue:,.0f}")
        print(f"📊 日次マーケティングROI: {daily_roi:.1%}")
        print(f"📊 年間純価値: ¥{annual_net_value:,.0f}")
        
        return value_metrics
    
    def calculate_service_optimization_value(self):
        """カスタマーサービス最適化価値を算出"""
        print("\n📞 カスタマーサービス最適化価値定量化")
        print("=" * 40)
        
        # 仮定パラメータ
        current_annual_cost = 15000000  # 現在の年間カスタマーサービスコスト
        avg_hourly_cost = 3000  # 平均時間コスト
        
        # 予測モデルに基づく最適化ポテンシャル
        if 'prediction_models' in self.insights and 'call_prediction' in self.insights['prediction_models']:
            model_accuracy = self.insights['prediction_models']['call_prediction'].get('model_score', 0)
            
            # 予測精度10%向上ごとに5%のコスト削減を仮定
            cost_reduction_rate = min(0.25, model_accuracy * 0.3)  # 最大25%
            
            annual_cost_savings = current_annual_cost * cost_reduction_rate
            
            # サービス品質改善価値
            service_improvement_value = current_annual_cost * 0.1
            
            total_value = annual_cost_savings + service_improvement_value
            
            value_metrics = {
                'current_annual_cost': current_annual_cost,
                'cost_reduction_rate': cost_reduction_rate,
                'annual_cost_savings': annual_cost_savings,
                'service_improvement_value': service_improvement_value,
                'total_annual_value': total_value,
                'model_accuracy': model_accuracy
            }
            
            print(f"📊 現在の年間コスト: ¥{current_annual_cost:,.0f}")
            print(f"📊 コスト削減率: {cost_reduction_rate:.1%}")
            print(f"📊 年間コスト削減: ¥{annual_cost_savings:,.0f}")
            print(f"📊 サービス改善価値: ¥{service_improvement_value:,.0f}")
            print(f"📊 総年間価値: ¥{total_value:,.0f}")
            
            return value_metrics
        else:
            print("❌ 予測モデルデータが不足")
            return {}
    
    def generate_executive_summary(self):
        """エグゼクティブサマリーを生成"""
        print("\n📋 エグゼクティブサマリー")
        print("=" * 50)
        
        # 全価値を集約
        marketing_value = self.calculate_marketing_value()
        service_value = self.calculate_service_optimization_value()
        
        total_annual_value = 0
        if marketing_value:
            total_annual_value += marketing_value.get('annual_net_value', 0)
        if service_value:
            total_annual_value += service_value.get('total_annual_value', 0)
        
        print(f"\n🎯 主要発見:")
        
        if marketing_value:
            print(f"   • マーケティングROI達成可能: {marketing_value['daily_roi']:.1%}")
            print(f"   • 年間マーケティング純価値: ¥{marketing_value['annual_net_value']:,.0f}")
        
        if service_value:
            print(f"   • カスタマーサービスコスト削減可能: {service_value['cost_reduction_rate']:.1%}")
            print(f"   • 年間カスタマーサービス最適化価値: ¥{service_value['total_annual_value']:,.0f}")
        
        print(f"\n💰 総商業価値:")
        print(f"   • 年間総価値: ¥{total_annual_value:,.0f}")
        print(f"   • 3年累計価値: ¥{total_annual_value * 3:,.0f}")
        
        print(f"\n🚀 優先推奨事項:")
        print(f"   1. マーケティング投放戦略の即時最適化")
        print(f"   2. カスタマーサービス需要予測システムの導入")
        print(f"   3. データ監視体制の構築")
        print(f"   4. データサイエンスチームへの投資")
        
        return {
            'marketing_value': marketing_value,
            'service_value': service_value,
            'total_annual_value': total_annual_value
        }

# 7. メイン実行関数
def main():
    """メイン実行関数"""
    print("🚀 POS業務データ分析開始...")
    
    # 1. データ読み込み
    loader = DataLoader()
    if not loader.load_all_data():
        print("❌ データ読み込み失敗、ファイルパスを確認してください")
        return
    
    # 2. データ品質チェック
    loader.check_data_quality()
    
    # 3. データ前処理
    preprocessor = DataPreprocessor(loader.data)
    master_data = preprocessor.create_master_dataset()
    featured_data = preprocessor.create_features()
    
    # 4. ビジネス分析
    analyzer = BusinessAnalyzer(featured_data)
    
    # 各種分析実行
    marketing_insights = analyzer.analyze_marketing_effectiveness()
    service_insights = analyzer.analyze_customer_service_patterns()
    search_insights = analyzer.analyze_search_business_correlation()
    model_insights = analyzer.build_prediction_models()
    
    # 5. データ可視化
    visualizer = BusinessVisualizer(featured_data, analyzer.insights)
    dashboard = visualizer.create_comprehensive_dashboard()
    
    # 6. 商業価値定量化
    value_quantifier = BusinessValueQuantifier(analyzer.insights)
    business_value = value_quantifier.generate_executive_summary()
    
    print("\n✅ 分析完了！")
    print("📊 全洞察と推奨事項が生成されました")
    print("💡 上記の可視化チャートと分析結果をご確認ください")
    
    return {
        'data': featured_data,
        'insights': analyzer.insights,
        'business_value': business_value,
        'dashboard': dashboard
    }

# 8. 分析実行
if __name__ == "__main__":
    results = main()
