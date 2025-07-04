# =====================================
# POS业务数据分析 - 顶级数据科学家方法
# =====================================

# 1. 环境配置和库导入
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

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

print("📊 POS业务数据分析系统")
print("=" * 50)
print("🔬 基于顶级数据科学家方法论")
print("📈 目标：挖掘商业价值，优化业务决策")
print("=" * 50)

# 2. 数据加载和初步检查
class DataLoader:
    def __init__(self):
        self.data = {}
        self.data_quality = {}
    
    def load_all_data(self):
        """加载所有数据文件"""
        print("\n📥 加载数据文件...")
        
        # 加载各个数据文件
        try:
            # 客服电话数据
            self.data['call'] = pd.read_csv('regi_call_data_transform .csv')
            self.data['call']['cdr_date'] = pd.to_datetime(self.data['call']['cdr_date'])
            print(f"✅ 客服电话数据: {len(self.data['call'])} rows")
            
            # 账户获取数据
            self.data['account'] = pd.read_csv('regi_acc_get_data_transform .csv')
            self.data['account']['cdr_date'] = pd.to_datetime(self.data['account']['cdr_date'])
            print(f"✅ 账户获取数据: {len(self.data['account'])} rows")
            
            # 营销投放数据
            self.data['campaign'] = pd.read_csv('cm_data .csv')
            self.data['campaign']['cdr_date'] = pd.to_datetime(self.data['campaign']['cdr_date'])
            print(f"✅ 营销投放数据: {len(self.data['campaign'])} rows")
            
            # 搜索趋势数据
            self.data['search'] = pd.read_csv('gt_service_name .csv')
            self.data['search']['week'] = pd.to_datetime(self.data['search']['week'])
            print(f"✅ 搜索趋势数据: {len(self.data['search'])} rows")
            
            # 日历数据
            self.data['calendar'] = pd.read_csv('calender_data .csv')
            self.data['calendar']['cdr_date'] = pd.to_datetime(self.data['calendar']['cdr_date'])
            print(f"✅ 日历数据: {len(self.data['calendar'])} rows")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def check_data_quality(self):
        """检查数据质量"""
        print("\n🔍 数据质量检查...")
        
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
                
                print(f"📊 {name}: 缺失率={missing_rate:.2%}, 时间范围={date_range}")
        
        return self.data_quality

# 3. 数据预处理和特征工程
class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.master_data = None
    
    def create_master_dataset(self):
        """创建主数据集"""
        print("\n🔧 创建主数据集...")
        
        # 以日历数据为基础
        master = self.data['calendar'].copy()
        
        # 合并客服电话数据
        master = master.merge(
            self.data['call'], 
            on='cdr_date', 
            how='left'
        )
        
        # 合并账户获取数据
        master = master.merge(
            self.data['account'], 
            on='cdr_date', 
            how='left'
        )
        
        # 合并营销投放数据
        master = master.merge(
            self.data['campaign'], 
            on='cdr_date', 
            how='left'
        )
        
        # 处理搜索数据（周度转日度）
        search_daily = self.convert_weekly_to_daily(self.data['search'])
        master = master.merge(
            search_daily, 
            on='cdr_date', 
            how='left'
        )
        
        # 填充缺失值
        master['call_num'] = master['call_num'].fillna(0)
        master['acc_get_cnt'] = master['acc_get_cnt'].fillna(0)
        master['cm_flg'] = master['cm_flg'].fillna(0)
        master['search_cnt'] = master['search_cnt'].fillna(master['search_cnt'].mean())
        
        print(f"✅ 主数据集创建完成: {master.shape}")
        self.master_data = master
        return master
    
    def convert_weekly_to_daily(self, weekly_data):
        """将周度数据转换为日度数据"""
        daily_search = []
        
        for _, row in weekly_data.iterrows():
            week_start = row['week']
            search_cnt = row['search_cnt']
            
            # 为这一周的每一天分配搜索量
            for i in range(7):
                daily_search.append({
                    'cdr_date': week_start + timedelta(days=i),
                    'search_cnt': search_cnt
                })
        
        return pd.DataFrame(daily_search)
    
    def create_features(self):
        """创建特征变量"""
        print("\n⚙️ 创建特征变量...")
        
        df = self.master_data.copy()
        
        # 时间特征
        df['year'] = df['cdr_date'].dt.year
        df['month'] = df['cdr_date'].dt.month
        df['day'] = df['cdr_date'].dt.day
        df['weekday'] = df['cdr_date'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # 滞后特征
        df['acc_get_cnt_lag1'] = df['acc_get_cnt'].shift(1)
        df['acc_get_cnt_lag3'] = df['acc_get_cnt'].shift(3)
        df['acc_get_cnt_lag7'] = df['acc_get_cnt'].shift(7)
        
        df['call_num_lag1'] = df['call_num'].shift(1)
        df['call_num_lag3'] = df['call_num'].shift(3)
        df['call_num_lag7'] = df['call_num'].shift(7)
        
        # 移动平均特征
        df['acc_get_cnt_ma7'] = df['acc_get_cnt'].rolling(window=7).mean()
        df['acc_get_cnt_ma30'] = df['acc_get_cnt'].rolling(window=30).mean()
        
        df['call_num_ma7'] = df['call_num'].rolling(window=7).mean()
        df['call_num_ma30'] = df['call_num'].rolling(window=30).mean()
        
        # 营销活动特征
        df['cm_flg_lag1'] = df['cm_flg'].shift(1)
        df['cm_flg_lag3'] = df['cm_flg'].shift(3)
        df['cm_flg_lag7'] = df['cm_flg'].shift(7)
        
        # 累计营销效果
        df['cm_cumulative_7d'] = df['cm_flg'].rolling(window=7).sum()
        df['cm_cumulative_30d'] = df['cm_flg'].rolling(window=30).sum()
        
        print(f"✅ 特征创建完成: {df.shape[1]} 个特征")
        self.master_data = df
        return df

# 4. 核心业务分析类
class BusinessAnalyzer:
    def __init__(self, data):
        self.data = data
        self.insights = {}
    
    def analyze_marketing_effectiveness(self):
        """营销效果分析"""
        print("\n🎯 营销效果分析")
        print("=" * 40)
        
        # 1. 基础统计
        campaign_days = self.data[self.data['cm_flg'] == 1]
        no_campaign_days = self.data[self.data['cm_flg'] == 0]
        
        avg_acquisition_campaign = campaign_days['acc_get_cnt'].mean()
        avg_acquisition_no_campaign = no_campaign_days['acc_get_cnt'].mean()
        
        # 2. 统计显著性检验
        t_stat, p_value = stats.ttest_ind(
            campaign_days['acc_get_cnt'].dropna(),
            no_campaign_days['acc_get_cnt'].dropna()
        )
        
        # 3. 效果量化
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
        
        print(f"📊 营销投放期间日均获客: {avg_acquisition_campaign:.1f}")
        print(f"📊 非投放期间日均获客: {avg_acquisition_no_campaign:.1f}")
        print(f"📈 绝对提升: {insights['absolute_lift']:.1f} 个客户/天")
        print(f"📈 相对提升: {lift:.1%}")
        print(f"🔬 统计显著性: p={p_value:.4f}")
        
        # 4. 投放时机分析
        campaign_by_weekday = self.data[self.data['cm_flg'] == 1].groupby('weekday')['acc_get_cnt'].mean()
        
        print(f"\n📅 不同星期投放效果:")
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for day, avg in campaign_by_weekday.items():
            print(f"   {weekday_names[day]}: {avg:.1f} 个客户/天")
        
        self.insights['marketing'] = insights
        return insights
    
    def analyze_customer_service_patterns(self):
        """客服需求模式分析"""
        print("\n📞 客服需求模式分析")
        print("=" * 40)
        
        # 1. 周度模式
        weekly_pattern = self.data.groupby('weekday')['call_num'].mean()
        weekly_pattern_normalized = weekly_pattern / weekly_pattern.mean()
        
        # 2. 月度模式
        monthly_pattern = self.data.groupby(self.data['cdr_date'].dt.day)['call_num'].mean()
        
        # 3. 节假日效应
        holiday_effect = self.data.groupby('holiday_flag')['call_num'].mean()
        before_holiday_effect = self.data.groupby('day_before_holiday_flag')['call_num'].mean()
        
        # 4. 季节性模式
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
        
        print(f"📊 日均客服电话: {insights['avg_daily_calls']:.1f}")
        print(f"📈 高峰日: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][insights['peak_day']]}")
        print(f"📉 低谷日: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][insights['low_day']]}")
        
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        print(f"\n📅 周度需求模式:")
        for day, multiplier in weekly_pattern_normalized.items():
            print(f"   {weekday_names[day]}: {multiplier:.2f}x (比平均{(multiplier-1)*100:+.0f}%)")
        
        if True in insights['holiday_effect']:
            holiday_multiplier = insights['holiday_effect'][True] / insights['holiday_effect'][False]
            print(f"🎌 节假日效应: {holiday_multiplier:.2f}x")
        
        self.insights['customer_service'] = insights
        return insights
    
    def analyze_search_business_correlation(self):
        """搜索趋势与业务关联分析"""
        print("\n🔍 搜索趋势与业务关联分析")
        print("=" * 40)
        
        # 过滤掉缺失值
        valid_data = self.data.dropna(subset=['search_cnt', 'acc_get_cnt'])
        
        if len(valid_data) < 10:
            print("❌ 有效数据不足，无法进行关联分析")
            return {}
        
        # 1. 当期相关性
        corr_current, p_current = pearsonr(valid_data['search_cnt'], valid_data['acc_get_cnt'])
        
        # 2. 滞后相关性分析
        lag_correlations = {}
        for lag in range(1, 8):  # 1-7天滞后
            if len(valid_data) > lag:
                search_lag = valid_data['search_cnt'].shift(lag)
                mask = ~(search_lag.isna() | valid_data['acc_get_cnt'].isna())
                if mask.sum() > 10:
                    corr_lag, p_lag = pearsonr(search_lag[mask], valid_data['acc_get_cnt'][mask])
                    lag_correlations[lag] = {'correlation': corr_lag, 'p_value': p_lag}
        
        # 3. 找出最佳滞后期
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
        
        print(f"📊 当期相关性: {corr_current:.3f} (p={p_current:.4f})")
        print(f"📈 最佳滞后期: {best_lag} 天")
        print(f"📈 最佳相关性: {best_corr:.3f}")
        
        if best_lag > 0:
            print(f"💡 洞察: 搜索量领先业务指标 {best_lag} 天")
        
        self.insights['search_correlation'] = insights
        return insights
    
    def build_prediction_models(self):
        """构建预测模型"""
        print("\n🤖 构建预测模型")
        print("=" * 40)
        
        # 准备特征和目标变量
        feature_cols = [
            'dow', 'woy', 'wom', 'doy', 'is_weekend',
            'holiday_flag', 'day_before_holiday_flag',
            'cm_flg', 'cm_flg_lag1', 'cm_flg_lag3', 'cm_flg_lag7',
            'search_cnt', 'acc_get_cnt_lag1', 'acc_get_cnt_lag3', 'acc_get_cnt_lag7',
            'call_num_lag1', 'call_num_lag3', 'call_num_lag7'
        ]
        
        # 过滤存在的特征
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        # 1. 客服需求预测模型
        call_model_results = self.build_call_prediction_model(available_features)
        
        # 2. 账户获取预测模型
        account_model_results = self.build_account_prediction_model(available_features)
        
        insights = {
            'call_prediction': call_model_results,
            'account_prediction': account_model_results,
            'features_used': available_features
        }
        
        self.insights['prediction_models'] = insights
        return insights
    
    def build_call_prediction_model(self, features):
        """构建客服需求预测模型"""
        print("📞 客服需求预测模型...")
        
        # 准备数据
        model_data = self.data[features + ['call_num']].dropna()
        
        if len(model_data) < 50:
            print("❌ 数据不足，无法构建模型")
            return {}
        
        X = model_data[features]
        y = model_data['call_num']
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 特征重要性
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
        """构建账户获取预测模型"""
        print("👥 账户获取预测模型...")
        
        # 准备数据
        model_data = self.data[features + ['acc_get_cnt']].dropna()
        
        if len(model_data) < 50:
            print("❌ 数据不足，无法构建模型")
            return {}
        
        X = model_data[features]
        y = model_data['acc_get_cnt']
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 特征重要性
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

# 5. 可视化分析类
class BusinessVisualizer:
    def __init__(self, data, insights):
        self.data = data
        self.insights = insights
    
    def create_comprehensive_dashboard(self):
        """创建综合分析仪表板"""
        print("\n📊 创建综合分析仪表板...")
        
        # 创建大图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('POS Business Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. 时间序列趋势
        self.plot_time_series_trends(axes[0, 0])
        
        # 2. 营销效果分析
        self.plot_marketing_effectiveness(axes[0, 1])
        
        # 3. 客服需求模式
        self.plot_service_patterns(axes[0, 2])
        
        # 4. 搜索趋势关联
        self.plot_search_correlation(axes[1, 0])
        
        # 5. 预测模型表现
        self.plot_model_performance(axes[1, 1])
        
        # 6. 关键指标总览
        self.plot_key_metrics_summary(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_time_series_trends(self, ax):
        """绘制时间序列趋势"""
        ax.set_title('Business Metrics Time Series', fontweight='bold')
        
        # 创建双y轴
        ax2 = ax.twinx()
        
        # 左轴：账户获取数
        line1 = ax.plot(self.data['cdr_date'], self.data['acc_get_cnt'], 
                       color='blue', label='Account Acquisition', linewidth=2)
        ax.set_ylabel('Account Acquisition', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 右轴：客服电话数
        line2 = ax2.plot(self.data['cdr_date'], self.data['call_num'], 
                        color='red', label='Service Calls', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Service Calls', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 标记营销投放期间
        campaign_periods = self.data[self.data['cm_flg'] == 1]['cdr_date']
        for date in campaign_periods:
            ax.axvline(x=date, color='green', alpha=0.3, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        # 图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def plot_marketing_effectiveness(self, ax):
        """绘制营销效果分析"""
        ax.set_title('Marketing Campaign Effectiveness', fontweight='bold')
        
        if 'marketing' in self.insights:
            insights = self.insights['marketing']
            
            categories = ['No Campaign', 'Campaign Days']
            values = [insights['avg_acquisition_no_campaign'], 
                     insights['avg_acquisition_campaign']]
            colors = ['lightblue', 'orange']
            
            bars = ax.bar(categories, values, color=colors)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # 添加提升幅度
            lift_text = f"Lift: {insights['relative_lift']:.1%}"
            ax.text(0.5, max(values) * 0.9, lift_text, 
                   transform=ax.transAxes, ha='center', 
                   fontsize=12, fontweight='bold', color='green')
            
            ax.set_ylabel('Avg Daily Acquisitions')
            ax.grid(True, alpha=0.3)
    
    def plot_service_patterns(self, ax):
        """绘制客服需求模式"""
        ax.set_title('Customer Service Demand Patterns', fontweight='bold')
        
        if 'customer_service' in self.insights:
            insights = self.insights['customer_service']
            
            weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            pattern_values = [insights['weekly_pattern'].get(i, 1) for i in range(7)]
            
            bars = ax.bar(weekdays, pattern_values, 
                         color=['lightcoral' if x > 1.1 else 'lightblue' for x in pattern_values])
            
            # 添加基准线
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Average')
            
            # 添加数值标签
            for bar, value in zip(bars, pattern_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}x',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Demand Multiplier')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_search_correlation(self, ax):
        """绘制搜索趋势关联"""
        ax.set_title('Search Trend vs Business Metrics', fontweight='bold')
        
        # 过滤有效数据
        valid_data = self.data.dropna(subset=['search_cnt', 'acc_get_cnt'])
        
        if len(valid_data) > 10:
            scatter = ax.scatter(valid_data['search_cnt'], valid_data['acc_get_cnt'], 
                               alpha=0.6, color='purple')
            
            # 添加趋势线
            if len(valid_data) > 2:
                z = np.polyfit(valid_data['search_cnt'], valid_data['acc_get_cnt'], 1)
                p = np.poly1d(z)
                ax.plot(valid_data['search_cnt'], p(valid_data['search_cnt']), 
                       "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Search Volume')
            ax.set_ylabel('Account Acquisitions')
            
            # 添加相关性信息
            if 'search_correlation' in self.insights:
                corr = self.insights['search_correlation']['current_correlation']
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.grid(True, alpha=0.3)
    
    def plot_model_performance(self, ax):
        """绘制模型表现"""
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
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontweight='bold')
                
                ax.set_ylabel('R² Score')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
    
    def plot_key_metrics_summary(self, ax):
        """绘制关键指标总览"""
        ax.set_title('Key Business Metrics Summary', fontweight='bold')
        
        # 计算关键指标
        metrics = {
            'Avg Daily Acquisitions': self.data['acc_get_cnt'].mean(),
            'Avg Daily Service Calls': self.data['call_num'].mean(),
            'Campaign Days': self.data['cm_flg'].sum(),
            'Total Days': len(self.data)
        }
        
        # 创建表格显示
        table_data = []
        for metric, value in metrics.items():
            if isinstance(value, float):
                table_data.append([metric, f"{value:.1f}"])
            else:
                table_data.append([metric, str(value)])
        
        # 添加营销提升信息
        if 'marketing' in self.insights:
            lift = self.insights['marketing']['relative_lift']
            table_data.append(['Marketing Lift', f"{lift:.1%}"])
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 隐藏坐标轴
        ax.axis('off')

# 6. 商业价值量化类
class BusinessValueQuantifier:
    def __init__(self, insights):
        self.insights = insights
    
    def calculate_marketing_value(self):
        """计算营销价值"""
        print("\n💰 营销价值量化")
        print("=" * 40)
        
        if 'marketing' not in self.insights:
            print("❌ 缺少营销分析数据")
            return {}
        
        marketing = self.insights['marketing']
        
        # 假设参数
        customer_ltv = 180000  # 客户生命周期价值（日元）
        daily_campaign_cost = 50000  # 日营销成本
        
        # 计算价值
        daily_incremental_customers = marketing['absolute_lift']
        daily_incremental_revenue = daily_incremental_customers * customer_ltv
        daily_roi = (daily_incremental_revenue - daily_campaign_cost) / daily_campaign_cost
        
        # 年度价值
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
        
        print(f"📊 日增量客户: {daily_incremental_customers:.1f} 个")
        print(f"📊 日增量收入: ¥{daily_incremental_revenue:,.0f}")
        print(f"📊 日营销ROI: {daily_roi:.1%}")
        print(f"📊 年净价值: ¥{annual_net_value:,.0f}")
        
        return value_metrics
    
    def calculate_service_optimization_value(self):
        """计算客服优化价值"""
        print("\n📞 客服优化价值量化")
        print("=" * 40)
        
        # 假设参数
        current_annual_cost = 15000000  # 当前年度客服成本
        avg_hourly_cost = 3000  # 平均小时成本
        
        # 基于预测模型的优化潜力
        if 'prediction_models' in self.insights and 'call_prediction' in self.insights['prediction_models']:
            model_accuracy = self.insights['prediction_models']['call_prediction'].get('model_score', 0)
            
            # 假设预测准确率每提升10%，可以节省5%的成本
            cost_reduction_rate = min(0.25, model_accuracy * 0.3)  # 最高25%
            
            annual_cost_savings = current_annual_cost * cost_reduction_rate
            
            # 服务质量改善价值
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
            
            print(f"📊 当前年度成本: ¥{current_annual_cost:,.0f}")
            print(f"📊 成本节约率: {cost_reduction_rate:.1%}")
            print(f"📊 年度成本节约: ¥{annual_cost_savings:,.0f}")
            print(f"📊 服务改善价值: ¥{service_improvement_value:,.0f}")
            print(f"📊 总年度价值: ¥{total_value:,.0f}")
            
            return value_metrics
        else:
            print("❌ 缺少预测模型数据")
            return {}
    
    def generate_executive_summary(self):
        """生成执行摘要"""
        print("\n📋 执行摘要")
        print("=" * 50)
        
        # 汇总所有价值
        marketing_value = self.calculate_marketing_value()
        service_value = self.calculate_service_optimization_value()
        
        total_annual_value = 0
        if marketing_value:
            total_annual_value += marketing_value.get('annual_net_value', 0)
        if service_value:
            total_annual_value += service_value.get('total_annual_value', 0)
        
        print(f"\n🎯 关键发现:")
        
        if marketing_value:
            print(f"   • 营销ROI可达 {marketing_value['daily_roi']:.1%}")
            print(f"   • 年度营销净价值 ¥{marketing_value['annual_net_value']:,.0f}")
        
        if service_value:
            print(f"   • 客服成本可节约 {service_value['cost_reduction_rate']:.1%}")
            print(f"   • 年度客服优化价值 ¥{service_value['total_annual_value']:,.0f}")
        
        print(f"\n💰 总商业价值:")
        print(f"   • 年度总价值: ¥{total_annual_value:,.0f}")
        print(f"   • 3年累计价值: ¥{total_annual_value * 3:,.0f}")
        
        print(f"\n🚀 优先建议:")
        print(f"   1. 立即优化营销投放策略")
        print(f"   2. 部署客服需求预测系统")
        print(f"   3. 建立数据监控体系")
        print(f"   4. 投资数据科学团队")
        
        return {
            'marketing_value': marketing_value,
            'service_value': service_value,
            'total_annual_value': total_annual_value
        }

# 7. 主执行函数
def main():
    """主执行函数"""
    print("🚀 开始POS业务数据分析...")
    
    # 1. 数据加载
    loader = DataLoader()
    if not loader.load_all_data():
        print("❌ 数据加载失败，请检查文件路径")
        return
    
    # 2. 数据质量检查
    loader.check_data_quality()
    
    # 3. 数据预处理
    preprocessor = DataPreprocessor(loader.data)
    master_data = preprocessor.create_master_dataset()
    featured_data = preprocessor.create_features()
    
    # 4. 业务分析
    analyzer = BusinessAnalyzer(featured_data)
    
    # 执行各项分析
    marketing_insights = analyzer.analyze_marketing_effectiveness()
    service_insights = analyzer.analyze_customer_service_patterns()
    search_insights = analyzer.analyze_search_business_correlation()
    model_insights = analyzer.build_prediction_models()
    
    # 5. 数据可视化
    visualizer = BusinessVisualizer(featured_data, analyzer.insights)
    dashboard = visualizer.create_comprehensive_dashboard()
    
    # 6. 商业价值量化
    value_quantifier = BusinessValueQuantifier(analyzer.insights)
    business_value = value_quantifier.generate_executive_summary()
    
    print("\n✅ 分析完成！")
    print("📊 所有洞察和建议已生成")
    print("💡 请查看上方的可视化图表和分析结果")
    
    return {
        'data': featured_data,
        'insights': analyzer.insights,
        'business_value': business_value,
        'dashboard': dashboard
    }

# 8. 执行分析
if __name__ == "__main__":
    results = main()
