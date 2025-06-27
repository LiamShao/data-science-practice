# HR数据分析价值发现框架
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HRValueDiscovery:
    def __init__(self, df):
        self.df = df.copy()
        self.insights = []
        self.recommendations = []
        
    def quick_business_value_scan(self):
        """快速扫描潜在业务价值点"""
        print("=== 快速业务价值扫描 ===\n")
        
        # 1. 流失率基本情况
        attrition_rate = (self.df['Attrition'] == 'Yes').mean()
        print(f"🔍 整体流失率: {attrition_rate:.2%}")
        
        if attrition_rate > 0.15:
            self.insights.append("高流失率警告: 超过15%的员工流失")
            self.recommendations.append("优先级1: 建立流失预警系统")
        
        # 2. 部门流失差异
        dept_attrition = self.df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
        print(f"\n📊 各部门流失率:")
        print(dept_attrition.round(3))
        
        # 3. 薪酬分布异常检测
        income_stats = self.df['MonthlyIncome'].describe()
        print(f"\n💰 薪酬分布:")
        print(f"中位数: ${income_stats['50%']:,.0f}")
        print(f"均值: ${income_stats['mean']:,.0f}")
        
        # 薪酬差异过大检测
        if income_stats['std'] / income_stats['mean'] > 0.5:
            self.insights.append("薪酬分布不均: 标准差过大，存在薪酬公平性问题")
            self.recommendations.append("优先级2: 薪酬体系审查与优化")
            
        # 4. 满意度与流失关系
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                           'RelationshipSatisfaction', 'WorkLifeBalance']
        
        for col in satisfaction_cols:
            if col in self.df.columns:
                avg_satisfaction_stay = self.df[self.df['Attrition'] == 'No'][col].mean()
                avg_satisfaction_leave = self.df[self.df['Attrition'] == 'Yes'][col].mean()
                diff = avg_satisfaction_stay - avg_satisfaction_leave
                
                if diff > 0.5:
                    self.insights.append(f"{col}显著影响流失: 差异达{diff:.2f}")
        
        return self.insights, self.recommendations
    
    def identify_high_value_segments(self):
        """识别高价值员工群体和风险群体"""
        print("\n=== 高价值群体识别 ===\n")
        
        # 定义高价值员工：高绩效 + 高薪酬 + 长期员工
        high_performers = self.df[
            (self.df['PerformanceRating'] >= 3) & 
            (self.df['MonthlyIncome'] > self.df['MonthlyIncome'].quantile(0.75)) &
            (self.df['YearsAtCompany'] >= 3)
        ]
        
        high_performer_attrition = (high_performers['Attrition'] == 'Yes').mean()
        print(f"💎 高价值员工流失率: {high_performer_attrition:.2%}")
        
        if high_performer_attrition > 0.1:
            self.insights.append("关键人才流失风险: 高价值员工流失率超10%")
            self.recommendations.append("紧急: 制定关键人才保留计划")
        
        # 识别高风险群体
        risk_factors = []
        if 'OverTime' in self.df.columns:
            overtime_attrition = self.df[self.df['OverTime'] == 'Yes']['Attrition'].apply(
                lambda x: x == 'Yes').mean()
            if overtime_attrition > 0.25:
                risk_factors.append("加班员工")
                
        print(f"📈 高风险群体识别完成，发现{len(risk_factors)}个风险因素")
        
    def calculate_business_impact(self):
        """计算潜在业务影响"""
        print("\n=== 业务影响计算 ===\n")
        
        total_employees = len(self.df)
        current_attrition = (self.df['Attrition'] == 'Yes').sum()
        
        # 假设替换成本为年薪的50%
        avg_annual_salary = self.df['MonthlyIncome'].mean() * 12
        replacement_cost_per_employee = avg_annual_salary * 0.5
        
        current_cost = current_attrition * replacement_cost_per_employee
        
        print(f"💸 当前年度流失成本: ${current_cost:,.0f}")
        
        # 如果流失率降低5%的潜在节省
        potential_reduction = total_employees * 0.05 * replacement_cost_per_employee
        print(f"💰 流失率降低5%的潜在节省: ${potential_reduction:,.0f}")
        
        self.insights.append(f"年度流失成本约${current_cost/1000000:.1f}M")
        self.recommendations.append(f"目标: 通过数据驱动策略节省${potential_reduction/1000000:.1f}M")
        
    def generate_actionable_insights(self):
        """生成可执行的洞察建议"""
        print("\n=== 可执行洞察与建议 ===\n")
        
        # 特征重要性分析（用于指导行动方向）
        X = self.df.drop('Attrition', axis=1)
        y = self.df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # 处理分类变量
        categorical_columns = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
        
        # 训练简单模型获取特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_encoded, y)
        
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🎯 影响流失的TOP5因素:")
        top_5_features = feature_importance.head(5)
        for idx, row in top_5_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
            
        # 基于TOP特征生成具体建议
        top_feature = top_5_features.iloc[0]['feature']
        
        action_map = {
            'MonthlyIncome': "薪酬调整：建立基于市场的薪酬体系",
            'Age': "年龄管理：关注不同年龄段员工需求",
            'JobSatisfaction': "满意度提升：改善工作内容和环境",
            'WorkLifeBalance': "平衡政策：推行弹性工作制度",
            'YearsAtCompany': "职业发展：建立明确的晋升通道"
        }
        
        if top_feature in action_map:
            self.recommendations.append(f"核心行动: {action_map[top_feature]}")
    
    def create_value_proposition(self):
        """创建价值主张报告"""
        print("\n" + "="*50)
        print("           数据驱动HR优化价值主张")
        print("="*50)
        
        print("\n🔍 核心发现:")
        for i, insight in enumerate(self.insights, 1):
            print(f"   {i}. {insight}")
            
        print("\n🎯 推荐行动:")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"   {i}. {rec}")
            
        print("\n📊 建议的分析产品:")
        products = [
            "员工流失风险预警仪表板",
            "薪酬公平性分析报告", 
            "部门绩效对比分析",
            "员工满意度改善路径图",
            "高潜人才识别与发展计划"
        ]
        
        for i, product in enumerate(products, 1):
            print(f"   {i}. {product}")
            
        return self.insights, self.recommendations

# 使用示例
def analyze_hr_dataset(df):
    """完整的HR数据集价值发现流程"""
    analyzer = HRValueDiscovery(df)
    
    # 执行分析
    analyzer.quick_business_value_scan()
    analyzer.identify_high_value_segments()
    analyzer.calculate_business_impact() 
    analyzer.generate_actionable_insights()
    
    # 生成最终报告
    insights, recommendations = analyzer.create_value_proposition()
    
    return insights, recommendations

# 如果有数据集，运行分析：
# insights, recs = analyze_hr_dataset(your_df)