# HR数据深度价值挖掘分析系统
# Phase 3: 流失预测、薪酬优化、组织效能全方位分析
# 专为Google Colab设计

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

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("🚀 HR数据深度价值挖掘分析系统")
print("="*80)
print("📋 分析模块: A.流失预测与成本优化 | B.薪酬优化分析 | C.组织效能提升")
print("="*80)

print(f"✅ 数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")

class HRValueMiner:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.results = {}
        
        print(f"📊 数据概览:")
        print(f"   流失率: {(self.df['Attrition'] == 'Yes').mean():.1%}")
        print(f"   部门数: {self.df['Department'].nunique()}")
        print(f"   岗位数: {self.df['JobRole'].nunique()}")
    
    # =================== A. 流失预测与成本优化 ===================
    
    def build_attrition_risk_model(self):
        """建立流失风险评分模型"""
        print("\n" + "="*60)
        print("🎯 A1. 流失风险评分模型构建")
        print("="*60)
        
        # 数据预处理
        X = self.df.copy()
        y = (X['Attrition'] == 'Yes').astype(int)
        X = X.drop(['Attrition'], axis=1)
        
        # 编码分类变量
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 训练多个模型
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        model_results = {}
        
        for name, model in models.items():
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 评估
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
        
        # 选择最佳模型
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
        best_model = model_results[best_model_name]['model']
        
        print(f"\n🏆 最佳模型: {best_model_name}")
        
        # 计算风险评分
        risk_scores = best_model.predict_proba(X)[:, 1]
        self.df['AttritionRiskScore'] = risk_scores
        
        # 风险等级分组
        self.df['RiskLevel'] = pd.cut(risk_scores, 
                                     bins=[0, 0.3, 0.6, 0.8, 1.0],
                                     labels=['低风险', '中等风险', '高风险', '极高风险'])
        
        # 风险分布统计
        risk_distribution = self.df['RiskLevel'].value_counts()
        print(f"\n📊 风险等级分布:")
        for level, count in risk_distribution.items():
            percentage = count / len(self.df) * 100
            print(f"   {level}: {count}人 ({percentage:.1f}%)")
        
        # 特征重要性（如果是随机森林）
        if best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 TOP10 流失预测关键因素:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"   {i}. {row['feature']}: {row['importance']:.3f}")
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 风险分布
        plt.subplot(1, 3, 1)
        risk_distribution.plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('员工流失风险等级分布')
        plt.ylabel('员工数量')
        plt.xticks(rotation=45)
        
        # ROC曲线
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {model_results[best_model_name]["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC曲线')
        plt.legend()
        
        # 风险评分分布
        plt.subplot(1, 3, 3)
        plt.hist(risk_scores, bins=30, alpha=0.7, color='orange')
        plt.xlabel('流失风险评分')
        plt.ylabel('员工数量')
        plt.title('风险评分分布')
        
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
        """计算各部门人才替换成本"""
        print("\n" + "="*60)
        print("💰 A2. 部门人才替换成本分析")
        print("="*60)
        
        # 基础成本假设
        RECRUITMENT_COST_RATIO = 0.3  # 招聘成本为年薪的30%
        TRAINING_COST_RATIO = 0.2     # 培训成本为年薪的20%
        PRODUCTIVITY_LOSS_RATIO = 0.25 # 生产力损失为年薪的25%
        
        # 按部门计算
        dept_analysis = self.df.groupby('Department').agg({
            'MonthlyIncome': ['mean', 'count'],
            'Attrition': lambda x: (x == 'Yes').sum(),
            'AttritionRiskScore': 'mean' if 'AttritionRiskScore' in self.df.columns else lambda x: 0
        }).round(2)
        
        dept_analysis.columns = ['平均月薪', '员工总数', '实际流失人数', '平均风险评分']
        
        # 计算替换成本
        dept_analysis['年薪'] = dept_analysis['平均月薪'] * 12
        dept_analysis['单人替换成本'] = dept_analysis['年薪'] * (RECRUITMENT_COST_RATIO + TRAINING_COST_RATIO + PRODUCTIVITY_LOSS_RATIO)
        dept_analysis['年度流失成本'] = dept_analysis['单人替换成本'] * dept_analysis['实际流失人数']
        dept_analysis['流失率'] = dept_analysis['实际流失人数'] / dept_analysis['员工总数']
        
        # 预测未来风险
        if 'AttritionRiskScore' in self.df.columns:
            dept_analysis['预测流失人数'] = (dept_analysis['员工总数'] * dept_analysis['平均风险评分']).round(0)
            dept_analysis['预测年度成本'] = dept_analysis['单人替换成本'] * dept_analysis['预测流失人数']
        
        print(f"📊 各部门人才替换成本分析:")
        print(dept_analysis[['员工总数', '流失率', '单人替换成本', '年度流失成本']].to_string())
        
        # 总体成本统计
        total_current_cost = dept_analysis['年度流失成本'].sum()
        total_predicted_cost = dept_analysis['预测年度成本'].sum() if 'AttritionRiskScore' in self.df.columns else 0
        
        print(f"\n💸 成本汇总:")
        print(f"   当前年度总流失成本: ${total_current_cost:,.0f}")
        if total_predicted_cost > 0:
            print(f"   预测年度总流失成本: ${total_predicted_cost:,.0f}")
            print(f"   成本变化: ${total_predicted_cost - total_current_cost:+,.0f}")
        
        # 可视化
        plt.figure(figsize=(15, 10))
        
        # 各部门流失成本
        plt.subplot(2, 2, 1)
        dept_analysis['年度流失成本'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('各部门年度流失成本')
        plt.ylabel('成本 ($)')
        plt.xticks(rotation=45)
        
        # 流失率对比
        plt.subplot(2, 2, 2)
        dept_analysis['流失率'].plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('各部门流失率')
        plt.ylabel('流失率')
        plt.xticks(rotation=45)
        
        # 成本构成饼图
        plt.subplot(2, 2, 3)
        cost_components = {
            '招聘成本': RECRUITMENT_COST_RATIO,
            '培训成本': TRAINING_COST_RATIO,
            '生产力损失': PRODUCTIVITY_LOSS_RATIO
        }
        plt.pie(cost_components.values(), labels=cost_components.keys(), autopct='%1.1f%%')
        plt.title('替换成本构成')
        
        # 部门员工数量
        plt.subplot(2, 2, 4)
        dept_analysis['员工总数'].plot(kind='bar', color='lightgreen', alpha=0.8)
        plt.title('各部门员工数量')
        plt.ylabel('员工数')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        self.results['replacement_costs'] = dept_analysis
        
        return dept_analysis
    
    def identify_hidden_flight_risk(self):
        """识别隐形离职员工"""
        print("\n" + "="*60)
        print("👻 A3. 隐形离职员工识别")
        print("="*60)
        
        # 定义隐形离职条件：低满意度但未离职
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_cols if col in self.df.columns]
        
        if len(available_satisfaction) == 0:
            print("❌ 缺少满意度数据")
            return None
        
        # 计算综合满意度
        self.df['OverallSatisfaction'] = self.df[available_satisfaction].mean(axis=1)
        
        # 隐形离职条件
        conditions = {
            '低满意度': self.df['OverallSatisfaction'] <= 2.0,
            '在职状态': self.df['Attrition'] == 'No',
            '高压力': self.df['StressRating'] > self.df['StressRating'].quantile(0.7) if 'StressRating' in self.df.columns else False
        }
        
        # 基础隐形离职群体：低满意度 + 在职
        hidden_flight_basic = self.df[conditions['低满意度'] & conditions['在职状态']]
        
        # 高风险隐形离职：基础条件 + 高压力
        if 'StressRating' in self.df.columns:
            hidden_flight_high_risk = self.df[
                conditions['低满意度'] & 
                conditions['在职状态'] & 
                conditions['高压力']
            ]
        else:
            hidden_flight_high_risk = hidden_flight_basic
        
        print(f"📊 隐形离职员工识别结果:")
        print(f"   基础隐形离职群体: {len(hidden_flight_basic)}人 ({len(hidden_flight_basic)/len(self.df)*100:.1f}%)")
        print(f"   高风险隐形离职群体: {len(hidden_flight_high_risk)}人 ({len(hidden_flight_high_risk)/len(self.df)*100:.1f}%)")
        
        # 分析隐形离职群体特征
        if len(hidden_flight_basic) > 0:
            print(f"\n🔍 隐形离职群体特征分析:")
            
            # 部门分布
            dept_distribution = hidden_flight_basic['Department'].value_counts()
            print(f"   部门分布: {dict(dept_distribution.head(3))}")
            
            # 岗位分布
            role_distribution = hidden_flight_basic['JobRole'].value_counts()
            print(f"   岗位分布: {dict(role_distribution.head(3))}")
            
            # 关键数值特征
            key_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
            available_features = [f for f in key_features if f in self.df.columns]
            
            if available_features:
                print(f"\n📈 关键特征对比 (隐形离职 vs 正常员工):")
                normal_employees = self.df[
                    (self.df['OverallSatisfaction'] > 2.5) & 
                    (self.df['Attrition'] == 'No')
                ]
                
                for feature in available_features:
                    hidden_mean = hidden_flight_basic[feature].mean()
                    normal_mean = normal_employees[feature].mean()
                    diff = hidden_mean - normal_mean
                    
                    print(f"   {feature}: {hidden_mean:.1f} vs {normal_mean:.1f} (差异: {diff:+.1f})")
        
        # 如果有风险评分，分析风险分布
        if 'AttritionRiskScore' in self.df.columns and len(hidden_flight_basic) > 0:
            avg_risk_hidden = hidden_flight_basic['AttritionRiskScore'].mean()
            avg_risk_normal = self.df[self.df['Attrition'] == 'No']['AttritionRiskScore'].mean()
            
            print(f"\n⚠️ 风险评分对比:")
            print(f"   隐形离职群体平均风险: {avg_risk_hidden:.3f}")
            print(f"   正常员工平均风险: {avg_risk_normal:.3f}")
            print(f"   风险差异: {avg_risk_hidden - avg_risk_normal:+.3f}")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 满意度分布对比
        axes[0, 0].hist(self.df[self.df['Attrition'] == 'No']['OverallSatisfaction'], 
                       alpha=0.7, label='正常员工', bins=20, color='green')
        axes[0, 0].hist(hidden_flight_basic['OverallSatisfaction'], 
                       alpha=0.7, label='隐形离职', bins=20, color='red')
        axes[0, 0].set_title('满意度分布对比')
        axes[0, 0].set_xlabel('综合满意度')
        axes[0, 0].legend()
        
        # 部门分布
        if len(hidden_flight_basic) > 0:
            dept_dist = hidden_flight_basic['Department'].value_counts()
            axes[0, 1].bar(range(len(dept_dist)), dept_dist.values, color='orange', alpha=0.8)
            axes[0, 1].set_title('隐形离职员工部门分布')
            axes[0, 1].set_xticks(range(len(dept_dist)))
            axes[0, 1].set_xticklabels(dept_dist.index, rotation=45)
        
        # 风险评分分布（如果有）
        if 'AttritionRiskScore' in self.df.columns:
            axes[1, 0].hist(self.df[self.df['Attrition'] == 'No']['AttritionRiskScore'], 
                           alpha=0.7, label='正常员工', bins=20, color='blue')
            if len(hidden_flight_basic) > 0:
                axes[1, 0].hist(hidden_flight_basic['AttritionRiskScore'], 
                               alpha=0.7, label='隐形离职', bins=20, color='red')
            axes[1, 0].set_title('风险评分分布对比')
            axes[1, 0].set_xlabel('流失风险评分')
            axes[1, 0].legend()
        
        # 年龄分布对比
        if 'Age' in self.df.columns:
            axes[1, 1].hist(self.df[self.df['Attrition'] == 'No']['Age'], 
                           alpha=0.7, label='正常员工', bins=20, color='green')
            if len(hidden_flight_basic) > 0:
                axes[1, 1].hist(hidden_flight_basic['Age'], 
                               alpha=0.7, label='隐形离职', bins=20, color='red')
            axes[1, 1].set_title('年龄分布对比')
            axes[1, 1].set_xlabel('年龄')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 生成干预建议
        print(f"\n💡 干预建议:")
        if len(hidden_flight_basic) > 0:
            recommendations = [
                f"立即关注{len(hidden_flight_high_risk)}名高风险隐形离职员工",
                "开展满意度提升专项行动，重点关注工作环境和生活平衡",
                "建立定期沟通机制，了解员工真实想法",
                "考虑岗位调整或职业发展机会"
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
    
    # =================== B. 薪酬优化与公平性分析 ===================
    
    def analyze_compensation_equity(self):
        """同岗位薪酬差异分析"""
        print("\n" + "="*60)
        print("⚖️ B1. 同岗位薪酬公平性分析")
        print("="*60)
        
        # 按岗位分析薪酬分布
        job_salary_stats = self.df.groupby('JobRole')['MonthlyIncome'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(0)
        
        job_salary_stats.columns = ['员工数', '均值', '中位数', '标准差', '最小值', '最大值']
        job_salary_stats['变异系数'] = (job_salary_stats['标准差'] / job_salary_stats['均值']).round(3)
        job_salary_stats['薪酬范围'] = job_salary_stats['最大值'] - job_salary_stats['最小值']
        
        # 筛选员工数量足够的岗位进行分析
        significant_roles = job_salary_stats[job_salary_stats['员工数'] >= 10]
        
        print(f"📊 主要岗位薪酬统计 (员工数≥10):")
        print(significant_roles[['员工数', '均值', '中位数', '变异系数']].to_string())
        
        # 识别薪酬差异过大的岗位
        high_variance_roles = significant_roles[significant_roles['变异系数'] > 0.3]
        
        if len(high_variance_roles) > 0:
            print(f"\n⚠️ 薪酬差异较大的岗位 (变异系数>0.3):")
            for role in high_variance_roles.index:
                cv = high_variance_roles.loc[role, '变异系数']
                range_val = high_variance_roles.loc[role, '薪酬范围']
                print(f"   {role}: 变异系数{cv:.3f}, 薪酬范围${range_val:,.0f}")
        
        # 性别薪酬公平性分析
        if 'Gender' in self.df.columns:
            print(f"\n👥 性别薪酬公平性分析:")
            
            gender_salary = self.df.groupby(['JobRole', 'Gender'])['MonthlyIncome'].mean().unstack()
            if gender_salary.shape[1] == 2:  # 确保有男女两个性别
                gender_salary['薪酬差异'] = gender_salary.iloc[:, 0] - gender_salary.iloc[:, 1]
                gender_salary['差异百分比'] = (gender_salary['薪酬差异'] / gender_salary.mean(axis=1) * 100).round(1)
                
                # 找出差异较大的岗位
                significant_gaps = gender_salary[abs(gender_salary['差异百分比']) > 10]
                
                if len(significant_gaps) > 0:
                    print(f"   发现{len(significant_gaps)}个岗位存在显著性别薪酬差异(>10%):")
                    for role in significant_gaps.index:
                        gap = significant_gaps.loc[role, '差异百分比']
                        print(f"   {role}: {gap:+.1f}%")
        
        # 学历与薪酬关系
        if 'Education' in self.df.columns:
            edu_salary = self.df.groupby('Education')['MonthlyIncome'].mean().sort_index()
            print(f"\n🎓 学历与薪酬关系:")
            for edu_level, salary in edu_salary.items():
                print(f"   学历等级{edu_level}: ${salary:,.0f}")
        
        # 可视化
        plt.figure(figsize=(18, 12))
        
        # 岗位薪酬分布箱线图
        plt.subplot(2, 3, 1)
        roles_to_plot = significant_roles.head(6).index
        salary_data = [self.df[self.df['JobRole'] == role]['MonthlyIncome'] for role in roles_to_plot]
        plt.boxplot(salary_data, labels=roles_to_plot)
        plt.title('主要岗位薪酬分布')
        plt.ylabel('月薪 ($)')
        plt.xticks(rotation=45)
        
        # 薪酬变异系数
        plt.subplot(2, 3, 2)
        significant_roles['变异系数'].plot(kind='bar', color='orange', alpha=0.8)
        plt.title('岗位薪酬变异系数')
        plt.ylabel('变异系数')
        plt.xticks(rotation=45)
        
        # 性别薪酬对比（如果有数据）
        if 'Gender' in self.df.columns:
            plt.subplot(2, 3, 3)
            self.df.boxplot(column='MonthlyIncome', by='Gender', ax=plt.gca())
            plt.title('性别薪酬分布对比')
            plt.suptitle('')
        
        # 学历薪酬关系
        if 'Education' in self.df.columns:
            plt.subplot(2, 3, 4)
            self.df.boxplot(column='MonthlyIncome', by='Education', ax=plt.gca())
            plt.title('学历与薪酬关系')
            plt.suptitle('')
        
        # 薪酬范围分析
        plt.subplot(2, 3, 5)
        significant_roles['薪酬范围'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('岗位薪酬范围')
        plt.ylabel('薪酬范围 ($)')
        plt.xticks(rotation=45)
        
        # 整体薪酬分布
        plt.subplot(2, 3, 6)
        plt.hist(self.df['MonthlyIncome'], bins=30, alpha=0.7, color='skyblue')
        plt.title('整体薪酬分布')
        plt.xlabel('月薪 ($)')
        plt.ylabel('员工数量')
        
        plt.tight_layout()
        plt.show()
        
        self.results['compensation_equity'] = {
            'job_salary_stats': job_salary_stats,
            'high_variance_roles': high_variance_roles,
            'gender_salary': gender_salary if 'Gender' in self.df.columns else None
        }
        
        return job_salary_stats, high_variance_roles
    
    def evaluate_performance_compensation_alignment(self):
        """绩效与薪酬匹配度评估"""
        print("\n" + "="*60)
        print("🎯 B2. 绩效与薪酬匹配度评估")
        print("="*60)
        
        # 绩效与薪酬相关性
        performance_cols = ['PerformanceRating', 'PerformanceIndex', 'MonthlyAchievement']
        available_perf = [col for col in performance_cols if col in self.df.columns]
        
        if len(available_perf) == 0:
            print("❌ 缺少绩效数据")
            return None
        
        print(f"📊 绩效与薪酬相关性分析:")
        correlations = {}
        
        for perf_col in available_perf:
            corr = self.df[perf_col].corr(self.df['MonthlyIncome'])
            correlations[perf_col] = corr
            print(f"   {perf_col} 与薪酬相关系数: {corr:.3f}")
        
        # 使用主要绩效指标进行深度分析
        main_perf_col = max(correlations, key=correlations.get)
        print(f"\n🎯 以{main_perf_col}为主要绩效指标进行深度分析")
        
        # 创建绩效分组
        perf_groups = pd.qcut(self.df[main_perf_col], q=4, labels=['低绩效', '中下绩效', '中上绩效', '高绩效'])
        self.df['PerformanceGroup'] = perf_groups
        
        # 各绩效组薪酬统计
        perf_salary_stats = self.df.groupby('PerformanceGroup')['MonthlyIncome'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(0)
        
        print(f"\n📈 各绩效组薪酬统计:")
        print(perf_salary_stats.to_string())
        
        # 识别薪酬不匹配情况
        # 高绩效低薪酬
        high_perf_threshold = self.df[main_perf_col].quantile(0.8)
        low_salary_threshold = self.df['MonthlyIncome'].quantile(0.3)
        
        high_perf_low_pay = self.df[
            (self.df[main_perf_col] >= high_perf_threshold) & 
            (self.df['MonthlyIncome'] <= low_salary_threshold)
        ]
        
        # 低绩效高薪酬
        low_perf_threshold = self.df[main_perf_col].quantile(0.2)
        high_salary_threshold = self.df['MonthlyIncome'].quantile(0.8)
        
        low_perf_high_pay = self.df[
            (self.df[main_perf_col] <= low_perf_threshold) & 
            (self.df['MonthlyIncome'] >= high_salary_threshold)
        ]
        
        print(f"\n⚠️ 薪酬不匹配情况:")
        print(f"   高绩效低薪酬: {len(high_perf_low_pay)}人 ({len(high_perf_low_pay)/len(self.df)*100:.1f}%)")
        print(f"   低绩效高薪酬: {len(low_perf_high_pay)}人 ({len(low_perf_high_pay)/len(self.df)*100:.1f}%)")
        
        # 计算薪酬公平性指数
        expected_salary = self.df.groupby('PerformanceGroup')['MonthlyIncome'].transform('mean')
        actual_salary = self.df['MonthlyIncome']
        fairness_index = 1 - abs(actual_salary - expected_salary) / expected_salary
        self.df['SalaryFairnessIndex'] = fairness_index
        
        avg_fairness = fairness_index.mean()
        print(f"\n📊 薪酬公平性指数: {avg_fairness:.3f} (1.0为完全公平)")
        
        # 不匹配员工的流失风险
        if len(high_perf_low_pay) > 0:
            high_perf_low_pay_attrition = (high_perf_low_pay['Attrition'] == 'Yes').mean()
            print(f"   高绩效低薪酬员工流失率: {high_perf_low_pay_attrition:.1%}")
        
        if len(low_perf_high_pay) > 0:
            low_perf_high_pay_attrition = (low_perf_high_pay['Attrition'] == 'Yes').mean()
            print(f"   低绩效高薪酬员工流失率: {low_perf_high_pay_attrition:.1%}")
        
        # 可视化
        plt.figure(figsize=(15, 10))
        
        # 绩效vs薪酬散点图
        plt.subplot(2, 3, 1)
        colors = ['red' if x == 'Yes' else 'blue' for x in self.df['Attrition']]
        plt.scatter(self.df[main_perf_col], self.df['MonthlyIncome'], c=colors, alpha=0.6)
        plt.xlabel(main_perf_col)
        plt.ylabel('月薪 ($)')
        plt.title(f'{main_perf_col} vs 薪酬 (红=离职)')
        
        # 各绩效组薪酬分布
        plt.subplot(2, 3, 2)
        self.df.boxplot(column='MonthlyIncome', by='PerformanceGroup', ax=plt.gca())
        plt.title('各绩效组薪酬分布')
        plt.suptitle('')
        
        # 薪酬公平性指数分布
        plt.subplot(2, 3, 3)
        plt.hist(fairness_index, bins=30, alpha=0.7, color='green')
        plt.xlabel('薪酬公平性指数')
        plt.ylabel('员工数量')
        plt.title('薪酬公平性指数分布')
        
        # 不匹配情况可视化
        plt.subplot(2, 3, 4)
        mismatch_data = [len(high_perf_low_pay), len(low_perf_high_pay)]
        mismatch_labels = ['高绩效低薪酬', '低绩效高薪酬']
        plt.bar(mismatch_labels, mismatch_data, color=['orange', 'red'], alpha=0.8)
        plt.title('薪酬不匹配员工数量')
        plt.ylabel('员工数')
        
        # 绩效组薪酬均值
        plt.subplot(2, 3, 5)
        perf_salary_stats['mean'].plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('各绩效组平均薪酬')
        plt.ylabel('平均月薪 ($)')
        plt.xticks(rotation=45)
        
        # 相关性热力图
        plt.subplot(2, 3, 6)
        corr_data = self.df[available_perf + ['MonthlyIncome']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
        plt.title('绩效指标与薪酬相关性')
        
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
        """市场薪酬竞争力分析"""
        print("\n" + "="*60)
        print("🏢 B3. 市场薪酬竞争力分析")
        print("="*60)
        
        # 模拟市场薪酬数据（实际应用中应该使用真实的市场调研数据）
        # 假设市场薪酬比公司内部薪酬高10-20%
        market_multiplier = {
            'Sales': 1.15,
            'Research & Development': 1.20,
            'Human Resources': 1.10,
            'Marketing': 1.18,
            'Finance': 1.22
        }
        
        print(f"📊 各部门市场竞争力分析:")
        print(f"注: 基于模拟市场数据，实际应用需要真实市场调研数据")
        
        dept_competitiveness = []
        
        for dept in self.df['Department'].unique():
            dept_data = self.df[self.df['Department'] == dept]
            internal_avg = dept_data['MonthlyIncome'].mean()
            
            # 使用预设的市场倍数
            multiplier = market_multiplier.get(dept, 1.15)
            market_avg = internal_avg * multiplier
            
            competitiveness_gap = (internal_avg - market_avg) / market_avg * 100
            
            dept_competitiveness.append({
                'Department': dept,
                'Internal_Avg': internal_avg,
                'Market_Avg': market_avg,
                'Gap_Percentage': competitiveness_gap,
                'Competitiveness': '竞争力强' if competitiveness_gap > -5 else 
                                 '一般' if competitiveness_gap > -15 else '竞争力弱'
            })
            
            print(f"   {dept}:")
            print(f"     内部平均: ${internal_avg:,.0f}")
            print(f"     市场平均: ${market_avg:,.0f}")
            print(f"     竞争力差距: {competitiveness_gap:+.1f}%")
        
        competitiveness_df = pd.DataFrame(dept_competitiveness)
        
        # 岗位级别竞争力分析
        if 'JobLevel' in self.df.columns:
            print(f"\n📈 不同岗位级别竞争力分析:")
            level_competitiveness = []
            
            for level in sorted(self.df['JobLevel'].unique()):
                level_data = self.df[self.df['JobLevel'] == level]
                internal_avg = level_data['MonthlyIncome'].mean()
                
                # 高级别岗位市场溢价更高
                market_multiplier_level = 1.1 + (level - 1) * 0.05
                market_avg = internal_avg * market_multiplier_level
                
                gap = (internal_avg - market_avg) / market_avg * 100
                
                level_competitiveness.append({
                    'Level': f'Level {level}',
                    'Internal_Avg': internal_avg,
                    'Market_Avg': market_avg,
                    'Gap': gap
                })
                
                print(f"   Level {level}: 内部${internal_avg:,.0f} vs 市场${market_avg:,.0f} ({gap:+.1f}%)")
        
        # 高风险流失的薪酬竞争力
        if 'AttritionRiskScore' in self.df.columns:
            high_risk_employees = self.df[self.df['AttritionRiskScore'] > 0.7]
            
            if len(high_risk_employees) > 0:
                print(f"\n⚠️ 高流失风险员工薪酬竞争力:")
                
                for dept in high_risk_employees['Department'].unique():
                    dept_high_risk = high_risk_employees[high_risk_employees['Department'] == dept]
                    if len(dept_high_risk) > 0:
                        avg_salary = dept_high_risk['MonthlyIncome'].mean()
                        dept_market_avg = competitiveness_df[
                            competitiveness_df['Department'] == dept
                        ]['Market_Avg'].iloc[0]
                        
                        gap = (avg_salary - dept_market_avg) / dept_market_avg * 100
                        print(f"   {dept}: {len(dept_high_risk)}人, 平均薪酬${avg_salary:,.0f} ({gap:+.1f}%)")
        
        # 可视化
        plt.figure(figsize=(15, 10))
        
        # 部门竞争力对比
        plt.subplot(2, 3, 1)
        x_pos = range(len(competitiveness_df))
        plt.bar(x_pos, competitiveness_df['Internal_Avg'], alpha=0.7, label='内部平均', color='blue')
        plt.bar(x_pos, competitiveness_df['Market_Avg'], alpha=0.7, label='市场平均', color='red')
        plt.xlabel('部门')
        plt.ylabel('平均月薪 ($)')
        plt.title('内部 vs 市场薪酬对比')
        plt.xticks(x_pos, competitiveness_df['Department'], rotation=45)
        plt.legend()
        
        # 竞争力差距
        plt.subplot(2, 3, 2)
        colors = ['green' if x > -5 else 'orange' if x > -15 else 'red' 
                 for x in competitiveness_df['Gap_Percentage']]
        plt.bar(x_pos, competitiveness_df['Gap_Percentage'], color=colors, alpha=0.8)
        plt.xlabel('部门')
        plt.ylabel('竞争力差距 (%)')
        plt.title('各部门薪酬竞争力差距')
        plt.xticks(x_pos, competitiveness_df['Department'], rotation=45)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 岗位级别竞争力（如果有数据）
        if 'JobLevel' in self.df.columns:
            plt.subplot(2, 3, 3)
            level_df = pd.DataFrame(level_competitiveness)
            plt.plot(level_df['Level'], level_df['Internal_Avg'], 'o-', label='内部平均', linewidth=2)
            plt.plot(level_df['Level'], level_df['Market_Avg'], 's-', label='市场平均', linewidth=2)
            plt.xlabel('岗位级别')
            plt.ylabel('平均月薪 ($)')
            plt.title('不同级别薪酬竞争力')
            plt.legend()
            plt.xticks(rotation=45)
        
        # 竞争力分布饼图
        plt.subplot(2, 3, 4)
        competitiveness_counts = competitiveness_df['Competitiveness'].value_counts()
        plt.pie(competitiveness_counts.values, labels=competitiveness_counts.index, autopct='%1.1f%%')
        plt.title('部门竞争力分布')
        
        # 薪酬分布与市场基准线
        plt.subplot(2, 3, 5)
        plt.hist(self.df['MonthlyIncome'], bins=30, alpha=0.7, color='skyblue', label='内部薪酬分布')
        
        # 添加市场基准线
        overall_market_avg = competitiveness_df['Market_Avg'].mean()
        plt.axvline(x=overall_market_avg, color='red', linestyle='--', linewidth=2, label=f'市场平均线')
        plt.xlabel('月薪 ($)')
        plt.ylabel('员工数量')
        plt.title('薪酬分布 vs 市场基准')
        plt.legend()
        
        # 高风险员工薪酬分布
        plt.subplot(2, 3, 6)
        if 'AttritionRiskScore' in self.df.columns:
            high_risk = self.df[self.df['AttritionRiskScore'] > 0.7]
            low_risk = self.df[self.df['AttritionRiskScore'] <= 0.3]
            
            plt.hist(low_risk['MonthlyIncome'], alpha=0.7, label='低风险员工', bins=20, color='green')
            plt.hist(high_risk['MonthlyIncome'], alpha=0.7, label='高风险员工', bins=20, color='red')
            plt.xlabel('月薪 ($)')
            plt.ylabel('员工数量')
            plt.title('不同风险员工薪酬分布')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 竞争力提升建议
        print(f"\n💡 薪酬竞争力提升建议:")
        
        weak_depts = competitiveness_df[competitiveness_df['Gap_Percentage'] < -10]
        if len(weak_depts) > 0:
            print(f"   优先调整部门: {', '.join(weak_depts['Department'].tolist())}")
            
        total_adjustment_cost = 0
        for _, dept in weak_depts.iterrows():
            dept_employees = len(self.df[self.df['Department'] == dept['Department']])
            monthly_adjustment = abs(dept['Internal_Avg'] - dept['Market_Avg'])
            annual_cost = monthly_adjustment * dept_employees * 12
            total_adjustment_cost += annual_cost
            
            print(f"   {dept['Department']}: 需调整${monthly_adjustment:,.0f}/月/人, 年成本${annual_cost:,.0f}")
        
        if total_adjustment_cost > 0:
            print(f"   总调整成本: ${total_adjustment_cost:,.0f}/年")
            
            # 计算ROI
            if 'AttritionRiskScore' in self.df.columns:
                current_attrition_cost = self.results.get('replacement_costs', {}).get('年度流失成本', pd.Series()).sum()
                if current_attrition_cost > 0:
                    roi = (current_attrition_cost * 0.3 - total_adjustment_cost) / total_adjustment_cost * 100
                    print(f"   预期ROI: {roi:+.1f}% (假设薪酬调整可降低30%流失成本)")
        
        self.results['market_competitiveness'] = competitiveness_df
        
        return competitiveness_df
    
    # =================== C. 组织效能提升 ===================
    
    def identify_high_performance_team_characteristics(self):
        """高绩效团队特征识别"""
        print("\n" + "="*60)
        print("🏆 C1. 高绩效团队特征识别")
        print("="*60)
        
        # 定义高绩效团队
        performance_cols = ['PerformanceRating', 'PerformanceIndex', 'MonthlyAchievement']
        available_perf = [col for col in performance_cols if col in self.df.columns]
        
        if len(available_perf) == 0:
            print("❌ 缺少绩效数据")
            return None
        
        # 计算综合绩效得分
        perf_data = self.df[available_perf].copy()
        # 标准化绩效指标
        for col in available_perf:
            perf_data[col] = (perf_data[col] - perf_data[col].mean()) / perf_data[col].std()
        
        self.df['OverallPerformance'] = perf_data.mean(axis=1)
        
        # 按部门计算平均绩效
        dept_performance = self.df.groupby('Department').agg({
            'OverallPerformance': 'mean',
            'MonthlyIncome': 'mean',
            'JobSatisfaction': 'mean',
            'Attrition': lambda x: (x == 'Yes').mean(),
            'OverTime': lambda x: (x == 1).mean() if self.df['OverTime'].dtype in [int, float] else (x == 'Yes').mean()
        }).round(3)
        
        dept_performance.columns = ['平均绩效', '平均薪酬', '平均满意度', '流失率', '加班比例']
        dept_performance = dept_performance.sort_values('平均绩效', ascending=False)
        
        print(f"📊 各部门绩效表现:")
        print(dept_performance.to_string())
        
        # 识别高绩效部门
        high_perf_threshold = dept_performance['平均绩效'].quantile(0.7)
        high_perf_depts = dept_performance[dept_performance['平均绩效'] >= high_perf_threshold]
        
        print(f"\n🏆 高绩效部门: {', '.join(high_perf_depts.index.tolist())}")
        
        # 分析高绩效团队特征
        high_perf_employees = self.df[self.df['Department'].isin(high_perf_depts.index)]
        normal_perf_employees = self.df[~self.df['Department'].isin(high_perf_depts.index)]
        
        print(f"\n🔍 高绩效团队特征分析:")
        
        # 工作模式特征
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
            print(f"   {feature}: 高绩效{high_perf_avg:.2%} vs 一般{normal_perf_avg:.2%} (差异: {diff:+.1%})")
        
        # 员工发展特征
        development_features = ['TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion']
        available_dev = [col for col in development_features if col in self.df.columns]
        
        if available_dev:
            print(f"\n📈 员工发展特征:")
            for feature in available_dev:
                high_perf_avg = high_perf_employees[feature].mean()
                normal_perf_avg = normal_perf_employees[feature].mean()
                diff = high_perf_avg - normal_perf_avg
                
                print(f"   {feature}: 高绩效{high_perf_avg:.1f} vs 一般{normal_perf_avg:.1f} (差异: {diff:+.1f})")
        
        # 员工满意度特征
        satisfaction_features = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
        available_satisfaction = [col for col in satisfaction_features if col in self.df.columns]
        
        if available_satisfaction:
            print(f"\n😊 员工满意度特征:")
            for feature in available_satisfaction:
                high_perf_avg = high_perf_employees[feature].mean()
                normal_perf_avg = normal_perf_employees[feature].mean()
                diff = high_perf_avg - normal_perf_avg
                
                print(f"   {feature}: 高绩效{high_perf_avg:.2f} vs 一般{normal_perf_avg:.2f} (差异: {diff:+.2f})")
        
        # 可视化
        plt.figure(figsize=(18, 12))
        
        # 部门绩效雷达图
        plt.subplot(2, 3, 1)
        dept_performance_top5 = dept_performance.head(5)
        categories = ['平均绩效', '平均满意度', '平均薪酬标准化']
        
        # 标准化薪酬数据用于雷达图
        dept_performance_top5['薪酬标准化'] = (dept_performance_top5['平均薪酬'] - dept_performance_top5['平均薪酬'].min()) / (dept_performance_top5['平均薪酬'].max() - dept_performance_top5['平均薪酬'].min())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, dept in enumerate(dept_performance_top5.index[:3]):  # 只显示前3个部门
            values = [
                dept_performance_top5.loc[dept, '平均绩效'],
                dept_performance_top5.loc[dept, '平均满意度'] / 4,  # 标准化到0-1
                dept_performance_top5.loc[dept, '薪酬标准化']
            ]
            values += values[:1]
            
            plt.subplot(2, 3, 1, projection='polar')
            plt.plot(angles, values, 'o-', linewidth=2, label=dept)
            plt.fill(angles, values, alpha=0.25)
        
        plt.xticks(angles[:-1], categories)
        plt.title('高绩效部门特征雷达图')
        plt.legend()
        
        # 部门绩效排名
        plt.subplot(2, 3, 2)
        dept_performance['平均绩效'].plot(kind='bar', color='gold', alpha=0.8)
        plt.title('各部门绩效排名')
        plt.ylabel('平均绩效得分')
        plt.xticks(rotation=45)
        
        # 绩效与流失率关系
        plt.subplot(2, 3, 3)
        plt.scatter(dept_performance['平均绩效'], dept_performance['流失率'], 
                   s=100, alpha=0.7, color='red')
        plt.xlabel('平均绩效')
        plt.ylabel('流失率')
        plt.title('部门绩效 vs 流失率')
        
        # 绩效与满意度关系
        plt.subplot(2, 3, 4)
        plt.scatter(dept_performance['平均绩效'], dept_performance['平均满意度'], 
                   s=100, alpha=0.7, color='blue')
        plt.xlabel('平均绩效')
        plt.ylabel('平均满意度')
        plt.title('部门绩效 vs 满意度')
        
        # 高绩效团队工作模式对比
        plt.subplot(2, 3, 5)
        if available_work_modes:
            work_mode_comparison = []
            labels = []
            
            for feature in available_work_modes[:3]:  # 只显示前3个
                if self.df[feature].dtype in [int, float]:
                    high_perf_avg = high_perf_employees[feature].mean()
                    normal_perf_avg = normal_perf_employees[feature].mean()
                else:
                    high_perf_avg = (high_perf_employees[feature] == 'Yes').mean()
                    normal_perf_avg = (normal_perf_employees[feature] == 'Yes').mean()
                
                work_mode_comparison.extend([high_perf_avg, normal_perf_avg])
                labels.extend([f'{feature}\n(高绩效)', f'{feature}\n(一般)'])
            
            colors = ['gold', 'lightblue'] * len(available_work_modes)
            plt.bar(range(len(work_mode_comparison)), work_mode_comparison, color=colors[:len(work_mode_comparison)])
            plt.title('工作模式对比')
            plt.ylabel('比例')
            plt.xticks(range(len(work_mode_comparison)), labels, rotation=45)
        
        # 绩效分布
        plt.subplot(2, 3, 6)
        plt.hist(high_perf_employees['OverallPerformance'], alpha=0.7, label='高绩效部门', bins=20, color='gold')
        plt.hist(normal_perf_employees['OverallPerformance'], alpha=0.7, label='一般部门', bins=20, color='lightblue')
        plt.xlabel('综合绩效得分')
        plt.ylabel('员工数量')
        plt.title('绩效分布对比')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 高绩效团队成功要素总结
        print(f"\n💡 高绩效团队成功要素:")
        
        success_factors = []
        
        # 工作模式要素
        for feature in available_work_modes:
            if self.df[feature].dtype in [int, float]:
                high_perf_rate = high_perf_employees[feature].mean()
                normal_perf_rate = normal_perf_employees[feature].mean()
            else:
                high_perf_rate = (high_perf_employees[feature] == 'Yes').mean()
                normal_perf_rate = (normal_perf_employees[feature] == 'Yes').mean()
            
            if high_perf_rate > normal_perf_rate * 1.2:
                success_factors.append(f"更多采用{feature}")
            elif high_perf_rate < normal_perf_rate * 0.8:
                success_factors.append(f"较少使用{feature}")
        
        # 满意度要素
        for feature in available_satisfaction:
            high_perf_avg = high_perf_employees[feature].mean()
            normal_perf_avg = normal_perf_employees[feature].mean()
            
            if high_perf_avg > normal_perf_avg + 0.3:
                success_factors.append(f"更高的{feature}")
        
        # 发展要素
        if 'TrainingTimesLastYear' in available_dev:
            high_perf_training = high_perf_employees['TrainingTimesLastYear'].mean()
            normal_perf_training = normal_perf_employees['TrainingTimesLastYear'].mean()
            
            if high_perf_training > normal_perf_training * 1.2:
                success_factors.append("更多的培训投入")
        
        for i, factor in enumerate(success_factors, 1):
            print(f"   {i}. {factor}")
        
        if not success_factors:
            print("   基于当前数据未发现显著差异要素")
        
        self.results['high_performance_teams'] = {
            'dept_performance': dept_performance,
            'high_perf_depts': high_perf_depts.index.tolist(),
            'success_factors': success_factors
        }
        
        return dept_performance, high_perf_depts
    
    def evaluate_work_mode_effectiveness(self):
        """工作模式效果评估"""
        print("\n" + "="*60)
        print("🏠 C2. 工作模式(远程/弹性)效果评估")
        print("="*60)
        
        work_modes = ['RemoteWork', 'FlexibleWork', 'OverTime']
        available_modes = [col for col in work_modes if col in self.df.columns]
        
        if len(available_modes) == 0:
            print("❌ 缺少工作模式数据")
            return None
        
        mode_effectiveness = {}
        
        for mode in available_modes:
            print(f"\n📊 {mode} 效果分析:")
            
            # 处理不同数据类型
            if self.df[mode].dtype in [int, float]:
                mode_yes = self.df[self.df[mode] == 1]
                mode_no = self.df[self.df[mode] == 0]
                yes_label, no_label = "是", "否"
            else:
                mode_yes = self.df[self.df[mode] == 'Yes']
                mode_no = self.df[self.df[mode] == 'No']
                yes_label, no_label = "Yes", "No"
            
            if len(mode_yes) == 0 or len(mode_no) == 0:
                print(f"   数据不足，跳过{mode}分析")
                continue
            
            # 效果指标对比
            metrics = {
                '员工数量': [len(mode_yes), len(mode_no)],
                '流失率': [
                    (mode_yes['Attrition'] == 'Yes').mean(),
                    (mode_no['Attrition'] == 'Yes').mean()
                ],
                '平均绩效': [
                    mode_yes['OverallPerformance'].mean() if 'OverallPerformance' in self.df.columns else 0,
                    mode_no['OverallPerformance'].mean() if 'OverallPerformance' in self.df.columns else 0
                ],
                '工作满意度': [
                    mode_yes['JobSatisfaction'].mean() if 'JobSatisfaction' in self.df.columns else 0,
                    mode_no['JobSatisfaction'].mean() if 'JobSatisfaction' in self.df.columns else 0
                ],
                '压力水平': [
                    mode_yes['StressRating'].mean() if 'StressRating' in self.df.columns else 0,
                    mode_no['StressRating'].mean() if 'StressRating' in self.df.columns else 0
                ]
            }
            
            mode_analysis = {}
            
            for metric, (yes_val, no_val) in metrics.items():
                if yes_val != 0 or no_val != 0:  # 确保有有效数据
                    diff = yes_val - no_val
                    if metric == '流失率' or metric == '压力水平':
                        improvement = "改善" if diff < 0 else "恶化"
                    else:
                        improvement = "改善" if diff > 0 else "恶化"
                    
                    mode_analysis[metric] = {
                        'yes': yes_val,
                        'no': no_val,
                        'diff': diff,
                        'improvement': improvement
                    }
                    
                    if metric == '流失率':
                        print(f"   流失率: {yes_label} {yes_val:.1%} vs {no_label} {no_val:.1%} ({improvement})")
                    elif metric == '员工数量':
                        print(f"   采用比例: {yes_val}/{yes_val+no_val} ({yes_val/(yes_val+no_val):.1%})")
                    else:
                        print(f"   {metric}: {yes_label} {yes_val:.2f} vs {no_label} {no_val:.2f} ({improvement})")
            
            mode_effectiveness[mode] = mode_analysis
        
        # 工作模式组合效果分析
        print(f"\n🔄 工作模式组合效果分析:")
        
        # 创建工作模式组合
        if len(available_modes) >= 2:
            mode1, mode2 = available_modes[0], available_modes[1]
            
            # 处理数据类型
            if self.df[mode1].dtype in [int, float]:
                mode1_condition = self.df[mode1] == 1
            else:
                mode1_condition = self.df[mode1] == 'Yes'
                
            if self.df[mode2].dtype in [int, float]:
                mode2_condition = self.df[mode2] == 1
            else:
                mode2_condition = self.df[mode2] == 'Yes'
            
            # 四种组合
            combinations = {
                f'都采用': mode1_condition & mode2_condition,
                f'仅{mode1}': mode1_condition & ~mode2_condition,
                f'仅{mode2}': ~mode1_condition & mode2_condition,
                f'都不采用': ~mode1_condition & ~mode2_condition
            }
            
            combo_results = {}
            
            for combo_name, combo_mask in combinations.items():
                combo_data = self.df[combo_mask]
                
                if len(combo_data) > 10:  # 样本量足够
                    combo_results[combo_name] = {
                        'count': len(combo_data),
                        'attrition_rate': (combo_data['Attrition'] == 'Yes').mean(),
                        'satisfaction': combo_data['JobSatisfaction'].mean() if 'JobSatisfaction' in self.df.columns else 0,
                        'performance': combo_data['OverallPerformance'].mean() if 'OverallPerformance' in self.df.columns else 0
                    }
                    
                    print(f"   {combo_name}: {len(combo_data)}人, 流失率{combo_results[combo_name]['attrition_rate']:.1%}")
        
        # 可视化
        plt.figure(figsize=(18, 12))
        
        plot_idx = 1
        
        for mode in available_modes:
            if mode in mode_effectiveness:
                # 流失率对比
                plt.subplot(3, len(available_modes), plot_idx)
                
                attrition_data = mode_effectiveness[mode].get('流失率', {})
                if attrition_data:
                    values = [attrition_data['yes'], attrition_data['no']]
                    labels = ['采用', '不采用']
                    colors = ['green' if attrition_data['improvement'] == '改善' else 'red', 'lightblue']
                    
                    plt.bar(labels, values, color=colors, alpha=0.8)
                    plt.title(f'{mode}\n流失率对比')
                    plt.ylabel('流失率')
                
                # 满意度对比
                plt.subplot(3, len(available_modes), plot_idx + len(available_modes))
                
                satisfaction_data = mode_effectiveness[mode].get('工作满意度', {})
                if satisfaction_data:
                    values = [satisfaction_data['yes'], satisfaction_data['no']]
                    labels = ['采用', '不采用']
                    colors = ['green' if satisfaction_data['improvement'] == '改善' else 'red', 'lightblue']
                    
                    plt.bar(labels, values, color=colors, alpha=0.8)
                    plt.title(f'{mode}\n满意度对比')
                    plt.ylabel('满意度')
                
                # 绩效对比
                plt.subplot(3, len(available_modes), plot_idx + 2*len(available_modes))
                
                performance_data = mode_effectiveness[mode].get('平均绩效', {})
                if performance_data:
                    values = [performance_data['yes'], performance_data['no']]
                    labels = ['采用', '不采用']
                    colors = ['green' if performance_data['improvement'] == '改善' else 'red', 'lightblue']
                    
                    plt.bar(labels, values, color=colors, alpha=0.8)
                    plt.title(f'{mode}\n绩效对比')
                    plt.ylabel('绩效得分')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        # 工作模式推荐
        print(f"\n💡 工作模式优化建议:")
        
        recommendations = []
        
        for mode, analysis in mode_effectiveness.items():
            attrition_improvement = analysis.get('流失率', {}).get('improvement')
            satisfaction_improvement = analysis.get('工作满意度', {}).get('improvement')
            performance_improvement = analysis.get('平均绩效', {}).get('improvement')
            
            positive_effects = sum(1 for imp in [attrition_improvement, satisfaction_improvement, performance_improvement] 
                                 if imp == '改善')
            
            if positive_effects >= 2:
                recommendations.append(f"推广{mode}政策，显示积极效果")
            elif positive_effects == 0:
                recommendations.append(f"重新评估{mode}政策，可能需要调整")
            else:
                recommendations.append(f"优化{mode}实施方式，平衡利弊")
        
        # 基于组合效果的建议
        if 'combo_results' in locals() and combo_results:
            best_combo = min(combo_results.items(), key=lambda x: x[1]['attrition_rate'])
            recommendations.append(f"推荐工作模式组合: {best_combo[0]} (流失率最低: {best_combo[1]['attrition_rate']:.1%})")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        self.results['work_mode_effectiveness'] = {
            'mode_analysis': mode_effectiveness,
            'recommendations': recommendations
        }
        
        return mode_effectiveness
    
    def analyze_training_roi(self):
        """培训ROI分析"""
        print("\n" + "="*60)
        print("📚 C3. 培训ROI分析")
        print("="*60)
        
        if 'TrainingTimesLastYear' not in self.df.columns:
            print("❌ 缺少培训数据")
            return None
        
        # 培训投入成本假设
        TRAINING_COST_PER_SESSION = 500  # 每次培训成本$500
        
        # 培训分组
        training_groups = pd.cut(
            self.df['TrainingTimesLastYear'],
            bins=[-1, 0, 2, 4, 20],
            labels=['无培训', '少量培训(1-2次)', '适量培训(3-4次)', '大量培训(5次以上)']
        )
        
        self.df['TrainingGroup'] = training_groups
        
        # 各培训组效果分析
        training_analysis = self.df.groupby('TrainingGroup').agg({
            'TrainingTimesLastYear': 'mean',
            'Attrition': lambda x: (x == 'Yes').mean(),
            'OverallPerformance': 'mean' if 'OverallPerformance' in self.df.columns else lambda x: 0,
            'JobSatisfaction': 'mean' if 'JobSatisfaction' in self.df.columns else lambda x: 0,
            'MonthlyIncome': 'mean',
            'YearsSinceLastPromotion': 'mean' if 'YearsSinceLastPromotion' in self.df.columns else lambda x: 0,
            'MonthlyIncome': ['count', 'mean']
        }).round(3)
        
        # 重新整理列名
        training_stats = self.df.groupby('TrainingGroup').agg({
            'TrainingTimesLastYear': 'mean',
            'Attrition': lambda x: (x == 'Yes').mean(),
            'JobSatisfaction': 'mean',
            'MonthlyIncome': ['count', 'mean']
        }).round(3)
        
        # 平铺多级索引
        training_stats.columns = ['平均培训次数', '流失率', '平均满意度', '员工数量', '平均薪酬']
        
        if 'OverallPerformance' in self.df.columns:
            perf_by_training = self.df.groupby('TrainingGroup')['OverallPerformance'].mean()
            training_stats['平均绩效'] = perf_by_training
        
        print(f"📊 各培训组效果统计:")
        print(training_stats.to_string())
        
        # 计算培训ROI
        print(f"\n💰 培训ROI计算:")
        
        baseline_group = '无培训'
        
        if baseline_group in training_stats.index:
            baseline_attrition = training_stats.loc[baseline_group, '流失率']
            baseline_performance = training_stats.loc[baseline_group, '平均绩效'] if '平均绩效' in training_stats.columns else 0
            baseline_satisfaction = training_stats.loc[baseline_group, '平均满意度']
            
            roi_analysis = {}
            
            for group in training_stats.index:
                if group != baseline_group:
                    group_data = training_stats.loc[group]
                    employees = group_data['员工数量']
                    avg_training = group_data['平均培训次数']
                    
                    # 培训成本
                    training_cost = employees * avg_training * TRAINING_COST_PER_SESSION
                    
                    # 收益计算
                    # 1. 流失率降低带来的节省
                    attrition_reduction = baseline_attrition - group_data['流失率']
                    avg_salary = group_data['平均薪酬'] * 12  # 年薪
                    replacement_cost_saving = attrition_reduction * employees * avg_salary * 0.5  # 替换成本为年薪50%
                    
                    # 2. 绩效提升带来的价值（假设绩效提升1个标准差价值年薪10%）
                    if '平均绩效' in training_stats.columns:
                        performance_improvement = group_data['平均绩效'] - baseline_performance
                        performance_value = performance_improvement * employees * avg_salary * 0.1
                    else:
                        performance_value = 0
                    
                    # 3. 满意度提升的间接价值（降低隐性成本）
                    satisfaction_improvement = group_data['平均满意度'] - baseline_satisfaction
                    satisfaction_value = satisfaction_improvement * employees * 1000  # 每提升1分价值$1000
                    
                    # 总收益和ROI
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
                    print(f"     培训成本: ${training_cost:,.0f}")
                    print(f"     流失节省: ${replacement_cost_saving:,.0f}")
                    print(f"     绩效价值: ${performance_value:,.0f}")
                    print(f"     满意度价值: ${satisfaction_value:,.0f}")
                    print(f"     总收益: ${total_benefit:,.0f}")
                    print(f"     ROI: {roi:+.1f}%")
        
        # 培训效果的统计显著性检验
        print(f"\n🔬 培训效果显著性检验:")
        
        no_training = self.df[self.df['TrainingTimesLastYear'] == 0]
        with_training = self.df[self.df['TrainingTimesLastYear'] > 0]
        
        if len(no_training) > 0 and len(with_training) > 0:
            # 流失率检验
            from scipy.stats import chi2_contingency
            
            contingency_table = pd.crosstab(
                self.df['TrainingTimesLastYear'] > 0,
                self.df['Attrition']
            )
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"   培训对流失率影响: p-value = {p_value:.4f} ({'显著' if p_value < 0.05 else '不显著'})")
            
            # 满意度检验
            if 'JobSatisfaction' in self.df.columns:
                from scipy.stats import ttest_ind
                
                t_stat, p_value_sat = ttest_ind(
                    with_training['JobSatisfaction'],
                    no_training['JobSatisfaction']
                )
                print(f"   培训对满意度影响: p-value = {p_value_sat:.4f} ({'显著' if p_value_sat < 0.05 else '不显著'})")
        
        # 可视化
        plt.figure(figsize=(18, 12))
        
        # 培训次数分布
        plt.subplot(2, 4, 1)
        self.df['TrainingTimesLastYear'].hist(bins=15, alpha=0.7, color='skyblue')
        plt.title('培训次数分布')
        plt.xlabel('年度培训次数')
        plt.ylabel('员工数量')
        
        # 各组流失率
        plt.subplot(2, 4, 2)
        training_stats['流失率'].plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title('各培训组流失率')
        plt.ylabel('流失率')
        plt.xticks(rotation=45)
        
        # 各组满意度
        plt.subplot(2, 4, 3)
        training_stats['平均满意度'].plot(kind='bar', color='lightgreen', alpha=0.8)
        plt.title('各培训组满意度')
        plt.ylabel('平均满意度')
        plt.xticks(rotation=45)
        
        # ROI对比
        plt.subplot(2, 4, 4)
        if 'roi_analysis' in locals():
            roi_values = [analysis['roi'] for analysis in roi_analysis.values()]
            roi_labels = list(roi_analysis.keys())
            colors = ['green' if roi > 0 else 'red' for roi in roi_values]
            
            plt.bar(range(len(roi_values)), roi_values, color=colors, alpha=0.8)
            plt.title('各培训组ROI')
            plt.ylabel('ROI (%)')
            plt.xticks(range(len(roi_labels)), roi_labels, rotation=45)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 培训次数vs流失率散点图
        plt.subplot(2, 4, 5)
        colors = ['red' if x == 'Yes' else 'blue' for x in self.df['Attrition']]
        plt.scatter(self.df['TrainingTimesLastYear'], self.df['JobSatisfaction'], c=colors, alpha=0.6)
        plt.xlabel('年度培训次数')
        plt.ylabel('工作满意度')
        plt.title('培训次数 vs 满意度')
        
        # 培训成本收益分解
        plt.subplot(2, 4, 6)
        if 'roi_analysis' in locals() and roi_analysis:
            best_group = max(roi_analysis.keys(), key=lambda k: roi_analysis[k]['roi'])
            best_analysis = roi_analysis[best_group]
            
            benefit_components = [
                best_analysis['replacement_saving'],
                best_analysis['performance_value'],
                best_analysis['satisfaction_value']
            ]
            component_labels = ['流失节省', '绩效价值', '满意度价值']
            
            plt.pie(benefit_components, labels=component_labels, autopct='%1.1f%%')
            plt.title(f'{best_group}\n收益构成')
        
        # 培训投入与产出关系
        plt.subplot(2, 4, 7)
        if 'roi_analysis' in locals():
            costs = [analysis['training_cost'] for analysis in roi_analysis.values()]
            benefits = [analysis['total_benefit'] for analysis in roi_analysis.values()]
            labels = list(roi_analysis.keys())
            
            plt.scatter(costs, benefits, s=100, alpha=0.7)
            
            for i, label in enumerate(labels):
                plt.annotate(label, (costs[i], benefits[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            # 添加盈亏平衡线
            max_cost = max(costs) if costs else 1
            plt.plot([0, max_cost], [0, max_cost], 'r--', alpha=0.5, label='盈亏平衡线')
            
            plt.xlabel('培训成本 ($)')
            plt.ylabel('总收益 ($)')
            plt.title('培训投入产出关系')
            plt.legend()
        
        # 培训频率vs绩效
        plt.subplot(2, 4, 8)
        if '平均绩效' in training_stats.columns:
            training_stats['平均绩效'].plot(kind='bar', color='gold', alpha=0.8)
            plt.title('各培训组绩效')
            plt.ylabel('平均绩效得分')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 培训策略建议
        print(f"\n💡 培训策略优化建议:")
        
        recommendations = []
        
        if 'roi_analysis' in locals():
            # 找出ROI最高的培训组
            best_roi_group = max(roi_analysis.keys(), key=lambda k: roi_analysis[k]['roi'])
            best_roi = roi_analysis[best_roi_group]['roi']
            
            if best_roi > 50:
                recommendations.append(f"重点推广{best_roi_group}模式，ROI达{best_roi:.1f}%")
            
            # 识别ROI为负的组
            negative_roi_groups = [group for group, analysis in roi_analysis.items() if analysis['roi'] < 0]
            if negative_roi_groups:
                recommendations.append(f"重新评估{', '.join(negative_roi_groups)}的培训效果")
        
        # 基于最优培训次数的建议
        optimal_training = training_stats.loc[training_stats['流失率'].idxmin(), '平均培训次数']
        recommendations.append(f"建议年度培训次数: {optimal_training:.0f}次左右")
        
        # 针对不同群体的培训建议
        if len(no_training) > 0:
            no_training_attrition = (no_training['Attrition'] == 'Yes').mean()
            if no_training_attrition > 0.2:
                recommendations.append(f"优先为无培训员工安排培训，当前流失率{no_training_attrition:.1%}")
        
        recommendations.append("建立培训效果跟踪机制，定期评估ROI")
        recommendations.append("根据岗位特点定制化培训内容")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        self.results['training_roi'] = {
            'training_stats': training_stats,
            'roi_analysis': roi_analysis if 'roi_analysis' in locals() else {},
            'recommendations': recommendations
        }
        
        return training_stats
    
    # =================== 综合报告生成 ===================
    
    def generate_comprehensive_report(self):
        """生成综合价值挖掘报告"""
        print("\n" + "="*80)
        print("📋 HR数据深度价值挖掘综合报告")
        print("="*80)
        
        # 执行摘要
        print(f"\n🎯 执行摘要:")
        
        current_attrition = (self.df['Attrition'] == 'Yes').mean()
        total_employees = len(self.df)
        
        print(f"   数据集规模: {total_employees}名员工")
        print(f"   当前流失率: {current_attrition:.1%}")
        
        # A部分总结
        if 'attrition_model' in self.results:
            high_risk_count = len(self.df[self.df['AttritionRiskScore'] > 0.7]) if 'AttritionRiskScore' in self.df.columns else 0
            print(f"   高风险员工: {high_risk_count}人 ({high_risk_count/total_employees:.1%})")
        
        if 'replacement_costs' in self.results:
            total_cost = self.results['replacement_costs']['年度流失成本'].sum()
            print(f"   年度流失成本: ${total_cost:,.0f}")
        
        if 'hidden_flight_risk' in self.results:
            hidden_count = self.results['hidden_flight_risk']['basic_count']
            print(f"   隐形离职风险: {hidden_count}人 ({hidden_count/total_employees:.1%})")
        
        # B部分总结
        if 'compensation_equity' in self.results:
            high_variance_roles = len(self.results['compensation_equity']['high_variance_roles'])
            print(f"   薪酬差异较大岗位: {high_variance_roles}个")
        
        if 'performance_compensation' in self.results:
            fairness_index = self.results['performance_compensation']['fairness_index']
            print(f"   薪酬公平性指数: {fairness_index:.3f}")
        
        # C部分总结
        if 'high_performance_teams' in self.results:
            high_perf_depts = len(self.results['high_performance_teams']['high_perf_depts'])
            print(f"   高绩效部门数: {high_perf_depts}个")
        
        if 'training_roi' in self.results and self.results['training_roi']['roi_analysis']:
            best_roi = max(self.results['training_roi']['roi_analysis'].values(), key=lambda x: x['roi'])['roi']
            print(f"   最佳培训ROI: {best_roi:.1f}%")
        
        # 关键发现
        print(f"\n🔍 关键发现:")
        
        key_findings = []
        
        # 流失预测发现
        if 'attrition_model' in self.results and self.results['attrition_model']['feature_importance'] is not None:
            top_factor = self.results['attrition_model']['feature_importance'].iloc[0]['feature']
            key_findings.append(f"流失的最大影响因素是{top_factor}")
        
        # 成本发现
        if 'replacement_costs' in self.results:
            highest_cost_dept = self.results['replacement_costs']['年度流失成本'].idxmax()
            highest_cost = self.results['replacement_costs'].loc[highest_cost_dept, '年度流失成本']
            key_findings.append(f"{highest_cost_dept}部门流失成本最高(${highest_cost:,.0f})")
        
        # 薪酬公平性发现
        if 'performance_compensation' in self.results:
            mismatch_high = len(self.results['performance_compensation']['high_perf_low_pay'])
            mismatch_low = len(self.results['performance_compensation']['low_perf_high_pay'])
            if mismatch_high > 0 or mismatch_low > 0:
                key_findings.append(f"发现{mismatch_high + mismatch_low}名员工薪酬绩效不匹配")
        
        # 工作模式发现
        if 'work_mode_effectiveness' in self.results:
            effective_modes = []
            for mode, analysis in self.results['work_mode_effectiveness']['mode_analysis'].items():
                if analysis.get('流失率', {}).get('improvement') == '改善':
                    effective_modes.append(mode)
            if effective_modes:
                key_findings.append(f"{', '.join(effective_modes)}有助于降低流失率")
        
        # 培训效果发现
        if 'training_roi' in self.results:
            positive_roi_groups = [group for group, analysis in self.results['training_roi'].get('roi_analysis', {}).items() 
                                 if analysis['roi'] > 0]
            if positive_roi_groups:
                key_findings.append(f"{len(positive_roi_groups)}个培训组显示正ROI")
        
        for i, finding in enumerate(key_findings, 1):
            print(f"   {i}. {finding}")
        
        # 行动建议优先级
        print(f"\n🎯 行动建议 (按优先级排序):")
        
        high_priority = []
        medium_priority = []
        low_priority = []
        
        # 高优先级：直接影响成本的措施
        if 'hidden_flight_risk' in self.results:
            high_risk_count = self.results['hidden_flight_risk']['high_risk_count']
            if high_risk_count > 0:
                high_priority.append(f"立即干预{high_risk_count}名高风险隐形离职员工")
        
        if 'performance_compensation' in self.results:
            high_perf_low_pay = len(self.results['performance_compensation']['high_perf_low_pay'])
            if high_perf_low_pay > 0:
                high_priority.append(f"调整{high_perf_low_pay}名高绩效低薪酬员工薪酬")
        
        # 中优先级：系统性改进措施
        if 'market_competitiveness' in self.results:
            weak_depts = len(self.results['market_competitiveness'][self.results['market_competitiveness']['Gap_Percentage'] < -10])
            if weak_depts > 0:
                medium_priority.append(f"提升{weak_depts}个部门的薪酬竞争力")
        
        if 'work_mode_effectiveness' in self.results:
            recommendations = self.results['work_mode_effectiveness'].get('recommendations', [])
            for rec in recommendations[:2]:  # 只取前2个
                medium_priority.append(rec)
        
        # 低优先级：长期优化措施
        if 'high_performance_teams' in self.results:
            success_factors = self.results['high_performance_teams'].get('success_factors', [])
            for factor in success_factors[:2]:  # 只取前2个
                low_priority.append(f"在全公司推广{factor}")
        
        if 'training_roi' in self.results:
            training_recs = self.results['training_roi'].get('recommendations', [])
            for rec in training_recs[:1]:  # 只取1个
                low_priority.append(rec)
        
        # 输出优先级建议
        print(f"\n   🔴 高优先级 (立即执行):")
        for i, action in enumerate(high_priority, 1):
            print(f"      {i}. {action}")
        
        print(f"\n   🟡 中优先级 (3个月内):")
        for i, action in enumerate(medium_priority, 1):
            print(f"      {i}. {action}")
        
        print(f"\n   🟢 低优先级 (6个月内):")
        for i, action in enumerate(low_priority, 1):
            print(f"      {i}. {action}")
        
        # ROI预估
        print(f"\n💰 投资回报预估:")
        
        # 计算潜在节省
        if 'replacement_costs' in self.results:
            current_total_cost = self.results['replacement_costs']['年度流失成本'].sum()
            
            # 假设措施效果
            risk_reduction = 0.05  # 降低5%流失率
            cost_saving = current_total_cost * risk_reduction
            
            # 投资成本估算
            investment_cost = 0
            
            # 薪酬调整成本
            if 'performance_compensation' in self.results:
                mismatch_employees = len(self.results['performance_compensation']['high_perf_low_pay'])
                avg_adjustment = 500  # 假设每人每月调整$500
                annual_adjustment_cost = mismatch_employees * avg_adjustment * 12
                investment_cost += annual_adjustment_cost
            
            # 培训投资
            if 'training_roi' in self.results:
                untrained_employees = len(self.df[self.df['TrainingTimesLastYear'] == 0])
                training_investment = untrained_employees * 2 * 500  # 每人2次培训，每次$500
                investment_cost += training_investment
            
            # 工作模式改进成本
            flexible_work_cost = total_employees * 100  # 每人$100的灵活工作支持
            investment_cost += flexible_work_cost
            
            # 计算ROI
            net_benefit = cost_saving - investment_cost
            roi_percentage = (net_benefit / investment_cost * 100) if investment_cost > 0 else 0
            
            print(f"   预计节省流失成本: ${cost_saving:,.0f}")
            print(f"   所需投资成本: ${investment_cost:,.0f}")
            print(f"   净收益: ${net_benefit:,.0f}")
            print(f"   预期ROI: {roi_percentage:+.1f}%")
        
        # 实施时间表
        print(f"\n📅 实施时间表:")
        print(f"   第1个月: 高风险员工干预，薪酬公平性调整")
        print(f"   第2-3个月: 工作模式优化，培训计划启动")
        print(f"   第4-6个月: 效果评估，政策调整")
        print(f"   第7-12个月: 持续优化，经验总结")
        
        print(f"\n✅ 综合报告生成完成！")
        print(f"🚀 建议定期(季度)重新评估指标，动态调整策略")
        
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

# 执行完整的价值挖掘分析
print("🚀 开始执行HR数据深度价值挖掘分析...")

# 创建分析器
miner = HRValueMiner(df)

# Phase A: 流失预测与成本优化
print("\n📊 Phase A: 流失预测与成本优化")
attrition_model, risk_scores = miner.build_attrition_risk_model()
replacement_costs = miner.calculate_replacement_costs()
hidden_flight_basic, hidden_flight_high_risk = miner.identify_hidden_flight_risk()

# Phase B: 薪酬优化与公平性分析
print("\n💰 Phase B: 薪酬优化与公平性分析")
job_salary_stats, high_variance_roles = miner.analyze_compensation_equity()
perf_comp_correlations, high_perf_low_pay, low_perf_high_pay = miner.evaluate_performance_compensation_alignment()
market_competitiveness = miner.market_competitiveness_analysis()

# Phase C: 组织效能提升
print("\n🏆 Phase C: 组织效能提升")
dept_performance, high_perf_depts = miner.identify_high_performance_team_characteristics()
work_mode_effectiveness = miner.evaluate_work_mode_effectiveness()
training_stats = miner.analyze_training_roi()

# 生成综合报告
comprehensive_report = miner.generate_comprehensive_report()

print("\n🎉 HR数据深度价值挖掘分析完成！")
print("📋 所有分析结果已保存在miner.results中，可以进一步导出或深入分析。")
        