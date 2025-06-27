import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HypothesisDrivenEDA:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        
    def test_h1_department_job_attrition(self):
        """H1: 某些岗位/部门流失率显著高于其他"""
        print("="*60)
        print("H1 验证: 岗位/部门流失率差异分析")
        print("="*60)
        
        # 1. 部门流失率分析
        dept_attrition = self.df.groupby('Department').agg({
            'Attrition': [
                lambda x: (x == 'Yes').mean(),  # 流失率
                lambda x: (x == 'Yes').sum(),   # 流失人数
                'count'                         # 总人数
            ]
        }).round(4)
        
        dept_attrition.columns = ['AttritionRate', 'AttritionCount', 'TotalCount']
        dept_attrition = dept_attrition.sort_values('AttritionRate', ascending=False)
        
        print("\n📊 各部门流失率:")
        print(dept_attrition)
        
        # 2. 岗位流失率分析
        job_attrition = self.df.groupby('JobRole').agg({
            'Attrition': [
                lambda x: (x == 'Yes').mean(),
                lambda x: (x == 'Yes').sum(),
                'count'
            ]
        }).round(4)
        
        job_attrition.columns = ['AttritionRate', 'AttritionCount', 'TotalCount']
        job_attrition = job_attrition.sort_values('AttritionRate', ascending=False)
        
        print("\n📊 各岗位流失率 (前10):")
        print(job_attrition.head(10))
        
        # 3. 统计显著性检验
        overall_attrition_rate = (self.df['Attrition'] == 'Yes').mean()
        print(f"\n📈 整体流失率: {overall_attrition_rate:.2%}")
        
        # 卡方检验
        from scipy.stats import chi2_contingency
        contingency_dept = pd.crosstab(self.df['Department'], self.df['Attrition'])
        chi2_dept, p_value_dept, dof, expected = chi2_contingency(contingency_dept)
        
        print(f"\n🔬 统计检验结果:")
        print(f"部门差异卡方检验 p-value: {p_value_dept:.4f}")
        
        # 4. 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 部门流失率
        dept_attrition['AttritionRate'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('各部门流失率')
        ax1.set_ylabel('流失率')
        ax1.axhline(y=overall_attrition_rate, color='red', linestyle='--', 
                   label=f'整体平均({overall_attrition_rate:.2%})')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 岗位流失率 (前8个)
        job_attrition.head(8)['AttritionRate'].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('各岗位流失率 (Top 8)')
        ax2.set_ylabel('流失率')
        ax2.axhline(y=overall_attrition_rate, color='red', linestyle='--', 
                   label=f'整体平均({overall_attrition_rate:.2%})')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 5. 结论
        max_dept_attrition = dept_attrition['AttritionRate'].max()
        min_dept_attrition = dept_attrition['AttritionRate'].min()
        
        h1_conclusion = {
            'hypothesis': 'H1: 某些岗位/部门流失率显著高于其他',
            'p_value': p_value_dept,
            'significant': p_value_dept < 0.05,
            'max_dept_rate': max_dept_attrition,
            'min_dept_rate': min_dept_attrition,
            'rate_difference': max_dept_attrition - min_dept_attrition,
            'conclusion': f"部门间流失率差异{'显著' if p_value_dept < 0.05 else '不显著'}"
        }
        
        self.results['H1'] = h1_conclusion
        print(f"\n✅ H1结论: {h1_conclusion['conclusion']}")
        print(f"   最高流失率: {max_dept_attrition:.2%}, 最低流失率: {min_dept_attrition:.2%}")
        
        return h1_conclusion
    
    def test_h2_compensation_performance_mismatch(self):
        """H2: 薪酬与绩效存在不匹配现象"""
        print("\n" + "="*60)
        print("H2 验证: 薪酬与绩效匹配度分析")
        print("="*60)
        
        # 1. 薪酬与绩效相关性
        correlation = self.df['MonthlyIncome'].corr(self.df['PerformanceRating'])
        print(f"\n📊 薪酬与绩效相关系数: {correlation:.4f}")
        
        # 2. 按绩效等级分组的薪酬分析
        perf_salary = self.df.groupby('PerformanceRating')['MonthlyIncome'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(0)
        
        print(f"\n📈 各绩效等级薪酬统计:")
        print(perf_salary)
        
        # 3. 识别薪酬异常情况
        # 高绩效低薪酬 vs 低绩效高薪酬
        high_perf_threshold = self.df['PerformanceRating'].quantile(0.75)
        low_perf_threshold = self.df['PerformanceRating'].quantile(0.25)
        high_salary_threshold = self.df['MonthlyIncome'].quantile(0.75)
        low_salary_threshold = self.df['MonthlyIncome'].quantile(0.25)
        
        # 不匹配情况
        high_perf_low_pay = self.df[
            (self.df['PerformanceRating'] >= high_perf_threshold) & 
            (self.df['MonthlyIncome'] <= low_salary_threshold)
        ]
        
        low_perf_high_pay = self.df[
            (self.df['PerformanceRating'] <= low_perf_threshold) & 
            (self.df['MonthlyIncome'] >= high_salary_threshold)
        ]
        
        print(f"\n🚨 薪酬不匹配情况:")
        print(f"高绩效低薪酬员工: {len(high_perf_low_pay)} 人 ({len(high_perf_low_pay)/len(self.df):.1%})")
        print(f"低绩效高薪酬员工: {len(low_perf_high_pay)} 人 ({len(low_perf_high_pay)/len(self.df):.1%})")
        
        # 4. 不匹配员工的流失率
        if len(high_perf_low_pay) > 0:
            high_perf_low_pay_attrition = (high_perf_low_pay['Attrition'] == 'Yes').mean()
            print(f"高绩效低薪酬员工流失率: {high_perf_low_pay_attrition:.2%}")
        
        if len(low_perf_high_pay) > 0:
            low_perf_high_pay_attrition = (low_perf_high_pay['Attrition'] == 'Yes').mean()
            print(f"低绩效高薪酬员工流失率: {low_perf_high_pay_attrition:.2%}")
        
        # 5. 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 散点图：薪酬 vs 绩效
        colors = ['red' if x == 'Yes' else 'blue' for x in self.df['Attrition']]
        ax1.scatter(self.df['PerformanceRating'], self.df['MonthlyIncome'], 
                   c=colors, alpha=0.6)
        ax1.set_xlabel('绩效评级')
        ax1.set_ylabel('月薪')
        ax1.set_title(f'薪酬 vs 绩效散点图\n(相关系数: {correlation:.3f})')
        
        # 箱线图：不同绩效等级的薪酬分布
        self.df.boxplot(column='MonthlyIncome', by='PerformanceRating', ax=ax2)
        ax2.set_title('各绩效等级薪酬分布')
        ax2.set_xlabel('绩效评级')
        ax2.set_ylabel('月薪')
        
        plt.tight_layout()
        plt.show()
        
        # 6. 结论
        mismatch_rate = (len(high_perf_low_pay) + len(low_perf_high_pay)) / len(self.df)
        
        h2_conclusion = {
            'hypothesis': 'H2: 薪酬与绩效存在不匹配现象',
            'correlation': correlation,
            'mismatch_rate': mismatch_rate,
            'high_perf_low_pay_count': len(high_perf_low_pay),
            'low_perf_high_pay_count': len(low_perf_high_pay),
            'significant_mismatch': mismatch_rate > 0.1,
            'conclusion': f"薪酬绩效不匹配比例: {mismatch_rate:.1%}"
        }
        
        self.results['H2'] = h2_conclusion
        print(f"\n✅ H2结论: {h2_conclusion['conclusion']}")
        
        return h2_conclusion
    
    def test_h3_satisfaction_drives_attrition(self):
        """H3: 工作满意度是流失的主要驱动因素"""
        print("\n" + "="*60)
        print("H3 验证: 满意度与流失关系分析")
        print("="*60)
        
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                           'RelationshipSatisfaction', 'WorkLifeBalance']
        
        # 1. 各满意度指标与流失的关系
        satisfaction_analysis = {}
        
        for col in satisfaction_cols:
            if col in self.df.columns:
                # 留任vs离职员工的满意度对比
                stay_avg = self.df[self.df['Attrition'] == 'No'][col].mean()
                leave_avg = self.df[self.df['Attrition'] == 'Yes'][col].mean()
                difference = stay_avg - leave_avg
                
                # t检验
                stay_scores = self.df[self.df['Attrition'] == 'No'][col].dropna()
                leave_scores = self.df[self.df['Attrition'] == 'Yes'][col].dropna()
                t_stat, p_value = stats.ttest_ind(stay_scores, leave_scores)
                
                satisfaction_analysis[col] = {
                    'stay_avg': stay_avg,
                    'leave_avg': leave_avg,
                    'difference': difference,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                print(f"\n📊 {col}:")
                print(f"   留任员工平均分: {stay_avg:.2f}")
                print(f"   离职员工平均分: {leave_avg:.2f}")
                print(f"   差异: {difference:.2f} (p-value: {p_value:.4f})")
        
        # 2. 满意度等级与流失率交叉分析
        print(f"\n📈 各满意度等级流失率:")
        for col in satisfaction_cols:
            if col in self.df.columns:
                satisfaction_attrition = self.df.groupby(col)['Attrition'].apply(
                    lambda x: (x == 'Yes').mean()
                ).round(4)
                print(f"\n{col}:")
                print(satisfaction_attrition)
        
        # 3. 综合满意度得分
        if all(col in self.df.columns for col in satisfaction_cols):
            self.df['OverallSatisfaction'] = self.df[satisfaction_cols].mean(axis=1)
            
            # 满意度分组
            self.df['SatisfactionLevel'] = pd.cut(
                self.df['OverallSatisfaction'], 
                bins=[0, 2, 3, 4, 5], 
                labels=['低', '中下', '中上', '高']
            )
            
            satisfaction_level_attrition = self.df.groupby('SatisfactionLevel')['Attrition'].apply(
                lambda x: (x == 'Yes').mean()
            ).round(4)
            
            print(f"\n📊 综合满意度等级流失率:")
            print(satisfaction_level_attrition)
        
        # 4. 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(satisfaction_cols):
            if col in self.df.columns and i < 4:
                # 箱线图
                self.df.boxplot(column=col, by='Attrition', ax=axes[i])
                axes[i].set_title(f'{col} vs 流失')
                axes[i].set_xlabel('是否流失')
                axes[i].set_ylabel('满意度评分')
        
        plt.tight_layout()
        plt.show()
        
        # 5. 满意度重要性排序
        importance_ranking = []
        for col, analysis in satisfaction_analysis.items():
            importance_ranking.append({
                'satisfaction_type': col,
                'effect_size': abs(analysis['difference']),
                'p_value': analysis['p_value'],
                'significant': analysis['significant']
            })
        
        importance_df = pd.DataFrame(importance_ranking).sort_values(
            'effect_size', ascending=False
        )
        
        print(f"\n🏆 满意度影响力排序:")
        print(importance_df)
        
        # 6. 结论
        significant_factors = sum(1 for analysis in satisfaction_analysis.values() 
                                if analysis['significant'])
        
        h3_conclusion = {
            'hypothesis': 'H3: 工作满意度是流失的主要驱动因素',
            'significant_factors': significant_factors,
            'top_factor': importance_df.iloc[0]['satisfaction_type'] if len(importance_df) > 0 else None,
            'max_effect_size': importance_df.iloc[0]['effect_size'] if len(importance_df) > 0 else 0,
            'satisfaction_analysis': satisfaction_analysis,
            'strong_evidence': significant_factors >= 3
        }
        
        self.results['H3'] = h3_conclusion
        print(f"\n✅ H3结论: 发现{significant_factors}个显著的满意度影响因素")
        
        return h3_conclusion
    
    def test_h4_flexible_work_improves_retention(self):
        """H4: 远程工作/弹性工作能改善员工保留"""
        print("\n" + "="*60)
        print("H4 验证: 工作模式与员工保留关系")
        print("="*60)
        
        work_flexibility_cols = ['RemoteWork', 'FlexibleWork', 'OverTime']
        
        # 1. 各工作模式的流失率对比
        for col in work_flexibility_cols:
            if col in self.df.columns:
                work_mode_attrition = self.df.groupby(col)['Attrition'].apply(
                    lambda x: (x == 'Yes').mean()
                ).round(4)
                
                print(f"\n📊 {col} 流失率:")
                print(work_mode_attrition)
                
                # 卡方检验
                contingency = pd.crosstab(self.df[col], self.df['Attrition'])
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                print(f"   统计显著性 p-value: {p_value:.4f}")
        
        # 2. 工作满意度与工作模式的关系
        if 'RemoteWork' in self.df.columns and 'JobSatisfaction' in self.df.columns:
            remote_satisfaction = self.df.groupby('RemoteWork')['JobSatisfaction'].mean()
            print(f"\n📈 远程工作与工作满意度:")
            print(remote_satisfaction.round(2))
        
        # 3. 加班与流失关系深度分析
        if 'OverTime' in self.df.columns:
            overtime_analysis = self.df.groupby(['OverTime', 'Attrition']).size().unstack()
            overtime_rates = self.df.groupby('OverTime')['Attrition'].apply(
                lambda x: (x == 'Yes').mean()
            )
            
            print(f"\n🔍 加班情况详细分析:")
            print(f"加班员工流失率: {overtime_rates.get('Yes', 0):.2%}")
            print(f"不加班员工流失率: {overtime_rates.get('No', 0):.2%}")
        
        # 4. 可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, col in enumerate(work_flexibility_cols):
            if col in self.df.columns and i < 3:
                # 流失率条形图
                attrition_by_mode = self.df.groupby(col)['Attrition'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                attrition_by_mode.plot(kind='bar', ax=axes[i], color='lightcoral')
                axes[i].set_title(f'{col} vs 流失率')
                axes[i].set_ylabel('流失率')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 5. 结论
        flexibility_effects = {}
        for col in work_flexibility_cols:
            if col in self.df.columns:
                mode_rates = self.df.groupby(col)['Attrition'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                flexibility_effects[col] = {
                    'rates': mode_rates.to_dict(),
                    'beneficial': any(rate < 0.15 for rate in mode_rates.values())
                }
        
        h4_conclusion = {
            'hypothesis': 'H4: 远程工作/弹性工作能改善员工保留',
            'flexibility_effects': flexibility_effects,
            'supports_hypothesis': any(effect['beneficial'] for effect in flexibility_effects.values())
        }
        
        self.results['H4'] = h4_conclusion
        print(f"\n✅ H4结论: 工作灵活性对员工保留的影响分析完成")
        
        return h4_conclusion
    
    def test_h5_training_development_correlation(self):
        """H5: 培训投入与员工发展正相关"""
        print("\n" + "="*60)
        print("H5 验证: 培训投入与员工发展关系")
        print("="*60)
        
        # 1. 培训次数与相关指标的关系
        if 'TrainingTimesLastYear' in self.df.columns:
            print(f"\n📊 培训次数基本统计:")
            print(self.df['TrainingTimesLastYear'].describe())
            
            # 培训次数与流失关系
            training_attrition = self.df.groupby('TrainingTimesLastYear')['Attrition'].apply(
                lambda x: (x == 'Yes').mean()
            ).round(4)
            
            print(f"\n📈 各培训次数流失率:")
            print(training_attrition)
            
            # 培训次数与绩效关系
            if 'PerformanceRating' in self.df.columns:
                training_performance_corr = self.df['TrainingTimesLastYear'].corr(
                    self.df['PerformanceRating']
                )
                print(f"\n📊 培训次数与绩效相关系数: {training_performance_corr:.4f}")
            
            # 培训次数与满意度关系
            if 'JobSatisfaction' in self.df.columns:
                training_satisfaction_corr = self.df['TrainingTimesLastYear'].corr(
                    self.df['JobSatisfaction']
                )
                print(f"培训次数与工作满意度相关系数: {training_satisfaction_corr:.4f}")
            
            # 培训次数与职业发展关系
            development_cols = ['YearsSinceLastPromotion', 'YearsInCurrentRole']
            
            for col in development_cols:
                if col in self.df.columns:
                    corr = self.df['TrainingTimesLastYear'].corr(self.df[col])
                    print(f"培训次数与{col}相关系数: {corr:.4f}")
        
        # 2. 培训投入分组分析
        if 'TrainingTimesLastYear' in self.df.columns:
            # 创建培训投入等级
            self.df['TrainingLevel'] = pd.cut(
                self.df['TrainingTimesLastYear'],
                bins=[-1, 0, 2, 4, 10],
                labels=['无培训', '少量培训', '适中培训', '大量培训']
            )
            
            training_level_analysis = self.df.groupby('TrainingLevel').agg({
                'Attrition': lambda x: (x == 'Yes').mean(),
                'PerformanceRating': 'mean',
                'JobSatisfaction': 'mean',
                'MonthlyIncome': 'mean'
            }).round(3)
            
            print(f"\n📊 不同培训水平员工表现:")
            print(training_level_analysis)
        
        # 3. 可视化
        if 'TrainingTimesLastYear' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 培训次数分布
            self.df['TrainingTimesLastYear'].hist(bins=20, ax=axes[0,0])
            axes[0,0].set_title('培训次数分布')
            axes[0,0].set_xlabel('培训次数')
            axes[0,0].set_ylabel('员工数量')
            
            # 培训次数vs流失率
            if len(training_attrition) > 1:
                training_attrition.plot(kind='bar', ax=axes[0,1], color='skyblue')
                axes[0,1].set_title('培训次数 vs 流失率')
                axes[0,1].set_ylabel('流失率')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # 培训水平vs绩效
            if 'TrainingLevel' in self.df.columns and 'PerformanceRating' in self.df.columns:
                self.df.boxplot(column='PerformanceRating', by='TrainingLevel', ax=axes[1,0])
                axes[1,0].set_title('培训水平 vs 绩效评级')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # 培训次数vs薪酬散点图
            if 'MonthlyIncome' in self.df.columns:
                axes[1,1].scatter(self.df['TrainingTimesLastYear'], self.df['MonthlyIncome'], alpha=0.6)
                axes[1,1].set_title('培训次数 vs 月薪')
                axes[1,1].set_xlabel('培训次数')
                axes[1,1].set_ylabel('月薪')
            
            plt.tight_layout()
            plt.show()
        
        # 4. 结论
        h5_conclusion = {
            'hypothesis': 'H5: 培训投入与员工发展正相关',
            'training_available': 'TrainingTimesLastYear' in self.df.columns,
            'positive_correlations': [],
            'supports_hypothesis': False
        }
        
        if 'TrainingTimesLastYear' in self.df.columns:
            # 检查正相关关系
            correlations = {}
            for col in ['PerformanceRating', 'JobSatisfaction', 'MonthlyIncome']:
                if col in self.df.columns:
                    corr = self.df['TrainingTimesLastYear'].corr(self.df[col])
                    correlations[col] = corr
                    if corr > 0.1:  # 阈值可调整
                        h5_conclusion['positive_correlations'].append(col)
            
            h5_conclusion['correlations'] = correlations
            h5_conclusion['supports_hypothesis'] = len(h5_conclusion['positive_correlations']) >= 2
        
        self.results['H5'] = h5_conclusion
        print(f"\n✅ H5结论: 培训与发展相关性分析完成")
        
        return h5_conclusion
    
    def generate_hypothesis_summary(self):
        """生成假设验证总结报告"""
        print("\n" + "="*80)
        print("                    假设验证总结报告")
        print("="*80)
        
        for hypothesis, result in self.results.items():
            print(f"\n{result['hypothesis']}")
            print("-" * len(result['hypothesis']))
            
            if hypothesis == 'H1':
                status = "✅ 支持" if result['significant'] else "❌ 不支持"
                print(f"结论: {status}")
                print(f"证据: 部门间流失率差异达{result['rate_difference']:.1%}")
                
            elif hypothesis == 'H2':
                status = "✅ 支持" if result['significant_mismatch'] else "❌ 不支持"
                print(f"结论: {status}")
                print(f"证据: {result['mismatch_rate']:.1%}的员工存在薪酬绩效不匹配")
                
            elif hypothesis == 'H3':
                status = "✅ 强支持" if result['strong_evidence'] else "⚠️ 部分支持"
                print(f"结论: {status}")
                print(f"证据: {result['significant_factors']}个满意度因素显著影响流失")
                
            elif hypothesis == 'H4':
                status = "✅ 支持" if result['supports_hypothesis'] else "❌ 不支持"
                print(f"结论: {status}")
                
            elif hypothesis == 'H5':
                if result['training_available']:
                    status = "✅ 支持" if result['supports_hypothesis'] else "❌ 不支持"
                    print(f"结论: {status}")
                    print(f"证据: {len(result['positive_correlations'])}个指标与培训正相关")
                else:
                    print("结论: ⚠️ 数据不足")
        
        # 整体洞察
        print(f"\n" + "="*50)
        print("🎯 核心洞察与建议")
        print("="*50)
        
        supported_hypotheses = sum(1 for result in self.results.values() 
                                 if result.get('significant', False) or 
                                    result.get('supports_hypothesis', False) or
                                    result.get('strong_evidence', False))
        
        print(f"✅ 验证通过的假设: {supported_hypotheses}/5")
        print(f"\n💡 关键发现:")
        
        # 基于验证结果生成建议
        recommendations = []
        
        if self.results.get('H1', {}).get('significant', False):
            recommendations.append("1. 针对高流失部门制定专项保留策略")
        
        if self.results.get('H2', {}).get('significant_mismatch', False):
            recommendations.append("2. 建立薪酬与绩效联动机制")
        
        if self.results.get('H3', {}).get('strong_evidence', False):
            recommendations.append("3. 实施员工满意度提升计划")
        
        if self.results.get('H4', {}).get('supports_hypothesis', False):
            recommendations.append("4. 推广灵活工作制度")
        
        if self.results.get('H5', {}).get('supports_hypothesis', False):
            recommendations.append("5. 加大员工培训投入")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        return self.results

# 使用示例函数
def run_hypothesis_driven_eda(df):
    """运行完整的假设驱动EDA流程"""
    
    print("🚀 开始假设驱动的EDA分析...")
    print("数据集基本信息:")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   目标变量(Attrition)分布: {df['Attrition'].value_counts().to_dict()}")
    
    # 创建分析器实例
    analyzer = HypothesisDrivenEDA(df)
    
    # 逐一验证假设
    print("\n🔬 开始假设验证...")
    
    try:
        analyzer.test_h1_department_job_attrition()
        analyzer.test_h2_compensation_performance_mismatch()
        analyzer.test_h3_satisfaction_drives_attrition()
        analyzer.test_h4_flexible_work_improves_retention()
        analyzer.test_h5_training_development_correlation()
        
        # 生成总结报告
        results = analyzer.generate_hypothesis_summary()
        
        return results
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return None

# 运行分析的示例代码
# results = run_hypothesis_driven_eda(your_dataframe)