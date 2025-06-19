"""
Titanic生存预测模型 - 优化重构版本
使用随机森林算法预测泰坦尼克号乘客的生存情况
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略警告信息
warnings.filterwarnings('ignore')

class TitanicSurvivalPredictor:
    """泰坦尼克号生存预测器类"""
    
    def __init__(self):
        """初始化预测器"""
        self.pipeline = None
        self.feature_names = None
        self.is_trained = False
    
    def load_data(self, train_path, test_path):
        """
        加载训练和测试数据
        
        Args:
            train_path (str): 训练数据文件路径
            test_path (str): 测试数据文件路径
        
        Returns:
            tuple: (训练数据, 测试数据)
        """
        print("正在加载数据...")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(f"训练数据形状: {train_df.shape}")
            print(f"测试数据形状: {test_df.shape}")
            return train_df, test_df
        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
            return None, None
    
    def data_preprocessing(self, train_df, test_df):
        """
        数据预处理和特征工程
        
        Args:
            train_df (pd.DataFrame): 训练数据
            test_df (pd.DataFrame): 测试数据
        
        Returns:
            tuple: 处理后的数据
        """
        print("开始数据预处理...")
        
        # 1. 处理异常值：将票价为0的值设为NaN，后续用中位数填充
        train_df.loc[train_df['Fare'] == 0, 'Fare'] = np.nan
        test_df.loc[test_df['Fare'] == 0, 'Fare'] = np.nan
        
        # 2. 特征工程：从姓名中提取称谓
        print("提取称谓特征...")
        train_df['Title'] = train_df['Name'].apply(self._extract_title)
        test_df['Title'] = test_df['Name'].apply(self._extract_title)
        
        # 3. 标准化称谓：将少见的称谓归类到常见类别
        train_df['Title'] = train_df['Title'].apply(self._standardize_title)
        test_df['Title'] = test_df['Title'].apply(self._standardize_title)
        
        # 4. 创建家庭规模特征
        print("创建家庭相关特征...")
        train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch'] + 1
        test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch'] + 1
        
        # 5. 基于家庭规模创建类别特征
        train_df['Family_Type'] = train_df['Family_Size'].apply(self._categorize_family_size)
        test_df['Family_Type'] = test_df['Family_Size'].apply(self._categorize_family_size)
        
        # 6. 从船票信息中提取特征
        print("提取船票特征...")
        train_df['Ticket_Length'] = train_df['Ticket'].apply(len)
        test_df['Ticket_Length'] = test_df['Ticket'].apply(len)
        
        # 提取船票前缀（前两个字符）
        train_df['Ticket_Prefix'] = train_df['Ticket'].apply(lambda x: x[:2] if len(x) >= 2 else x)
        test_df['Ticket_Prefix'] = test_df['Ticket'].apply(lambda x: x[:2] if len(x) >= 2 else x)
        
        # 7. 从客舱信息中提取特征（如果存在）
        print("处理客舱信息...")
        train_df['Cabin_Letter'] = train_df['Cabin'].apply(self._extract_cabin_letter)
        test_df['Cabin_Letter'] = test_df['Cabin'].apply(self._extract_cabin_letter)
        
        # 8. 创建是否独自旅行的特征
        train_df['Is_Alone'] = (train_df['Family_Size'] == 1).astype(int)
        test_df['Is_Alone'] = (test_df['Family_Size'] == 1).astype(int)
        
        print("数据预处理完成！")
        return train_df, test_df
    
    def _extract_title(self, name):
        """从姓名中提取称谓"""
        try:
            return name.split(',')[1].split('.')[0].strip()
        except:
            return 'Unknown'
    
    def _standardize_title(self, title):
        """标准化称谓"""
        # 女性称谓统一
        if title in ['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona']:
            return 'Miss'
        # 男性尊称统一
        elif title in ['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer']:
            return 'Mr'
        # 保留常见称谓
        elif title in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev']:
            return title
        else:
            return 'Other'
    
    def _categorize_family_size(self, size):
        """基于家庭规模分类"""
        if size == 1:
            return 'Solo'
        elif size <= 4:
            return 'Small'
        elif size <= 7:
            return 'Medium'
        else:
            return 'Large'
    
    def _extract_cabin_letter(self, cabin):
        """提取客舱首字母"""
        if pd.isna(cabin):
            return 'Unknown'
        else:
            return cabin[0]
    
    def prepare_features(self, train_df, test_df):
        """
        准备用于建模的特征
        
        Args:
            train_df (pd.DataFrame): 训练数据
            test_df (pd.DataFrame): 测试数据
        
        Returns:
            tuple: (X_train, y_train, X_test, test_passenger_ids)
        """
        print("准备建模特征...")
        
        # 选择用于建模的特征
        self.feature_names = [
            'Pclass',           # 票舱等级
            'Sex',              # 性别
            'Age',              # 年龄
            'Fare',             # 票价
            'Embarked',         # 登船港口
            'Title',            # 称谓
            'Family_Type',      # 家庭类型
            'Ticket_Length',    # 船票长度
            'Ticket_Prefix',    # 船票前缀
            'Cabin_Letter',     # 客舱首字母
            'Is_Alone'          # 是否独自旅行
        ]
        
        # 准备训练数据
        X_train = train_df[self.feature_names].copy()
        y_train = train_df['Survived'].copy()  # 修正：使用Survived作为目标变量
        
        # 准备测试数据
        X_test = test_df[self.feature_names].copy()
        test_passenger_ids = test_df['PassengerId'].copy()
        
        print(f"特征数量: {len(self.feature_names)}")
        print("特征列表:", self.feature_names)
        
        return X_train, y_train, X_test, test_passenger_ids
    
    def build_pipeline(self):
        """
        构建机器学习Pipeline
        包含数据预处理和模型训练步骤
        """
        print("构建机器学习Pipeline...")
        
        # 定义数值型特征和类别型特征
        numeric_features = ['Age', 'Fare', 'Ticket_Length', 'Is_Alone']
        categorical_features = [
            'Pclass', 'Sex', 'Embarked', 'Title', 
            'Family_Type', 'Ticket_Prefix', 'Cabin_Letter'
        ]
        
        # 数值型特征预处理：用中位数填充缺失值
        numeric_transformer = SimpleImputer(strategy='median')
        
        # 类别型特征预处理：用最频繁值填充缺失值，然后进行独热编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 组合预处理步骤
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # 创建完整的机器学习Pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                random_state=42,
                n_estimators=100,    # 决策树数量
                max_depth=10,        # 最大深度
                min_samples_split=5, # 分割内部节点所需的最小样本数
                min_samples_leaf=2,  # 叶节点所需的最小样本数
                class_weight='balanced'  # 处理类别不平衡
            ))
        ])
        
        print("Pipeline构建完成！")
    
    def train_model(self, X_train, y_train, use_validation=True):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            use_validation (bool): 是否使用验证集
        """
        print("开始训练模型...")
        
        if use_validation:
            # 分割训练集和验证集
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # 训练模型
            self.pipeline.fit(X_train_split, y_train_split)
            
            # 在验证集上评估
            val_predictions = self.pipeline.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            
            print(f"验证集准确率: {val_accuracy:.4f}")
            print("\n验证集分类报告:")
            print(classification_report(y_val, val_predictions))
            
        else:
            # 使用全部训练数据
            self.pipeline.fit(X_train, y_train)
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print(f"\n5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        print("模型训练完成！")
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        print("开始超参数调优...")
        
        # 定义参数网格
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        # 网格搜索
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
        
        # 使用最佳参数更新Pipeline
        self.pipeline = grid_search.best_estimator_
        self.is_trained = True
        
        print("超参数调优完成！")
    
    def make_predictions(self, X_test):
        """
        进行预测
        
        Args:
            X_test: 测试特征
        
        Returns:
            np.array: 预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练！请先调用train_model()方法。")
        
        print("正在进行预测...")
        predictions = self.pipeline.predict(X_test)
        probabilities = self.pipeline.predict_proba(X_test)[:, 1]  # 生存概率
        
        print(f"预测完成！生存预测数量: {sum(predictions)}")
        return predictions, probabilities
    
    def save_predictions(self, passenger_ids, predictions, filename='titanic_predictions.csv'):
        """
        保存预测结果到CSV文件
        
        Args:
            passenger_ids: 乘客ID
            predictions: 预测结果
            filename (str): 输出文件名
        """
        output_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        
        output_df.to_csv(filename, index=False)
        print(f"预测结果已保存到: {filename}")
    
    def plot_feature_importance(self):
        """绘制特征重要性图"""
        if not self.is_trained:
            print("模型尚未训练，无法显示特征重要性")
            return
        
        # 获取特征重要性
        feature_importance = self.pipeline.named_steps['classifier'].feature_importances_
        
        # 获取特征名称（需要考虑独热编码后的特征名称）
        preprocessor = self.pipeline.named_steps['preprocessor']
        feature_names = []
        
        # 数值特征名称
        numeric_features = ['Age', 'Fare', 'Ticket_Length', 'Is_Alone']
        feature_names.extend(numeric_features)
        
        # 类别特征名称（经过独热编码）
        categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        feature_names.extend(categorical_features)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        print("Top 10 重要特征:")
        print(importance_df.head(10))


def main():
    """主函数：完整的模型训练和预测流程"""
    
    # 创建预测器实例
    predictor = TitanicSurvivalPredictor()
    
    # 1. 加载数据
    train_df, test_df = predictor.load_data('train.csv', 'test.csv')
    
    if train_df is None or test_df is None:
        print("数据加载失败，请检查文件路径")
        return
    
    # 2. 数据预处理
    train_df, test_df = predictor.data_preprocessing(train_df, test_df)
    
    # 3. 准备特征
    X_train, y_train, X_test, test_passenger_ids = predictor.prepare_features(train_df, test_df)
    
    # 4. 构建Pipeline
    predictor.build_pipeline()
    
    # 5. 训练模型（可选择是否进行超参数调优）
    use_tuning = input("是否进行超参数调优？(y/n): ").lower() == 'y'
    
    if use_tuning:
        predictor.hyperparameter_tuning(X_train, y_train)
    else:
        predictor.train_model(X_train, y_train)
    
    # 6. 进行预测
    predictions, probabilities = predictor.make_predictions(X_test)
    
    # 7. 保存结果
    predictor.save_predictions(test_passenger_ids, predictions)
    
    # 8. 显示特征重要性
    show_importance = input("是否显示特征重要性？(y/n): ").lower() == 'y'
    if show_importance:
        predictor.plot_feature_importance()
    
    print("\n预测流程完成！")


if __name__ == "__main__":
    main()
