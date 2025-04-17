# 导入库文件
import numpy as np
import pandas as pd
import time
import csv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

# ---------------------------------数据读入
print('数据读入ing:')
df = pd.read_csv('xx.csv',encoding='GBK') # GBK or UTF-8
# df = df.set_index('sample_id')
print("原始数据示例\n",df.head(3))

# 长度统计特征
df['lengths'] = df['lengths'].apply(lambda x: list(map(int, x.split(';')))) # 分号处理为逗号
df['length_mean'] = df['lengths'].apply(np.mean)
df['length_std'] = df['lengths'].apply(np.std)
df['length_max'] = df['lengths'].apply(max)
print("df\n",df.head(3))

# 宽度统计特征
df['widths'] = df['widths'].apply(lambda x: list(map(int, x.split(';'))))
df['width_mean'] = df['widths'].apply(np.mean)
df['width_std'] = df['widths'].apply(np.std)
df['width_max'] = df['widths'].apply(max)
print("df\n",df.head(3))

# 长与宽的比值
df['L/W'] = df['L/W'].apply(lambda x: list(map(float, x.split(';'))))  # 分割并转为浮点列表
df['长宽比极差'] = df['L/W'].apply(max)-df['L/W'].apply(min)
print("df\n",df.head(3))

# 面积
df['L*W'] = df['L*W'].apply(lambda x: list(map(float, x.split(';'))))
df['面积标准差'] = df['L*W'].apply(np.std)
# df['面积最大值'] = df['面积'].apply(max) # 2475×1220=3019500
print("df\n",df.head(3))

# 删除非特征列
id_list = df['id_list']
columns_to_drop = [
    'boards_used', 'calculation_time', 'materials',
    'id_list' ,'lengths',
    'widths', 'L/W', 'L*W'
]
df = df.drop(columns=columns_to_drop)
print("测试集列名:", df.columns.tolist())

# -----------------------------------初始分割
'''
防止数据泄露：先分割原始数据为训练集（含验证）和测试集（保持测试集纯净）
'''
print('初始分割数据集ing：')
label = 'utilization'  # 假设标签列名为'利用率'

# 假设测试集比例为20%，可根据需要调整
train_data, test_data = train_test_split(
    df,
    test_size=0.25,
    random_state=42,  # 固定随机种子确保可复现
)

print(f"\n数据集分割结果：")
print(f"训练集: {train_data.shape[0]}个样本（{train_data.index.nunique()}个批次）")
print(f"测试集: {test_data.shape[0]}个样本（{test_data.index.nunique()}个批次）")

# --------------------------------特征标签分离
# 训练集的特征和标签
X_train = train_data.drop(['sample_id', 'utilization'], axis=1)
print('train数据集\n',train_data.head(3))
y_train = train_data['utilization']
print('train特征\n',X_train.head(3))
print('train标签（利用率）\n',y_train.head(3))

# 测试集的特征标签分离
X_id = test_data['sample_id']
X_test = test_data.drop(['sample_id','utilization'],axis=1)
y_test = test_data['utilization']

print('测试集的特征',X_test.head(3))

# --------------------------------XGboost预测模型
start_time = time.time()
print('设置XGBoost各项参数ing：')

params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'gamma': 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 行采样（样本）
    'colsample_bytree':0.7, # 列采样（特征）
    'min_child_weight': 3,
    'verbosity': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.05,  # 学习率
    'seed': 1000,
    'nthread': 12,  # cpu 线程数
    'eval_metric': 'rmse'
}

# -----------------------------交叉验证：评估模型泛化能力
print("开始交叉验证：")
dtrain = xgb.DMatrix(X_train, label=y_train)
# 执行交叉验证（调整参数）
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,      # 设置较大的迭代次数，由早停控制实际轮数
    nfold=5,                   # 5折交叉验证
    early_stopping_rounds=50,  # 早停轮数
    metrics='rmse',             # 根据任务类型选择指标（回归用 rmse/mae）
    maximize=False,             # 若指标为 auc 则设为 True
    verbose_eval=10,           # 每10轮输出一次日志
    seed=42                    # 固定随机种子
)

print("\n=== 交叉验证结果 ===")
print(cv_results.tail())  # 查看最后几轮的结果
# 最佳训练轮数（根据早停自动选择）
best_round = cv_results.shape[0]
print(f"交叉验证最佳轮数: {best_round}")
# 提取最佳验证指标
best_rmse = cv_results['test-rmse-mean'].min()
print(f"验证集平均RMSE: {best_rmse:.4f}")


# --------------------------------使用最佳轮次训练最终模型
print('使用最佳轮次训练最终模型：')
# 合并训练集和验证集（最大化数据利用）
model_final = xgb.train(
    params,
    dtrain,
    num_boost_round=best_round
)

# 预测测试集
xgb_test = xgb.DMatrix(X_test)
preds_test = model_final.predict(xgb_test)
# 计算 RMSE
rmse_test = np.sqrt(metrics.mean_squared_error(y_test, preds_test))
print(f"测试集 RMSE: {rmse_test:.4f}")
# 计算 MAE
mae_test= metrics.mean_absolute_error(y_test, preds_test)
print(f"测试集 MAE: {mae_test:.4f}")
# 保存结果
test_results = test_data[['sample_id']].copy()
test_results['预测值'] = preds_test

# test_results.to_csv("../output/测试集结果.csv", index=None, encoding='utf-8')

# -------特征重要性评分(用来看对各个特征的敏感度)
feature_score = model_final.get_score(importance_type='weight')
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
csv_path = '../feature_score/特征得分.csv'
# 确保目录存在
import os
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, 'w', newline='', encoding='GBK') as f:
    writer = csv.writer(f)
    writer.writerow(['feature', 'score'])
    for feature, score in feature_score:
        writer.writerow([feature, score])

cost_time = time.time() - start_time
print("模型运行时间", '\n', "cost time:", cost_time, "(s)")

model_final.save_model('../model/trained_model.json')  # 保存模型到文件
print('已保存训练好的模型')
