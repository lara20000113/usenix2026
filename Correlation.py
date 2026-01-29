import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau

# 教育程度数据(1-高中,2-大专,3-本科,4-硕士,5-博士)
education_level = np.array([1.82, 1.54, 2.18, 1.57, 1.65, 1.90, 1.15, 1.87, 1.52, 1.44, 1.45, 1.39, 1.54, 1.47, 1.52,
                            1.48, 1.43, 1.73])

# 收入满意度数据(1-非常不满意,2-不满意,3-一般,4-满意,5-非常满意)
income_satisfaction = np.array([22.49, 34.12, 24.09, 39.37, 29.93, 25.13, 77.71, 42.15, 34.70, 41.24, 35.92,
                                43.49, 40.24, 45.24, 36.05, 58.73, 40.76, 29.06])

# 计算斯皮尔曼相关系数和p值
correlation, p_value = spearmanr(education_level, income_satisfaction)

print(f"斯皮尔曼相关系数: {correlation:.4f}")
print(f"p值: {p_value:.4f}")

correlation, p_value = kendalltau(education_level, income_satisfaction)
print(f"肯德尔相关系数: {correlation:.4f}")
print(f"p值: {p_value:.4f}")

# 教育程度数据(1-高中,2-大专,3-本科,4-硕士,5-博士)
education_level = np.array([8.68, 8.73, 8.94, 9.24, 8.32, 9.02, 8.01, 8.63, 7.95, 7.88, 8.05, 7.54, 7.72, 7.59, 7.83,
                            7.14, 8.21, 7.95])

# 收入满意度数据(1-非常不满意,2-不满意,3-一般,4-满意,5-非常满意)
income_satisfaction = np.array([59.36, 60.41, 68.37, 63.71, 56.67, 75.33, 29.46, 68.07, 64.07, 62.90, 58.08, 60.36,
                                53.34, 56.77, 56.09, 52.93, 60.47, 60.36])

# 计算斯皮尔曼相关系数和p值
correlation, p_value = spearmanr(education_level, income_satisfaction)

print(f"斯皮尔曼相关系数: {correlation:.3f}")
print(f"p值: {p_value:.3f}")

correlation, p_value = kendalltau(education_level, income_satisfaction)
print(f"肯德尔相关系数: {correlation:.3f}")
print(f"p值: {p_value:.3f}")