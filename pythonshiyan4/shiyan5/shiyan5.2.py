import numpy as np
import matplotlib.pyplot as plt

# 提供的数据
X = np.array([
    [25, 2, 3],
    [30, 5, 8],
    [35, 8, 12],
    [28, 3, 4],
    [32, 6, 10],
    [40, 10, 15],
    [45, 12, 20],
    [38, 9, 13],
    [29, 4, 6],
    [33, 7, 11]
])  # 年龄、教育水平、工作经验
Y = np.array([4.5, 7.2, 9.6, 5.0, 8.3, 12.5, 15.2, 10.0, 6.8, 8.9])  # 工资

# 初始化系数 a 和 b
a = np.zeros(X.shape[1])
b = 0.0

# 学习率
alpha = 0.00001

# 迭代次数
iterations = 100000

n = len(X)

# 梯度下降迭代
for _ in range(iterations):
    Y_pred = np.dot(X, a) + b
    error = Y_pred - Y
    gradient_a = (1/n) * np.dot(X.T, error)
    gradient_b = (1/n) * np.sum(error)
    a -= alpha * gradient_a
    b -= alpha * gradient_b

# 打印结果
print("斜率 a:", a)
print("截距 b:", b)

# 根据模型方程，预测一位年龄为 30 岁、教育水平为 5 年、工作经验为 8 岁的员工的工资
age = 30
education = 5
experience = 8
predicted_salary = np.dot([age, education, experience], a) + b
print("预测工资:", predicted_salary)

# 根据模型方程，预测如果自己毕业与读研时期分别加入这个公司后的工资差别
age_graduated = 22
education_graduated = 4
experience_graduated = 0
predicted_salary_graduated = np.dot([age_graduated, education_graduated, experience_graduated], a) + b
print("毕业时预测工资:", predicted_salary_graduated)

age_masters = 25
education_masters = 6
experience_masters = 1
predicted_salary_masters = np.dot([age_masters, education_masters, experience_masters], a) + b
print("读研时期预测工资:", predicted_salary_masters)

# 绘制散点图
plt.scatter(X[:, 0], Y, label="年龄")
plt.scatter(X[:, 1], Y, label="教育水平")
plt.scatter(X[:, 2], Y, label="工作经验")

# 绘制拟合的线性回归曲线
X_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
Y_pred = a[0] * X_range + b
plt.plot(X_range, Y_pred, color='red', label="年龄拟合")

X_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
Y_pred = a[1] * X_range + b
plt.plot(X_range, Y_pred, color='blue', label="教育水平拟合")

X_range = np.linspace(np.min(X[:, 2]), np.max(X[:, 2]), 100)
Y_pred = a[2] * X_range + b
plt.plot(X_range, Y_pred, color='green', label="工作经验拟合")
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.xlabel("特征值")
plt.ylabel("工资")
plt.legend()
plt.show()
