'''
https://blog.csdn.net/CodeSamer/article/details/81191487
卡尔曼滤波（Kalman filtering）是一种利用线性系统状态方程，通过系统输入输出观测数据，对系统状态进行最优估计的算法。由于观测数据中包括系统中的噪声和干扰的影响，所以最优估计也可看作是滤波过程。

卡尔曼系数的作用主要有两个方面：
    第一是权衡预测状态协方差矩阵P和观察量协方差矩阵R的大小，来决定我们是相信预测模型多一点还是观察模型多一点。如果相信预测模型多一点，这个残差的权重就会小一点，如果相信观察模型多一点，权重就会大一点。
    第二就是把残差的表现形式从观察域转换到状态域，在我们这个例子中，观察值z表示的是小车的位置，只是一个一维向量，而状态向量是一个二维向量，它们所使用的单位和描述的特征都是不同的。而卡尔曼系数就是要实现这样将一维的观测向量转换为二维的状态向量的残差，在本例中我们只观测了小车的位置，而在K中已经包含了协方差矩阵P的信息，所以就利用了位置和速度这两个维度的相关性，从位置的信息中推测出了速度的信息，从而让我们可以对状态量x的两个维度同时进行修正。

    在这一轮里噪声的不确定性是减小的，而在下一轮迭代中，由于传递噪声的引入，不确定性又会增大，卡尔曼滤波器就是在这种不确定性的变化中寻求一种平衡。

协方差
    从直观上来看，协方差表示的是两个变量总体误差的期望。
    如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值时另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值；如果两个变量的变化趋势相反，即其中一个变量大于自身的期望值时另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。
    如果X与Y是统计独立的，那么二者之间的协方差就是0，因为两个独立的随机变量满足E[XY]=E[X]E[Y]。
    但是，反过来并不成立。即如果X与Y的协方差为0，二者并不一定是统计独立的。
    协方差Cov(X,Y)的度量单位是X的协方差乘以Y的协方差。
    协方差为0的两个随机变量称为是不相关的。    
'''
import numpy as np
import matplotlib.pyplot as plt

# 创建一个0-99的一维矩阵
z_watch = np.arange(100)
# print(z_watch)

# 创建一个标准差为1的高斯噪声，精确到小数点后两位
noise = np.round(np.random.normal(0, 1, 100), 2)
noise_mat = np.mat(noise)

# 将z的观测值和噪声相加
z_mat = z_watch + noise_mat
# print(z_mat)

# 定义x的初始状态
x_mat = np.mat([[0, ], [0, ]])
# 定义初始状态协方差矩阵
p_mat = np.mat([[1, 0], [0, 1]])
# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.mat([[1, 1], [0, 1]])
# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.0001, 0], [0, 0.0001]])
# 定义观测矩阵
h_mat = np.mat([1, 0])
# 定义观测噪声协方差
r_mat = np.mat([1])

x = list()
p = list()
for i in range(100):
    x_predict = f_mat * x_mat
    p_predict = f_mat * p_mat * f_mat.T + q_mat
    kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
    x_mat = x_predict + kalman * (z_mat[0, i] - h_mat * x_predict)
    p_mat = (np.eye(2) - kalman * h_mat) * p_predict
    print(x_mat, '\n', p_mat, '\n------------------')

    x.append(x_mat[0, 0])
    p.append(x_mat[1, 0])
    # plt.plot(x_mat[0, 0], x_mat[1, 0], 'ro', markersize=1)

# print([round(a - b, 2) for a, b in zip(x, z_watch.tolist()[0])])
# print([round(a - 1, 2) for a in y])
plt.plot(x, p, 'go', markersize=1)

# 只与上一点相关的简单预测
z_np = z_mat.tolist()[0]
z_ns = [0]
i = 1
for p in range(len(z_np)-1):
    z_ns.append(z_np[i]-z_np[i-1])
    i += 1
plt.plot(z_np, z_ns, 'ro', markersize=1)

plt.show()
