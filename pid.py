
'''
https://cloud.tencent.com/developer/article/1456305

PID 控制算法通常分为位置式 PID 控制算法和增量式 PID 控制算法
PID调试的一般原则：
    在输出不震荡时，增大比例增益；
    在输出不震荡时，减少积分时间常数；
    在输出不震荡时，增大微分时间常数；

PID调节口诀：
    参数整定找最佳，从小到大顺序查
    先是比例后积分，最后再把微分加
    曲线振荡很频繁，比例度盘要放大
    曲线漂浮绕大湾，比例度盘往小扳
    曲线偏离回复慢，积分时间往下降
    曲线波动周期长，积分时间再加长
    曲线振荡频率快，先把微分降下来
    动差大来波动慢，微分时间应加长
    理想曲线两个波，前高后低四比一
    一看二调多分析，调节质量不会低
'''
import time
import matplotlib.pyplot as plt
import numpy as np


class PID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.output = 0.0

    def update(self, feedback_value):
        error = self.SetPoint - feedback_value
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error  # 比例
            self.ITerm += error * delta_time  # 积分
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time  # 微分
            self.last_time = self.current_time
            self.last_error = error
            self.output = self.PTerm + \
                (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time


def test_pid(P, I, D, L):

    pid = PID(P, I, D)

    pid.SetPoint = 1.1
    pid.setSampleTime(0.01)

    END = L
    feedback = 0
    feedback_list = []
    time_list = []
    setpoint_list = []

    for i in range(1, END):
        pid.update(feedback)
        output = pid.output
        feedback += output  # PID控制系统的函数
        time.sleep(0.01)
        feedback_list.append(feedback)
        setpoint_list.append(pid.SetPoint)
        time_list.append(i)

    plt.figure(0)
    plt.grid(True)
    plt.plot(time_list, feedback_list, 'b-')
    plt.plot(time_list, setpoint_list, 'r')
    plt.xlim((0, L))
    plt.ylim((min(feedback_list)-0.5, max(feedback_list)+0.5))
    plt.xlabel('time (s)')
    plt.ylabel('PID (PV)')
    plt.title('TEST PID', fontsize=15)

    plt.ylim((1-0.5, 1+0.5))

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_pid(1.2, 1, 0.001, L=100)
