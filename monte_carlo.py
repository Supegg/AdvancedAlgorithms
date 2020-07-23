'''
https://www.jianshu.com/p/3430b69e0a92
蒙特卡罗方法（英語：Monte Carlo method），也称统计模拟方法，是1940年代中期由于科学技术的发展和电子计算机的发明，而提出的一种以概率统计理论为指导的数值计算方法。 是指使用随机数（或更常见的伪随机数）来解决很多计算问题的方法。
'''

import random
import math


def main():
    '''
    一个圆半径R，它有一个外切正方形边长2R。
    可以知道：
    圆面积=Pi*R^2
    正方形面积 2R * 2R=4R^2
    从这个正方形内随机抽取一个点，对这个点的要求是在正方形内任意一点的概率平均分布。
    那么这个点在圆以内的概率是pi*R^2/4R^2=pi/4
    生成若干个这样的点，利用平面上两点间距离公式计算这个点到圆心的距离来判断是否在圆内。
    当足够多的点来进行统计时，得到的概率值趋近pi/4
    '''
    print('请输入迭代的次数：')
    n = int(input())  # n是随机的次数
    total = 0  # total是所有落入圆内的随机点
    for i in range(n):
        x = random.random()
        y = random.random()
        if math.sqrt(x**2+y**2) < 1.0:  # 判断是否落入圆内
            total += 1
    mypi = 4.0*total/n  # 得到Pi值

    print('迭代次数：', n, '  mypi值：', mypi)
    print('数学pi：', math.pi)
    print('误差：  ', abs(math.pi-mypi)/math.pi)  # 计算误差


if __name__ == "__main__":
    main()
