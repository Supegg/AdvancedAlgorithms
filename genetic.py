'''
https://zhuanlan.zhihu.com/p/43546261
遗传算法（Genetic Algorithm，GA）最早是由美国的 John holland于20世纪70年代提出,该算法是根据大自然中生物体进化规律而设计提出的。模拟达尔文生物进化论的自然选择和遗传学机理的生物进化过程搜索最优解的计算模型。

http://tinyurl.com/y39pwomt
关键术语：
    种群（Population） 参与演化的生物群体，即解的搜索空间
        个体（Individual） 种群的每一个成员，对应每一个可能的解
            染色体（Chromosome） 对应问题的解向量
            基因（Gene） 解向量的一个分量，或者编码后的解向量的一位
        适应度（Fitness） 体现个体的生存能力，与目标函数相关的函数
    遗传算子（Operator） 个体的演化操作，包括选择、交叉、变异
        选择(Selection) 基于适应度的优胜劣汰，以一定的概率从种群中选择若干个体
        交叉(Crossover) 两个染色体进行基因重组
        变异(Mutation)：单个染色体的基因以较低概率发生随机变化

初始种群产生了一系列随机解，选择操作保证了搜索的方向性，交叉和变异拓宽了搜索空间，其中交叉操作延续父辈个体的优良基因，变异操作则可能产生比当前优势基因更优秀的个体。变异操作有利于跳出局部最优解，同时增加了随机搜索的概率，即容易发散。

遗传算法需要在过早收敛（早熟）和发散、精度和效率之间平衡。当物种多样性迅速降低即个体趋于一致，例如选择操作时过分突出优势基因的地位，结果可能只是收敛于局部最优解。当物种持续保持多样性，例如选择力度不大、变异概率太大，结果可能很难收敛，即算法效率较低。
'''
import math
import random


class Population:
    '''
    种群的设计
    size,种群的个体数量\n
    chrom_size,染色体长度\n
    cp,交叉概率为\n
    mp,变异概率\n
    gen_max,进化最大世代数\n
    '''

    def __init__(self, size, chrom_size, cp, mp, gen_max):
        # 种群信息合
        self.individuals = []          # 个体集合
        self.fitness = []              # 个体适应度集
        self.selector_probability = []  # 个体选择概率集合
        self.new_individuals = []      # 新一代个体集合

        self.elitist = {'chromosome': [0, 0],
                        'fitness': 0, 'age': 0}  # 最佳个体的信息

        self.size = size  # 种群所包含的个体数
        self.chromosome_size = chrom_size  # 个体的染色体长度
        self.crossover_probability = cp   # 个体之间的交叉概率
        self.mutation_probability = mp    # 个体之间的变异概率

        self.generation_max = gen_max  # 种群进化的最大世代数
        self.age = 0                  # 种群当前所处世代

        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以 0 值进行初始化
        v = 2 ** self.chromosome_size - 1
        for i in range(self.size):
            self.individuals.append(
                [random.randint(0, v), random.randint(0, v)])
            self.new_individuals.append([0, 0])
            self.fitness.append(0)
            self.selector_probability.append(0)

    # 基于轮盘赌博机的选择
    def decode(self, interval, chromosome):
        '''将一个染色体 chromosome 映射为区间 interval 之内的数值'''
        d = interval[1] - interval[0]
        n = float(2 ** self.chromosome_size - 1)
        return (interval[0] + chromosome * d / n)

    def fitness_func(self, chrom1, chrom2):
        '''适应度函数，可以根据个体的两个染色体计算出该个体的适应度'''
        interval = [-10.0, 10.0]
        (x, y) = (self.decode(interval, chrom1),
                  self.decode(interval, chrom2))

        def n(x, y): return math.sin(math.sqrt(x*x + y*y)) ** 2 - 0.5
        def d(x, y): return (1 + 0.001 * (x*x + y*y)) ** 2
        def func(x, y): return 0.5 - n(x, y)/d(x, y)
        return func(x, y)

    def evaluate(self):
        '''用于评估种群中的个体集合 self.individuals 中各个个体的适应度'''
        sp = self.selector_probability  # sp 引用 self.selector_probability
        for i in range(self.size):
            self.fitness[i] = self.fitness_func(self.individuals[i][0],   # 将计算结果保存在 self.fitness 列表中
                                                self.individuals[i][1])

        # 马太效应
        m = max(self.fitness)
        p = [v * v / m for v in self.fitness]

        ft_sum = sum(p)
        for i in range(self.size):
            sp[i] = p[i] / float(ft_sum)   # 得到各个个体的生存概率
        for i in range(1, self.size):
            sp[i] = sp[i] + sp[i-1]   # 需要将个体的生存概率进行叠加，从而计算出各个个体的选择概率

    # 轮盘赌博机（选择）
    def select(self):
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        return i

    # 交叉
    def cross(self, chrom1, chrom2):
        p = random.random()    # 随机概率
        n = 2 ** self.chromosome_size - 1
        if chrom1 != chrom2 and p < self.crossover_probability:
            t = random.randint(1, self.chromosome_size - 1)   # 随机选择一点（单点交叉）
            mask = n << t    # << 左移运算符
            # & 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
            (r1, r2) = (chrom1 & mask, chrom2 & mask)
            mask = n >> (self.chromosome_size - t)
            (l1, l2) = (chrom1 & mask, chrom2 & mask)
            (chrom1, chrom2) = (r1 + l2, r2 + l1)
        return (chrom1, chrom2)

    # 变异
    def mutate(self, chrom):
        p = random.random()
        if p < self.mutation_probability:
            t = random.randint(1, self.chromosome_size)
            mask1 = 1 << (t - 1)
            mask2 = chrom & mask1
            if mask2 > 0:
                chrom = chrom & (~mask2)  # ~ 按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1
            else:
                chrom = chrom ^ mask1   # ^ 按位异或运算符：当两对应的二进位相异时，结果为1
        return chrom

    # 保留最佳个体
    def reproduct_elitist(self):
        # 与当前种群进行适应度比较，更新最佳个体
        j = -1
        for i in range(self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness'] = self.fitness[i]
        if (j >= 0):
            self.elitist['chromosome'][0] = self.individuals[j][0]
            self.elitist['chromosome'][1] = self.individuals[j][1]
            self.elitist['age'] = self.age

    # 进化过程
    def evolve(self):
        indvs = self.individuals
        new_indvs = self.new_individuals
        # 计算适应度及选择概率
        self.evaluate()
        # 最佳个体保留
        # 如果在选择之前保留当前最佳个体，最终能收敛到全局最优解。
        self.reproduct_elitist()

        # 进化操作
        i = 0
        while True:
            # 选择两个个体，进行交叉与变异，产生新的种群
            idv1 = self.select()
            idv2 = self.select()
            # 交叉
            (idv1_x, idv1_y) = (indvs[idv1][0], indvs[idv1][1])
            (idv2_x, idv2_y) = (indvs[idv2][0], indvs[idv2][1])
            (idv1_x, idv2_x) = self.cross(idv1_x, idv2_x)
            (idv1_y, idv2_y) = self.cross(idv1_y, idv2_y)
            # 变异
            (idv1_x, idv1_y) = (self.mutate(idv1_x), self.mutate(idv1_y))
            (idv2_x, idv2_y) = (self.mutate(idv2_x), self.mutate(idv2_y))
            # 将计算结果保存于新的个体集合self.new_individuals中
            (new_indvs[i][0], new_indvs[i][1]) = (idv1_x, idv1_y)
            (new_indvs[i+1][0], new_indvs[i+1][1]) = (idv2_x, idv2_y)
            # 判断进化过程是否结束
            i = i + 2         # 循环self.size/2次，每次从self.individuals 中选出2个
            if i >= self.size:
                break

        # 更新换代：用种群进化生成的新个体集合 self.new_individuals 替换当前个体集合
        for i in range(self.size):
            self.individuals[i][0] = self.new_individuals[i][0]
            self.individuals[i][1] = self.new_individuals[i][1]

    def run(self):
        '''根据种群最大进化世代数设定了一个循环。
        在循环过程中，调用 evolve 函数进行种群进化计算，并输出种群的每一代的个体适应度最大值、平均值和最小值。'''
        for i in range(self.generation_max):
            self.evolve()
            self.age += 1
            print(i, max(self.fitness), sum(self.fitness) /
                  self.size, min(self.fitness))


if __name__ == '__main__':
    # 种群的个体数量为 50，染色体长度为 25，交叉概率为 0.8，变异概率为 0.1,进化最大世代数为 150
    pop = Population(50, 24, 0.8, 0.1, 150)
    pop.run()
    print(pop.elitist)

    interval = [-10.0, 10.0]
    (x, y) = (pop.decode(interval, pop.elitist['chromosome'][0]),
              pop.decode(interval, pop.elitist['chromosome'][1]))
    print(x, y, pop.elitist['fitness'])
