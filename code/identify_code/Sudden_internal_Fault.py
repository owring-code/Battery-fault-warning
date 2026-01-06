import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
import time
from matplotlib.pyplot import MultipleLocator


def distance(point1, point2):
    return abs(point1[1] - point2[1])


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]


def sigma_m(m, k, sigma):
    return math.sqrt(pow(sigma, 2) * ((m - k) / (m - 1)) / k)


def k(m, sigma, Z=1.96):
    return (m * pow(Z, 2) * pow(sigma, 2)) / ((m - 1) * pow(0.1 * sigma, 2) + pow(sigma, 2) * pow(Z, 2))


def D(v, o):
    d = np.zeros((v.shape[0], o.shape[0]))
    for i in range(v.shape[0]):
        for j in range(o.shape[0]):
            d[i, j] = distance(v[i], o[j])
    return d


def id_n(d, X=6):
    identifier = np.zeros((d.shape[0], X))
    for x in range(X):
        min_indices = d.argmin(axis=1)
        for j in range(len(min_indices)):
            identifier[j, x] = min_indices[j]
            d[j, min_indices[j]] = float('inf')
    return identifier


def p(identifier, o, K):
    P = np.zeros(K)
    for j in range(K):
        P[j] = np.sum(identifier == j)

    ro = np.percentile(P, 33)
    tem = []
    for j in range(K):
        if P[j] < ro:
            tem.append(j)
    P = np.delete(P, tem, axis=0)
    o = np.delete(o, tem, axis=0)
    return P, o


def Y(v, o, identifier, X=6):
    avg = np.average(v[:, 1])
    y = []
    m = v.shape[0]
    for i in range(m):
        if v[i][1] > avg:
            a = 1
        elif v[i][1] == avg:
            a = 0
        else:
            a = -1
        yi = 0
        for x in range(X):
            yi += distance(v[i], o[identifier[i][x]])
        y.append(a * yi / X)
    return y


def SDO(V):
    V = np.array([[i, V[i]] for i in range(m)])  # 取最小电压序列
    K = int(round(k(m, np.std(V[:, 1]))))  # 随机选取观测点
    o1 = rand_row(V, K)
    d = D(V, o1)
    ID = id_n(d)

    p_cnt, o2 = p(ID, o1, K)  # 剔除异常观测点
    d2 = D(V, o2)
    ID2 = id_n(d2)

    y = Y(V, o2, identifier=np.round(ID2).astype(int))  # 计算异常程度
    return y


if __name__ == '__main__':
    t0 = time.time()
    u = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15), (16, 17, 18, 19, 20), (21, 22, 23, 24, 25),
         (26, 27, 28, 29, 30), (31, 32, 33, 34, 35), (36, 37, 38, 39, 40), (41, 42, 43, 44, 45), (46, 47, 48, 49, 50),
         (51, 52, 53, 54, 55, 56), (57, 58, 59, 60, 61, 62), (63, 64, 65, 66, 67, 68), (69, 70, 71, 72, 73, 74),
         (75, 76, 77, 78, 79, 80), (81, 82, 83, 84, 85, 86), (87, 88, 89, 90, 91, 92)]
    t = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
         (25, 26), (27, 28), (29, 30), (31, 32), (33, 34)]
    # csv_file = '/tiaozhanbei/code/Data_set/WCVT1000000000127.csv'
    for csv_file in glob.glob('/tiaozhanbei/code/Data_set/' + '*.csv'):
        name = csv_file.split('/')[-1].split('.')[0]
        data = pd.read_csv(csv_file)
        all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
        all_t = np.nan_to_num(data.loc[:, 'T_1':'T_34'].to_numpy())
        Time = np.nan_to_num(data.loc[:, 'TIME'].to_numpy())
        if max(all_u[0]) > 1000:
            all_u = all_u / 1000

        n = len(u)
        m = all_u.shape[0]
        data['sudden_short_circuit'] = 0
        for i in range(n):
            min_u = np.min(all_u[:, u[i][0] - 1:u[i][len(u[i]) - 1]], axis=1)
            max_t = np.max(all_t[:, t[i][0] - 1:t[i][1]], axis=1)
            score = SDO(min_u)
            for a in range(m):
                if max_t[a] >= 100 or (score[a] <= -1 and min_u[a] != 0):
                    data.loc[a, 'sudden_short_circuit'] = 1
                    if score[a] <= -1:
                        index = np.argmin(all_u[a, u[i][0] - 1:u[i][len(u[i]) - 1]], axis=0)
                        print(f'{name} Sudden short circuit: '
                              f'U_{u[i][index-1]},TIME {time.localtime(Time[a])}')
                    else:
                        print(f'{name} Sudden short circuit: '
                              f'U_{u[i]},TIME {time.localtime(Time[a])}')
                elif data.loc[a, 'sudden_short_circuit'] == 1:
                    data.loc[a, 'sudden_short_circuit'] = 1
                else:
                    data.loc[a, 'sudden_short_circuit'] = 0
        t1 = time.time()
        spend1 = t1 - t0
        print('模型运行时间：', spend1)

        # original_data = pd.read_csv(csv_file)
        # original_data['sudden_short_circuit'] = data['sudden_short_circuit']
        # original_data.to_csv(csv_file, index=False)
        # print(csv_file.split('/')[-1].split('.')[0], "数据已成功写回原文件")




        # plt.figure(figsize=(15, 6))
        # plt.plot(result, c='r', label='result')
        # # plt.scatter(o2[:, 0], o2[:, 1], c='g', label='observe')
        # plt.title(name)
        # plt.grid(True)
        #
        # # 坐标轴的实例
        # ax = plt.gca()
        #
        # # 取消突起刻度线
        # ax.tick_params(axis='both', which='both', length=0)
        #
        # # 设置边框颜色为灰色
        # ax.spines['top'].set_color('gray')
        # ax.spines['right'].set_color('gray')
        # ax.spines['left'].set_color('gray')
        # ax.spines['bottom'].set_color('gray')
        #
        # plt.show()
