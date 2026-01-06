import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


if __name__ == '__main__':
    t0 = time.time()
    for csv_file in glob.glob('/tiaozhanbei/code/battery2/battery2/' + '*.csv'):
        # csv_file = '/home/data_disk/user5/cdata/battery/WCVT1000000000293.csv'
        name = csv_file.split('/')[-1].split('.')[0]
        plt.figure(figsize=(12, 9))
        data = pd.read_csv(csv_file)
        all_u = np.nan_to_num(data.loc[:, 'U_1':'U_92'].to_numpy())
        time_pd = np.nan_to_num(data.loc[:, 'TIME'].to_numpy())
        for i in range(all_u.shape[0]):
            for j in range(all_u.shape[1]):
                if all_u[i, j] == 0:
                    all_u[i, j] = np.median(all_u[:, j])
        if max(all_u[0]) > 1000:
            all_u = all_u / 1000

        l = all_u.shape[0]
        w_l = 300
        all_uf = []

        avg_tem = np.median(all_u, axis=0)
        all_u_centered = all_u - avg_tem

        # 使用NumPy的滚动窗口平均函数，如果需要的话，需要安装scipy库
        from scipy.signal import convolve

        weights = np.ones(w_l) / w_l

        # 扩展weights以匹配all_u_centered的形状
        weights_2d = weights[:, np.newaxis]

        # 使用卷积计算滚动平均
        all_u_rolling_avg = convolve(all_u_centered, weights_2d, mode='valid')

        for j in range(all_u.shape[1]):
            uf = np.empty(l)
            valid_length = l - w_l + 1
            uf[:valid_length] = all_u_centered[:valid_length, j] - all_u_rolling_avg[:, j]
            if valid_length < l:
                remaining_avg = np.mean(all_u_centered[valid_length - 1:, j])
                uf[valid_length:] = all_u_centered[valid_length:, j] - remaining_avg
            all_uf.append(uf.tolist())

        all_uf_flat = np.concatenate(all_uf)
        f1 = np.percentile(all_uf_flat, 25)
        f2 = np.percentile(all_uf_flat, 75)
        iqr = f2 - f1

        all_th_bottom = f1 - 3 * iqr
        all_th_top = f2 + 3 * iqr

        min_all_uf = [min(col) for col in zip(*all_uf)]

        y = np.zeros((l, len(all_uf)))
        for j in range(len(all_uf)):
            for i in range(l):
                if (all_uf[j][i] < all_th_bottom or all_uf[j][i] > all_th_top) and all_uf[j][i] == min_all_uf[i]:
                    y[i][j] = 1

        result = np.sum(y, axis=0)
        threshold = max(result) * 0.7
        for j in range(len(all_uf)):
            if result[j] > threshold:
                index = j
                for i in range(l):
                    if y[i][index] == 1:
                        if type(time_pd[i]) == np.int64:
                            t = time.localtime(time_pd[i])
                            print(
                                f'{name} Abnormal_self_discharge: U_{index + 1},TIME:{t.tm_year}-{t.tm_mon}-{t.tm_mday} {t.tm_hour}:{t.tm_min}:{t.tm_sec}')
                        else:
                            print(f'{name} Abnormal_self_discharge: U_{index + 1},TIME:{time_pd[i]}')
                        break
    t1 = time.time()
    spend1 = t1 - t0
    print('模型运行时间：', spend1)
        # plt.figure(figsize=(15, 6))
        # plt.plot(result, c='b')
        # plt.title(name)
        # plt.show()
