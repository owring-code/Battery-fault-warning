import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import glob
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

import math
import random
import numpy as np
import tensorflow as tf
from tcn import TCN
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# %%
with open('./pkl_data/balanced_data.pkl', 'rb') as f:
    balanced_data = pkl.load(f)
# %%
predict_interval = 90
features = [f'U_{i}' for i in range(1, 93)]

x_train_all = np.array(balanced_data.drop('Abnormal_self_discharge', axis=1))
y_train_all = np.array(balanced_data['Abnormal_self_discharge'])

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all,
                                                    random_state=42,
                                                    test_size=0.2)

window_size = 500
stride = 1


# %%
# 滑窗函数
def sliding_window(x_data, y_data, window_size, stride=1):
    n_samples, n_features = x_data.shape
    n_windows = (n_samples - window_size) // stride + 1
    x_split = np.zeros((n_windows, window_size, n_features))
    y_split = np.zeros(n_windows)
    for i in range(n_windows):
        x_split[i] = x_data[i * stride:i * stride + window_size]
        y_split[i] = y_data[i * stride + window_size - 1]
    return x_split, y_split


x_train_split, y_train_split = sliding_window(x_train, y_train, window_size, stride)
x_test_split, y_test_split = sliding_window(x_test, y_test, window_size, stride)


# %%
# 定义TCN网络模型
def create_tcn_model():
    model = tf.keras.Sequential([
        TCN(input_shape=[x_train_split.shape[1], x_train_split.shape[2]],
            nb_filters=64,
            kernel_size=3,
            nb_stacks=3,
            dilations=(1, 2, 4, 8),
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=False,
            # activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=True,
            use_weight_norm=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    return model


# 交叉函数
def crossover(weights1, weights2):
    crossover_point = random.randint(0, len(weights1))
    weights1[:crossover_point], weights2[:crossover_point] = weights2[:crossover_point], weights1[:crossover_point]
    return weights1


# 变异函数
def mutate(weights):
    t = random.randint(0, len(weights) - 1)
    weights[t] *= 1.25
    return weights


# 计算准确率
def get_accuracy(model):
    result = model.predict(x_test_split)
    result_final = np.zeros(len(result))
    for i in range(len(result)):
        if result[i] >= 0.5:
            result_final[i] = 1
    sum = 0
    for i in range(len(result)):
        if result_final[i] == y_test_split[i]:
            sum += 1
    accuracy = sum / len(result)

    return accuracy


# %%
# 定义种群数量
pop_size = 5
pop = []

# 定义GA超参数
parent_selection_pressure = 0.5
mutation_rate = 0.1
accuracy_tar = 0.32

# 权重初始化
ori_weights = []
for i in range(pop_size):
    tf.keras.backend.clear_session()
    model = create_tcn_model()
    ori_weights.append(model.get_weights())

# 生成第一代种群
print("Generation_1")
fitness = []
for i in range(pop_size):
    model.set_weights(ori_weights[i])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    print(f"Individual_{i + 1}:")
    model.fit(x_train_split, y_train_split, epochs=4,
              validation_data=(x_val_split, y_val_split),
              batch_size=16, verbose=1)
    accuracy = get_accuracy(model)
    pop.append(model.get_weights())
    print(accuracy)
    fitness.append(accuracy)

generation = 1
# 开始迭代
while accuracy >= accuracy_tar:
    generation += 1
    # 选择父代
    parents = []
    for _ in range(pop_size):
        # 轮盘赌徒法
        idx1 = np.random.choice(np.arange(pop_size), p=fitness / np.sum(fitness))
        idx2 = np.random.choice(np.arange(pop_size), p=fitness / np.sum(fitness))
        if fitness[idx1] > fitness[idx2]:
            parents.append(pop[idx1])
        else:
            parents.append(pop[idx2])
    # 生产新子代
    offspring = []
    for i in range(pop_size):
        # 交叉
        if np.random.random() < parent_selection_pressure:
            weight1 = parents[i]
            weight2 = parents[(i + 1) % pop_size]
            offspring.append(crossover(weight1, weight2))
        # 变异
        elif np.random.random() < mutation_rate:
            offspring.append(mutate(parents[i]))
        # 复制
        else:
            offspring.append(parents[i])

    # 替换新种群
    pop = offspring

    # 计算适应度
    print(f"第{generation}代")
    fitness = []
    for i in range(len(pop)):
        model.set_weights(pop[i])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
        print(f"Individual_{i + 1}:")
        model.fit(x_train_split, y_train_split, epochs=2,
                  validation_data=(x_val_split, y_val_split),
                  batch_size=16, verbose=0)
        accuracy = get_accuracy(model)
        pop[i] = model.get_weights()
        fitness.append(accuracy)

    # 选择最优模型
    model.set_weights(pop[np.argmax(fitness)])
    accuracy = get_accuracy(model)
    print(accuracy)

print(f"best accuracy:{accuracy}")
