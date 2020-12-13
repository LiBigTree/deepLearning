#!/usr/bin/env python
# -*- coding:utf-8 -*-
# datetime:2020/2/21 10:43
# ----导--包--区----

import numpy as np
import matplotlib.pyplot as plt  # painting
import h5py  # data management
import skimage.transform as tf  # zoom image

# ----说--明--区--/参--数--预--设--区----
# 写一个简单的神经网络，判断图片是不是猫咪


# ----def--core--问--题--分--解----

# 1------加--载--数--据--并--对--数--据--预--处--理------
def load_dataset():
    """正确加载训练数据特征、标签，加载测试数据特征、标签、类别。
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# index = 30 plt.imshow(train_set_x_orig[index])
# plt.show()
# print("label:" + str(train_set_y[:, index]) + " classes:"
# + classes[np.squeeze(train_set_y[:, index])].decode("utf-8"))

# 变量的维度
print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_orig: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# 提取样本数、图片长宽像素

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

# 将样本数据进行扁平化和转置
# 处理后：（图片数据， 样本数）

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print(str(train_set_x_flatten.shape))
print(str(test_set_x_flatten.shape))

# 标准化处理，值在[0, 1]之间
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# 2----构--建--网--络--模--型--训--练----
# 说明：

def sigmoid(z):
    """
    参数：
    z： numpy数组或数值
    s： 经过计算sigmoid的值，在[0， 1]区间
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    功能： 初始化权重w、阈值b
    参数：
    w：权重数组
    b：偏置bias
    """
    w = np.zeros((dim, 1))
    b = 0

    return w, b


def propagate(w, b, X, Y):
    """
    参数：
    w：权重 （12288，1）
    b：偏置
    X：图片特征数据 （12288，209）
    Y：图片标签

    返回值：
    cost: 成本
    dw: w的梯度
    db: b的梯度
    """

    m = X.shape[1]  # 样本数

    # 前向传播
    A = sigmoid(np.dot(w.T, X))
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

    # 反向传播
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    # 将dw 和 db 保存到字典里
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    参数：
    w： 权重
    b： 偏置
    X：图片特征
    Y: 标签
    num_iterations: 优化次数
    learning_rates: 每次的步进
    print_cost: 每一百次变为true，用于打印出来观察分析

    返回值：
    params: 优化后的w、b
    costs: 每100次的优化成本
    """

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)  # 得出梯度，成本

        dw = grads["dw"]
        db = grads["db"]

        # 梯度更新，优化参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 成本记录：
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("优化%i次后成本是： %f" % (i, cost))

    params = {"w": w,
              "b": b}
    return params, costs


# 3----测--试--训--练--的--模--型-------


def predict(w, b, X):
    """
    参数：
    w：权重
    b：偏置
    X: 图片特征

    返回值：
    Y_prediction: 预测结果
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)  # 预测

    # 转化为 0 or 1
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    # print()
    # print("Y_prediction:", Y_prediction)

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    参数：
    X_train: 训练图片 （12288，209）
    Y_train：训练图片对应的标签 （1，209）
    X_test: 测试图片  （12288， 50）
    Y_test： 测试图片对应的标签 （1，50）
    num_iterations: 训练/优化次数
    learning_rate: 学习步进
    print_cost: True时打印成本

    返回值：
    d： 返回一些信息
    """

    # 参数初始化：
    w, b = initialize_with_zeros(X_train.shape[0])

    # 参数优化
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']

    # 预测图片
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 打印预测的准确率
    print("训练图片的准确率为： {}%.".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试图片的准确率为： {}%.".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}

    return d


# ----运--行--区----
if __name__ == "__main__":
    d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

    # index = 8
    # plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    # plt.show()
    # print("标签：", str(test_set_y[0, index]), " 预测：", str(int(d["Y_prediction_test"][0, index])))

    # 测试自己的图片
    my_image = "2.jpg"
    fname = "images/" + my_image

    image = np.array(plt.imread(fname))
    my_image = tf.resize(image, (num_px, num_px), mode='reflect').reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    print("预测结果为" + str(int(np.squeeze(my_predicted_image))))

    plt.imshow(image)
    plt.show()


