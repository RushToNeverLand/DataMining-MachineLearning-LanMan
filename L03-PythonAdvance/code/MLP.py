#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-

# 导入库
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import theano
import tensorflow

# 导入数据
from sklearn import datasets
dataset = datasets.load_boston() 

X_train = dataset.data
y_train = dataset.target
X_test = dataset.data
y_test = dataset.target

# 简单搭建一个MLP（多层感知机）
model = Sequential()  # 模型初始化
model.add(Dense(32,input_dim=13))  # 添加输入层（20个节点）、第一隐含层（64节点）的连接
model.add(Activation('tanh')) # 第一隐含层使用tanh作为激活函数
model.add(Dropout(0.5))   # 使用Dropout防止过拟合
model.add(Dense(32, 32))   # 添加第一隐含层（64节点）、第二隐含层（64节点）的连接
model.add(Activation('tanh'))   # 第二隐含层使用tanh作为激活函数
model(Dropout(0.5))  # 使用Dropout防止过拟合
model.add(Dense(32, 1))  # 添加第二隐含层（64节点）、输出层（1节点）的连接
model(Activation('sigmoid'))  # 输出层用sigmoid作为激活函数

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # 定义求解算法
model.compile(loss='mean_squared_error', optimizer=sgd) # 编译生成模型，损失函数均方误差

model.fit(X_train, y_train, nb_epoch=20, batch_size=16)  # 训练模型
score = model.evaluation(X_test, y_test, batch_size=16) # 测试模型