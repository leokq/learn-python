#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:LiQing
# datetime:2018/4/17 21:31
# software: PyCharm
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets

def knn(x_train,x_test,y_train,y_test,k):
    x_trainSize = len(x_train)
    ex_x_test = np.tile(x_test,(x_trainSize,1))
    distant = ((ex_x_test-x_train)**2).sum(axis=1)**0.5
    sorteDistInd = distant.argsort()
    classcount = {}
    for i in range(k):
        indic = sorteDistInd[i]
        lable = y_train[indic]
        classcount[lable] = classcount.get(lable,0) + 1  #采集k个最近邻的标签

        #dic.items():将字典转化为list；lambda classcount:classcount[1]：函数-取classcount的第二域的数据；reverse默认false升序排列
        lableSorte = sorted(classcount.items(),key=lambda classcount:classcount[1],reverse=True)
        #lambda classcount:classcount[1] = operator.itergetter(1)
    return lableSorte[0][0]

if __name__ == '__main__':
    data = sklearn.datasets.load_iris()
    x = data.data
    y = data.target
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1)
    predict = knn(x_train,x_test,y_train,y_test,k=15)
    # print(predict)
    # print(data.target_names[predict])
    print('预测类别：',data.target_names[predict])
    print('实际类别：',data.target_names[y_test])
    print('预测结果：',predict==y_test)
