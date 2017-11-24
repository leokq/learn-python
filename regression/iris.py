import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.linear_model import LogisticRegression
import sn

def iris_type(s):
    it = {b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return it[s]

#加载数据
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = urllib.request.urlopen(url)
dataset = np.loadtxt(data,delimiter=',',converters={4:iris_type})



dataset_x = dataset[:,:4]
dataset_y = dataset[:,4]

#为了可视化，只选前两个特征
dataset_x = dataset_x[:,:2]

#建立logistic模型
logr = LogisticRegression()
logr.fit(dataset_x,dataset_y)

N,M = 500,500
x1_min,x1_max = dataset_x[:,0].min(),dataset_x[:,0].max()
x2_min,x2_max = dataset_x[:,1].min(),dataset_x[:,1].max()
t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_max,M)
x1,x2 = np.meshgrid(t1,t2)
x_test = np.stack((x1.flat,x2.flat),axis=1)

y_hat = logr.predict(x_test)
y_hat = y_hat.reshape(x1.shape)

plt.pcolormesh(x1,x2,y_hat,cmap=plt.cm.prism)
plt.scatter(dataset_x[:,0],dataset_x[:,1],c=dataset_y,edgecolors='k',cmap=plt.cm.prism)

#样本显示
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.grid()
plt.show()



