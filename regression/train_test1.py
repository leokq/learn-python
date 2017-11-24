import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

#加载数据
data = pd.read_csv('Advertising.csv')
dataset = np.array(data)
dataset_x = dataset[:,1:4]
dataset_y = dataset[:,4]

#划分训练样本集和测试样本集
x_train,x_test,y_train,y_test = train_test_split(dataset_x,dataset_y,test_size=0.2,random_state=1)

#训练模型
linreg = LinearRegression()
linreg_model = linreg.fit(x_train,y_train)

#测试
y_pred = linreg.predict(x_test)

a = (y_pred-y_test)**2
mse = np.sqrt(np.sum(a))
print('mse=',mse)

#绘图
plt.figure()
plt.plot(np.arange(len(y_pred)),y_test,'r',label = 'y_test')
plt.plot(np.arange(len(y_pred)),y_pred,'b',label = 'y_pred')
plt.legend(loc='best')
plt.grid()
plt.show()
