import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np

#载入数据
data = pd.read_csv('C:\\Program Files\\Python36\\regression\\Advertising.csv')
data_x1 = np.mat([np.ones(200),data['TV']])
data_x2 = np.mat([np.ones(200),data['radio']])
data_x3 = np.mat([np.ones(200),data['newspaper']])
data_y = np.mat(data['sales'])

weights1 = data_x1*data_x1.T
weights1 = weights1.I*data_x1*data_y.T
weights2 = data_x2*data_x2.T
weights2 = weights2.I*data_x2*data_y.T
weights3 = data_x3*data_x3.T
weights3 = weights3.I*data_x3*data_y.T

data1 = data.describe()

weights1 = np.array(weights1)
y_hat1 = weights1[0]+weights1[1]*np.arange(data1['TV']['max'])
weights2 = np.array(weights2)
y_hat2 = weights2[0]+weights2[1]*np.arange(data1['radio']['max'])
weights3 = np.array(weights3)
y_hat3 = weights3[0]+weights3[1]*np.arange(data1['newspaper']['max'])

plt.figure()
plt.subplot(3,1,1)
plt.plot(data['TV'],data['sales'],'ro',label='TV')
plt.plot(np.arange(data1['TV']['max']),y_hat1,'r')
plt.subplot(3,1,2)
plt.plot(data['radio'],data['sales'],'g*',label='radio')
plt.plot(np.arange(data1['radio']['max']),y_hat2,'g')
plt.subplot(3,1,3)
plt.plot(data['newspaper'],data['sales'],'b.',label='newspaper')
plt.plot(np.arange(data1['newspaper']['max']),y_hat3,'b')
plt.show()
