import scipy.stats as sp
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./output.csv")
print (df)
mean=df.mean(axis=0,skipna=False)
std=df.std(axis=0,skipna=False)
print(mean['optimalDistanceBetweenEyes'])
print(std['optimalDistanceBetweenEyes'])

print("optimalDistanceBetweenEyes_min : {}".format(
    df['optimalDistanceBetweenEyes'].min()))
print("optimalDistanceBetweenEyes_max : {}".format(
    df['optimalDistanceBetweenEyes'].max()))

print("optimalConfidence_min : {}".format(
    df['optimalConfidence'].min()))
print("optimalConfidence_max : {}".format(
    df['optimalConfidence'].max()))

# rv=sp.norm(loc=mean,scale=std)
# x = np.arange(df['optimalDistanceBetweenEyes'].min(),
#               df['optimalDistanceBetweenEyes'].max(), 0.00000000000001)
# print(x.shape)
# print(x)
# y = rv.pdf(x)
# fig,ax=plt.subplots(1,1)
# ax.plot(x,y,'bo',ms=8,label='normal pdf')
# ax.vlines(x,0,y,colors='b',lw=5,alpha=0.5)
# ax.set_ylim([0,1.05])
# plt.show()

# rv = norm(loc=0, scale=1)  # 평균 0이고 표준편차 1인 정규분포 객체 만들기
# x = np.arange(-3, 3, 0.1)  # X 확률변수 범위
# print(x.shape)
# y = rv.pdf(x)  # X 범위에 따른 정규확률밀도값
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, y, 'bo', ms=8, label='normal pdf')
# ax.vlines(x, 0, y, colors='b', lw=5, alpha=0.5)  # 결과는
# ax.set_ylim([0, 1.05])  # y축 범위
# plt.show()

# rv = sp.norm(loc=mean, scale=std)
# xx = np.linspace(int(df['optimalDistanceBetweenEyes'].min()),int(df['optimalDistanceBetweenEyes'].max()), 1)
# pdf=rv.pdf(xx)
# plt.plot(xx,pdf)
# plt.show()

