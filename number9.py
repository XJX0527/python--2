# -*- codeing = utf-8 -*-
# @Time:2021/8/19 17:27
# @Author:A20190277
# @File:number9.py
# @Software:PyCharm

#kd-tree：快速查找最邻近的点
#求点[2,2.5]到[0,2],[1,3],[2,4],[3,5]哪个点最近，并求出最小距离
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
p1=np.array([[0,2],[1,3],[2,4],[3,5]])
findp=np.array([2,2.5])
kt=spt.KDTree(p1)
d,p=kt.query(findp)
print('最近邻点距离:',d)
print('最近邻点坐标：',p)
plt.plot(p1[:,0],p1[:,1],'go')
x=2
y=2.5
plt.plot(x,y,'rp')
plt.plot([1,2],[3,2.5],'k--')
plt.grid()
plt.show()

#卷积：已知函数1，函数2，根据1，2得出函数3
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
fig=plt.figure(figsize=(12,4))
plt.subplot(131)
sig=np.repeat([0.,1.,0.],100)
plt.title('方波脉冲信号')
plt.plot(sig)
plt.subplot(132)
h_win=signal.hann(50)
plt.title('汉宁窗脉冲信号')
plt.plot(h_win)
plt.subplot(133)
f=signal.convolve(sig,h_win,mode='same')/sum(h_win) #mode指定函数的输出形式，method卷积方法默认为傅里叶变换
plt.title('卷积后的脉冲信号')
plt.plot(f)
fig.tight_layout()
plt.show()


#插值计算
#一维插值
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(19750105)
x=np.random.randn(10)
y=np.random.randn(10)
f=interpolate.interp1d(x,y)      #计算插值，返回f
xnew=np.linspace(x.min(),x.max(),80)  #设置最大最小值防止插值越界,共80个数
ynew=f(xnew)                          #根据x得出y
plt.plot(x,y,'o',xnew,ynew,'-')
plt.title('10个随机离散点的一维插值曲线图')   #插了80个数
plt.show()


#二维插值：当y的值由多个自变量决定时

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
def func(x,y):
    return x*(1-x)*np.cos(4*np.pi*x)*np.sin(4*np.pi*y*2)**2
grid_x,grid_y=np.mgrid[0:1:100j,0:1:200j]   #网格点坐标，两个100*200矩阵，
points=np.random.rand(1000,2)     #1000行，2列实际点坐标
values=func(points[:,0],points[:,1])  #获取实际点的值
from scipy.interpolate import griddata
grid_z0=griddata(points,values,(grid_x,grid_y),method='nearest')  #利用最近邻补法，进行插值填充
#point：数据点的坐标，values：数据值，(grid_x,grid_y)：插值数据点
grid_z1=griddata(points,values,(grid_x,grid_y),method='linear')   #线性插补法
grid_z2=griddata(points,values,(grid_x,grid_y),method='cubic')   #三次插补法
plt.subplot(221)
plt.imshow(func(grid_x,grid_y).T,origin='lower')
plt.plot(points[:,0],points[:,1],'k.',ms=1)
plt.title('原始网格数据')
plt.subplot(222)
plt.imshow(grid_z0.T,origin='lower')
plt.title('最近邻补法插值')
plt.subplot(223)
plt.imshow(grid_z1.T,origin='lower')
plt.title('线性插值法插值')
plt.subplot(224)
plt.imshow(grid_z2.T,origin='lower')
plt.title('三次插补法插值')
plt.gcf().set_size_inches(6,6)
plt.show()


#样条插值
#一维平滑样条插值
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(29292)
x=np.linspace(-5,5,20)
y=np.exp(-x**2)+0.01*np.random.randn(20)+1
spl=InterpolatedUnivariateSpline(x,y)   #插值处理
plt.plot(x,y,'rp')
xs=np.linspace(-5,5,100)
plt.plot(xs,spl(xs),'g')
plt.title('一维平滑样条插值')
plt.show()

#优化与拟合

#最小二乘法拟合
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
X=np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])    #共记录7天的数据，分布为每天的最高温度和最低温度
Y=np.array([7,16,10,19,12,20.5,16,25,17,27,18,27,20,31])
def P(x):
    a,b=x
    return Y-(a*X+b)    #误差值
o_line=least_squares(P,[1,0])  #指定初始ab值，进行最小二乘误差计算
#print(o_line)
y=(Y+o_line.fun)   #o_line.fun：函数返回值
plt.plot(X,Y,'o',label='温度实测点')
plt.plot(X,y,label='带残差值温度线')
plt.plot([0,6],o_line.x+11.5,'--',label='线性拟合')  #11.5:取第一天的最高最低温度中值
plt.legend()  #显示图例
plt.show()

#B-样条组合
def B(x,k,i,t):   #B-样条的数学公式
    if k==0:
        return 1.0 if t[i]<=x<t[i+1] else 0.0
    if t[i+k]==t[i]:
        c1=0
    else:
        c1=(x-t[i])/(t[i+k]-t[i])*B(x,k-1,i,t)
    if t[i+k+1]==t[i+1]:
        c2=0.0
    else:
        c2=(t[i+k+1]-x)/(t[i+k+1]-t[i+1])*B(x,k-1,i+1,t)
    return c1+c2
def bspline(x,t,c,k):   #简单的B-样条公式（性能一般）
    n=len(t)-k-1
    assert (n>k+1) and (len(c)>=n)
    return sum(c[i]*B(x,k,i,t) for i in range(n))
import numpy as np
from scipy.interpolate import BSpline
k=2  #指定阶数为2阶
x=np.linspace(0,np.pi,7)
t=x*1.8     #原始数据采样
c=[-1,2,0,-1]  #样条系数
spl=BSpline(t,c,k)   #t:多维数组，原始节点数，c:多维数组，样条系数，k:整数，样条阶数
spl(2.5)
bspline(2.5,t,c,k)   #自定义待评估B-样条曲线
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
fig,ax=plt.subplots()
xx=np.linspace(1.5,4.5,50)
ax.plot(xx,[bspline(x,t,c,k) for x in xx],'r--',label='自定义样条评估曲线')
ax.plot(xx,spl(xx),'b--',label='样条拟合曲线')
ax.grid(True)
ax.legend(loc='best')
plt.show()


#聚类
#K-Means算法：优点快速，缺点，该算法的前提是要知道有多少类
import numpy as np
from scipy.cluster.vq import vq,kmeans,whiten
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
#将随机产生的100个数变为3类数
datapoints=100
a=np.random.multivariate_normal([0,0],[[4,1],[1,4]],size=datapoints)   #生成正态分布数，[0,0]为均值，[[4,1],[1,4]]为cov(x,x)
b=np.random.multivariate_normal([30,10],[[10,2],[2,1]],size=datapoints)
c=np.random.multivariate_normal([20,2],[[8,2],[2,1]],size=datapoints)
features=np.concatenate((a,b,c))   #将a,b,c排列拼接，axis=0（按列）
whitened=whiten(features)          #白化数据处理
codebook,distortion=kmeans(whitened,3)  #指定3中心点的两类聚类计算
plt.scatter(whitened[:,0],whitened[:,1],c='b')  #白化后的样本数据点
plt.scatter(codebook[:,0],codebook[:,1],c='r') #根据均值计算后的3个类
# plt.text(0,-0.7,r'A聚类')
# plt.text(0,-0.7,r'B聚类')
# plt.text(0,-0.7,r'C聚类')
plt.title('K-Means聚类计算')
plt.show()


#分层聚类算法
#优点：可以自动推算聚类数，不需要提前指定，缺点：计算效率低
import numpy as np
from scipy.cluster.vq import vq,kmeans,whiten
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
#将随机产生的100个数变为3类数
datapoints=100
a=np.random.multivariate_normal([0,0],[[4,1],[1,4]],size=datapoints)   #生成正态分布数，[0,0]为均值，[[4,1],[1,4]]为cov(x,x)
b=np.random.multivariate_normal([30,10],[[10,2],[2,1]],size=datapoints)
c=np.random.multivariate_normal([20,2],[[8,2],[2,1]],size=datapoints)
features=np.concatenate((a,b,c))   #将a,b,c排列拼接，axis=0（按列）
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.spatial.distance import pdist
x=linkage(features,'weighted')    #对数据进行分层聚类处理
plt.scatter(x[:,0],x[:,1],c='b')  #绘制分层聚类的数据点图
plt.show()
fig=plt.figure(figsize=(20,8))
dn=dendrogram(x)   #将层次聚类绘制为树状图
plt.show()


#根据y值：y=np.array([9,8,7,5,4,3,3,2,1,6,7,8,9,10,11,12,14,17,19,21])，x值为默认序号值
#利用样条插值法连接个点
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
y=np.array([9,8,7,5,4,3,3,2,1,6,7,8,9,10,11,12,14,17,19,21])
x=np.arange(1,21,1)
spl=InterpolatedUnivariateSpline(x,y)
plt.plot(x,y,'rp')
xs=np.linspace(1,21,500)
plt.plot(xs,spl(xs),'g')
plt.title('一维平滑样条插值')
plt.show()



