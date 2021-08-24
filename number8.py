# -*- codeing = utf-8 -*-
# @Time:2021/8/19 15:00
# @Author:A20190277
# @File:number8.py
# @Software:PyCharm


from scipy import special
import scipy
import scipy.constants as cn
cn.golden   #黄金比例
cn.c        #真空光速
cn.degree   #角度圆对应弧度
cn.minute   #一分钟秒数
cn.day      #一天秒数
cn.inch     #一英寸米数
cn.light_year #一光年米数
#等等

#special：数学模块
#logit逻辑回归函数
import matplotlib.pyplot as plt
# from scipy.special import logit
import numpy as np
# x=np.linspace(0,1,40)
# y=logit(x)
# print(y)
# plt.plot(x,y,'mo')
# plt.grid()
# plt.show()

#立方根函数:y=cbrt(x*64)   :x的64次方

#stats模块
#抽取随机变量函数rvs
from scipy.stats import norm
x=np.linspace(0,1,1000)
y=norm.rvs(size=1000)
plt.plot(x,y,'mo')
plt.grid()
plt.show()
norm.rvs([2,3])
#概率密度函数pdf()
norm.pdf(20,20,10)   #正态连续随机变量在20处的概率密度值
norm.pdf(20,[20,10,30,10])



