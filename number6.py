# -*- codeing = utf-8 -*-
# @Time:2021/8/18 20:16
# @Author:A20190277
# @File:number6.py
# @Software:PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.linspace(-np.pi,np.pi,100)
yc,ys=np.cos(x),np.sin(x)
plt.plot(x,yc)
plt.plot(x,ys)
plt.show()

#plot绘图：plot(*args,fmt,data,**kwargs)
#*args:坐标值.fmt：指定线条颜色、线性等等。data：数据。**kwargs：线型的属性，比如线宽等等。

plt.plot(10,10,'o')  #绘制点（10，10）用圆圈表示此点

x=np.array([0,13,14])
y=np.array([1,3,4])
plt.plot(x,y,'o')   #绘制点
plt.plot(x,y)       #绘制点之间的线
plt.show()

#text注释：x,y,s,fontdict,withdash,**kwargs
#x,y：所注释的范围。s：设置文本内容。fontdict：覆盖默认文本属性的字典（比如：字体的颜色，粗细，大小等等）。
#withdash：暂时不管。**kwargs：与fontdict用法相同
plt.text(-1.5,1.5,r'Blue Sin() Cos()',fontsize=20,fontweight='heavy')

#绘图中显示中文：fontproperties（设置中文属性）
from matplotlib.font_manager import FontProperties
x=np.linspace(-np.pi,np.pi,100)
yc,ys=np.cos(x),np.sin(x)
font=FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf',size=14)   #C:\Windows\Font电脑字体文件库，这里选取楷体
plt.xlabel('x轴 楷体',fontproperties=font)
plt.ylabel('y轴 楷体',fontproperties=font)
plt.title('绘制函数图 楷体',fontproperties=font)
plt.plot(x,yc)
plt.show()

#annotate标注：s,xy,*args,**kwargs
#s：注释信息，字符串类型。xy：注释箭头开始坐标。*args：注释文本左边坐标。**kwargs：箭头参数
plt.annotate('Top max',xy=(0,1),xytext=(1,1),arrowprops=dict(facecolor='m',shrink=0.01),fontsize=10)

#在一个面板上绘制多个图
plt.subplot(222)   #将面板分为（2*2），将该图绘制到2区域，？？？没有图？
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
x=np.linspace(-np.pi,np.pi,100)
yc,ys=np.cos(x),np.sin(x)
font=FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf',size=14)   #C:\Windows\Font电脑字体文件库，这里选取楷体
plt.annotate('Top max',xy=(0,1),xytext=(1,1),arrowprops=dict(facecolor='m',shrink=0.01),fontsize=10)
plt.xlabel('x轴 楷体',fontproperties=font)
plt.ylabel('y轴 楷体',fontproperties=font)
plt.title('绘制函数图 楷体',fontproperties=font)
plt.plot(x,yc)
plt.subplot(222)   #将面板分为（2*2），将该图绘制到2区域
plt.show()
'''

print('=================================================================')
#绘制直方图hist：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
d1=np.random.random(1000)
font=FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf',size=14)   #C:\Windows\Font电脑字体文件库，这里选取楷体
plt.xlabel('概率分布区间',fontproperties=font)
plt.ylabel('频率',fontproperties=font)
plt.title('频率分布直方图',fontproperties=font)
plt.hist(d1,bins=40,facecolor='blue',edgecolor='black',alpha=0.9)
#d1：数据。bins：条形数量。等等
plt.show()

#饼图pie：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
size=[48,21,18,13]   #占比
explode=[0.1,0,0,0]  #每块饼到圆心距离的设置
color=['orange','red','yellow','green']
label=['感冒','肚子痛','发烧','咽喉痛']
plt.pie(size,colors=color,explode=explode,labels=label)
plt.title(u'疾病分布饼状图')
plt.axis('equal')
plt.legend()
plt.show()

#图像处理，感兴趣的话可以看看

