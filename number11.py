# -*- codeing = utf-8 -*-
# @Time:2021/8/23 10:07
# @Author:A20190277
# @File:number11.py
# @Software:PyCharm

import numpy as np
import pandas as pd
df=pd.read_csv(r'')
#缺失值的判断
nan=df.isnull()  #缺失值的判断，与isna()等价，若values为缺失值则返回True，
nan=df.notnull() #缺失值的判断，与notna()等价，若values为缺失值则返回False

#缺失值的处理
new_df=df.fillna(value=100)  #替代缺失值
new_df=df.dropna()           #将存在缺失值的行进行删除处理
new_df=df.replace(to_replace=[pd.NaT,None,np.NAN],value=100)  #替换缺失值

#合并数据
full=pd.merge(df,new_df,on='age',how='inner')  #on：连接的关键字，how：连接的方法，默认为内连接（交集）
full=new_df.join(df)   #以行索引为关键进行行横向连接
full=pd.concat([new_df,df],axis=1)  #axis=1：行

#数据转置
new_df=df.pivot(index=['','',''],columns=['ID','Age'])  #默认为数据框原先的索引，但也可以自行进行修改
new_df=df.stack(level=0,dropna=True)  #从列转置为行#level：指定列索引层级的位置，dropna：默认为真，意思是转制后的数据行中若有缺失值进行删除处理
new_df=df.unstack(level=0,fill_value=100) #从行转置为列#fill_value=100：将缺失值用100填充
new_df=df.melt(id_vars=['',''],var_name='分类')  #局部转置，带新索引列进行转置，新索引列名称为‘分类'

#数据统计
ds=df.describe()  #描述性统计分析
ino=df.info()     #数据介绍

#数据分组
g1=df.groupby(level=1)  #按第一层分组
g1.sum()  #对分组结果，对数据进行求和运算

#数据聚合
ar=df.agg('sum')
ar=df.agg(np.mean)
ar=df.aggregate(['sum','mean'])
df.groupby(level=0).agg(['sum','mean'])  #先分组，再聚合

#分组转化
df.groupby(level=0).transform(lambda x:x-5)  #将value值均减5

#查找数据
df.filter(like='苗',axis=0)  #模糊查找#行索引中含有苗的字符串
df.filter(items=['',''])    #列索引为’‘或’‘
df.filter(regex='A\笔')     #列索引以笔为开头的

#数据可视化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1975)
ll=pd.DataFrame({'A':np.random.randn(200),'B':np.random.randn(200)})   #正态随机分布
l2=ll.cumsum()  #累计和
# l2.plot(color=['g','b'])
plt.show()
l2.plot(subplots=True,figsize=(6,6),color=['g','b'],marker='v',style='-')
plt.show()

#点图
l2.plot.scatter(x='A',y='B')
plt.show()

#条形图
l2[:8].plot.bar(x='A',y='B')  #取前8行值
plt.show()

#水平条形图
l2[:8].plot.barh(x='A',y='B')  #取前8行值
plt.show()

#直方图
l2.plot.hist()   #默认直方图为10个
plt.show()

#箱线图
l2[:8].plot.box()
plt.show()

#饼图
l3=np.abs(l2[:8])
l3.plot.pie(subplots=True)
plt.show()

#正则表达式
import re
df=pd.read_csv(r'C:\Users\18356\Desktop\软件学习\Python1.csv')

#简单分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
df=pd.read_csv(r'C:\Users\18356\Desktop\软件学习\Python1\resume.csv',index_col=False)
# print(df.head())
# print(df.columns)
# print(df.shape)
#数据过滤
new_df=df[(df.应聘职位=='数据分析')|(df.应聘职位=='系统运维')]
new_df['年龄']=new_df['年龄'].str.replace('岁','')   #将岁替换为空
new_df=new_df.sort_values('年龄')
print(new_df)
#对年龄进行归一化处理
row,col=new_df.shape
weight=np.zeros(row)  #转为数组，数组内有row个数均为0
print(weight)
first=np.arange(row)
i=0
mid=0
mid_i=row//2
if row%2==0:
    while i<mid_i:
        weight[mid_i-i-1]=first[row-i*2-1]
        weight[mid_i+i] = first[row - i * 2 - 2]
        i+=1
else:
    weight[mid_i]=row
    while i<mid_i:
        weight[mid_i-i-1]=first[row-i*2-2]
        weight[mid_i+i+1]=first[row-i*2-3]
        i+=1
print(weight)

t_w=weight/weight.sum()   #归一化处理
print(t_w)
new_df['年龄']=t_w
new_df.plot.bar(x='姓名',y='年龄',title='应聘者年龄优势比较图')
plt.show()


import pandas as pd
A=pd.DataFrame({'姓名':['三酷猫','加菲猫','凯蒂猫','机械猫'],'性别':['男','男','女','男'],'国籍':['中国','美国','日本','日本'],'年龄':[18,19,16,17]})
B=pd.DataFrame({'姓名':['三酷猫','加菲猫','凯蒂猫','机械猫'],'班级':[1,2,1,1],'总分':[290,270,280,260]})
print(A)
print(B)
C=pd.merge(A,B,on='姓名',how='right')
print(C)



