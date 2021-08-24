# -*- codeing = utf-8 -*-
# @Time:2021/8/23 16:51
# @Author:A20190277
# @File:number12.py
# @Software:PyCharm

#pandas基于时间处理
import pandas as pd
import numpy as np
from datetime import datetime,date,time

np.random.seed(192220)
si=pd.period_range('2010',periods=10,freq='Y')  #生成时间序列，从2010开始，生成10年的，2019结束，为Y
year=[]
for d in si:
    year.append(np.repeat(d,100))       #每年为100个学生记录成绩，所以每年产生100个重复的年份
y1=np.array(year).reshape(1000)         #生成1000个时间一维数组
Score=pd.DataFrame(
    {'语文':np.random.randint(0,150,1000),'数学':np.random.randint(0,150,1000),'英语':np.random.randint(0,150,1000)},
    index=y1
)  #生成成绩，索引为年份

s1=Score['语文']+Score['数学'],Score['英语']
s2=pd.DataFrame({'总分':s1[1]})  #每个同学的总成绩，索引为成绩行对应的年份
s3=s2.T
s4=s3['2010'].T.sort_values(by='总分',ascending=False,axis=0)
s4['总分'][9]    #第十名同学的成绩为，当年录取分数线
Score_line={}
for y in s3:
    s4=s3[y].T.sort_values(by='总分',ascending=False,axis=0)
    Score_line[y]=s4['总分'][9]
print('历年分数线为：',Score_line)














