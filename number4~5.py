# -*- codeing = utf-8 -*-
# @Time:2021/8/18 16:54
# @Author:A20190277
# @File:number4.py
# @Software:PyCharm

import pandas as pd
import numpy as np

#求三维空间面积
#矩阵可以实现二维空间，三维空间的点坐标的表示。（空间位置标定，图片处理方面有重大应用）
#利用矩阵求三维空间任意三角形的面积
#已知点A（1，1，1），点B（3，1.5，1.5），C（4，2，2）
A=np.array([1,1,1])
B=np.array([3,1.5,1.5])
C=np.array([4,2,2])
#求三边长
a=np.sqrt(np.sum(np.square(B-C)))
b=np.sqrt(np.sum(np.square(A-C)))
c=np.sqrt(np.sum(np.square(A-B)))
#求面积（海伦公式）
s=(a+b+c)/2
area=np.sqrt(s*(s-a)*(s-b)*(s-c))
print(area)

#求某个矩阵的特征值，特征向量
A=np.matrix([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 4]])
print(A)
w,v=np.linalg.eig(A)
print('特征值',w)
print('特征向量',v)

#绘制围棋格子
#围棋：18*18=324，利用函数计算所有格子坐标。利用二维数组的形式输出所有坐标


#将数据保存到表格中csv文件
#data.to_csv(r'C:\Users\18356\pythoxjx\new_xjx.csv',index=False)

#广播计算
#将矩阵B转为广播计算规则行事，建立数组，计算A-B
A=np.matrix([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
B=np.matrix([[1],
       [1]])
C=np.matrix([[0,0],[0,0]])
D=np.matrix([[0,0,0]])
#广播原理：处理不同形状数组之间的运算法则，提高了数组的计算速度
B=np.append(B,C,axis=1)
B=np.append(B,D,axis=0)
print(A-B)


data='I am from China. My name is Tom.Alias is “Three cool cat.I like Programming.Lift is short ,I select Python.'

# fh=open('C:/Users/18356/Desktop/Kaggle/AWEE.txt','w', encoding='utf-8')
# fh.write(data)
# fh.close()

f=open("C:/Users/18356/Desktop/Kaggle/AWEE.txt","r")   #C:\Users\18356\python_study1~5
content=f.readlines()
print(content)
f.close()

m=data.count('a')
print(m)

