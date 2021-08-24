# -*- codeing = utf-8 -*-
# @Time:2021/8/12 17:23
# @Author:A20190277
# @File:number 3.py
# @Software:PyCharm

#数组的合并
import numpy as np
student1=np.array([['1','加菲猫','男'],['2','凯蒂猫','女'],['3','波斯猫','男']])
print(student1)
score1=np.array([[1,66,99,100],[2,100,99,100],[3,99,100,100]])
print(score1)

full=np.append(student1,score1,axis=1)
print(full)
table_title=np.array(['序号','名字','性别','语文分数','数学分数','英语分数'])
arr=np.delete(full,3,axis=1)
new_full=np.insert(arr,[0,],table_title,axis=0)
print(arr)
print(new_full)

#种树问题
#长100米，宽5米，每隔5米种一棵树。
# 一半树苗5元一棵，一半树苗10元一棵。
#用数组表示种植树苗的间隔点米数；在上面的基础上记录树苗的单价，一棵低价，一棵高价，依次间隔；
#当天树苗每棵涨价2倍，求每棵树苗的价格；高价树苗记为1，低价记为0；
#实现对树苗和间隔米数的对齐。
trees=np.linspace(0,100,21)  #生成序列：0开始，100：结束，21个数
print(trees)
price=np.full((21),10)       #给每棵树附加为10，full(shape, fill_value, dtype, order)
i=0
while i<21:
    if i%2==0:
        price[i]=5
    i+=1
print(price)             #一半5元，一半10元
p1=price*2   #涨价2倍
p1[p1==20]=1
p1[p1==10]=0
print(p1)
print('------')
trees1=np.vstack((price,trees))   #np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组，数组类型相同
print(trees1)
print('-------------------------------------------------------------------------------')
#班级成绩分析
#用随机函数生成40名学生的成绩单，语文，数学，英语（成绩为整数，分数不低于55，成绩由高到低进行排序）
#取成绩最低的20%确定分数线，并获取成绩单
#对全班的分数按照语文、数学、英语进行最低分数和最高分数以及平均分进行统计，并写前三名的分数

score=np.random.randint(55,101,(3,40))
score.sort(axis=1)
print(score)
p20=np.percentile(score,20,axis=1)   #20%分位数
print(p20)
s1=score[0]
y1=s1[s1<p20[0]]
print(y1)                      #生成语文的20%成绩单
s2=score[1]
y2=s2[s2<p20[1]]
print(y2)                      #生成数学的20%成绩单
s3=score[2]
y3=s3[s3<p20[2]]
print(y3)                      #生成英语的20%成绩单

maxv=np.max(score,axis=1)   #以行为单位进行统计（axis=1）
print(maxv)
minv=np.min(score,axis=1)
print(minv)
meanv=np.mean(score,axis=1)
print(meanv)
sumv=score[0]+score[1]+score[2]   #每科目成绩进行累加
sumv.sort
print(sumv)

sw=sumv[-3:]  #取前三名成绩
print(sw)

#numpy可以较好地统计数值型数组，pandas才能更好的处理字符串混合型数组

#每天进货100支雪糕，成本价1元一支
#卖雪糕时，每天卖出50%时，每支1.2元，卖出支数大于50%时，每支1.3元
#求，卖出20支，18，36，26支时的利润
ice=np.arange(1,101,1)
price=np.full((100),1)
ice=np.arange(1,101,1)
ice=np.where(ice<=50,1.2,1.3)
i=0
sum1=0
while i<20:
    sum1=ice[i]+sum1
    i+=1
print(round(sum1)-20)

i=0
sum2=0
while i<(20+18):
    sum2=ice[i]+sum2
    i+=1
print(round(sum2)-18-round(sum1))

#对36，26同理

#数字图像可以用二维数组表示，0：空白，1：数字形状
# 000000
# 001010
# 010100
# 011110
# 000100
# 000100
#该数字图像表示4
#将上述0，1用二维数组表示，统计每列数字（每行数字）的和

score=np.random.randint(0,2,(5,6))
sum=np.sum(score,axis=1)   #每行
sum=np.sum(score,axis=0)   #每列


'''
np.mgrid[start：end：step]
开始坐标，结束坐标，步长（等分）
np.mgrid[-5:5:5j]
5j：5个点
步长为复数表示点数，左闭右闭
步长为实数表示间隔，左闭右开


>>> import numpy as np
>>> x=np.mgrid[-5:5:5j]
>>> x
array([-5. , -2.5,  0. ,  2.5,  5. ])

>>> import numpy as np
>>> x,y=np.mgrid[-5:5:3j,-2:2:3j]  #两个x*y矩阵
>>> x
array([[-5., -5., -5.],
       [ 0.,  0.,  0.],
       [ 5.,  5.,  5.]])
>>> y
array([[-2.,  0.,  2.],
       [-2.,  0.,  2.],
       [-2.,  0.,  2.]])
'''