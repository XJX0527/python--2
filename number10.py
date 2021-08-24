# -*- codeing = utf-8 -*-
# @Time:2021/8/22 16:04
# @Author:A20190277
# @File:number10.py
# @Software:PyCharm

import pandas as pd
import numpy as np
#pandas数据类型分为Series与DataFrame两类
s1=pd.Series(np.array([1,10,101,1001]))   #默认索引：0，1，...
s1.index   #索引范围
s1.values  #值
s2=pd.Series(np.array([1,0,5,7]),index=['one','two','three','four'])  #指定索引
s3=pd.Series({'Tom':34,'Aol':56,'Woe':90})
s4=pd.Series(np.ones(3),dtype=bool,name='No')   #将数值转化为布尔值，索引名称为No

#查询数据方法依靠索引值
#删除数据依靠索引删除
data=np.array([[18,19,20],[2,3,4],[32,45,67]])   #默认索引
A=pd.DataFrame(data)

data={'Tom':[100,88,99],'Join':[21,45,98],'Alic':[34,67,99]}  #列索引默认，行索引命名
pd.DataFrame(data)

s1=pd.Series(np.array(['Tom','Join','Alic','Jack']))
s2=pd.Series(np.array([1,10,101,1001]))
dic={'Name':s1,'Room No':s2}
B=pd.DataFrame(dic)

C=pd.DataFrame(np.arange(9).reshape(3,3))

Score=np.array([[95,100,99],[90,88,100],[85,45,20]])
pd.DataFrame(Score,columns=['数学','语文','英语'],index=['Tom','Join','Jack'])

#根据索引进行查询，删除，修改，增加（append,insert）

#读写数据
# df=pd.read_csv(r'C:\Users\18356\pythoxjx\Online_Retail.csv')
# df.to_csv(r'C:\Users\18356\pythoxjx\new_xjx.csv',index=False)
import sqlite3
conn=sqlite3.connect('First.db')
data.to_sql('test',conn,if_exists='replace')
conn.close()

# #利用pycharm与mysql链接从而建表的总体思路
# conn=sqlite3.connect("test.db")    #打开或者创建数据库文件
# print("打开数据库")
# c=conn.cursor()     #获取游标
# sql=""     #输入字符串
# c.execute(sql)  #执行sql语句
# conn.commit()   #提交数据操作
# conn.close()    #关闭数据库链接
# print("建表成功")

#1、利用pycharm与mysql链接从而建表
conn=sqlite3.connect("test.db")
print("打开数据库")
c=conn.cursor()
sql='''
    create table company
        (id int primary key not null,
        name text not null,
        age int not null,
        address varchar(50) not null,
        salary real);
'''
c.execute(sql)
conn.commit()
conn.close()
print("建表成功")

#
#2、插入数据
conn=sqlite3.connect("test.db")
print("打开数据库")
c=conn.cursor()
sql1='''
   insert into company(id,name,age,address,salary)
   values(1,'张三',32,'成都',80000);
'''
sql2='''
   insert into company(id,name,age,address,salary)
   values(2,'张',32,'成',8000);
'''
c.execute(sql1)
c.execute(sql2)
conn.commit()
conn.close()
print("建表成功")

#3、查询数据
conn=sqlite3.connect("test.db")
print("打开数据库")
c=conn.cursor()
sql="select*from company"
cursor=c.execute(sql)
for row in cursor:
    print("id=", row[0])
    print("name=", row[1])
    print("address=", row[2])
    print("salary=", row[3],"\n")
conn.close()
print("建表成功")



