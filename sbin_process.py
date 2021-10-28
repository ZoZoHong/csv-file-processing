# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

# from time import sleep
# from tqdm import tqdm

# for i in tqdm(range(1, 500)):
#     sleep(0.01)
import timeit
from pandas.core.arrays.sparse import dtype

from pandas.core.tools.numeric import to_numeric
start = timeit.default_timer()
# 获取文件地址
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root, file))
    return L
ls = file_name('./data')
# search_first_row 找到想要的起始行
def search_first_row(files, str):
    res = []
    i = 0
    for file in files:
        temp = pd.read_csv(file, sep='None')
        # 读行开头是否为未知的值
        for index, row in temp.iterrows():
            if(row.to_string().find(str) != -1):
                res.append(index)
                break
    return res
# 读数据
rowarr = search_first_row(ls, 'Pass__Default')
siteNum_row = search_first_row(ls, 'SITE_NUM')
# 总数
Total,Pass,Fail = 0,0,0
for index in range(len(ls)):
    test = pd.read_csv(ls[index], sep='\s+',names=['Name','Count'],skiprows=rowarr[0],nrows=3)
    Total += test.loc[0,'Count']
    Pass += test.loc[1,'Count']
    Fail += test.loc[2,'Count']
print(Total,Pass,Fail)
columns = ['Name','Count','percent','other']
temp = pd.read_csv(ls[0], sep='\s+',names=columns,skiprows=rowarr[0]+3,nrows=(siteNum_row[0]-rowarr[0]))
# 转换列数据格式
temp['Count'] = temp['Count'].apply(pd.to_numeric, errors='ignore')

# 数据相加
for index in range(1,len(ls)):
    temp_after = pd.read_csv(ls[index],sep='\s+',names=columns,skiprows=rowarr[index]+3,nrows=(siteNum_row[index]-rowarr[index]))
    temp_after['Count'] = temp_after['Count'].apply(pd.to_numeric)
    for row_index,row in temp_after.iterrows():
        temp.loc[row_index,'Count'] += row[1]

print(temp)

# 计算占比
temp['percent'] = temp['Count']/Total
temp['percent'] = temp['percent'].apply(lambda x:format(x,'.2%'))

temp.to_csv('./output/Total%s.csv'%Total)


# # 加个更新文件的判断，像服务器那样，判断文件时间戳，太久了就更新一下，不过这种文件一般都不会更新
# data_startAtSiteNum = read_csv_data(ls, 'SITE_NUM')
# data_startAtSiteNum.to_csv('./output/data_startAtSiteNum.csv')
# data_startAtSBin = read_csv_data(ls, 'SBin[1]')
# print(data_startAtSBin)
# data_startAtSBin.to_csv('./output/data_startAtSBin.csv')
end = timeit.default_timer()
print('Running time: %s Seconds' % (end-start))
