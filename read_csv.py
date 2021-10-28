# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

# from time import sleep
# from tqdm import tqdm

# for i in tqdm(range(1, 500)):
#     sleep(0.01)
import timeit
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
def read_csv_data(files, str):
    rowarr = search_first_row(files, str)
    data = pd.DataFrame()
    siteNum_row = search_first_row(files, 'SITE_NUM')
    for index, file in enumerate(files):
        if(str == 'SITE_NUM'):
            temp = pd.read_csv(file, skiprows=rowarr[index]-1+5, index_col=0)
            if(index == 0):
                data = data.append(temp)
            else:
                data = data.append(temp[3:])
        else:
            temp = pd.read_csv(
                file, sep='None', skiprows=rowarr[index]-1, nrows=(siteNum_row[index]-rowarr[index]-1))
            data = pd.merge(data, temp, left_index=True,
                            right_index=True, how="outer")
    return data
# 加个更新文件的判断，像服务器那样，判断文件时间戳，太久了就更新一下，不过这种文件一般都不会更新
data_startAtSiteNum = read_csv_data(ls, 'SITE_NUM')
data_startAtSiteNum.to_csv('./output/data_startAtSiteNum.csv')
data_startAtSBin = read_csv_data(ls, 'SBin')
data_startAtSBin.to_csv('./output/data_startAtSBin.csv')
end = timeit.default_timer()
print('Running time: %s Seconds' % (end-start))
