# 可视化尝试
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import timeit
from matplotlib.ticker import FuncFormatter
from adjustText import adjust_text
from scipy import stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import ticker 
import math
import timeit
start = timeit.default_timer()

# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False

mkdir('./output')
mkdir('./output/picture')
mkdir('./output/major_fail/picture')
mkdir('./output/site_gap/picture')
mkdir('./output/site_gap/major')
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

def read_sbin_csv():
    rowarr = search_first_row(ls, 'Pass__Default')
    siteNum_row = search_first_row(ls, 'SITE_NUM')
    # 总数
    Total,Pass,Fail = 0,0,0
    for index in range(len(ls)):
        test = pd.read_csv(ls[index], sep='\s+',names=['Name','Count'],skiprows=rowarr[index],nrows=3)
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
    major_fail = []
    for row_index,row in temp.iterrows():
        if(temp.loc[row_index,'percent'] >=  0.01 and temp.loc[row_index,'percent']<=0.5):
            # 找到__AllFaill 并删除 
            major_fail.append((temp.loc[row_index,'Name'])[:-9])
    print(major_fail)
    temp['percent'] = temp['percent'].apply(lambda x:format(x,'.2%'))
    temp.to_csv('./output/Total%s.csv'%Total)
    return major_fail

# 加个更新文件的判断，像服务器那样，判断文件时间戳，太久了就更新一下，不过这种文件一般都不会更新
data_startAtSiteNum = read_csv_data(ls, 'SITE_NUM')
data_startAtSiteNum.to_csv('./output/data_startAtSiteNum.csv')
# data_startAtSBin = read_sbin_csv()
# data_startAtSBin.to_csv('./output/data_startAtSBin.csv')


# 绘图
# 取数据
np.set_printoptions(threshold=20)
start = timeit.default_timer()
df = pd.read_csv('./output/data_startAtSiteNum.csv')

Unit = df.iloc[0]
df = df.apply(pd.to_numeric, errors='coerce').fillna(df.mean())
df = df.loc[:, ~df.columns.str.contains('Unnamed')]
df1 = df[:3]

LimitL = df1.iloc[1]
LimitU = df1.iloc[2]

df_bin1 = df[df.SOFT_BIN == 1]
df_bin1.to_csv('./output/bin1.csv')
df2 = df.astype(float)
df2 = df2[3:]

# site num
df_site1 = df_bin1[df_bin1.SITE_NUM == 1]
df_site2 = df_bin1[df_bin1.SITE_NUM == 2]
df_major_site1 = df2 [df2.SITE_NUM == 1]
df_major_site2 = df2 [df2.SITE_NUM == 2]


# 正态分布图

def normfun(x, mu, sigma):
    pdf = np.exp(-(x-mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    return pdf

def isNorm(data,mu,sigma):
    s,p = stats.kstest(data,'norm',(mu,sigma))
    print('p = %s'%p)
    if p > 0.05 :
        return 1
    else :
        return -1

def to_percent(y, position):
    return str(round(100*y, 2))+"%"  # 这里可以用round（）函数设置取几位小数

def cpk(USL,LSL,sigma,mu):
    cpu = abs(USL -mu ) / (3*sigma)
    cpl = abs(mu - LSL) / (3*sigma)
    return min(cpu,cpl)

def cp(USL,LSL,sigma):
    return (USL-LSL)/(6*sigma)

def x_axis_scale(max, min):
    # 设置一个合理的刻度 , 考虑负数和极小数，10-3 数量级的
    if(abs(max - min) < 10):
        temp = round((max-min))
        return abs(round((max-min), 3))
    return abs(round((max-min)/20, 0))

def plotOfMe(data, Name):
    result = data[Name]
    mu_all = data.mean()
    std_all = data.std()
    result_max = max(result)
    result_min = min(result)
    # 先统计，计算各种描述值
    mu = mu_all[Name]
    sigma = std_all[Name]
    L = LimitL[Name]
    U = LimitU[Name]
    Cpk = cpk(U,L,sigma,mu)
    Cp = cp(U,L,sigma)
    texts = []
    # x 设置 正态分布曲线的范围 , 符合正态分布再生成
    # if(isNorm(result,mu,sigma)):
    #     x = np.arange(result_min, result_max,0.001)
    #     y = normfun(x, mu, sigma)
    #     plt.plot(x, y)
    # 正态分布曲线有bug 需调整
    # 生成 正态分布曲线
    # 划分上下限区间
    # 权重修改
    weights = np.ones_like(result)/float(len(result))
    # 保存数据到ans中,完成图形自适应设置
    ans = plt.hist(result, 20, weights=weights,
                   rwidth=0.7, color='orange')
    # 坐标轴刻度  自适应, x轴即可
    # ax = plt.gca()
    # temp = x_axis_scale(result_max, result_min)
    # x_major_locator = MultipleLocator(temp)
    # ax.xaxis.set_major_locator(x_major_locator)
    # 标签数值
    for i in range(len(ans[0])):
        # 显示失效
        if((ans[1][i] < L or ans[1][i]> U )and ans[0][i] >= 0.01):
            datatostr = str(round(ans[0][i]*100, 3))+"%"
            texts.append(plt.text(ans[1][i], ans[0][i], datatostr,color='red'))
        # 显示多数
        elif(ans[0][i] >= 0.05):
            datatostr = str(round(ans[0][i]*100, 2))+"%"
            texts.append( plt.text(ans[1][i], ans[0][i], datatostr))           
    # y轴 设置为 % 显示 , 设置副刻度值和等比例输出图形
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    
    if(result_max-result_min != 0):
        major_ax = ax.get_xticks()
        # major_ax = pow(10,(math.floor(math.log10(abs(result_max-result_min)))))
        xminorLocator   = MultipleLocator(round((major_ax[1]-major_ax[0])/4,4))
        ax.xaxis.set_minor_locator(xminorLocator)
    # 文字重叠 adjust_text
    adjust_text(texts,lim=50,arrowprops=dict( arrowstyle='->', lw= 1, color='red'))
    # 网格线 , 坐标轴标签, 标题
    plt.grid()
    # plt.xlabel('Measure')
    # plt.ylabel('Percent')
    plt.title(Name+ ' , Unit : %s \n μ : %4.3f , sigma : %4.3f , 6sigma : (%4.3f,%4.3f) \n max : %4.3f , min : %4.3f \n LimitL : %.3f , LimitU : %.3f , Cp : %.3f , Cpk : %.3f' 
              % (Unit[Name],mu, sigma, mu-6*sigma, mu+6*sigma, result_max, result_min,L,U,Cp,Cpk),loc='left')


def generate_visible_data(data,start):
    # mkdir('./output/%s/picture'%data)
    for col in data.loc[:, start:]:
        plotOfMe(data, col)
        print('save picture :', './output/picture/%s.png' % col)
        plt.savefig('./output/picture/%s.png' % col, bbox_inches='tight')
        # plt.show()
        plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
        plt.close()  # Close a figure window

# plot site gap
def Statistical_parameters(data,Name):
    result = data[Name]
    mu_all = data.mean()
    std_all = data.std()
    result_max = max(result)
    result_min = min(result)
    # 先统计，计算各种描述值
    mu = mu_all[Name]
    sigma = std_all[Name]
    L = LimitL[Name]
    U = LimitU[Name]
    Cpk = cpk(U,L,sigma,mu)
    Cp = cp(U,L,sigma)
    return result,mu,sigma,result_max,result_min,L,U,Cpk,Cp
def plotOfsite(site1,site2, Name):
    result1 = site1[Name]
    result2 = site2[Name]
    # print(getattr(site1, Name))
    
    fail_rate1 = (len(site1[getattr(site1, Name) < LimitL[Name]][Name]) + len(site1[getattr(site1, Name) > LimitU[Name]][Name]) ) /len(site1[Name])
    fail_rate2 = (len(site2[getattr(site2, Name) < LimitL[Name]][Name]) + len(site2[getattr(site2, Name) > LimitU[Name]][Name]) ) /len(site2[Name])
    
    fail_rate1,fail_rate2 = str(round(100*fail_rate1, 2))+"%" , str(round(100*fail_rate2, 2))+"%"
    # y轴 设置为 % 显示
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    xminorLocator = ticker.AutoMinorLocator(4) 
    # major_ax = pow(10,(math.floor(math.log10(abs(max(result1))))))
    # xminorLocator   = MultipleLocator(round(major_ax/5,2))
    ax.xaxis.set_minor_locator(xminorLocator)

    # 划分上下限区间
    # 权重修改
    weights1 = np.ones_like(result1)/float(len(result1))
    weights2 = np.ones_like(result2)/float(len(result2))

    # 保存数据到ans中,完成图形自适应设置
    # plt.hist(result1, 100, weights=weights1,histtype='barstacked',
    #                rwidth=1, alpha= 0.6,label='SITE1')
    # plt.hist(result2, 100, weights=weights2,histtype='barstacked',
    #                rwidth=1, alpha= 0.6,label= 'SITE2' )
    plt.hist([result1,result2],20,weights=[weights1,weights2],label=['SITE1','SITE2'])
    
    # 网格线 , 坐标轴标签, 标题
    plt.grid()
    # plt.ylabel('Percent')
    plt.legend(loc='upper right')
    plt.title(Name + '\nFail_rate: SITE1 -> %s SITE2 ->%s ' %(fail_rate1,fail_rate2) )



generate_visible_data(df_bin1,'Pin10BST')
major_fail = read_sbin_csv();  


for col in major_fail:
    plotOfMe(df2, col)
    print('save picture :', './output/picture/%s.png' % col)
    plt.savefig('./output/major_fail/picture/%s.png' % col, bbox_inches='tight')
    # plt.show()
    plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    plt.close()  # Close a figure window

for col in df_site1:
    plotOfsite(df_site1,df_site2,col)
    print('save picture :', './output/site_gap/picture/%s.png' % col)
    plt.savefig('./output/site_gap/picture/%s.png' % col, bbox_inches='tight')
    # plt.show()
    plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    plt.close()  # Close a figure window


for col in major_fail:
    plotOfsite(df_major_site1,df_major_site2,col)
    print('save picture :', './output/site_gap/major/%s.png' % col)
    plt.savefig('./output/site_gap/major/%s.png' % col, bbox_inches='tight')
    # plt.show()
    plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    plt.close()  # Close a figure window




end = timeit.default_timer()
print('Running time: %s Seconds' % (end-start))
