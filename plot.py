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

print('\n Start... \n')
np.set_printoptions(threshold=20)
start = timeit.default_timer()
df = pd.read_csv('./output/data_startAtSiteNum.csv')
major_fail = ['ReEfuseOK','LS_LKG_12V','ICS_final','Efuse31_16']

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

mkdir("./output")
mkdir("./output/picture")    

# 取数据
Unit = df.iloc[0]
df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
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

    # y轴 设置为 % 显示
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)

    # 划分上下限区间
    # 权重修改
    weights = np.ones_like(result)/float(len(result))
    # 保存数据到ans中,完成图形自适应设置
    ans = plt.hist(result, 100, weights=weights,
                   rwidth=0.7, color='orange')
    # 坐标轴刻度  自适应, x轴即可
    # ax = plt.gca()
    # temp = x_axis_scale(result_max, result_min)
    # x_major_locator = MultipleLocator(temp)
    # ax.xaxis.set_major_locator(x_major_locator)
    # 标签数值
    for i in range(len(ans[0])):
        # 显示多数
        if(ans[0][i] >= 0.05):
            datatostr = str(round(ans[0][i]*100, 2))+"%"
            texts.append( plt.text(ans[1][i], ans[0][i], datatostr))
        # 显示失效
        if((ans[1][i] < L or ans[1][i]> U )and ans[0][i] >= 0.01):
            datatostr = str(round(ans[0][i]*100, 3))+"%"
            texts.append(plt.text(ans[1][i], ans[0][i], datatostr,color='red'))
        # 文字重叠 adjust_text
    adjust_text(texts,lim=50,arrowprops=dict( arrowstyle='->', lw= 1, color='red'))
    # 网格线 , 坐标轴标签, 标题
    plt.grid()
    plt.xlabel('Measure')
    plt.ylabel('Percent')
    plt.title(Name+ ' , Unit : %s \n μ : %4.3f , sigma : %4.3f , 6sigma : (%4.3f,%4.3f) \n max : %4.3f , min : %4.3f \n LimitL : %.3f , LimitU : %.3f , Cp : %.3f , Cpk : %.3f' 
              % (Unit[Name],mu, sigma, mu-6*sigma, mu+6*sigma, result_max, result_min,L,U,Cp,Cpk))


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
    # result1,mu1,sigma1,result_max1,result_min1,L1,U1,Cpk1,Cp1 = Statistical_parameters(site1,Name)
    # result2,mu2,sigma2,result_max2,result_min2,L2,U2,Cpk2,Cp2 = Statistical_parameters(site2,Name)
    result1 = site1[Name]
    result2 = site2[Name]
    # y轴 设置为 % 显示
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)

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
    plt.xlabel('Measure')
    plt.ylabel('Percent')
    plt.legend(loc='upper right')
    plt.title(Name)


   
generate_visible_data(df_bin1,'Pin10BST')

# mkdir('./output/major_fail/picture')
# for col in major_fail:
#     plotOfMe(df2, col)
#     print('save picture :', './output/picture/%s.png' % col)
#     plt.savefig('./output/major_fail/picture/%s.png' % col, bbox_inches='tight')
#     # plt.show()
#     plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
#     plt.close()  # Close a figure window

mkdir('./output/site_gap/picture')
for col in df_site1:
    plotOfsite(df_site1,df_site2,col)
    print('save picture :', './output/site_gap/picture/%s.png' % col)
    plt.savefig('./output/site_gap/picture/%s.png' % col, bbox_inches='tight')
    # plt.show()
    plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    plt.close()  # Close a figure window

end = timeit.default_timer()
print('Running time: %s Seconds' % (end-start))
