# 可视化尝试
# -*- coding: utf-8 -*-
from typing import NewType
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import timeit
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator, plot

np.set_printoptions(threshold=20)
start = timeit.default_timer()
df = pd.read_csv('./output/data_startAtSiteNum.csv')

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
df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

df_bin1 = df[df.SOFT_BIN == 1]
print(df_bin1)

df1 = df[:3]
Unit = df1.iloc[0]
LimitL = df1.iloc[1]
LimitU = df1.iloc[2]

df2 = df.astype(float)
df2 = df2[3:]


# 正态分布图

def normfun(x, mu, sigma):
    pdf = np.exp(-(x-mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    return pdf

def to_percent(y, position):
    return str(round(100*y, 2))+"%"  # 这里可以用round（）函数设置取几位小数


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

    # x 设置 正态分布曲线的范围
    x = np.arange(result_min, result_max)
    y = normfun(x, mu, sigma)
    # y轴 设置为 % 显示
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    # plt.plot(x, y)
    # 正态分布曲线有bug 需调整
    # 生成 正态分布曲线
    # 划分上下限区间
    # 权重修改
    weights = np.ones_like(result)/float(len(result))
    # 保存数据到ans中,完成图形自适应设置
    ans = plt.hist(result, 100, weights=weights,
                   rwidth=1, color='lightskyblue')
    # 坐标轴刻度  自适应, x轴即可
    # ax = plt.gca()
    # temp = x_axis_scale(result_max, result_min)
    # x_major_locator = MultipleLocator(temp)
    # ax.xaxis.set_major_locator(x_major_locator)
    # 标签数值
    count = 0
    for i in range(len(ans[0])):
        count += ans[0][i]
        if(ans[0][i] >= 0.05):
            datatostr = str(round(ans[0][i]*100, 3))+"%"
            plt.text(ans[1][i], ans[0][i], datatostr)
    # 网格线 , 坐标轴标签, 标题
    plt.grid()
    plt.xlabel('Measure')
    plt.ylabel('Percent')
    # plt.axvline(LimitL[Name])
    # plt.axvline(LimitU[Name])
    plt.title(Name + '\n μ : %4.3f , sigma : %4.3f , 6sigma : (%4.3f,%4.3f) \n max : %4.3f , min : %4.3f'
              % (mu, sigma, mu-6*sigma, mu+6*sigma, result_max, result_min))


def generate_visible_data(data,start):
    for col in data.loc[:, start:]:
        plotOfMe(data, col)
        print('save picture :', './output/picture/%s.png' % col)
        plt.savefig('./output/picture/%s.png' % col, bbox_inches='tight')
        # plt.show()
        plt.clf()   # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
        plt.close()  # Close a figure window
    
generate_visible_data(df_bin1,'Pin10BST')

# plotOfMe(df2, 'Ilk_SW_L')
# # plotOfMe(df2, 'Ilk_SW_H')
# plt.show()
# plt.savefig('./output/picture/%s.png' % 'Ilk_SW_L')

# plotOfMe(df2, 'Ilk_SW_H')
# plt.show()
# plt.savefig('./output/picture/%s.png' % 'Ilk_SW_L')
# # x = np.arange(min['Ilk_SW_L'], max['Ilk_SW_L'],
# #               (min['Ilk_SW_L']-min['Ilk_SW_L'])/100)
# x = np.arange(23, 35, (35-23)/100)

# # 设定y轴，载入正态分布函数
# # print(mu['Ilk_SW_L'], std['Ilk_SW_L'])
# y = normfun(x, result.mean(), result.std())
# plt.plot(x, y)
# # bins个柱状图,宽度是rwidth(0~1),=1没有缝隙
# plt.hist(result, bins=10, rwidth=0.5, density=True)
# plt.title('distribution')
# plt.xlabel('Measure')
# plt.ylabel('Count')
# plt.xlim(23, 35)
# plt.ylim()


end = timeit.default_timer()
print('Running time: %s Seconds' % (end-start))
