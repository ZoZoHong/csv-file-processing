#Python绘制正态分布曲线
import numpy as np
import matplotlib.pyplot as plt

#正态分布的概率密度函数
def normpdf(x,mu,sigma):       
    pdf=np.exp(-(x-mu)**2/(2*sigma**2))/(sigma * np.sqrt(2 * np.pi))
    return pdf

mu,sigma=eval(input()) #mu:期望;sigma:标准差  
x= np.arange(mu-3*sigma,mu+3*sigma,0.01) #生成数据，步长越小，曲线越平滑
y=normpdf(x,mu,sigma)

#概率分布曲线
plt.plot(x,y,'g--',linewidth=2)
plt.title('Normal Distribution: mu = {:.2f}, sigma={:.2f}'.format(mu,sigma))
plt.vlines(mu, 0, normpdf(mu,mu,sigma), colors = "c", linestyles = "dotted")
plt.vlines(mu+sigma, 0, normpdf(mu+sigma,mu,sigma), colors = "y", linestyles = "dotted")
plt.vlines(mu-sigma, 0, normpdf(mu-sigma,mu,sigma), colors = "y", linestyles = "dotted")
plt.xticks ([mu-sigma,mu,mu+sigma],['μ-σ','μ','μ+σ'])

#输出
plt.grid()
plt.show()

