#coding:UTF-8  
''''' 
Created on 2015年5月19日 
 
@author: zhaozhiyong 
'''  
  
from numpy import *  
import matplotlib.pyplot as plt 

#  T={(1,2),(3,4),(1,2)}
    
def loadDataSet():
    """
    加载数据集
 
    :return:输入向量矩阵和输出向量
    """
    dataMat = []; labelMat = []
    fr = open('./test.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0])]) #X0设为1.0，构成拓充后的输入向量
        labelMat.append(int(lineArr[2]))
    #print(dataMat, labelMat)
    return dataMat,labelMat

# 计算p(x,y)经验分布和p(x)的经验分布
# 输入：训练数据集， 训练数据集类别， 
# 输出：训练数据中X=x的经验分布，训练数据中X=x,Y=y的经验分布
def distribution(dataMatIn, classLabels, x, y):
    dataMatrix = mat(dataMatIn)
    m,n = shape(dataMatrix)
    count1 = 0
    count2 = 0
    for i in range(m):
        if dataMatrix[i] == x:
            count1 += 1
        if classLabels[i] == y and dataMatrix[i] == x:
            count2 += 1
    px=count1/m
    py=count2/m
    return  px, py

# 求解无约束具体的优化问题
# min f(x)=100(x1**2-x2**2)+(x1-1)**2
#fun  
def fun(x):
    return 100 * (x[0,0] ** 2 - x[1,0]) ** 2 + (x[0,0] - 1) ** 2  
  
#gfun  求梯度
def gfun(x): 
    result = zeros((2, 1))  
    result[0, 0] = 400 * x[0,0] * (x[0,0] ** 2 - x[1,0]) + 2 * (x[0,0] - 1)  
    result[1, 0] = -200 * (x[0,0] ** 2 - x[1,0])  
    return result  


def dfp(fun, gfun, x0):  
    result = []  
    maxk = 500  
    rho = 0.55  
    sigma = 0.4  
    m = shape(x0)[0]  
    Hk = eye(m)  
    k = 0  
    while (k < maxk):  
        gk = mat(gfun(x0))                          # 计算梯度  
        dk = -mat(Hk)*gk                            # 计算搜索方向
        m = 0  
        mk = 0  
        while (m < 20):  
            newf = fun(x0 + rho ** m * dk)          # 使得f(xk + λk*pk)=min f(xk + λ*pk)
            oldf = fun(x0)  
            if (newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0,0]):  
                mk = m  
                break  
            m = m + 1  
          
        #DFP校正  
        # G(k+1)=Gk - (Gk*yk*yk.T*Gk)/(yk.T*Gk*yk) + (sk*sk.T)/(sk.T*yk)
        x = x0 + rho ** mk * dk  
        sk = x - x0  
        yk = gfun(x) - gk  
        if (sk.T * yk > 0):  
            Hk = Hk - (Hk * yk * yk.T * Hk) / (yk.T * Hk * yk) + (sk * sk.T) / (sk.T * yk)  
          
        k = k + 1  
        x0 = x  
        result.append(fun(x0))  
      
    return result  
'''    

#定义DFP方法
def dfp(x0, dataMatIn, classLabels):  
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    
    result = []  
    maxk = 500                          # 最大迭代次数
    lamda = 0.01                        # λ
    
    Gk = eye(m)                         # 生成对角矩阵,在实数范围内，对角矩阵相似正定矩阵
    k = 0  
    weights = ones((n,1))
    px, py= distribution(dataMatIn, classLabels, 0.828534, 0)
    
    for k in range(maxk):  
        
        f = px*log(exp(weights)) - py*weights   # 定义函数f(x)
        grad = mat(exp(m*px*weights)/exp(weights)*px*m)   #定义梯度
        f2 = px*log(exp(weights+lamda*grad) - py*(weights+lamda*grad))
        # 使得f(xk + λk*pk)=min f(xk + λ*pk)
        if ( f > f2):
            lamda += 0.01
        # 置x(k+1)=xk + λk*pk
        weights += lamda * grad            
        sk = lamda * grad
        gk = grad
        yk = f-gk
        # DFP校正  计算Gk+1
        # G(k+1)=Gk - (Gk*yk*yk.T*Gk)/(yk.T*Gk*yk) + (sk*sk.T)/(sk.T*yk)       
        if (sk.T * yk > 0):
            Gk = Gk - (Gk * yk * yk.T * Gk) / (yk.T * Gk * yk) + (sk * sk.T) / (sk.T * yk)
        k = k + 1  
        
        gk = mat(gfun(x0))              # 计算梯度  
        pk = -mat(Gk)*gk                # 计算搜索方向 pk
        i = 0  
        mk = 0                          # mk是满足下列不等式的最小非负整数m  
        
        while (i < 20):  
            newf = fun(x0 + delta ** i * pk)  #f(xk+δ**m*pk)
            oldf = fun(x0)  
            if (newf < oldf + sigma * (delta ** i) * (gk.T * pk)[0,0]):  
                mk = i  
                break  
            i = i + 1  
          
        # DFP校正  计算Gk+1
        # G(k+1)=Gk - (Gk*yk*yk.T*Gk)/(yk.T*Gk*yk) + (sk*sk.T)/(sk.T*yk)
        x = x0 + delta ** mk * pk  
        sk = x - x0  
        yk = gfun(x) - gk  
        if (sk.T * yk > 0):  
            Gk = Gk - (Gk * yk * yk.T * Gk) / (yk.T * Gk * yk) + (sk * sk.T) / (sk.T * yk)  
          
        k = k + 1  
        x0 = x  
        result.append(fun(x0))  
      
    return result 
   
    return weights
'''
if __name__ == '__main__': 
    x0 = mat([[-1.2], [1]])  
    result = dfp(fun, gfun, x0)  
    #dataMatIn, classLabels = loadDataSet()
    #weight = dfp(x0, dataMatIn, classLabels) 
    n = len(result)  
    ax = plt.figure().add_subplot(111)  
    x = arange(0, n, 1)  
    y = result  
    #y = weight
    ax.plot(x,y)  
  
    plt.show()  