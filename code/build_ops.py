import numpy as np
import pandas as pd
import random
import math

#获取N个建筑的静态属性值
def random_blds(bld_num):
    #设置随机种子，当种子一样时，保证每次random的结果是一样的
    random.seed(2000)
    #创建存储各建筑面积、高度、层数、设定温度、商业建筑类型的list
    area_A = []
    hight_h = []
    layers_M = []
    T_set = []
    build_type = []
    #获取面积、高度、层数、设定温度、商业建筑类型的值
    for i in range(bld_num):
        ty = random.randrange(1,4) 
        if ty == 1:  #写字办公楼
            A = random.randrange(3000,4001)
            h = 4
            M = random.randrange(60,100) 
        elif ty == 2:  #商业建筑
            A = random.randrange(8000,16001)
            h = 6
            M = random.randrange(12,24)
        else:   #酒店
            A = random.randrange(3000,4001)
            h = 3
            M = random.randrange(30,100)
        #建筑的设定温度范围：20~23度，精度为0.5
        T = random.randrange(40,47) / 2  
        #把建筑的各项属性存进list
        area_A.append(A)
        hight_h.append(h)
        layers_M.append(M)
        T_set.append(T)  
        build_type.append(ty)
        
    #list变成nparray，便于矩阵计算   
    area_A = np.array(area_A)
    hight_h = np.array(hight_h)
    layers_M = np.array(layers_M)
    T_set =np.array(T_set)
    build_type = np.array(build_type) 
    #计算建筑的体积V(m³)、传热表面积S(㎡)
    volume_V = area_A * hight_h * layers_M
    a = np.array([math.sqrt(i) for i in area_A])
    surface_area_S = 2 * area_A + 4 * a * hight_h * layers_M
    #返回tuple
    return build_type, area_A, hight_h, layers_M, T_set, volume_V, surface_area_S

#根据建筑类型生成人流随机分布
def random_ee(seed0, build_type):
    #获取建筑数量
    num = len(build_type)
    #设置随机种子，当种子一样时，保证每次random的结果是一样的
    random.seed(seed0)
    #创建存储各建筑一天里每分钟换气量的list
    bld_ee = []
    
    #给定几种典型的小时级人流分布采样均值(数据来源：高德)
    #腾讯大厦
    ee1 = [0,0,0,0,0,0.06,0.12,0.3,0.6,0.63,0.71,0.88,0.62,0.65,0.6,0.72,0.84,0.91,0.6,0.38,0.17,0.08,0.05,0] 
    #珠海西南大厦
    ee2 = [0,0,0,0,0,0.11,0.12,0.28,0.41,0.63,0.71,0.9,0.86,0.57,0.4,0.51,0.6,0.86,1,1.05,0.72,0.51,0.32,0]
    #横琴国贸大厦
    ee3 = [0,0,0,0,0,0.04,0.12,0.36,0.51,0.49,0.47,0.45,0.38,0.36,0.58,0.53,0.5,0.35,0.21,0.15,0.14,0.11,0.08,0]
    #珠海站
    ee4 = [0.15,0.03,0.01,0,0.02,0.15,0.3,0.45,0.6,0.48,0.45,0.47,0.62,0.58,0.66,0.98,0.82,0.65,0.6,0.54,0.55,0.28,0.18,0.06]
    #横琴中央美食汇广场
    ee5 = [0,0,0,0,0,0.12,0.23,0.35,0.46,0.51,0.63,0.92,0.71,0.59,0.56,0.60,0.64,0.83,0.81,0.76,0.64,0.42,0.27,0.13]
    #梧桐树大厦
    ee6 = [0,0,0,0,0,0.06,0.16,0.36,0.42,0.46,0.45,0.41,0.38,0.36,0.35,0.51,0.50,0.46,0.35,0.18,0.14,0.10,0.08,0.06]
    #酒店
    ee7 = [0,0,0,0,0,0.05,0.12,0.34,0.32,0.54,0.42,0.34,0.52,0.68,0.64,0.34,0.39,0.36,0.49,0.54,0.60,0.4,0.21,0.12]
    ee = [ee1, ee2, ee3, ee4, ee5, ee6, ee7]
    
    #确定每栋建筑采用的人流分布类型
    for i in range(num):
        if build_type[i] == 1:  #写字办公楼
           k  = random.randrange(0,3) 
        elif build_type[i] == 2:  #商业建筑
           k  = random.randrange(3,5)
        else: #酒店
           k  = random.randrange(5,7)
        #把每栋建筑的24小时人流分布存进list   
        bld_ee.append(ee[k])
        
    #变成nparray方便插入数据 
    bld_ee = np.array(bld_ee)   
    #把每个建筑人流量24个小时扩展成1440个分钟点
    for i in range(24):
        add = [bld_ee[:,i*60]] * 59
        bld_ee = np.insert(bld_ee, i*60, values=add, axis=1)   
    #转置变成每分钟每个建筑对应的ee值
    bld_ee = np.transpose(bld_ee)
    
    #在每个小时内，每15分钟随机震荡一次，振幅20%
    N = 1/5
    for i in range(24):
        #随机k值，表示以均值为基准，向上or向下震荡
        k = random.randrange(0,2)     
        change1 = ((-1)**k) * random.random() * N * bld_ee[i*60]  #乘上N表示以当前小时总量的N倍幅度波动 
        #0~15分钟赋相同的值
        for j in range(1,16):
            bld_ee[i*60+j] = bld_ee[i*60] + change1      
        #15~30分钟赋相同的值，上下通过'k'值体现，与0~15分钟相反
        change2 = ((-1)**(1-k)) * random.random() * N * bld_ee[i*60]
        for j in range(16,31):
            bld_ee[i*60+j] = bld_ee[i*60+15] + change2
        #重新随机k值
        k = random.randrange(0,2)
        #30~45分钟赋相同的值
        change3 = ((-1)**k) * random.random() * N * bld_ee[i*60]
        for j in range(31,46):
            bld_ee[i*60+j] = bld_ee[i*60+30] + change3
        #45~59分钟赋值赋相同的值，上下通过'k'值体现，与30~45分钟相反.
        #因为第60分钟要作为下一次的震荡均值，因此暂时不能覆盖
        change4 = ((-1)**(1-k)) * random.random() * N * bld_ee[i*60]
        for j in range(46,60):
            bld_ee[i*60+j] = bld_ee[i*60+45] + change4
    #非临界点全部订正完，处理每小时临界点的跳变，取上一个小时最后一分钟的值
    for i in range(1,24):
        bld_ee[i*60] = bld_ee[i*60-1]
        
            
    # #把数据写到Excel里面去
    # data = pd.DataFrame(bld_ee)
    # #写入Excel文件
    # writer = pd.ExcelWriter(excel_name)		
    # # ‘ee’是写入excel的sheet名
    # data.to_excel(writer, 'ee', float_format='%.4f')		
    # writer.save()
    # writer.close()
        
    return bld_ee
        
        



