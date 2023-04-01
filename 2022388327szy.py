import numpy as np

def test():
    # 初始化配置
    x = np.mat([[1, 1], 
                [-1, 1], 
                [-1, -0.5], 
                [-1, 1.5]])
    t = np.mat([[0], [1], [0], [1]])
    
    w = np.mat([1, 1])
    b = 0
    lr = 0.25 
    loss = 0
    
    for i in range(100):
        s = x.shape[0]
        y = np.where((np.dot(x, w.T) + b) > 0, 1, 0)
        
        gradw = - ((t - y).T * x) / s
        gradb = - (np.sum((t-y))) / s
        w = w - lr * gradw
        b = b - lr * gradb
        loss = (np.sum(-(t-y).T * (x * w.T + b))) / s
        
        print("训练轮数：{}, 损失值：{}, w = {}, b = {}".format(i, loss, w, b))
        if(loss == 0):
            break

test()