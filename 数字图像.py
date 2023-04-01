import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


def powers():
    x = np.array([[1., 1.], [-1., 1.], [-1., -0.5], [-1., 1.5]])
    t = np.array([0, 1., 0, 1.]).T
    w = np.array([1., 1.])
    b = 0
    lr = 0.25
    loss = 0
    for i in range(100):
        y = np.where((np.dot(x, w.T) + b) > 0, 1, 0)
        m = x.shape[0]
        grad_w = -np.dot((t - y).T, x)*(1/m)
        grad_b = -np.mean((t - y))
        w = w - lr * grad_w
        b = b - lr * grad_b
        loss = np.mean(loss_function(t,y,x,w,b))
        print(f"epoch:{i}",loss)

    fig = plt.figure(figsize=(8, 8))  # 创建画布
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)  # 111 代表1行1列的第1个，subplot()可以用于绘制多个子图
    fig.add_axes(ax)  # 将绘图区对象添加到画布中

    # ----------2. 绘制带箭头的x-y坐标轴#通过set_visible方法设置绘图区所有坐标轴隐藏-------
    ax.axis[:].set_visible(False)  # 隐藏了四周的方框
    # ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"].set_axisline_style("->", size=1.0)  # 给x坐标轴加上箭头
    ax.axis["y"] = ax.new_floating_axis(1, 0)  # 添加y坐标轴，且加上箭头
    ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # 设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")

    # -----------3. 在带箭头的x-y坐标轴背景下，绘制函数图像#生成x步长为0.1的列表数据----------

    # 设置x、y坐标轴的范围
    plt.xlim(-5, 5)
    plt.ylim(-3, 3)
    # 绘制图形
    a = np.linspace(-4, 4, 10)
    print(w,b)
    plt.plot(a,(-w[0]*a-b)/w[1])
    #plt.plot(b,b*(np.dot(-w[0],x[0])/w[1]),linestyle='--')
    plt.plot(1, 1, c='b',marker=m)
    plt.plot(-1, 1, c='b',marker=m)
    plt.plot(-1, -0.5, c='b',marker=m)
    plt.plot(-1, 1.5, c='b',marker=m)
    plt.show()



def loss_function(t,y,x,w,b):
    loss_f = -(t-y)*(np.dot(x,w.T)+b)
    return loss_f

powers()
