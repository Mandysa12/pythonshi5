import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mhumps(x):
    return abs(-1/((x-0.3)**2+0.01)+1/((x-0.9)**2+0.04)-6)


def achley(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 *
                                                                                    np.pi * x)+np.cos(2 * np.pi * y))) + np.exp(1) + 20


def rastrigin(x, y):
    return 20 + x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)


# plot mhumps
def df(x):
    return (mhumps(x + 1e-5) - mhumps(x)) / 1e-5
def opt(x0):
    max_iter = 100  # 最大迭代次数
    tol = 1e-6  # 收敛阈值

    x_old = x0
    for i in range(max_iter):
        y_temp=0
        x_new = x_old - mhumps(x_old) / df(x_old)
        if abs(x_new - x_old) < tol:
            print("迭代次数为：",i)
            break
        x_old = x_new
    return x_new


x_new=opt(0.5)
print("函数的极值点为：", opt(x_new))
plt.scatter(x_new, mhumps(x_new), color='red')
a = np.arange(-5, 5, 0.01)
b = mhumps(a)
plt.plot(a, b)
plt.show()

# plot achley function
def gradient(x, y):
    dx = 1e-6
    dy = 1e-6
    df_dx = (achley(x + dx, y) - achley(x - dx, y)) / (2 * dx)
    df_dy = (achley(x, y + dy) - achley(x, y - dy)) / (2 * dy)
    return df_dx, df_dy  # 由于y关于x的导数与x关于y的导数相同，所以这里直接使用df_dx


def hessian(x, y):
    dx = 1e-6
    dy = 1e-6
    d2f_dx2 = (gradient(x + dx, y)[0] - gradient(x - dx, y)[0]) / (2 * dx)
    d2f_dy2 = (gradient(x, y + dy)[1] - gradient(x, y - dy)[1]) / (2 * dy)
    d2f_dxdy = (gradient(x + dx, y)[1] - gradient(x - dx, y)[1]) / (2 * dx)
    return d2f_dx2, d2f_dy2, d2f_dxdy


def opt(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    for i in range(max_iter):
        df_dx, df_dy = gradient(x, y)
        d2f_dx2, d2f_dy2, d2f_dxdy = hessian(x, y)
        delta_x = -df_dx / d2f_dx2
        delta_y = -df_dy / d2f_dy2
        x += delta_x
        y += delta_y
        if abs(delta_x) < tol and abs(delta_y) < tol:
            print("迭代次数:", i)
            break
    return x, y


x0, y0 = 2, 2
x_opt, y_opt = opt(x0, y0)
print("最优解：", x_opt, y_opt)
r_min, r_max = -32.768, 32.768
xaxis = np.arange(r_min, r_max, 2.0)
yaxis = np.arange(r_min, r_max, 2.0)
x, y = np.meshgrid(xaxis, yaxis)
results = achley(x, y)
figure = plt.figure()
axis = figure.add_axes(Axes3D(figure))
axis.plot_surface(x, y, results, cmap='jet', shade="false", alpha=0.5)
axis.scatter(x_opt, y_opt, achley(x_opt, y_opt), color='red')
plt.show()
plt.contour(x, y, results)
plt.show()


# plot rastrigin function
def gradient(x, y):
    dx = 1e-6
    dy = 1e-6
    df_dx = (rastrigin(x + dx, y) - rastrigin(x - dx, y)) / (2 * dx)
    df_dy = (rastrigin(x, y + dy) - rastrigin(x, y - dy)) / (2 * dy)
    return df_dx, df_dy


def hessian(x, y):
    dx = 1e-6
    dy = 1e-6
    d2f_dx2 = (gradient(x + dx, y)[0] - gradient(x - dx, y)[0]) / (2 * dx)
    d2f_dy2 = (gradient(x, y + dy)[1] - gradient(x, y - dy)[1]) / (2 * dy)
    d2f_dxdy = (gradient(x + dx, y)[1] - gradient(x - dx, y)[1]) / (2 * dx)
    return d2f_dx2, d2f_dy2, d2f_dxdy


def opt(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    for i in range(max_iter):
        df_dx, df_dy = gradient(x, y)
        d2f_dx2, d2f_dy2, d2f_dxdy = hessian(x, y)
        delta_x = -df_dx / d2f_dx2
        delta_y = -df_dy / d2f_dy2
        x += delta_x
        y += delta_y
        if abs(delta_x) < tol and abs(delta_y) < tol:
            print("迭代次数:", i)
            break
    return x, y


x0, y0 = -5, 3
x_opt, y_opt = opt(x0, y0)
print("最优解：", x_opt, y_opt)

r_min, r_max = -5, 5
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
x, y = np.meshgrid(xaxis, yaxis)
results1 = rastrigin(x, y)
figure = plt.figure()
axis = figure.add_axes(Axes3D(figure))
axis.plot_surface(x, y, results1, cmap='jet', shade="false", alpha=0.5)
axis.scatter(x_opt, y_opt, rastrigin(x_opt, y_opt), color='black')
plt.show()
plt.contour(x, y, results1)
plt.show()
