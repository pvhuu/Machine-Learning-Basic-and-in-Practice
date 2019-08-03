from __future__ import division,print_function,unicode_literals
import numpy as np
import math as m
import matplotlib.pyplot as plt


'''
1.grad() : Tính đạo hàm
2.cost():Tính giá trị hàm số.Dùng để kiểm tra việc tính đạo hàm có đúng không hoặc xem giá trị 
của hàm số có giảm theo mỗi vòng lặp hay không
3.myGD1():Thực hiện thuật toán Gradient Descent .Đầu vào là learning rate và điểm bắt đầu.Thuật 
toán dừng lại khi đạo hàm có độ lớn đủ nhỏ
'''

def grad(x):
    return 2*x + 5*np.cos(x)


def cost(x):
    return x**2 + 5*np.sin(x)


def myGD1(eta,x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return(x,it)
np.random.seed(2)    
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 

# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w



