import matplotlib.pyplot as plt
import random
import numpy as np
x_train=np.array([i for i in range(10)]) #features
y_train=np.array([random.randrange(1,10,1) for i in range(10)]) #targetvalues
def model(x,y,w,b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb
def compute_value(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb-y[i])**2
    total_cost=1/(2*m)*cost
    return total_cost
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=(f_wb-y[i])
        dj_dw+=dj_dw_i
        dj_db+=dj_db_i
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db
def gradient_descent(x,y,alpha,w_in,b_in,num_iter):
    b=b_in
    w=w_in
    for i in range(num_iter):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        b-=alpha*dj_db
        w-=alpha*dj_dw
    return w,b
w_intial=0
b_intial=0
alpha_temp=1.0e-2 #learning rate
w_final,b_final=gradient_descent(x_train,y_train,alpha_temp,w_intial,b_intial,10000)
print(w_final,b_final)

plt.scatter(x_train,y_train,marker="x",c='r')
plt.plot(x_train,model(x_train,y_train,w_final,b_final))
plt.show()