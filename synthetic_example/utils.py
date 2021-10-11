import numpy as np
import math

def covariance_function(x1,x2,length,signal_var,noise_var):
    size1 = x1.shape[0]
    size2 = x2.shape[0]
    out = np.zeros((size1,size2))
    for i in range(size1):
        for j in range(size2): 
            xp = x1[i]
            xq = x2[j]
            out[i,j] =signal_var * math.exp(-1/2*(abs(xp-xq)**2)/length)+noise_var*(i==j)
    return out

def polar2xy(theta,radius):
    return np.cos(theta)*radius,np.sin(theta)*radius

def transform(alpha,x,y):
    x_ = x*np.cos(alpha)-y*np.sin(alpha)
    y_ = x*np.sin(alpha)+y*np.cos(alpha)
    return x_, y_

def get_action(x,y):
    dx = x[:-1]- x[1:]
    dy = y[:-1] - y[1:]
    return dx.tolist()+[0],dy.tolist()+[0]