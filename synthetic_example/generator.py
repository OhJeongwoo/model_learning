import numpy as np
import math
import matplotlib.pyplot as plt
from utils import covariance_function,polar2xy,transform,get_action
from logger import logger
from config import PATH_SIGMA,anchor_num,vel_types,length_scale,num_traj_type

saver = logger()

'''
parameters
'''
for sigma in PATH_SIGMA:
    for num in anchor_num:
        for l in length_scale:
            noise_sigma = 1e-3
            signal_sigma = 1
            noise = sigma/l

            '''
            grp interporlation
            '''
            
            x_data = np.asarray([i*2*math.pi/(num) for i in range(num+1)])
            y_data = np.asarray([1]*(num+1)) + sigma*np.random.normal(size=num+1)

            x_ = np.linspace(0,2*math.pi,100)
            prior_mean = np.zeros(100)
            noise_var = noise_sigma**2
            signal_var = signal_sigma**2

            post_mean = np.matmul(np.matmul(covariance_function(x_,x_data,l,signal_var,noise_var),
                                            np.linalg.inv(covariance_function(x_data,x_data,l,signal_var,noise_var))),y_data)
            '''
            transform trajectory
            '''
            x,y = polar2xy(x_,post_mean) 
            y = y +1
            for vel in vel_types:
                filter_index = np.asarray([int(i*vel) for i in range(int(100/vel))])
                x_1 = x[filter_index]
                y_1 = y[filter_index]
                action_1 = get_action(x_1,y_1)
                saver.log_traj(vel,noise,1,(x_1.tolist(),y_1.tolist()),action_1,l,sigma,num)

                for t in range(2,num_traj_type+1):
                    x_2,y_2 = transform(2*(t-1)*math.pi/num_traj_type,x_1,y_1)
                    action_2 = get_action(x_2,y_2)
                    saver.log_traj(vel,noise,t,(x_2.tolist(),y_2.tolist()),action_2,l,sigma,num)

saver.save_log()
print("%d samples saved!"%(saver.id))

'''
visualizing some trajectories
'''

plt.figure(figsize=(15,10))
plt.xlim((-4,4))
plt.ylim((-4,4))
for key in saver.logs:
    if key>300:
        break
    if saver.logs[key]['vel']==1: #saver.logs[key]['type']==1 
        (x,y) = saver.logs[key]['states']
        config = saver.configs[key]
        plt.plot(x,y,marker='o',linewidth=1, markersize=2,label = 'lengthscale:{}, sigma: {} , anchor_num:{}'.format(config['lengthscale'],config['sigma'],config['anchor_num']))
plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig('./data_vis.png')
plt.show()