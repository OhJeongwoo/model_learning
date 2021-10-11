import json
import matplotlib.pyplot as plt
class logger():
    '''
    each trajectory has
    1. id: id
    2. vel: velocity
    3. noise: noise rate
    4. type: trajectory type
    5. states: x,y
    6. actions: dx,dy
    '''
    def __init__(self,path='./data'):
        self.path = path + '/sdata.json'
        self.id = 0
        self.logs = {}
        self.configs = {}

    def log_traj(self,vel,noise,type,states,actions,lengthscale,sigma,anchor_num):
        self.logs[self.id] = {
            'vel':vel, 'noise':noise, 'type':type,
            'states': states, 'actions':actions, 
        }
        self.configs[self.id]={
            'lengthscale': lengthscale,
            'sigma': sigma,
            'anchor_num': anchor_num
        }
        self.id+=1
    
    def save_log(self):
        with open(self.path,'w') as jf:
            json.dump(self.logs,jf)
    