import json
import torch
import numpy as np


class Dataloader:
    def __init__(self, path, N):
        self.trj_path = path
        self.n_trj = N
        self.n_data = 0
        self.obs = []
        self.act = []
        self.cls = []

        self.load()


    def load(self):
        for seq in range(self.n_trj):
            trj_file = self.trj_path + str(seq).zfill(6) + ".json"
            with open(trj_file,'r') as jf:
                data = json.load(jf)
            
            # update obs, act, cls array
            n = data['length']
            self.n_trj += n
            self.obs += data['state']
            self.act += data['action']
            self.cls += [seq for i in range(n)]
            self.n_data += n
        
        self.obs = np.array(self.obs)
        self.act = np.array(self.act)
        self.cls = np.array(self.cls)
        self.obs_mean = np.mean(self.obs, axis = 0)
        self.obs_std = np.std(self.obs, axis = 0)
        self.act_mean = np.mean(self.act, axis = 0)
        self.act_std = np.std(self.act, axis = 0)

        self.obs = (self.obs - self.obs_mean) / self.obs_std
        self.act = (self.act - self.act_mean) / self.act_std

        print("Finish Load!!!")
        print("# of state: %d" %(self.obs.shape[0]))
        print("# of action: %d" %(self.act.shape[0]))
        print("state dim: %d" %(self.obs.shape[1]))
        print("action dim: %d" %(self.act.shape[1]))

        
    def shuffle(self, size):
        arr = np.arange(self.n_data)
        np.random.shuffle(arr)
        return self.obs[arr[:size]], self.act[arr[:size]], self.cls[arr[:size]]
