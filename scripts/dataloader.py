import json
import torch
import numpy as np


class Dataloader:
    def __init__(self, path, N, is_sequence, consecutive_frame):
        self.trj_path = path
        self.n_trj = N
        self.n_data = 0
        self.obs = []
        self.act = []
        self.cls = []
        self.is_seq = is_sequence
        self.M = consecutive_frame
        self.traj_index = []

        self.load()


    def load(self):
        for seq in range(self.n_trj):
            trj_file = self.trj_path + str(seq).zfill(6) + ".json"
            with open(trj_file,'r') as jf:
                data = json.load(jf)
            
            # update obs, act, cls array
            n = data['length']
            if n < self.M:
                continue
            self.traj_index.append((self.n_data, self.n_data + n - self.M + 1))
            for k in range(0, n - self.M +1):
                obs = []
                for j in range(0,self.M - 1):
                    obs += data['state'][k + j]
                    obs += data['action'][k + j]
                obs += data['state'][k + self.M - 1]
                self.obs.append(obs)
                self.act.append(data['action'][k + self.M - 1])
            self.cls += [seq for i in range(n - self.M + 1)]
            self.n_data += n - self.M + 1
        
        self.n_cls = len(self.traj_index)
        self.train_cls = [i for i in range(0, int(self.n_cls * 0.8))]
        self.test_cls = [i for i in range(int(self.n_cls * 0.8), self.n_cls)]
        self.dataset_boundary = int(self.n_cls * 0.8)
        self.n_data_boundary = self.traj_index[self.dataset_boundary][0]
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
        self.state_dim = self.obs.shape[1]
        self.action_dim = self.act.shape[1]

        
    def shuffle(self, size, type):
        if type == 0:
            arr = np.arange(self.n_data_boundary)
        else :
            arr = np.arange(self.n_data_boundary, self.n_data)
        np.random.shuffle(arr)
        return self.obs[arr[:size]], self.act[arr[:size]], self.cls[arr[:size]]

    
    def sample_same_traj(self, size):
        cls_list = np.random.randint(self.dataset_boundary, self.n_cls, size = size)
        
        P_idx_list = []
        Q_idx_list = []
        for i in range(size):
            cls_num = cls_list[i]
            arr = np.arange(self.traj_index[cls_num][0], self.traj_index[cls_num][1])
            np.random.shuffle(arr)
            P_idx_list.append(arr[0])
            Q_idx_list.append(arr[1])
        return self.obs[P_idx_list], self.act[P_idx_list], self.obs[Q_idx_list], self.act[Q_idx_list]


    def sample_diff_traj(self, size):
        cls_arr = np.arange(self.n_cls)
        P_idx_list = []
        Q_idx_list = []
        for i in range(size):
            np.random.shuffle(cls_arr)
            P_cls = cls_arr[0]
            Q_cls = cls_arr[1]
            P_idx = np.random.randint(self.traj_index[P_cls][0], self.traj_index[P_cls][1], size = 1)[0]
            Q_idx = np.random.randint(self.traj_index[Q_cls][0], self.traj_index[Q_cls][1], size = 1)[0]
            P_idx_list.append(P_idx)
            Q_idx_list.append(Q_idx)
        return self.obs[P_idx_list], self.act[P_idx_list], self.obs[Q_idx_list], self.act[Q_idx_list]

