
##### add python path #####
import sys
import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import random
import copy
import time
import os

import json
import yaml
import rospkg

from model import Encoder
from model import MDN
from dataloader import Dataloader

##### Hyperparameter setting #####

##################################

if __name__ == "__main__":
    start_time = time.time()

    # device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ = torch.device("cpu")
    project_path = os.path.abspath("..")
    data_path = project_path + "/data/"
    
    # set yaml path
    yaml_file = project_path + "/test.yaml"
    with open(yaml_file) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        print(args)
    n_epoch_ = args['n_epoch']
    batch_size_ = args['batch_size']
    n_trj_ = args['n_trj']
    state_dim_ = args['state_dim']
    action_dim_ = args['action_dim']
    latent_dim_ = args['latent_dim']
    encoder_hidden_layers_ = args['encoder_hidden_layers']
    mdn_hidden_layers_ = args['mdn_hidden_layers']
    encoder_learning_rate_ = args['encoder_learning_rate']
    policy_learning_rate_ = args['policy_learning_rate']
    K_ = args['K']
    is_deterministic_ = args['is_deterministic']

    dataset = Dataloader(data_path, n_trj_)
    n_batches_ = int(dataset.n_data / batch_size_)

    encoder = Encoder(state_dim = state_dim_, action_dim = action_dim_, hidden_layers = encoder_hidden_layers_, latent_dim = latent_dim_, learning_rate = encoder_learning_rate_, is_deterministic = is_deterministic_)
    policy = MDN(state_dim = state_dim_, action_dim = action_dim_, hidden_layers = mdn_hidden_layers_, latent_dim = latent_dim_, K = K_, learning_rate = policy_learning_rate_)
    

    encoder_loss = 0.0
    policy_loss = 0.0
    for i_epoch in range(n_epoch_):
        for i in range(n_batches_):
            # update encoder
            encoder.optimizer.zero_grad()
            policy.optimizer.zero_grad()
            
            P_obs, P_act, P_cls = dataset.shuffle(batch_size_) # shuffle 2N array, but 2*k, 2*k+1 index are not same
            Q_obs, Q_act, Q_cls = dataset.shuffle(batch_size_) # shuffle 2N array, but 2*k, 2*k+1 index are not same
            P_obs = torch.from_numpy(P_obs).type(torch.FloatTensor).to(device = device_)
            P_act = torch.from_numpy(P_act).type(torch.FloatTensor).to(device = device_)
            P_cls = torch.from_numpy(P_cls).type(torch.IntTensor).to(device = device_)
            Q_obs = torch.from_numpy(Q_obs).type(torch.FloatTensor).to(device = device_)
            Q_act = torch.from_numpy(Q_act).type(torch.FloatTensor).to(device = device_)
            Q_cls = torch.from_numpy(Q_cls).type(torch.IntTensor).to(device = device_)
            cls_dir = torch.FloatTensor(2.0 * (P_cls == Q_cls) - 1.0)

            if is_deterministic_:
                P_mu = encoder(P_obs, P_act)
                Q_mu = encoder(Q_obs, Q_act)
            else:
                P_mu, P_sigma = encoder(P_obs, P_act)
                Q_mu, Q_sigma = encoder(Q_obs, Q_act)
            
            if is_deterministic_:
                dist = encoder.hellinger_distance(P_mu, Q_mu, None, None)
            else :
                dist = encoder.hellinger_distance(P_mu, Q_mu, P_sigma, Q_sigma)

            loss = torch.mean(cls_dir * dist) + encoder.regularization_loss(P_mu, P_sigma) + encoder.regularization_loss(Q_mu, Q_sigma)
            loss.backward()
            encoder.optimizer.step()
            encoder_loss = loss

            # update policy
            encoder.optimizer.zero_grad()
            policy.optimizer.zero_grad()
            
            batch_obs, batch_act, _ = dataset.shuffle(batch_size_)
            batch_obs = torch.from_numpy(batch_obs).type(torch.FloatTensor).to(device = device_)
            batch_act = torch.from_numpy(batch_act).type(torch.FloatTensor).to(device = device_)
            if is_deterministic_:
                out_mu = encoder(batch_obs, batch_act)
            else:
                out_mu, out_sigma = encoder(batch_obs, batch_act)
            # out_mu = out_mu.detach()
            # out_sigma = out_sigma.detach()
            if is_deterministic_:
                batch_lat = encoder.sample_latent_vector(out_mu, None)
            else :    
                batch_lat = encoder.sample_latent_vector(out_mu, out_sigma)
            out_pi, out_mu, out_sigma = policy(batch_lat, batch_obs)
            out_pi = torch.reshape(out_pi, (-1, policy.K))
            out_mu = torch.reshape(out_mu, (-1, policy.K, action_dim_))
            out_sigma = torch.reshape(out_sigma, (-1, policy.K, action_dim_))
            batch_act = torch.reshape(batch_act, (-1, 1, action_dim_))
            
            loss = policy.mdn_loss(out_pi, out_mu, out_sigma, batch_act)
            loss.backward()
            encoder.optimizer.step()
            policy.optimizer.step()
            policy_loss = loss

        end_time = time.time()
        if i_epoch % 10 == 0:
            print("[%04d-th Epoch, %.2lf] encoder loss: %.4f, policy loss: %.4f" %(i_epoch, end_time - start_time, encoder_loss, policy_loss))
        




    end_time = time.time()
    print("ELAPSED TIME")
    print(end_time - start_time)
    

    