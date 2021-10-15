
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

import matplotlib.pyplot as plt

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

TRAIN = 0
TEST = 1

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
    is_sequence_ = args['is_sequence']
    n_consecutive_frame_ = args['n_consecutive_frame']
    n_same_traj_samples_ = args['n_same_traj_samples']
    n_diff_traj_samples_ = args['n_diff_traj_samples']

    dataset = Dataloader(data_path, n_trj_, is_sequence_, n_consecutive_frame_)
    state_dim_ = dataset.state_dim
    action_dim_ = dataset.action_dim
    n_batches_ = int(dataset.n_data / batch_size_)

    encoder = Encoder(state_dim = state_dim_, action_dim = action_dim_, hidden_layers = encoder_hidden_layers_, latent_dim = latent_dim_, learning_rate = encoder_learning_rate_, is_deterministic = is_deterministic_)
    policy = MDN(state_dim = state_dim_, action_dim = action_dim_, hidden_layers = mdn_hidden_layers_, latent_dim = latent_dim_, K = K_, learning_rate = policy_learning_rate_)
    

    encoder_loss = 0.0
    distance_loss = 0.0
    regularizer_loss = 0.0
    policy_loss = 0.0
    for i_epoch in range(n_epoch_):
        for i in range(n_batches_):
            # update encoder
            encoder.optimizer.zero_grad()
            policy.optimizer.zero_grad()
            
            P_obs, P_act, P_cls = dataset.shuffle(batch_size_, TRAIN) # shuffle 2N array, but 2*k, 2*k+1 index are not same
            Q_obs, Q_act, Q_cls = dataset.shuffle(batch_size_, TRAIN) # shuffle 2N array, but 2*k, 2*k+1 index are not same
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

            distance_loss = torch.mean(cls_dir * dist)
            regularizer_loss = encoder.regularization_loss(P_mu, P_sigma) + encoder.regularization_loss(Q_mu, Q_sigma)

            #### mdn loss
            if is_deterministic_:
                batch_P_lat = encoder.sample_latent_vector(P_mu, None)
                batch_Q_lat = encoder.sample_latent_vector(Q_mu, None)
            else :    
                batch_P_lat = encoder.sample_latent_vector(P_mu, P_sigma)
                batch_Q_lat = encoder.sample_latent_vector(Q_mu, Q_sigma)

            out_P_pi, out_P_mu, out_P_sigma = policy(batch_P_lat, P_obs)
            out_P_pi = torch.reshape(out_P_pi, (-1, policy.K))
            out_P_mu = torch.reshape(out_P_mu, (-1, policy.K, action_dim_))
            out_P_sigma = torch.reshape(out_P_sigma, (-1, policy.K, action_dim_))
            P_act = torch.reshape(P_act, (-1, 1, action_dim_))
            out_Q_pi, out_Q_mu, out_Q_sigma = policy(batch_Q_lat, Q_obs)
            out_Q_pi = torch.reshape(out_Q_pi, (-1, policy.K))
            out_Q_mu = torch.reshape(out_Q_mu, (-1, policy.K, action_dim_))
            out_Q_sigma = torch.reshape(out_Q_sigma, (-1, policy.K, action_dim_))
            Q_act = torch.reshape(Q_act, (-1, 1, action_dim_))
            


            policy_loss = policy.mdn_loss(out_P_pi, out_P_mu, out_P_sigma, P_act) + policy.mdn_loss(out_Q_pi, out_Q_mu, out_Q_sigma, Q_act)
            encoder_loss = distance_loss + regularizer_loss

            loss = distance_loss + regularizer_loss + policy_loss
            loss.backward()
            encoder.optimizer.step()
            policy.optimizer.step()


            # update policy
            # encoder.optimizer.zero_grad()
            # policy.optimizer.zero_grad()
            
            # batch_obs, batch_act, _ = dataset.shuffle(batch_size_)
            # batch_obs = torch.from_numpy(batch_obs).type(torch.FloatTensor).to(device = device_)
            # batch_act = torch.from_numpy(batch_act).type(torch.FloatTensor).to(device = device_)
            # if is_deterministic_:
            #     out_mu = encoder(batch_obs, batch_act)
            # else:
            #     out_mu, out_sigma = encoder(batch_obs, batch_act)
            # # out_mu = out_mu.detach()
            # # out_sigma = out_sigma.detach()
            # if is_deterministic_:
            #     batch_lat = encoder.sample_latent_vector(out_mu, None)
            # else :    
            #     batch_lat = encoder.sample_latent_vector(out_mu, out_sigma)
            # out_pi, out_mu, out_sigma = policy(batch_lat, batch_obs)
            # out_pi = torch.reshape(out_pi, (-1, policy.K))
            # out_mu = torch.reshape(out_mu, (-1, policy.K, action_dim_))
            # out_sigma = torch.reshape(out_sigma, (-1, policy.K, action_dim_))
            # batch_act = torch.reshape(batch_act, (-1, 1, action_dim_))
            
            # loss = policy.mdn_loss(out_pi, out_mu, out_sigma, batch_act)
            # loss.backward()
            # encoder.optimizer.step()
            # policy.optimizer.step()
            # policy_loss = loss

        end_time = time.time()
        if i_epoch % 10 == 0:
            print("[%04d-th Epoch, %.2lf] encoder loss: %.4f, distance loss: %.4f, regularizer loss: %.4f, policy loss: %.4f" %(i_epoch, end_time - start_time, encoder_loss, distance_loss, regularizer_loss, policy_loss))
        
        # evaluation

        # first, evaluate encoder performance
        # sample (s,a) pair from same trajectory and calculate latent vector distance
        # sample (s,a) pair from different trajectory and calculate latent vector distance
        idx = [i for i in range(n_same_traj_samples_ + n_diff_traj_samples_)]
        color = ['red' if i < n_same_traj_samples_ else 'blue' for i in range(n_same_traj_samples_ + n_diff_traj_samples_)]
        latent_distance = []
        test_policy_loss = 0.0

        P_obs, P_act, Q_obs, Q_act = dataset.sample_same_traj(n_same_traj_samples_)
        P_obs = torch.from_numpy(P_obs).type(torch.FloatTensor).to(device = device_)
        P_act = torch.from_numpy(P_act).type(torch.FloatTensor).to(device = device_)
        Q_obs = torch.from_numpy(Q_obs).type(torch.FloatTensor).to(device = device_)
        Q_act = torch.from_numpy(Q_act).type(torch.FloatTensor).to(device = device_)
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
        same_cls_dist = torch.mean(dist)
        if is_deterministic_:
            batch_P_lat = encoder.sample_latent_vector(P_mu, None)
            batch_Q_lat = encoder.sample_latent_vector(Q_mu, None)
        else :    
            batch_P_lat = encoder.sample_latent_vector(P_mu, P_sigma)
            batch_Q_lat = encoder.sample_latent_vector(Q_mu, Q_sigma)

        out_P_pi, out_P_mu, out_P_sigma = policy(batch_P_lat, P_obs)
        out_P_pi = torch.reshape(out_P_pi, (-1, policy.K))
        out_P_mu = torch.reshape(out_P_mu, (-1, policy.K, action_dim_))
        out_P_sigma = torch.reshape(out_P_sigma, (-1, policy.K, action_dim_))
        P_act = torch.reshape(P_act, (-1, 1, action_dim_))
        out_Q_pi, out_Q_mu, out_Q_sigma = policy(batch_Q_lat, Q_obs)
        out_Q_pi = torch.reshape(out_Q_pi, (-1, policy.K))
        out_Q_mu = torch.reshape(out_Q_mu, (-1, policy.K, action_dim_))
        out_Q_sigma = torch.reshape(out_Q_sigma, (-1, policy.K, action_dim_))
        Q_act = torch.reshape(Q_act, (-1, 1, action_dim_))
        
        policy_loss = policy.mdn_loss(out_P_pi, out_P_mu, out_P_sigma, P_act) + policy.mdn_loss(out_Q_pi, out_Q_mu, out_Q_sigma, Q_act)
        test_policy_loss += policy_loss

        latent_distance += dist.detach().cpu().tolist()
        
        
        P_obs, P_act, Q_obs, Q_act = dataset.sample_diff_traj(n_diff_traj_samples_)
        P_obs = torch.from_numpy(P_obs).type(torch.FloatTensor).to(device = device_)
        P_act = torch.from_numpy(P_act).type(torch.FloatTensor).to(device = device_)
        Q_obs = torch.from_numpy(Q_obs).type(torch.FloatTensor).to(device = device_)
        Q_act = torch.from_numpy(Q_act).type(torch.FloatTensor).to(device = device_)
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
        diff_cls_dist = torch.mean(dist)

        if is_deterministic_:
            batch_P_lat = encoder.sample_latent_vector(P_mu, None)
            batch_Q_lat = encoder.sample_latent_vector(Q_mu, None)
        else :    
            batch_P_lat = encoder.sample_latent_vector(P_mu, P_sigma)
            batch_Q_lat = encoder.sample_latent_vector(Q_mu, Q_sigma)

        out_P_pi, out_P_mu, out_P_sigma = policy(batch_P_lat, P_obs)
        out_P_pi = torch.reshape(out_P_pi, (-1, policy.K))
        out_P_mu = torch.reshape(out_P_mu, (-1, policy.K, action_dim_))
        out_P_sigma = torch.reshape(out_P_sigma, (-1, policy.K, action_dim_))
        P_act = torch.reshape(P_act, (-1, 1, action_dim_))
        out_Q_pi, out_Q_mu, out_Q_sigma = policy(batch_Q_lat, Q_obs)
        out_Q_pi = torch.reshape(out_Q_pi, (-1, policy.K))
        out_Q_mu = torch.reshape(out_Q_mu, (-1, policy.K, action_dim_))
        out_Q_sigma = torch.reshape(out_Q_sigma, (-1, policy.K, action_dim_))
        Q_act = torch.reshape(Q_act, (-1, 1, action_dim_))
        
        policy_loss = policy.mdn_loss(out_P_pi, out_P_mu, out_P_sigma, P_act) + policy.mdn_loss(out_Q_pi, out_Q_mu, out_Q_sigma, Q_act)
        test_policy_loss += policy_loss

        latent_distance += dist.detach().cpu().tolist()
        
        print("same class distance: %.4lf, diff class distance: %.4f, test policy loss: %.4f" %(same_cls_dist, diff_cls_dist, test_policy_loss))
        plt.clf()
        plt.scatter(idx, latent_distance, color = color)
        plt.savefig(project_path + "/result/encoder/" + str(i_epoch).zfill(6) + '.png')

        # second, show 




    end_time = time.time()
    print("ELAPSED TIME")
    print(end_time - start_time)
    

    