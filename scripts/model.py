#!/usr/bin/env python
from __future__ import print_function

##### add python path #####
import sys
import os

from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.utils import shuffle



ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
EPS = 1e-6

class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, latent_dim, learning_rate, is_deterministic):
        super(Encoder, self).__init__()
        self.is_deterministic = is_deterministic
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.input_dim = self.state_dim + self.action_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.z_mu = nn.Linear(self.hidden_layers[self.H - 1], latent_dim)
        if not self.is_deterministic:
            self.z_sigma = nn.Linear(self.hidden_layers[self.H - 1], latent_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        

    def forward(self, state, action):
        # concatenate state and action
        x = torch.cat((state, action), axis = 1)

        # forward network and return
        for i in range(0,self.H):
            x = F.relu(self.fc[i](x))
        mu = F.normalize(self.z_mu(x), dim = 1)
        if not self.is_deterministic:
            sigma = torch.exp(self.z_sigma(x))
            return mu, sigma
        return mu

    def sample_latent_vector(self, mu, sigma):
        if not self.is_deterministic:
            epsilon = torch.normal(0.0,1.0,size=mu.shape)
            return mu + epsilon * sigma
        return mu

    def hellinger_distance(self, mu1, mu2, sigma1, sigma2):
        # return H^2(P,Q)
        # # mu1: N * D
        if not self.is_deterministic:
            return torch.mean(1.0 - torch.sqrt(2.0 * sigma1 * sigma2 / (sigma1 * sigma1 + sigma2 * sigma2)) * torch.exp(-0.25 * (mu1 - mu2)* (mu1 - mu2) / (sigma1 * sigma1 + sigma2 * sigma2)), dim =1)
            # return torch.norm(mu1 - mu2, dim = 1) + torch.norm(sigma1 - sigma2, dim =1)
        return torch.norm(mu1 - mu2, dim = 1)
        
    def regularization_loss(self, mu, sigma):
        return torch.mean(0.5 * torch.norm(mu, dim = 1) ** 2 + torch.norm(sigma, dim = 1) ** 2 - torch.sum(torch.log(sigma), dim = 1)) - 0.5

class MDN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, latent_dim, K, learning_rate):
        super(MDN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.input_dim = self.state_dim + self.latent_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.K = K
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.z_pi = nn.Linear(self.hidden_layers[self.H - 1], self.K)
        self.z_mu = nn.Linear(self.hidden_layers[self.H - 1], self.K * self.action_dim)
        self.z_sigma = nn.Linear(self.hidden_layers[self.H - 1], self.K * self.action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, state, z):
        # concatenate state and action
        x = torch.cat((state, z), axis = 1)

        # forward network and return
        for i in range(0,self.H):
            x = F.relu(self.fc[i](x))
        z_pi = torch.softmax(self.z_pi(x), dim=1)
        z_mu = self.z_mu(x)
        z_sigma = torch.exp(self.z_sigma(x))
        return z_pi, z_mu, z_sigma


    def gaussian_dist(self, mu, sigma, y):
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * ONEOVERSQRT2PI

    def mdn_loss(self, pi, mu, sigma, action):
        # pi: N * K
        # mu: N * K * A
        # sigma: N * K * A
        # action: N * 1 * A
        # rt = self.gaussian_dist(mu, sigma, action)
        # rt = torch.sum(torch.log(rt + EPS), dim = 2)
        # rt = torch.sum(torch.exp(rt) * pi, dim=1)
        # rt = -torch.log(rt + EPS)
        data_exp = action.expand_as(sigma) # [N x K x D]
        probs = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data_exp-mu)/sigma)**2) / sigma # [N x K x D]
        probs_prod = torch.prod(probs,2) # [N x K]
        prob = torch.sum(probs_prod*pi,dim=1) # [N]
        prob = torch.clamp(prob,min=1e-8) # Clamp if the prob is to small
        nll = -torch.log(prob)
        return torch.mean(nll)

