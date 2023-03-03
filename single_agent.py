import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.algorithm import Algorithm

import gym
from gym.core import ActionWrapper
import glob



import shutil
import os 

import numpy as np

import json
import random
import ast
from scipy import interpolate
import math
from typing import Any, Dict, Union
import time
import pathlib
from functools import partial
import datetime

import sys
import matplotlib.pyplot as plt


from sim import cstr_model as cstr


class CSTREnv(gym.Env):
    def __init__(self, env_config):
      #n_vars = 5 #T,Tc,Ca,Cr,Tr
      #n_actions = 1 #dTc

      self.Cref_signal = env_config['Cref_signal']
      self.selector = env_config['selector']
      self.model = env_config['model']

      if self.selector == True:
        self.action_space = gym.spaces.Discrete(3) ##add number of concepts
        self.models = []
        
        for i in self.model:
          self.models.append(Policy.from_checkpoint(glob.glob(i+"/*")[0])['default_policy'] )

        self.model = self.models

      else:
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0)

      self.observation_space = gym.spaces.Box(low=np.array([200, 200, 0, 0, 200]), high=np.array([500, 500, 12, 12, 500]))   

    def reset(self):
      noise_percentage = 0
      Cref_signal = self.Cref_signal

      #initial conditions
      Ca0: float = 8.5698 #kmol/m3
      T0: float = 311.2639 #K
      Tc0: float = 292 #K
      
      self.T = T0
      self.Tc = Tc0
      self.Ca = Ca0
      self.ΔTc = 0

      self.cnt = 0

      self.Cref_signal = Cref_signal
      self.noise_percentage = noise_percentage
      if self.noise_percentage > 0:
          self.noise_percentage = self.noise_percentage/100
      
      if self.Cref_signal == 2:
          self.Cref = 2
          self.Tref = 373.1311
          self.Ca = 2
          self.T = 373.1311
      else:
          self.Cref = 8.5698
          self.Tref = 311.2612

      self.rms = 0
      self.y_list = []
      self.obs = np.array([self.T,self.Tc,self.Ca,self.Cref,self.Tref], dtype=np.float32)
      return self.obs

    def step(self, action):
      if self.Cref_signal == 0:
        self.Cref = 0
        self.Tref = 0
      elif self.Cref_signal == 1: #transition
        #update Cref an Tref
        time = 90
        p1 = 22 
        p2 = 74
        k = self.cnt+p1
        ceq = [8.57,6.9275,5.2850,3.6425,2]
        teq = [311.2612,327.9968,341.1084,354.7246,373.1311]
        C = interpolate.interp1d([0,p1,p2,time], [8.57,8.57,2,2])
        self.Cref = float(C(k))
        T_ = interpolate.interp1d([0,p1,p2,time], [311.2612,311.2612,373.1311,373.1311])
        self.Tref = float(T_(k))
      elif self.Cref_signal == 2: #steady state 1
        self.Cref = 2
        self.Tref = 373.1311
      elif self.Cref_signal == 3: #steady state 2
        self.Cref = 8.5698
        self.Tref = 311.2612
      elif self.Cref_signal == 4: #full sim
        k = self.cnt
        time = 90
        #update Cref an Tref
        p1 = 22 
        p2 = 74 
        ceq = [8.57,6.9275,5.2850,3.6425,2]
        teq = [311.2612,327.9968,341.1084,354.7246,373.1311]
        C = interpolate.interp1d([0,p1,p2,time], [8.57,8.57,2,2])
        self.Cref = float(C(k))
        T_ = interpolate.interp1d([0,p1,p2,time], [311.2612,311.2612,373.1311,373.1311])
        self.Tref = float(T_(k))

      if self.selector == True:
        self.ΔTc = self.model[action].compute_single_action(self.obs)[0][0]
      else:
        self.ΔTc = action
        
      error_var = self.noise_percentage
      σ_max1 = error_var * (8.5698 - 2)
      σ_max2 = error_var * ( 373.1311 - 311.2612)

      σ_Ca = random.uniform(-σ_max1, σ_max1)
      σ_T = random.uniform(-σ_max2, σ_max2)
      mu = 0

      #calling the CSTR python model
      sim_model = cstr.CSTRModel(T = self.T, Ca = self.Ca, Tc = self.Tc, ΔTc = self.ΔTc)

      #Tc
      self.Tc += self.ΔTc

      #Tr
      self.T = sim_model.T + σ_T

      #Ca
      self.Ca = sim_model.Ca + σ_Ca
      self.y_list.append(self.Ca)

      #Increase time
      self.cnt += 1
      
      self.rms = math.sqrt( (self.Ca - self.Cref)**2 )

      reward = float(1/self.rms)

      done = False

      #end the simulation
      if self.cnt == 90 or self.T >= 400 or (self.Cref_signal == 1 and self.cnt == 68): 
        done = True

      info = {}
      return self.obs, reward, done, info



ray.shutdown()
ray.init(ignore_reinit_error=False)

config = ppo.PPOConfig()  
#print(config.to_dict()) 

algo = ppo.PPOTrainer(env=CSTREnv, config={
    "env_config": {"Cref_signal":4},  # config to pass to env class
})

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
for n in range(100):
  result = algo.train()
  print(result)

  print(s.format(
      n+1,
      result["episode_reward_min"],
      result["episode_reward_mean"],
      result["episode_reward_max"],
      result["episode_len_mean"]
      ))
