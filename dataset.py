import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np

import pdb
import statistics
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class dataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc

    buffer_size = buffer_state.shape[0]

    self.traj_len = self.state.shape[1]
    self.length = self.state.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.8

  def geometric_sampling_fn(self):
    start_index = np.random.randint(0, self.traj_len)
    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index >= 19:
      return self.geometric_sampling_fn()
    else:
      return start_index, pos_index

  def __getitem__(self,idx):

    start_index, pos_index = self.geometric_sampling_fn()

    state = self.state[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc

  def __len__(self):
    return self.length


class ActionDataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, buffer_action, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc
    self.action = buffer_action

    buffer_size = buffer_state.shape[0]

    self.traj_len = self.state.shape[1]
    self.length = self.state.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.8

  def geometric_sampling_fn(self):
    start_index = np.random.randint(0, self.traj_len)
    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index >= 19:
      return self.geometric_sampling_fn()
    else:
      return start_index, pos_index

  def __getitem__(self,idx):

    start_index, pos_index = self.geometric_sampling_fn()


    state = self.state[idx, start_index, ...]
    action = self.action[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc, action

  def __len__(self):
    return self.length


class ReverseActionDataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, buffer_action, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc
    self.action = buffer_action

    buffer_size = buffer_state.shape[0]

    self.traj_len = self.state.shape[1]
    self.length = self.state.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.8

  def geometric_sampling_fn(self):



    start_index = np.random.randint(0, self.traj_len)
    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index <= 0:
      return self.geometric_sampling_fn()
    else:
      return start_index, pos_index

  def __getitem__(self,idx):

    pos_index = (self.desc[idx] != 48).nonzero()[0][0]
    start_index = np.random.randint(0,pos_index +1)

    state = self.state[idx, start_index, ...]
    action = self.action[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc, action

  def __len__(self):
    return self.length

class Reversedataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc

    buffer_size = buffer_state.shape[0]

    self.traj_len = self.state.shape[1]
    self.length = self.state.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.8

  def geometric_sampling_fn(self):
    start_index = np.random.randint(0, self.traj_len)
    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index <= 0:
      return self.geometric_sampling_fn()
    else:
      return start_index, pos_index

  def __getitem__(self,idx):

    pos_index = (self.desc[idx] != 48).nonzero()[0][0]
    start_index = np.random.randint(0,pos_index +1)

    state = self.state[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc

  def __len__(self):
    return self.length

class ProperSampleActionDataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, buffer_action, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc
    self.action = buffer_action

    buffer_size = buffer_desc.shape[0]

    self.traj_len = buffer_desc.shape[1]
    self.length = buffer_desc.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.8

  def geometric_sampling_fn(self, start_index):

    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index >= 19:
      return start_index + 1
    else:
      return pos_index

  def __getitem__(self,idx):

    wherepick = self.action[idx,:] == 6

    ind = wherepick.nonzero()
    ind = ind[0][0]

    start_index = np.random.randint(0, ind+1)

    pos_index = self.geometric_sampling_fn(start_index)

    state = self.state[idx, start_index, ...]
    action = self.action[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc, action

  def __len__(self):
    return self.length


