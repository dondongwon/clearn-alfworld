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
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from torchvision import transforms




from constants import local2globaldict #obj_list, unique_objs,

import json

from model import *
import argparse

from collections import deque 
import pickle
import copy
from utils import *




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-use_action', required = True)
parser.add_argument('-reverse_geom', required = True)
parser.add_argument('-room', required = False)
parser.add_argument('-lr', required = True)
parser.add_argument('-obj_num', default = "999")
parser.add_argument('-epochs', default = "5001")
parser.add_argument('-subset', default = "10")
parser.add_argument('-baseline', default = "LCRL")

args = parser.parse_args()

subset = int(args.subset)
state_input = 'Image'
reverse_geom = int(args.reverse_geom)
use_action = int(args.use_action)
obj_num = int(args.obj_num)
epochs = int(args.epochs)
baseline = args.baseline

#description

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

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


class LCRL_Dataset(Dataset):
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


    neg_ind = np.random.randint(0, ind)

    pos_state = self.state[idx, ind, ...]
    pos_action = self.action[idx, ind, ...]

    neg_state_1 = self.state[idx, neg_ind, ...]
    neg_action_1 = self.action[idx, neg_ind, ...]

    neg_state_2 = self.state[idx, neg_ind + 1, ...]
    neg_action_2 = self.action[idx, neg_ind + 1, ...]

    desc = self.desc[idx, ind, ...]


    return pos_state, pos_action, neg_state_1, neg_action_1, neg_state_2, neg_action_2, desc

  def __len__(self):
    return self.length





# Model , Optimizer, Loss


# args.room = ""
data_root = "./our_data/pick2/{}/".format(str(args.room))
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

writer = SummaryWriter("tb_results/" + str(args.room) + "/use_action_{action}/lr_{lr}/{now}".format(lr = args.lr, action = str(bool(use_action)), now = dt_string))

#better dataloading - so it fits in RAM
state_ds = np.zeros((100*subset, 20, 300, 300, 3), dtype='uint8')
action_ds = np.zeros((100*subset, 20), dtype='uint8')
desc_ds = np.zeros((100*subset, 20 ), dtype='uint8')

if state_input == 'Image':
  print("\n Starting Dataloading... \n")
  for i in range(1,subset+1):
    state_sub = np.load(data_root + "state1k_part{}.npy".format(i)).astype(np.int8)
    desc_sub = np.load(data_root + "desc1k_part{}.npy".format(i)).astype(np.int8)
    action_sub = np.load(data_root + "action1k_part{}.npy".format(i)).astype(np.int8)

    state_ds[100*(i-1):100*i] = state_sub
    action_ds[100*(i-1):100*i] = action_sub
    desc_ds[100*(i-1):100*i] = desc_sub
    print(i)
  

  tmp = action_ds == 6
  tmp = torch.Tensor(tmp)
  idx = torch.arange(tmp.shape[1], 0, -1)
  tmp2= tmp * idx
  indices = torch.argmax(tmp2, 1, keepdim=True)
  indices.numpy()

  unique, counts = np.unique(indices, return_counts=True)

  #make such that pick doesn't take place at the very end 
  ind_greater = (indices.numpy() > 0).flatten()
  #make such that pick doesn't take place at the very end 
  ind_smaller = (indices.numpy() < 19).flatten()
  ind_inrange = np.logical_and(ind_greater, ind_smaller)

  #np.delete 
  desc_ds = desc_ds[ind_inrange,:]
  state_ds = state_ds[ind_inrange, ...]
  action_ds = action_ds[ind_inrange,:]

  
  #good until here


  # if obj_num != 999:
  #   # object_7 = ((desc_ds == 7).sum(1) > 0)
  #   # object_47 = ((desc_ds == 47).sum(1) > 0)
  #   # object_41 = ((desc_ds == 41).sum(1) > 0)
  #   # object_specific = object_7 + object_47 + object_41

  #   # object_specific =  ((desc_ds == 47).sum(1) > 0)
  #   state_ds = state_ds[object_specific]
  #   desc_ds = desc_ds[object_specific]
  #   action_ds = action_ds[object_specific]


  print("\n Dataloading Complete... \n")

  # local_desc_ds = desc_ds.copy()

  # for loc_idx, glob_idx in local2globaldict.items():
  #   desc_ds[local_desc_ds == loc_idx] = glob_idx

  ds_split = (state_ds.shape[0] // 10) * 9
  train_action = action_ds[:ds_split]
  train_state = state_ds[:ds_split]
  train_desc = desc_ds[:ds_split]
  val_action = action_ds[ds_split:]
  val_state = state_ds[ds_split:]
  val_desc = desc_ds[ds_split:]
  

#get rid of sparsity
  desc_all = np.unique(desc_ds)

  desc_num_classes = len(desc_all)
    
  for ind, obj in enumerate(desc_all):
    
    obj_contained = ((desc_ds == obj).sum(1) > 0)
    obj_contained_traj = desc_ds[obj_contained]
    length = round((obj_contained_traj != obj).sum()/obj_contained_traj.shape[0],2)
    
    print('Obj {}:{}'.format(ind, obj))
    print('length = {}'.format(length))
    print('num_traj = {}'.format(obj_contained_traj.shape[0]))

  #desc_num_classes = np.max(desc_ds) + 1
  action_num_classes = np.max(action_ds) + 1
  action_dim = 3
  desc_dim = 3

  if use_action == 1:
    
    if baseline == 'LCRL':

      q = QFunc(desc_num_classes, action_num_classes)
      target_q = copy.deepcopy(q)
      target_q.eval()
      gamma = 0.8 

    #ClassifierCRActionBasic
    #ClassifierCRActionResnetDropout - best so far
    #ClassifierCRActionResnetDropout_FuseAction - looks even better for now

    trainset = LCRL_Dataset(train_state, train_desc, train_action)
    valset = LCRL_Dataset(val_state, val_desc, val_action)


if use_cuda:
  q = q.cuda()
  target_q = target_q.cuda()



learning_rate = float(args.lr)
optimizer = torch.optim.Adam(q.parameters(), lr=learning_rate, weight_decay=0.01)
batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers = 4)
scale_img =  transforms.Compose([transforms.Scale((64,64))])
i = 0
for epoch in range(epochs):
  print('Epoch: {}'.format(epoch))
  losses = AverageMeter()
  neg_q = AverageMeter()
  pos_q = AverageMeter()

  
  for j,batch in enumerate(tqdm(trainloader)):
    
    #send batch to GPU
    for i in range(len(batch)): batch[i] = batch[i].cuda()


    #pos_state, pos_action: s_T, a_T (where T is timestep for terminal transition, a is always 6, pick in our case)
    #neg_state_1, neg_action_1: s_t, a_t (where t is a random timestep)
    #neg_state_2, neg_action_2: s_t+1, a_t+1 (s,a pair one timestep after neg_state_1, neg_action_1 - could be 6 as well?)

    pos_state, pos_action, neg_state_1, neg_action_1, neg_state_2, neg_action_2, desc = batch

    #scale image 
    
    pos_state = scale_img(pos_state.permute(0,3,1,2))
    neg_state_1 = scale_img(neg_state_1.permute(0,3,1,2))
    neg_state_2 = scale_img(neg_state_2.permute(0,3,1,2))


    # pos_state = (pos_state.permute(0,3,1,2))
    # neg_state_1 = (neg_state_1.permute(0,3,1,2))
    # neg_state_2 = (neg_state_2.permute(0,3,1,2))
    #visualize heatmap, permutation doesn't anything weird - integer divsion by 255, check range. 

    batch_size = pos_action.shape[0]

    #convert actions to one_hot
    pos_action = torch.nn.functional.one_hot(pos_action.long(), action_num_classes)
    neg_action_1 = torch.nn.functional.one_hot(neg_action_1.long(), action_num_classes)
    neg_action_2 = torch.nn.functional.one_hot(neg_action_2.long(), action_num_classes)
    

    #convert desc to one hot and unify 
    for i, val in enumerate(desc):
      desc[i] = int(np.where(desc_all == int(val))[0])
    desc = torch.nn.functional.one_hot(desc.long(), num_classes=desc_num_classes )

    # q values for neg and pos
    # negs is q value at (t)
    negs = q(neg_state_1, desc, neg_action_1)
    pos = q(pos_state, desc, pos_action)

    
    

    NUM_ACTS = action_num_classes

    # get target q values
    with torch.no_grad():
      a_dist = torch.eye(action_num_classes).cuda()
      #resizing
      neg_state_2_t = neg_state_2.unsqueeze(1).repeat(1, NUM_ACTS, 1, 1, 1).reshape(batch_size*NUM_ACTS, 3, 64, 64)
      desc_t = desc.unsqueeze(1).repeat(1, NUM_ACTS, 1).reshape(batch_size*NUM_ACTS, desc_num_classes)
      a_dist_t = a_dist.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size*NUM_ACTS, NUM_ACTS)
      #q value for all actions at (t+1)
      tgq_lng = target_q(neg_state_2_t, desc_t, a_dist_t).reshape(batch_size, NUM_ACTS, -1)
      # get max q value
      best_tgq_lng, _ = tgq_lng.max(1)



    #discounted q value (reward + gamma*max Q val)
    y_negs = gamma * best_tgq_lng
    #termianal q value = 1 
    y_pos = torch.ones(y_negs.shape).cuda()

    # y_negs - negs = 0 since reward = 0, thus negs = 0  + gamma*max Q val
    y = torch.cat([y_negs,y_pos], 0)
    qs = torch.cat([negs,pos], 0)

    #randomly initialzied q funciton, does it ignore inputs 
    # how we're calling vs how we train

    loss = ((y - qs)**2).mean()   

    #conservative regulation

    losses.update(loss, batch_size)
    neg_q.update(negs.mean(), batch_size)
    pos_q.update(pos.mean(), batch_size)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    i += 1 

    if (i % 10 == 0):
      target_q.load_state_dict(q.state_dict())
      target_q.eval()


  writer.add_scalar("Loss/train", losses.avg.item(), epoch)
  writer.add_scalar("pos q-values/train", pos_q.avg.item(), epoch)
  writer.add_scalar("neg q-values/train", neg_q.avg.item(), epoch)
  print(pos_q.avg.item())



  if epoch % 200 == 0:
    model = q
        #make data directory
    weight_dir = "./model_weights/pick_LCRL_Suraj_2/{}/".format(args.room)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        final_desc_path = weight_dir + "pick_desc_all.npy"
        np.save(final_desc_path, desc_all) 

    final_weight_path = weight_dir + "{model}_{obj_num}_pick.pth".format(model = model.__class__.__name__+"_epoch["+str(epoch)+"]", obj_num = obj_num)
    torch.save(model.state_dict(), final_weight_path)