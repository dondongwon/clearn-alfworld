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

from constants import local2globaldict #obj_list, unique_objs,

import json

from model import *
import argparse

from collections import deque 
import pickle




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-use_action', required = True)
parser.add_argument('-reverse_geom', required = True)
parser.add_argument('-room', required = False)
parser.add_argument('-lr', required = True)
parser.add_argument('-obj_num', default = "999")
parser.add_argument('-epochs', default = "10001")
parser.add_argument('-subset', default = "10")
parser.add_argument('-baseline', default = "LCBC")
parser.add_argument('-save_every', default = "100")
parser.add_argument('-sample_more_good', action='store_true')
parser.add_argument('-sample_more_bad', action='store_true')
parser.add_argument('-fraction', required = True)
parser.add_argument('-exp', required = True)
parser.add_argument('-binaryCE', action='store_true')
parser.add_argument('-weighted_BCE', action='store_true')
parser.add_argument('-mask_off_diags', action='store_true')


args = parser.parse_args()

subset = int(args.subset)
state_input = 'Image'
reverse_geom = int(args.reverse_geom)
use_action = int(args.use_action)
obj_num = int(args.obj_num)
epochs = int(args.epochs)
baseline = args.baseline
save_every = int(args.save_every)

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
    print(start_index)
    print(pos_index)
    print("next")
    return state, desc, action

  def __len__(self):
    return self.length





# Model , Optimizer, Loss


# args.room = ""
data_root = "../our_data/pick2/{}/".format(str(args.room))
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

writer = SummaryWriter("../tb_results/" + str(args.room) + "/use_action_{action}/lr_{lr}/{exp}/{now}".format(exp = args.exp, lr = args.lr, action = str(bool(use_action)), now = dt_string))

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

  #make such that pick doesn't take place at the very beginning 
  ind_greater = (indices.numpy() > 0).flatten()
  #make such that pick doesn't take place at the very end 
  ind_smaller = (indices.numpy() < 19).flatten()
  ind_inrange = np.logical_and(ind_greater, ind_smaller)

  #np.delete 
  desc_ds = desc_ds[ind_inrange,:]
  state_ds = state_ds[ind_inrange, ...]
  action_ds = action_ds[ind_inrange,:]

  if args.sample_more_good:
    tmp = action_ds == 6
    tmp = torch.Tensor(tmp)
    idx = torch.arange(tmp.shape[1], 0, -1)
    tmp2= tmp * idx
    indices = torch.argmax(tmp2, 1, keepdim=True)
    indices.numpy()


    ind_good = (indices.numpy() < 5).flatten()

    ind_ok = (indices.numpy() <= 10).flatten()

    ind_bad = (indices.numpy() > 10).flatten()

    ind_bad.sum()

    #find trajectories that end with less than 5
    desc_ds_good = desc_ds[ind_good,:]
    state_ds_good = state_ds[ind_good, ...]
    action_ds_good = action_ds[ind_good,:]

    #sample number of bad trajectories greater than 5
    desc_ds_good = desc_ds_good[np.random.randint(desc_ds_good.shape[0], size=ind_bad.sum()), :]
    state_ds_good = state_ds_good[np.random.randint(state_ds_good.shape[0], size=ind_bad.sum()), :]
    action_ds_good = action_ds_good[np.random.randint(action_ds_good.shape[0], size=ind_bad.sum()), :]

    #dataset with less than 10 
    desc_ds = desc_ds[ind_ok,:]
    state_ds = state_ds[ind_ok, ...]
    action_ds = action_ds[ind_ok,:]

    desc_ds = np.concatenate([desc_ds, desc_ds_good])
    state_ds = np.concatenate([state_ds, state_ds_good])
    action_ds = np.concatenate([action_ds, action_ds_good])

  if args.sample_more_bad:
    tmp = action_ds == 6
    tmp = torch.Tensor(tmp)
    idx = torch.arange(tmp.shape[1], 0, -1)
    tmp2= tmp * idx
    indices = torch.argmax(tmp2, 1, keepdim=True)
    indices.numpy()


    ind_good = (indices.numpy() < 5).flatten()

    ind_ok = (indices.numpy() <= 10).flatten()

    ind_bad = (indices.numpy() > 10).flatten()

    
    ind_bad.sum()

    #find trajectories that end with less than 5
    desc_ds_bad = desc_ds[ind_bad,:]
    state_ds_bad = state_ds[ind_bad, ...]
    action_ds_bad = action_ds[ind_bad,:]

    desc_ds_good = desc_ds[ind_good,:]
    state_ds_good = state_ds[ind_good, ...]
    action_ds_good = action_ds[ind_good,:]

    #10% trajs are good 
    frac_good = int(desc_ds.shape[0] * float(args.fraction))
    frac_bad = desc_ds.shape[0] - frac_good


    


    #sample number of bad trajectories greater than 5
    desc_ds_good = desc_ds_good[np.random.randint(desc_ds_good.shape[0], size=frac_good), :]
    state_ds_good = state_ds_good[np.random.randint(state_ds_good.shape[0], size=frac_good), :]
    action_ds_good = action_ds_good[np.random.randint(action_ds_good.shape[0], size=frac_good), :]


    desc_ds_bad = desc_ds_bad[np.random.randint(desc_ds_bad.shape[0], size=frac_bad), :]
    state_ds_bad = state_ds_bad[np.random.randint(state_ds_bad.shape[0], size=frac_bad), :]
    action_ds_bad = action_ds_bad[np.random.randint(action_ds_bad.shape[0], size=frac_bad), :]
    
    
    desc_ds = np.concatenate([desc_ds_bad, desc_ds_good])
    state_ds = np.concatenate([state_ds_bad, state_ds_good])
    action_ds = np.concatenate([action_ds_bad, action_ds_good])
  
  

  
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

  if use_action == 1:
    
    if baseline == 'LCBC':

      model = LCBC(desc_num_classes, action_num_classes)
    
    if baseline == 'LCBC_OuterProduct':
      model = LCBC_OuterProduct(desc_num_classes, action_num_classes)


    #ClassifierCRActionBasic
    #ClassifierCRActionResnetDropout - best so far
    #ClassifierCRActionResnetDropout_FuseAction - looks even better for now

    trainset = ProperSampleActionDataset(train_state, train_desc, train_action)
    valset = ProperSampleActionDataset(val_state, val_desc, val_action)



  elif use_action == 0:
    model = ClassifierCRDropout(desc_num_classes)

    # model_path = "./model_weights/pick2/pick_and_place_simple-AlarmClock-None-Desk-307/ClassifierCRDropout_300_moredata_fixed.pth"
    # model.load_state_dict(torch.load(model_path))

    trainset = ProperSampleActionDataset(train_state, train_desc, train_action)
    valset = ProperSampleActionDataset(val_state, val_desc, val_action)

  elif use_action == 2:
    model = NewModel2(desc_num_classes, action_num_classes)
    # model_path = "./model_weights/pick2/pick_and_place_simple-AlarmClock-None-Desk-307/ClassifierCRDropout_300_moredata_fixed.pth"
    # model.load_state_dict(torch.load(model_path))

    # model = nn.Sequential(*list(model.children())[:-1])

    trainset = ProperSampleActionDataset(train_state, train_desc, train_action)
    valset = ProperSampleActionDataset(val_state, val_desc, val_action)




if use_cuda:
  model = model.cuda()



learning_rate = float(args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
pos_weight = torch.ones([1]).to(device)
loss_fn = nn.NLLLoss()

if args.binaryCE:
  if args.weighted_BCE or args.mask_off_diags: 
    loss_fn = nn.BCEWithLogitsLoss(reduce=False)
  else: 
    loss_fn = nn.BCEWithLogitsLoss()



batch_size = 64

for epoch in range(epochs):
  print('Epoch: {}'.format(epoch))
  losses = []
  accur = []
  accur_cond = []
  accur_marg = []
  losses_cond = []
  losses_marg = []

  prop = []

  trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers = 4)
  valloader =  DataLoader(valset, batch_size=batch_size,shuffle=True, num_workers = 4)
  for j,batch in enumerate(tqdm(trainloader)):

    state, desc, action = batch
    state = state.to(device)
    desc = desc.to(device)
    action = action.to(device)
    action_one_hot = torch.nn.functional.one_hot(action.long(), action_num_classes)


    #desc = desc.sum(dim=1)
    #sparsity
    for i, val in enumerate(desc):
      desc[i] = int(np.where(desc_all == int(val))[0])

    desc = torch.nn.functional.one_hot(desc.long(), num_classes=desc_num_classes ) #check

    if use_action == 1:

      output = model(state, desc, action_one_hot)

    #calculate loss
    
    if args.binaryCE:

      if baseline == "LCBC_OuterProduct":
          curr_batch_size = desc.shape[0]
          mask = torch.eye(curr_batch_size).unsqueeze(2).expand(-1,-1,7).cuda()
          action =  torch.nn.functional.one_hot(action.long(), num_classes=action_num_classes ).float()
          label = action.unsqueeze(1).expand(-1,curr_batch_size,-1) *  mask

          pdb.set_trace()

          # Y = torch.zeros((batch_size, batch_size, action_num_classes))
          # for i in range(batch_size):
          #     Y[i, i] = action[i]

          #check with (Y.cuda() != label).sum()
          loss = loss_fn(output.squeeze(), label)
          if args.mask_off_diags:
            loss = loss * mask
            loss = (loss * mask).sum()/(mask.sum())

          if args.weighted_BCE: 
            loss = loss * (label*454)
            loss = loss.mean()

      else:
        loss = loss_fn(output.squeeze(), torch.nn.functional.one_hot(action.long(), num_classes=action_num_classes ).float())
    else:
      pdb.set_trace()
      loss = loss_fn(output.squeeze(), action.long())

    #accuracy
    with torch.no_grad():
      predictions = torch.argmax(output, dim = 1)
      acc = torch.mean((predictions == action).float())


    prop_ones =  predictions.float().mean().detach()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    prop.append(prop_ones)
    losses.append(loss.detach())
    accur.append(acc.detach())


  epoch_loss = torch.mean(torch.stack(losses))
  epoch_acc = torch.mean(torch.stack(accur))
  epoch_prop = torch.mean(torch.stack(prop))




  # with torch.set_grad_enabled(False):
  #   val_losses = []
  #   val_accur = []
  #   val_prop = []

  #   val_accur_cond = []
  #   val_accur_marg = []
  #   val_losses_cond = []
  #   val_losses_marg = []


  #   for i,batch in enumerate(tqdm(valloader)):
  #     if use_action:
  #       state, desc, action = batch
  #       state = state.to(device)
  #       desc = desc.to(device)
  #       action = action.to(device)
  #       action_one_hot = torch.nn.functional.one_hot(action.long(), action_num_classes)
        

  #     #desc = desc.sum(dim=1)

  #     #sparsity
  #     for i, val in enumerate(desc):
  #       desc[i] = int(np.where(desc_all == int(val))[0])

  #     desc = torch.nn.functional.one_hot(desc.long(), num_classes=desc_num_classes)
  #     if use_action:
  #       output = model(state, desc, action_one_hot)
  #     elif use_action == 0:
  #       output = model(state, desc)

  #     elif use_action == 2:
  #       output = model(state, desc)
  #     #calculate loss

  #     if args.binaryCE:

  #       if baseline == "LCBC_OuterProduct":
  #           curr_batch_size = desc.shape[0]
  #           mask = torch.eye(curr_batch_size).unsqueeze(2).expand(-1,-1,7).cuda()
  #           action =  torch.nn.functional.one_hot(action.long(), num_classes=action_num_classes ).float()
  #           label = action.unsqueeze(1).expand(-1,curr_batch_size,-1) *  mask
  #           val_loss = loss_fn(output.squeeze(), label)

  #       else:
  #         val_loss = loss_fn(output.squeeze(), torch.nn.functional.one_hot(action.long(), num_classes=action_num_classes ).float())
  #     else:
  #       val_loss = loss_fn(output.squeeze(), action.long())

      
  #     #accuracy
  #     val_predicted = torch.argmax(output, dim = 1)
  #     val_acc = torch.mean((val_predicted == action).float())
  #     val_prop_ones =  val_predicted.float().mean().detach()



  #     val_losses.append(val_loss.detach())
  #     val_prop.append(val_prop_ones)
  #     val_accur.append(val_acc.detach())



  #   #
  #   epoch_val_loss = torch.mean(torch.stack(val_losses))
  #   epoch_val_acc = torch.mean(torch.stack(val_accur))

  # print("accuracy:", epoch_acc)
  # print("loss:", epoch_loss)
  # print("val_accuracy:", epoch_val_acc)
  # print("val_loss:", epoch_val_loss)
  # writer.add_scalar("Accuracy/train", epoch_acc, epoch)
  # writer.add_scalar("Accuracy/test", epoch_val_acc, epoch)

  # # writer.add_scalar("Accuracy_Cond/train", epoch_acc_cond, epoch)
  # # writer.add_scalar("Accuracy_Cond/test", epoch_val_acc_cond, epoch)

  # # writer.add_scalar("Accuracy_Marg/train", epoch_acc_marg, epoch)
  # # writer.add_scalar("Accuracy_Marg/test", epoch_val_acc_marg, epoch)

  # writer.add_scalar("Loss/train", epoch_loss, epoch)
  # writer.add_scalar("Loss/test", epoch_val_loss, epoch)

  # writer.add_scalar("Loss_Cond/train", epoch_loss_cond, epoch)
  # writer.add_scalar("Loss_Cond/test", epoch_val_loss_cond, epoch)

  # writer.add_scalar("Loss_Marg/train", epoch_loss_marg, epoch)
  # writer.add_scalar("Loss_Marg/test", epoch_val_loss_marg, epoch)

  # writer.add_scalar("Loss_Cond_Over_Marg/train", epoch_loss_cond/epoch_loss_marg, epoch)
  # writer.add_scalar("Loss_Cond_Over_Marg/test", epoch_val_loss_cond/epoch_val_loss_marg, epoch)




  if epoch % save_every == 0:

        #make data directory
    weight_dir = "../model_weights/clean/pick/{}/{}/".format(args.exp, args.room)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        final_desc_path = weight_dir + "pick_desc_all.npy"
        np.save(final_desc_path, desc_all) 

    final_weight_path = weight_dir + "{model}_{obj_num}_pick.pth".format(model = model.__class__.__name__+"_epoch["+str(epoch)+"]", obj_num = obj_num)
    

    
    torch.save(model.state_dict(), final_weight_path)

    # queue_path = './queue.pkl'
    # if not os.path.exists(queue_path):
    #   queue = deque([]) 
    #   queue.append(final_weight_path)
    #   with open(queue_path, 'wb') as f:
    #     pickle.dump(queue, f, pickle.HIGHEST_PROTOCOL)
    #   #make queue
    #   #save queue
       
    # else: 
    #   #Open existing
    #   with open(queue_path, 'r+b') as h:
        
    #     queue = pickle.load(h)
    #     pdb.set_trace()
    #   #Update the values
    #     queue.append(final_weight_path)
    #     pickle.dump(queue, h)

      # #Save
      # with open(queue_path, 'wb') as h:
          


      #read queue
      #add to queue

    #{model}_moredata_intermediate.pth