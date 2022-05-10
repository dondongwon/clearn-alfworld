
import os 
import time 
import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import pdb
from collections import Counter
from random import sample
from tqdm import tqdm
import alfworld.gen.constants as constants
import torch
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import PIL.Image
import PIL.ImageDraw
import tqdm
import json
from constants import *
from alfworld.gen.utils import game_util
import random
import pandas as pd
from IPython.display import display, HTML
import pickle5 as pickle
from natsort import natsorted

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from model import *
from IPython.display import HTML

#open counter file to measure oOD 
with open('state_act_pair_10_imbalanced_dict.pickle', 'rb') as handle:
    counter = pickle.load(handle)

trajs = ['pick_and_place_simple-Watch-None-CoffeeTable-201',
 'pick_and_place_simple-CellPhone-None-SideTable-317',
 'pick_and_place_simple-Tomato-None-Microwave-13',
 'pick_and_place_simple-Statue-None-CoffeeTable-222',
 'pick_and_place_simple-Newspaper-None-Ottoman-203',
 'pick_and_place_simple-SoapBottle-None-GarbageCan-421',
 'pick_and_place_simple-Candle-None-Toilet-405',
 'pick_and_place_simple-AlarmClock-None-Desk-307/trial_T20190907_072303_146844',
 'pick_and_place_simple-Knife-None-SideTable-3',
 'pick_and_place_simple-Plunger-None-Cabinet-403']

# load config
traj_name = trajs[7]

config = {'dataset': {'data_path': '$ALFWORLD_DATA/json_2.1.1/train/{}'.format(traj_name), 'eval_id_data_path': '$ALFWORLD_DATA/json_2.1.1/valid_seen', 'eval_ood_data_path': '$ALFWORLD_DATA/json_2.1.1/valid_unseen', 'num_train_games': 1, 'num_eval_games': -1}, 'logic': {'domain': '$ALFWORLD_DATA/logic/alfred.pddl', 'grammar': '$ALFWORLD_DATA/logic/alfred.twl2'}, 'env': {'type': 'AlfredThorEnv', 'regen_game_files': False, 'domain_randomization': False, 'task_types': [1, 2, 3, 4, 5, 6], 'expert_timeout_steps': 150, 'expert_type': 'handcoded', 'goal_desc_human_anns_prob': 0.0, 'hybrid': {'start_eps': 100000, 'thor_prob': 0.5, 'eval_mode': 'tw'}, 'thor': {'screen_width': 300, 'screen_height': 300, 'smooth_nav': False, 'save_frames_to_disk': False, 'save_frames_path': './videos/'}}, 'controller': {'type': 'oracle_astar', 'debug': False, 'load_receps': True}, 'mask_rcnn': {'pretrained_model_path': '$ALFWORLD_DATA/detectors/mrcnn.pth'}, 'general': {'random_seed': 42, 'use_cuda': True, 'visdom': False, 'task': 'alfred', 'training_method': 'dagger', 'save_path': './training/', 'observation_pool_capacity': 3, 'hide_init_receptacles': False, 'training': {'batch_size': 10, 'max_episode': 50000, 'smoothing_eps': 0.1, 'optimizer': {'learning_rate': 0.001, 'clip_grad_norm': 5}}, 'evaluate': {'run_eval': True, 'batch_size': 10, 'env': {'type': 'AlfredTWEnv'}}, 'checkpoint': {'report_frequency': 1000, 'experiment_tag': 'test', 'load_pretrained': False, 'load_from_tag': 'not loading anything'}, 'model': {'encoder_layers': 1, 'decoder_layers': 1, 'encoder_conv_num': 5, 'block_hidden_dim': 64, 'n_heads': 1, 'dropout': 0.1, 'block_dropout': 0.1, 'recurrent': True}}, 'rl': {'action_space': 'admissible', 'max_target_length': 20, 'beam_width': 10, 'generate_top_k': 3, 'training': {'max_nb_steps_per_episode': 50, 'learn_start_from_this_episode': 0, 'target_net_update_frequency': 500}, 'replay': {'accumulate_reward_from_final': True, 'count_reward_lambda': 0.0, 'novel_object_reward_lambda': 0.0, 'discount_gamma_game_reward': 0.9, 'discount_gamma_count_reward': 0.5, 'discount_gamma_novel_object_reward': 0.5, 'replay_memory_capacity': 500000, 'replay_memory_priority_fraction': 0.5, 'update_per_k_game_steps': 5, 'replay_batch_size': 64, 'multi_step': 3, 'replay_sample_history_length': 4, 'replay_sample_update_from': 2}, 'epsilon_greedy': {'noisy_net': False, 'epsilon_anneal_episodes': 1000, 'epsilon_anneal_from': 0.3, 'epsilon_anneal_to': 0.1}}, 'dagger': {'action_space': 'generation', 'max_target_length': 20, 'beam_width': 10, 'generate_top_k': 5, 'unstick_by_beam_search': False, 'training': {'max_nb_steps_per_episode': 50}, 'fraction_assist': {'fraction_assist_anneal_episodes': 50000, 'fraction_assist_anneal_from': 1.0, 'fraction_assist_anneal_to': 0.01}, 'fraction_random': {'fraction_random_anneal_episodes': 0, 'fraction_random_anneal_from': 0.0, 'fraction_random_anneal_to': 0.0}, 'replay': {'replay_memory_capacity': 500000, 'update_per_k_game_steps': 5, 'replay_batch_size': 64, 'replay_sample_history_length': 4, 'replay_sample_update_from': 2}}, 'vision_dagger': {'model_type': 'resnet', 'resnet_fc_dim': 64, 'maskrcnn_top_k_boxes': 10, 'use_exploration_frame_feats': False, 'sequence_aggregation_method': 'average'}}
env_type = 'AlfredThorEnv' # config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'



# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
    
import pandas as pd


# weights_dir = "./model_weights/clean/pick/LCBC/pick_and_place_simple-AlarmClock-None-Desk-307"
# weights_dir = "./model_weights/clean/pick/LCBC_100_2/pick_and_place_simple-AlarmClock-None-Desk-307"
weights_dir = "./model_weights/clean/pick/LCBC_sample_more_bad/pick_and_place_simple-AlarmClock-None-Desk-307"
#[ 0,  2,  6,  7,  8, 21, 22, 25, 29, 30, 33, 41, 43, 44, 47]
objs_100 = [ 0,  2,  6,  7,  8, 21, 22, 25, 29, 30, 33, 41, 43, 44, 47]
objs_good = [ 0,  2,  6,  7,  8, 21, 22, 25, 29, 30, 33, 41, 43, 44, 47]
objs_bad = [ 0,  2,  6,  7,  8, 21, 22, 25, 29, 33, 41, 43, 44, 47]
objs = objs_bad

res_path = weights_dir + '/results_ood_bad2.csv'

if os.path.exists(res_path) == False:
    Done = []
    col = ['model','epoch']

    #objs = [ 0,  2,  6,  7,  8, 12, 21, 22, 25, 29, 30, 33, 41, 43, 44, 47]
    
    # objs = [ 0,  2,  6,  7,  8, 21, 22, 25, 29, 30, 33, 41, 43, 44, 47]
    for obj in range(len(objs)):
        obj = str(obj)
        col.append("pick_{}".format(obj))
        
    df = pd.DataFrame(columns=col)

else: 
    df = pd.read_csv(res_path)
    Done = df.model.to_list()
    

NotDone = os.listdir(weights_dir)
NotDone = [x for x in NotDone if '.npy' not in x]
NotDone = [x for x in NotDone if '.csv' not in x]
NotDone = natsorted(NotDone, reverse = True)

pdb.set_trace()


def no_recep_print_frame(oracleagent, loc):
    inst_color_count, inst_color_to_object_id = oracleagent.get_instance_seg()
    #recep_object_id = recep['object_id']

    # for each unique seg add to object dictionary if it's more visible than before
    visible_objects = []
    visible_objects_id = [] 
    visible_recep = [] 
    visible_recep_id = [] 
    
    #want it to be visible 

    c = inst_color_count
    new_inst_color_count = Counter({k: c for k, c in c.items() if c >= 10})
    
    for color, num_pixels in new_inst_color_count.most_common():
        if color in inst_color_to_object_id:
            
            
            object_id = inst_color_to_object_id[color]
            object_type = object_id.split("|")[0]
            object_metadata = oracleagent.get_obj_id_from_metadata(object_id)
            is_obj_in_recep = (object_metadata and object_metadata['parentReceptacles'] and len(object_metadata['parentReceptacles']) > 0) # and recep_object_id in object_metadata['parentReceptacles'])
            if object_id in oracleagent.receptacles.keys():
                
                visible_recep_id.append(object_id)
                visible_recep.append(oracleagent.receptacles[object_id]['num_id'])
            
            
            if object_type in oracleagent.OBJECTS and object_metadata and (not oracleagent.use_gt_relations or is_obj_in_recep):
                if object_id not in oracleagent.objects:
                    oracleagent.objects[object_id] = {
                        'object_id': object_id,
                        'object_type': object_type,
                        #'parent': recep['object_id'],
                        'loc': loc,
                        'num_pixels': num_pixels,
                        'num_id': "%s %d" % (object_type.lower() if "Sliced" not in object_id else "sliced-%s" % object_type.lower(),
                                             oracleagent.get_next_num_id(object_type, oracleagent.objects))
                    }
                elif object_id in oracleagent.objects and num_pixels > oracleagent.objects[object_id]['num_pixels']:
                    oracleagent.objects[object_id]['loc'] = loc
                    oracleagent.objects[object_id]['num_pixels'] = num_pixels

                if oracleagent.objects[object_id]['num_id'] not in oracleagent.inventory:
                    visible_objects.append(oracleagent.objects[object_id]['num_id'])
                    visible_objects_id.append(object_id)

    visible_objects_with_articles = ["a %s," % vo for vo in visible_objects]
    feedback = ""
    if len(visible_objects) > 0:
        feedback = "On the Receptacle, you see %s" % (oracleagent.fix_and_comma_in_the_end(' '.join(visible_objects_with_articles))) #recep['num_id']
    elif len(visible_objects) == 0: #not recep['closed'] and
        feedback = "On the Receptacle, you see nothing." #% (recep['num_id'])
    #pdb.set_trace()
    #print(visible_recep)
    return visible_objects, visible_objects_id, visible_recep, visible_recep_id, feedback



def results_success(model,desired_desc_ind, traj_len, rand = False, objs = objs):
    pick_success = 0
    nav_success = 0
    
    actions = [{'action': 'RotateRight'}, {'action': 'RotateLeft'}, {'action': 'MoveBack', 'moveMagnitude': 0.25}, {'action': 'MoveAhead', 'moveMagnitude': 0.25}, {'action': 'LookUp'},  {'action': 'LookDown'}]
    list_receps = ['Bed|+00.27|+00.00|+01.33', 'LaundryHamper|-01.44|+00.01|+01.77', 'Desk|-00.87|-00.01|-02.44', 'Shelf|-01.02|+01.35|-02.54', 'Drawer|-01.41|+00.73|-02.32', 'Drawer|-01.41|+00.48|-02.32', 'Drawer|-01.41|+00.20|-02.32', 'SideTable|+00.25|00.00|-02.38']
    img = [] # some array of images
    frames = [] # for storing the generated images

    traj = 0

    list_receps = None
    desired_desc_ind = desired_desc_ind
    #items in dataset
    obj_ds = objs
    obj_ds_len = len(obj_ds) + 1 
    print(unique_objs[local2globaldict[obj_ds[desired_desc_ind]]])

    traj = 0
    obs, info = env.reset()
    state_vec = []
    desc_vec = []
    action_vec = []
    pickable = []

    #initialize all of this here.
    game_state = env.envs[0].controller.navigator.game_state
    event = game_state.env.last_event
    pose = game_util.get_pose(event)
    oracleagent = env.envs[0].controller

    actions = [{'action': 'RotateRight'}, {'action': 'RotateLeft'}, {'action': 'MoveBack', 'moveMagnitude': 0.25}, {'action': 'MoveAhead', 'moveMagnitude': 0.25}, {'action': 'LookUp'},  {'action': 'LookDown'}, {'action': 'PickupObject'}]
    
    picked = False

    not_in_counter = 0 
    in_counter = 0
    infrequent = 0 


    while traj < traj_len:
        traj += 1 

        desired_desc = torch.zeros(obj_ds_len) #17
        desired_desc[desired_desc_ind] = 1


        #current state
        image = env.get_frames()
        image = torch.from_numpy(image.squeeze(0))

        
        all_objs = [x['objectId'] for x in oracleagent.env.last_event.metadata['objects']]
        pickable_objs = [x['objectId'] for x in oracleagent.env.last_event.metadata['objects'] if x['pickupable'] == True]
        visible_objs = [x['objectId'] for x in oracleagent.env.last_event.metadata['objects'] if x['visible'] == True]
        
        pick_objs = {}
        for x in oracleagent.env.last_event.metadata['objects']:
            if x['pickupable'] == True and x['visible'] == True:
                pick_objs[x['objectId']] = x['distance']
        pick_objs = sorted(pick_objs.items(), key=lambda x: x[1])

    

        if rand == False:
            
            state = image.expand(1, -1,-1,-1)
            desired_desc = desired_desc

            #print(obj_list[desired_desc_ind])
            #pdb.set_trace()

        #     output = model(state, desired_desc, action)

            with torch.no_grad():
                output = model(state, desired_desc, 0)
            #action_ind = torch.argmax(output[:,1])

        #     
        #     

            #p = output[:,1].numpy()
        #     temp_output = torch.exp(output[:,1]*1)
        #     p = temp_output.numpy()
        #     p /= p.sum()
        #     print(p)
        #     action_ind = np.random.choice(len(actions), 1, replace=False, p=p).item()
            action_ind = torch.argmax(output)

            ss = image.flatten().tolist()
            ss = tuple(ss)

            if traj == 1:
                if ss not in counter:
                    env.reset()
                    traj = 0
                    print('weird init!!')
                    continue

            if ss in counter:
                if action_ind in counter[ss]:
                    in_counter += 1
            else:
                not_in_counter += 1
                
    
    
        else:
            action_ind = random.randrange(len(actions))
#             print(actions[action_ind]['action'])


        #select pickup action
        if actions[action_ind]['action'] == 'PickupObject':

            #find pickable objects
            if len(pick_objs) == 0:
                #do nothing
                traj += 1 
                continue

            else:
#                 print(curr_pickable)
                picked_obj = pick_objs[0][0] #sample(curr_pickable, 1)[0]

                action = {'action': "PickupObject",'objectId': picked_obj, 'forceAction': True}
                try:
                    game_state.step(action)
                except Exception:
                    continue 


                image = env.get_frames()
                image = torch.from_numpy(image.squeeze(0))
                plt.imshow(image, interpolation='nearest')
                desired_obj = unique_objs[local2globaldict[obj_ds[desired_desc_ind]]]
                if picked_obj.split("|")[0] == desired_obj:
#                     print("success")
                    pick_success = 1
                
                if desired_obj in [x[0].split("|")[0] for x in pick_objs]:
#                     print("navigation success")
                    nav_success = 1

#                 image = env.get_frames()
#                 image = torch.from_numpy(image.squeeze(0))


                break


        else: 
            game_state.step(actions[action_ind])

        
    print(action_ind)
            
#     plt.imshow(image, interpolation='nearest')
#     plt.show()
#     pdb.set_trace()
    
    return traj, pick_success, nav_success, not_in_counter, in_counter, infrequent


def sim(model, desired_desc_ind, simulations = 1, rand = False):
    #non random
    pick_list = []
    nav_list = []
    traj_list = []
    for traj_len in [20]:
        pick_success_count = 0
        nav_success_count = 0 
#         print('traj_len {}'.format(traj_len))
        for i in range(simulations):
#             print("simulation no: {}".format(i))
            traj, success, nav_success, not_in_counter, in_counter, infrequent = results_success(model, desired_desc_ind, traj_len, rand = rand)
#             print("simulation is successful: {}".format(success))
#             print("navigation is successful: {}".format(nav_success))
            pick_list.append(success)
            nav_list.append(nav_success)
            traj_list.append(traj)
        


    return {'pick_'+ str(desired_desc_ind): pick_list,
            'nav_'+ str(desired_desc_ind): nav_list, 
            'traj_'+ str(desired_desc_ind): traj_list,
            'in_dist_'+ str(desired_desc_ind): in_counter,
            'out_dist_'+ str(desired_desc_ind): not_in_counter,
            }

def dict_mean(dict_list):
    mean_dict = {}
    for key, value in dict_list.items():
        if 'pick' in key:
            mean_dict[key] = sum(value) / len(value)
        if 'traj' in key:
            mean_dict[key] = sum(value) / len(value)
        if 'dist' in key:
            mean_dict[key] = value
    return mean_dict

while True:
    NotDone = os.listdir(weights_dir)
    NotDone = [x for x in NotDone if '.npy' not in x]
    NeedsEval = set(NotDone) - set(Done)
    if len(NeedsEval) > 0:
        for curr_file in NeedsEval:
            try: 
                start_time = time.time()
                full_res_dict = {}
                full_res_dict['model'] = curr_file
                full_res_dict['epoch'] = curr_file[curr_file.find("[")+1 : curr_file.find("]")]
                curr_path = os.path.join(weights_dir, curr_file )
                model_state = torch.load(curr_path, map_location=torch.device('cpu'))
                desc_num_classes = len(objs) + 1 #17
                action_num_classes = 7
                model = LCBC(desc_num_classes, action_num_classes)
                model.load_state_dict(model_state)
                model.eval()

                for obj_ind in range(len(objs)):
                    results = sim(model, obj_ind, simulations = 1, rand = False)
                    results_dict = dict_mean(results)
                    full_res_dict.update(results_dict)
                    
                df2 = full_res_dict
                df = df.append(df2, ignore_index = True)
                
                
                df['epoch']=df.epoch.astype('int64')
                df = df.sort_values(by=['epoch'])
                display(df)
                
                df.to_csv(res_path)
                
                #delete file after eval 
                Done.append(curr_file)
                print("--- %s seconds ---" % (time.time() - start_time))
            except Exception:
                print('bad file!')
                Done.append(curr_file)
                continue
#     if len(NeedsEval) == 0:
#         pdb.set_trace()
        
    
    