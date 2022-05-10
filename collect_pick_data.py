import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import pdb
from matplotlib import pyplot as plt

from collections import Counter
from random import sample
from tqdm import tqdm
import alfworld.gen.constants as constants
from alfworld.gen.utils import game_util
import os
import argparse
from natsort import natsorted

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-room_ind', required = True)
parser.add_argument('-subset', default = "200")

args = parser.parse_args()
room_ind = int(args.room_ind)
subset = int(args.subset)

trajs = [ 
 'pick_and_place_simple-AlarmClock-None-Desk-307/trial_T20190907_072303_146844',
 'pick_and_place_simple-Tomato-None-Microwave-13/trial_T20190908_125158_098734',
 'pick_and_place_simple-Plunger-None-Cabinet-403/trial_T20190909_054715_217064']

# load config
traj_name = trajs[room_ind]

config = {'dataset': {'data_path': '$ALFWORLD_DATA/json_2.1.1/train/{}'.format(traj_name), 'eval_id_data_path': '$ALFWORLD_DATA/json_2.1.1/valid_seen', 'eval_ood_data_path': '$ALFWORLD_DATA/json_2.1.1/valid_unseen', 'num_train_games': 1, 'num_eval_games': -1}, 'logic': {'domain': '$ALFWORLD_DATA/logic/alfred.pddl', 'grammar': '$ALFWORLD_DATA/logic/alfred.twl2'}, 'env': {'type': 'AlfredThorEnv', 'regen_game_files': False, 'domain_randomization': False, 'task_types': [1, 2, 3, 4, 5, 6], 'expert_timeout_steps': 150, 'expert_type': 'handcoded', 'goal_desc_human_anns_prob': 0.0, 'hybrid': {'start_eps': 100000, 'thor_prob': 0.5, 'eval_mode': 'tw'}, 'thor': {'screen_width': 300, 'screen_height': 300, 'smooth_nav': False, 'save_frames_to_disk': False, 'save_frames_path': './videos/'}}, 'controller': {'type': 'oracle_astar', 'debug': False, 'load_receps': True}, 'mask_rcnn': {'pretrained_model_path': '$ALFWORLD_DATA/detectors/mrcnn.pth'}, 'general': {'random_seed': 42, 'use_cuda': True, 'visdom': False, 'task': 'alfred', 'training_method': 'dagger', 'save_path': './training/', 'observation_pool_capacity': 3, 'hide_init_receptacles': False, 'training': {'batch_size': 10, 'max_episode': 50000, 'smoothing_eps': 0.1, 'optimizer': {'learning_rate': 0.001, 'clip_grad_norm': 5}}, 'evaluate': {'run_eval': True, 'batch_size': 10, 'env': {'type': 'AlfredTWEnv'}}, 'checkpoint': {'report_frequency': 1000, 'experiment_tag': 'test', 'load_pretrained': False, 'load_from_tag': 'not loading anything'}, 'model': {'encoder_layers': 1, 'decoder_layers': 1, 'encoder_conv_num': 5, 'block_hidden_dim': 64, 'n_heads': 1, 'dropout': 0.1, 'block_dropout': 0.1, 'recurrent': True}}, 'rl': {'action_space': 'admissible', 'max_target_length': 20, 'beam_width': 10, 'generate_top_k': 3, 'training': {'max_nb_steps_per_episode': 50, 'learn_start_from_this_episode': 0, 'target_net_update_frequency': 500}, 'replay': {'accumulate_reward_from_final': True, 'count_reward_lambda': 0.0, 'novel_object_reward_lambda': 0.0, 'discount_gamma_game_reward': 0.9, 'discount_gamma_count_reward': 0.5, 'discount_gamma_novel_object_reward': 0.5, 'replay_memory_capacity': 500000, 'replay_memory_priority_fraction': 0.5, 'update_per_k_game_steps': 5, 'replay_batch_size': 64, 'multi_step': 3, 'replay_sample_history_length': 4, 'replay_sample_update_from': 2}, 'epsilon_greedy': {'noisy_net': False, 'epsilon_anneal_episodes': 1000, 'epsilon_anneal_from': 0.3, 'epsilon_anneal_to': 0.1}}, 'dagger': {'action_space': 'generation', 'max_target_length': 20, 'beam_width': 10, 'generate_top_k': 5, 'unstick_by_beam_search': False, 'training': {'max_nb_steps_per_episode': 50}, 'fraction_assist': {'fraction_assist_anneal_episodes': 50000, 'fraction_assist_anneal_from': 1.0, 'fraction_assist_anneal_to': 0.01}, 'fraction_random': {'fraction_random_anneal_episodes': 0, 'fraction_random_anneal_from': 0.0, 'fraction_random_anneal_to': 0.0}, 'replay': {'replay_memory_capacity': 500000, 'update_per_k_game_steps': 5, 'replay_batch_size': 64, 'replay_sample_history_length': 4, 'replay_sample_update_from': 2}}, 'vision_dagger': {'model_type': 'resnet', 'resnet_fc_dim': 64, 'maskrcnn_top_k_boxes': 10, 'use_exploration_frame_feats': False, 'sequence_aggregation_method': 'average'}}
env_type = 'AlfredThorEnv' # config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()



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



# to get data




actions_other = [       
        {'action': 'OpenObject'},
        {'action': 'CloseObject'},
        {'action': 'PutObject'},
        {'action': 'ToggleObjectOn'},
        {'action': 'ToggleObjectOff'},
        {'action': 'SliceObject'},
]

all_objects = None


dataset_recep = []
dataset_state = []
dataset_desc = []
dataset_action= []

num_traj = 100
parts = subset

list_receps = None



prev_index = natsorted(os.listdir('./our_data/pick2/{}'.format(traj_name.split("/")[0])))
start = int(prev_index[-1][12:-4])


for part in range(start, parts + 1):
    print("Part {}/{} is being downloaded".format(part, parts))
    dataset_state = []
    dataset_desc = []
    dataset_action= []
    for i in tqdm(range(0, num_traj)):
#         print(i)
        traj = 0
        obs, info = env.reset()
        state_vec = []
        desc_vec = []
        recep_vec = []
        action_vec = []
        pickable = []
        
        #initialize all of this here.
        game_state = env.envs[0].controller.navigator.game_state
        event = game_state.env.last_event
        pose = game_util.get_pose(event)
        oracleagent = env.envs[0].controller

        actions = [{'action': 'RotateRight'}, {'action': 'RotateLeft'}, {'action': 'MoveBack', 'moveMagnitude': 0.25}, {'action': 'MoveAhead', 'moveMagnitude': 0.25}, {'action': 'LookUp'},  {'action': 'LookDown'}, {'action': 'PickupObject'}]
        
        picked = False
        while traj < 20:
            all_objs = [x['objectId'] for x in oracleagent.env.last_event.metadata['objects']]

            pick_objs = {}
    
            for x in oracleagent.env.last_event.metadata['objects']:
                if x['pickupable'] == True and x['visible'] == True:
                    pick_objs[x['objectId']] = x['distance']
            pick_objs = sorted(pick_objs.items(), key=lambda x: x[1])

            #current state
            image = env.get_frames()
            image = image.squeeze(0)
#             

            visible_objects, visible_objects_id, visible_recep, visible_recep_id, feedback =  no_recep_print_frame(oracleagent, pose)
            
        
            #pickup 
            
            action_idx = np.random.randint(len(actions))
            action = actions[action_idx]
            
            
            #select pickup action
            if action['action'] == 'PickupObject': 
                #find pickable objects
                
                if len(pick_objs) == 0:
                    #do nothing
                    curr_lang = len(all_objs)
                    
                else:
                    picked_obj = pick_objs[0][0] #sample(curr_pickable, 1)[0]
                    action = {'action': "PickupObject",'objectId': picked_obj, 'forceAction': True}
                    
                    curr_lang = all_objs.index(picked_obj)

                    
                    #no more picks - hardcoded
                    actions = actions[:-1]
                    try:
                        game_state.step(action)
                        picked = True 
                        picked_idx = curr_lang
                    except Exception:
                        print("error HERE")
                        continue 
                    
                
            else: 
                curr_lang = len(all_objs)
                game_state.step(action)
            
            if picked == True:
                curr_lang = picked_idx
            
            # print("lang:", curr_lang)
            # print("actions:",action_idx)
                
                
            action_vec.append(action_idx)
            desc_vec.append(curr_lang)
            state_vec.append(image)
            
#             plt.imshow(image, interpolation='nearest')
#             plt.show()
        
#             print(curr_lang)
            
            traj += 1 
        
        

        state_vec = np.stack(state_vec)
        desc_vec = np.stack(desc_vec)
        action_vec = np.stack(action_vec)

        dataset_state.append(state_vec)
        dataset_desc.append(desc_vec) 
        dataset_action.append(action_vec)
        
        if (i+1) % 50 == 0: 
            path = './our_data/pick2/{}'.format(traj_name.split("/")[0])
            os.makedirs(path, exist_ok=True) 
            
            np.save('./our_data/pick2/{}/state1k_part{}'.format(traj_name.split("/")[0], part), np.stack(dataset_state))
            np.save('./our_data/pick2/{}/desc1k_part{}'.format(traj_name.split("/")[0], part), np.stack(dataset_desc))
            np.save('./our_data/pick2/{}/action1k_part{}'.format(traj_name.split("/")[0], part), np.stack(dataset_action))

