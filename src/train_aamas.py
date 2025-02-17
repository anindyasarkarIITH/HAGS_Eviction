import torch
import pandas as pd
from torch import nn
import random
import sys
from src import models
import torch.nn.functional as F
from torch.autograd import Variable
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import pickle
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from torch.distributions.categorical import Categorical
from copy import deepcopy #as c

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


def feat_ext_pretrained():
    res34_model = torchmodels.resnet34(pretrained=True)
    agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
    for param in agent.parameters():
        param.requires_grad = False
    return agent

feat_model = feat_ext_pretrained()

def norm_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train

trans_img = norm_transforms()


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    with open("./image_dict.pkl" , 'rb') as f:
        image_dict = pickle.load(f)
    
    with open("./df_model_train.pkl" , 'rb') as g:
        tab_dict = pickle.load(g)
        
    with open("./df_model_test.pkl" , 'rb') as h:
        tab_dict_test = pickle.load(h)
    
    with open("./np_test.pkl" , 'rb') as i:
        task_dict_test = pickle.load(i)
        
    with open("./np_train.pkl" , 'rb') as j:
        task_dict_train = pickle.load(j)
        
    model_pred = getattr(models, hyp_params.model+'Model')(hyp_params)
    model_search = getattr(models, hyp_params.model_search)()
    if hyp_params.use_cuda:
        model_pred = model_pred.cuda()
        model_search = model_search.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model_pred.parameters(), lr=hyp_params.lr)
    optimizer_search = getattr(optim, hyp_params.optim)(model_search.parameters(), lr=hyp_params.lr)
    
    criterion = getattr(nn, hyp_params.criterion)()
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model_pred': model_pred,
                'model_search': model_search,
                'optimizer': optimizer,
                'optimizer_search': optimizer_search,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader, image_dict, tab_dict, tab_dict_test, task_dict_test, task_dict_train)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader, image_dict, tab_dict, tab_dict_test, task_dict_test, task_dict_train):
    model_pred = settings['model_pred']
    model_search = settings['model_search']
    optimizer = settings['optimizer']
    optimizer_search = settings['optimizer_search']
    criterion = settings['criterion']      
    scheduler = settings['scheduler']  
    def train(er, epoch, model_pred, model_search, optimizer, optimizer_search, criterion):
        # select a random search budget within a range at the start of every training epochs
        search_budget = random.randint(12, 28) 
        epoch_loss = 0
        model_pred.train()
        model_search.train()
        start_time = time.time()
        print ("training starts")
        # initialize lists that holds the record of search performance
        rewards, total_reward, policies = [], [], []
        #for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        #### Create the batch data
        task_tab_ft = list(); task_label_ft = list()      
        batch_img = list(); batch_tab = list(); batch_target = list()
        counter = 0 ; task_size = 100
        batch_id = 0; batch_size = 1;evaluate_counter = 1
        eval_img_data = list(); eval_tab_data = list(); eval_label_data = list();
        for t_num in range(int(task_dict_train.shape[0])):
            
            batch_flag = True
            task_img_ft = list()
            tolemi_id_list = task_dict_train[t_num,:,er]
            task_tab_ft = tab_dict[tab_dict['ta_id'].isin(tolemi_id_list)].iloc[:,1:43]
            task_label_ft = tab_dict[tab_dict['ta_id'].isin(tolemi_id_list)]['target']
            task_tab_ft = task_tab_ft.to_numpy()
            task_label_ft = task_label_ft.to_numpy()
            for tolemi_id in tolemi_id_list: 
                img_ft = np.reshape(image_dict[tolemi_id], (int(image_dict[tolemi_id].shape[1]), int(image_dict[tolemi_id].shape[2]), int(image_dict[tolemi_id].shape[0])))
                img_ft = cv2.resize(img_ft, dsize=(448, 448)) #fixed reshape size
                # Transform the image
                img_ft = trans_img(img_ft).unsqueeze(0)
                out_embedding = feat_model.forward(img_ft.cuda())
                out_img_ft = F.adaptive_avg_pool2d(out_embedding, (1, 1)).squeeze()
                task_img_ft.append(out_img_ft.cpu())
            img_ft_i = np.stack(task_img_ft, axis = 0) 
            vis_i = torch.from_numpy(img_ft_i).unsqueeze(0).cuda()
            target_label = torch.from_numpy(task_label_ft).unsqueeze(0).cuda()
            tab_i = torch.from_numpy(task_tab_ft).unsqueeze(0).cuda() 
            
            model_pred.zero_grad()
            model_search.zero_grad()
            batch_size = 1 
            combined_loss = 0
            
            #data parallel model for faster training
            model_pred = nn.DataParallel(model_pred) if batch_size > 10 else model_pred
            model_search = nn.DataParallel(model_search) if batch_size > 10 else model_search
            
            ## create a copy of the prediction module for meta learning
            model_pred_meta = deepcopy(model_pred)
            optimizer_pred_meta = optim.Adam(model_pred_meta.parameters(), lr=0.00001)
            
            # stores the information of previous search queries
            search_info = torch.zeros(int(vis_i.shape[0]), int(vis_i.shape[1])).cuda()
            #store the information about the remaining query
            query_info = torch.zeros(int(vis_i.shape[0])).cuda()   
            # stores the information of previously selected grids as target 
            mask_info = torch.ones((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            
            # Active Target Information representation
            act_target_label = torch.zeros((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            final_target_label = torch.zeros((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
           
            # initialize lists to store intermediate training trends
            store_policy_out = []; adv_store = [] ; log_prob_store = []; policy_loss = []; search_history = []; ce_loss = []
            
            # Start Active Search Iteration
            for step_ in range(search_budget):
                query_remain = search_budget - step_
                #store the information about the remaining query
                query_left = torch.add(query_info, query_remain).cuda()
                
                # Forward pass through the prediction module
                preds_i, hiddens_i = model_pred_meta(tab_i.float(), vis_i.float(), search_info)
                # Forward pass through the search module
                logit = model_search(search_info, query_left, preds_i.detach())
                
                # Apply a softmax to obtain a probability distribution over grids
                grid_prob_net = preds_i.view(preds_i.size(0), -1)
                probs = F.softmax(logit, dim = 1)
                    
                # we mask the probability dist. and assign 0 probability to the grids that is already selected by the agent 
                mask_probs = (probs * mask_info.clone())
                
                # sample a grid to query
                distr = Categorical(mask_probs)
                policy_sample = distr.sample()
                store_policy_out.append(policy_sample)

                # Random policy - used as baseline policy in the training step
                policy_map = torch.randint(0, int(vis_i.shape[1]), (int(vis_i.shape[0]),)) 
                
                # Find the reward for the baseline policy
                reward_map = compute_reward(target_label, policy_map.data)
                # Find the reward for the sampled policy
                reward_sample = compute_reward(target_label, policy_sample.data)
                rewards.append(reward_sample)
                
                for sample_id in range(int(vis_i.shape[0])):
                    # Update the search history and active target label based on the current reward and action
                    if (int(reward_sample[sample_id]) == 1):
                        search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_sample[sample_id])
                        act_target_label[sample_id, int(policy_sample[sample_id].data)] = 1
                    else:
                        search_info[sample_id, int(policy_sample[sample_id].data)] = -1
                        act_target_label[sample_id, int(policy_sample[sample_id].data)] = 0
                        
                    # Update the mask info based on the current action taken by the agent
                    mask_info[sample_id, int(policy_sample[sample_id].data)] = 0.0001
                    ## final active target representation to train prediction module
                    for out_idx in range(int(vis_i.shape[1])):
                        if (mask_info[sample_id, out_idx] == 0):
                            final_target_label[sample_id, out_idx] = act_target_label[sample_id, out_idx].data
                        else:
                            final_target_label[sample_id, out_idx] = preds_i[sample_id, out_idx].data
                 
                # BCE loss for the prediction module
                loss_cls_ml = (0.1) * (criterion(grid_prob_net.float(), final_target_label.float().cuda()))  
                # update the pred policy network parameters after every query
                optimizer_pred_meta.zero_grad()
                loss_cls_ml.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model_pred_meta.parameters(), hyp_params.clip)
                optimizer_pred_meta.step()   
                
                # Compute the advantage value
                advantage = reward_sample.cuda().float() #- reward_map.cuda().float()
                adv_store.append(advantage)
                # Find the loss for only the policy network (REINFORCE objective -> grad_log_pi)
                loss = -distr.log_prob(policy_sample)           
                log_prob_store.append(loss)
                    
        
                policy_loss = []; temp = []
            # Compute the sum of future discound reward
            temp = torch.zeros(int(vis_i.shape[0])).cuda()
            for t in range(search_budget)[::-1]:  
                adv_store[t] = adv_store[t] + (0.01) * temp  
                temp = adv_store[t]     
              
            # Final loss according to REINFORCE objective to train the policy/agent
            for log_prob, disc_return in zip(log_prob_store, adv_store): # returns
                policy_loss.append(log_prob * Variable(disc_return).expand_as(log_prob))          
                
            # Final REINFORCE loss
            loss_rl = torch.cat(policy_loss).mean() 
            # update the search policy network parameters 
            optimizer_search.zero_grad()
            loss_rl.backward()
            torch.nn.utils.clip_grad_norm_(model_search.parameters(), hyp_params.clip)
            optimizer_search.step()
        
            # update the initial prediction policy network parameters following metaa learning algorithm
            grid_prob, hiddens_i = model_pred(tab_i.float(), vis_i.float(), search_info)
            grid_prob_net = grid_prob.view(grid_prob.size(0), -1)
            loss_pred = (criterion(grid_prob_net.float(), target_label.float().cuda())) 
            loss_p = (0.1) * loss_pred  #(0.1) * 
            optimizer.zero_grad()
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(model_pred.parameters(), hyp_params.clip)
            optimizer.step()
            # store the search result in the following lists
            batch_reward = torch.cat(rewards).mean() 
            total_reward.append(batch_reward.cpu())
        
            policies.append(policy_sample.data.cpu())
                
    
            reward = performance_stats_search(policies, total_reward)
    
        with open('log.txt','a') as f:
            f.write('Train: %d | Rw: %.2f \n' % (epoch, reward))
    
        print('Train: %d | Rw: %.2E | CE: %.2E | RL: %.2E' % (epoch, reward, loss_pred, loss_rl))
      
        return reward
        
    def evaluate(er, best_sr, epoch, model_pred, model_search, optimizer, criterion, test=False):
        search_budget = 20 
        num_image = 0
        # set the agent in evaluation mode
        model_search.eval()
        model_pred.eval()
        # initialize lists to store search outcomes
        targets_found, metrics, policies, set_labels, num_targets, num_search = list(), [], [], [], list(), list()
        acc_steps = []; tpr_steps =[];
        # iterate over the test data
        #for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        #for vis_i, tab_i, target_label in zip(eval_img, eval_tab, eval_label):# in range(5):
        #    num_image += 1
        #### Create the batch data
        task_tab_ft = list(); task_label_ft = list()      
        batch_img = list(); batch_tab = list(); batch_target = list()
        counter = 0 ; task_size = 100
        batch_id = 0; batch_size = 1; search_stat = list()
        # Start a test search episode
        policy_loss = []; search_history = []; reward_history = []; batch_search = list(); batch_true_label = list()
        
        for t_num in range(int(task_dict_test.shape[0])):
            
            batch_flag = True
            task_img_ft = list(); 
            reward_history = list()
            tolemi_id_list = task_dict_test[t_num,:,er]
            task_tab_ft = tab_dict_test[tab_dict_test['ta_id'].isin(tolemi_id_list)].iloc[:,1:43]
            task_label_ft = tab_dict_test[tab_dict_test['ta_id'].isin(tolemi_id_list)]['target']
            #pos = pos + task_label_ft.sum()
            task_tab_ft = task_tab_ft.to_numpy()
            task_label_ft = task_label_ft.to_numpy()
            for tolemi_id in tolemi_id_list: 
                #tab_ft = tab_dict[tab_dict['ta_id'] == tolemi_id]
                img_ft = np.reshape(image_dict[tolemi_id], (int(image_dict[tolemi_id].shape[1]), int(image_dict[tolemi_id].shape[2]), int(image_dict[tolemi_id].shape[0])))
                img_ft = cv2.resize(img_ft, dsize=(448, 448)) #fixed reshape size
                # Transform the image
                img_ft = trans_img(img_ft).unsqueeze(0)
                out_embedding = feat_model.forward(img_ft.cuda())
                out_img_ft = F.adaptive_avg_pool2d(out_embedding, (1, 1)).squeeze()
                task_img_ft.append(out_img_ft.cpu())
                #tab_ft_ = tab_ft.to_numpy()
                #tab_ft = tab_ft_[0, 1:43] #omit the 1st column which contains tolemi id
                #tab_ft = np.array(tab_ft, dtype=np.float)
                #target_ft = tab_ft_[0,43]
                #task_tab_ft.append(tab_ft)
                #task_label_ft.append(target_ft)
            img_ft_i = np.stack(task_img_ft, axis = 0) 
            vis_i = torch.from_numpy(img_ft_i).unsqueeze(0).cuda()
            target_label = torch.from_numpy(task_label_ft).unsqueeze(0).cuda()
            tab_i = torch.from_numpy(task_tab_ft).unsqueeze(0).cuda() 
            
            
            '''
            reward_history = list()
            tolemi_id_list = task_dict_test[t_num,:,er]
            for tolemi_id in tolemi_id_list: 
                tab_ft = tab_dict_test[tab_dict_test['ta_id'] == tolemi_id]
                img_ft = np.reshape(image_dict[tolemi_id], (int(image_dict[tolemi_id].shape[1]), int(image_dict[tolemi_id].shape[2]), int(image_dict[tolemi_id].shape[0])))
                img_ft = cv2.resize(img_ft, dsize=(448, 448)) #fixed reshape size
                # Transform the image
                img_ft = trans_img(img_ft)
                #img_ft = 0 * img_ft ### for debugging
                task_img_ft.append(img_ft)
                tab_ft_ = tab_ft.to_numpy()
                tab_ft = tab_ft_[0, 1:43] #omit the 1st column which contains tolemi id
                tab_ft = np.array(tab_ft, dtype=np.float)
                #tab_ft = 0 * tab_ft_[0, 1:17] #omit the 1st column which contains tolemi id
                target_ft = tab_ft_[0,43]
                task_tab_ft.append(tab_ft)
                task_label_ft.append(target_ft)
                
            batch_img_ft = np.stack(task_img_ft[len(task_img_ft) - task_size:], axis=0)
            batch_tab_ft = np.stack(task_tab_ft[len(task_img_ft) - task_size:], axis=0)
            batch_label_ft = np.stack(task_label_ft[len(task_label_ft) - task_size:], axis=0)
            out_embedding = feat_model.forward(torch.from_numpy(batch_img_ft).cuda())
            out_img_ft = F.adaptive_avg_pool2d(out_embedding, (1, 1)).squeeze()
            batch_img.append(out_img_ft.cpu())
            batch_tab.append(batch_tab_ft)
            batch_target.append(batch_label_ft)
            batch_id += 1
            if batch_id % batch_size == 0 and batch_id != 0:
                batch_img_data = np.stack(batch_img[len(batch_img) - batch_size:], axis = 0)
                batch_tab_data = np.stack(batch_tab[len(batch_tab) - batch_size:], axis = 0)
                batch_label_data = np.stack(batch_target[len(batch_target) - batch_size:], axis = 0)
                batch_flag = False
                    
            if batch_flag: 
                    continue
            else:
                '''
            #print ("batch search")
            ## create a copy of the pretrained agent
            p_Agent = deepcopy(model_pred)
            optimizer = optim.Adam(p_Agent.parameters(), lr=0.00002)
            
            # dummy batch data
            #vis_i = torch.from_numpy(np.array(batch_img_data)).cuda() #torch.ones([1, 100, 512]).cuda()
            #print (vis_i.shape)
            #tab_i = torch.from_numpy(np.array(batch_tab_data)).cuda() #torch.ones([1, 100, 16]).cuda()
            #target_label = torch.from_numpy(np.array(batch_label_data)).cuda() #torch.zeros((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            batch_true_label.append(target_label.cpu())
            
            # stores the information of previous search queries
            search_info = torch.zeros((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            # Active Target Information representation
            act_target_label = torch.zeros((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            final_target_label = torch.zeros((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            # stores the information of previously selected grids as target 
            mask_info = torch.ones((int(vis_i.shape[0]), int(vis_i.shape[1]))).cuda()
            #store the information about the remaining query
            query_info = torch.zeros(int(vis_i.shape[0])).cuda()

                
                
            
            for step_ in range(search_budget): 
                query_remain = search_budget - step_
                # number of query left
                query_left = torch.add(query_info, query_remain).cuda()
                # action taken by agent
                grid_prob_, _ = p_Agent.forward(tab_i.float(), vis_i.float(), search_info)
                logit = model_search.forward(search_info, query_left, grid_prob_)            
                grid_prob_net = grid_prob_.view(grid_prob_.size(0), -1)            
                grid_prob = F.sigmoid(grid_prob_net)
                # get the prediction of target from the agents intermediate output
                policy_pred = grid_prob_net.data.clone()
                policy_pred[policy_pred<0.5] = 0.0
                policy_pred[policy_pred>=0.5] = 1.0
                policy_pred = Variable(policy_pred)
                
                acc, tpr = acc_calc(target_label, policy_pred.data)
                acc_steps.append(acc)
                tpr_steps.append(tpr)
                
                # get the probability distribution over grids
                probs = F.softmax(logit, dim=1)
            
                # assign 0 probability to those grids that is already queried by agent
                mask_probs = probs * mask_info.clone()  
                # Sample the grid that corresponds to highest probability of being target
                policy_sample = torch.argmax(mask_probs, dim=1) 
                
                # compute the reward for the agent's action
                reward_update = compute_reward(target_label, policy_sample.data)
                # get the outcome of an action in order to compute ESR/SR 
                reward_sample = compute_reward_batch(target_label, policy_sample.data)
                    
                # Update search info and mask info after every query
                for sample_id in range(int(vis_i.shape[0])):
                    # update the search info based on the reward
                    if int(reward_update[sample_id]) == 1:
                        search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_update[sample_id])
                    else:
                        search_info[sample_id, int(policy_sample[sample_id].data)] = -1
                        
                    if (int(reward_update[sample_id]) == 1):
                        act_target_label[sample_id, int(policy_sample[sample_id].data)] = 1
                    else: 
                        act_target_label[sample_id, int(policy_sample[sample_id].data)] = 0
                
                    # update the mask info based on the current action
                    mask_info[sample_id, int(policy_sample[sample_id].data)] = 0
        
                    for out_idx in range(int(vis_i.shape[1])):
                        if (mask_info[sample_id, out_idx] == 0):
                            final_target_label[sample_id, out_idx] = act_target_label[sample_id, out_idx].data
                        else:
                            final_target_label[sample_id, out_idx] = grid_prob[sample_id, out_idx].data

                # store the episodic reward in the list
                    
                reward_history.append(reward_sample.cpu())
                loss_cls = criterion(grid_prob_net.float(), final_target_label.float().cuda()) 
                # update the policy network parameters 
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()
                    
            temp = torch.zeros(int(vis_i.shape[0])).cpu() 
            for s_step in range(search_budget):
                temp = torch.add(reward_history[s_step].cpu(), temp)
            batch_search.append(temp.numpy())
            #print ("batch done")
                    
            
        #print (batch_search)
        search_out_ = np.array(batch_search)
        #print (search_out_.shape)
        search_stat.append(search_out_)
        
        true_target = np.array(batch_true_label)
        #print ("111111111", true_target.shape)
        #true_target = np.sum(true_target)
        #print (true_target.shape)
        true_stat = true_target #.flatten()
        #print ("true_target", true_stat.shape)
        
        search_stat_ = np.array(search_stat).flatten()
        recall = int(np.sum(search_stat_))
        #np.save("er_2.5_sb_20_our.npy", search_stat_)
        #print ("search_stat", search_stat_.shape)
        print ("ANT:", recall)
        #print (search_stat_)
        #print (true_stat)
        
        if (recall> best_sr):
            print ("best_SR for SB 20 is:", recall)
            #np.save("er_2.5_sb_20_our.npy", search_stat_)
            best_sr = recall
            # save the model --- search agent
            #agent_state_dict = model_search.module.state_dict() if args.parallel else search_Agent.state_dict()
            agent_state_dict = model_search.state_dict()
            state = {
                    'agent': agent_state_dict,
                    'epoch': epoch,
                    }
              
            torch.save(state, "./model_search_10")
            # save the model --- PRED agent
            #agent_state_dict = model_pred.module.state_dict() if args.parallel else pred_Agent.state_dict()
            agent_state_dict = model_pred.state_dict()
            state = {
                    'agent': agent_state_dict,
                    'epoch': epoch,
                    }
            # uncomment the following line and provide a path where you want to save the trained model
            #torch.save(state, args.cv_dir+'/ckpt_E_%d_R_%.2E'%(epoch, success)) 
            torch.save(state,"./model_pred_10")
            df_search_stat = pd.DataFrame(np.array([true_stat, search_stat_]).transpose(), columns = ['total_positives', 'found_positives'])
            df_search_stat['method'] = 'our'
            df_search_stat['budget'] = 20
            df_search_stat['positive_rate'] = 10
            df_search_stat.to_pickle("./pickle/exp_our_20_100.pkl")
            df_search_stat.to_csv("./csv/exp_our_20_100.csv")
    
    
        print('Test - Recall: %.2E | SB: %.2F' % (recall,search_budget))
        return best_sr       

    best_valid = 1e8
    best_sr = 0
    er = 2
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        _ = train(er, epoch, model_pred, model_search, optimizer, optimizer_search, criterion)
        if (epoch % 1 == 0):
            
            best_sr = evaluate(er, best_sr, epoch, model_pred, model_search, optimizer, criterion, test=True)
        