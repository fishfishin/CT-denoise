
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from PIL import RL
from PIL import  PSNR
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Bm3d,NLM
import skimage
import copy

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
print(dev)
device = torch.device(dev)  
a = torch.zeros(4,3)    
a = a.to(device)

####  1st path
Para = ['sigma', 'size','dist']
sigma = 1
Beta_Kaiser = 2.0

####  2nd path
Actions = ['plus_half', 'plus_tenth', 'null', 'minus_tenth', 'minus_half']

MAX_EPOCHS = 10
DISCOUNT_RATE = 0.99
RESOLUTION = 256
PATCH_SIZE = [9,9]
Patch_num = RESOLUTION **2
PATCH_reward = 5
TARGET_UPDATE_STEP = 300
MAXSTEPS_BM3D = 20
REPLAY_MEMORY = 5000000  ######### buffer？
BATCH_SIZE = 128

TUNING_STEP= 20

def replay_train(mainDQN: RL.doubleDQN, targetDQN: RL.doubleDQN, states, next_states, action,parameter, rewards) -> np.float64:

    X = torch.zeros(states.shape[0], 1, PATCH_SIZE[0],PATCH_SIZE[0])
    X1 = torch.zeros(states.shape[0], 1, PATCH_SIZE[0],PATCH_SIZE[0])
    X[:,0,:,:] = torch.from_numpy(states)
    X1[:,0,:,:] = torch.from_numpy(next_states)

    t1, t2 = targetDQN(X1)  #### old version of NET
    y1, y2 = mainDQN(X)

    ####  1st path
    tem = torch.max(t1, axis=1)
    temp = tem.values
    Q_target = rewards + DISCOUNT_RATE * temp.detach().numpy()
    for i in range(y1.shape[0]):
        y1[i,int(parameter[i])] = Q_target[i]

    ####  2nd path
    tem = torch.max(t2, axis=1)
    temp = tem.values
    Q_target = rewards + DISCOUNT_RATE * temp.detach().numpy()
    for i in range(y2.shape[0]):
        y2[i,int(action[i])] = Q_target[i]

    return mainDQN.update(X, y1,y2)



def divide_patch(fimg) -> np.float64:

    fimgpad = np.zeros((RESOLUTION + PATCH_SIZE[0] -1, RESOLUTION + PATCH_SIZE[0] -1),dtype =np.float64)
    fimgpad[int((PATCH_SIZE[0]+1)/2)-1:RESOLUTION+int((PATCH_SIZE[0]+1)/2)-1,int((PATCH_SIZE[0]+1)/2)-1:RESOLUTION+int((PATCH_SIZE[0]+1)/2)-1]=fimg
    state = np.zeros((Patch_num,PATCH_SIZE[0], PATCH_SIZE[0]))
    count = 0
    for xcord in range(RESOLUTION):
        for ycord in range(RESOLUTION):
            temp=fimgpad[ycord:ycord+PATCH_SIZE[0],xcord:xcord+PATCH_SIZE[0]]
            state[count,:,: ] = temp
            count += 1
    return state


   #state : patches in one image
def Denoise(state, parameter, action, parameter_value, GroundTruth, original_image) -> np.float64:
    

    # tuning parameter
    for idx in range(Patch_num):
        if parameter[idx] ==0:
            if action[idx]==0:
                parameter_value[idx,0] = parameter_value[idx,0] *1.5
            if action[idx]==1:
                parameter_value[idx,0] = parameter_value[idx,0] *1.1
            if action[idx]==3:
                parameter_value[idx,0] = parameter_value[idx,0]*0.9
            if action[idx]==4:
                parameter_value[idx,0]= parameter_value[idx,0] *0.5
        if parameter[idx]==1:
            if action[idx]==0:
                parameter_value[idx,1] = parameter_value[idx,1] *1.5
            if action[idx]==1:
                parameter_value[idx,1] = parameter_value[idx,1] *1.1
            if action[idx]==3:
                parameter_value[idx,1] = parameter_value[idx,1]*0.9
            if action[idx]==4:
                parameter_value[idx,1] = parameter_value[idx,1] *0.5
        if parameter[idx]==2:
            if action[idx]==0:
                parameter_value[idx,2] = parameter_value[idx,2] *1.5
            if action[idx]==1:
                parameter_value[idx,2] = parameter_value[idx,2] *1.1
            if action[idx]==3:
                parameter_value[idx,2] = parameter_value[idx,2]*0.9
            if action[idx]==4:
                parameter_value[idx,2] = parameter_value[idx,2] *0.5
    
    current_patch = np.zeros((Patch_num, PATCH_SIZE[0], PATCH_SIZE[0]))
    """
    # noise removal for each patch
    for  idx in range(Patch_num):
        basic_patch = Bm3d.BM3D_1st_step(original_image, parameter_value[idx,0], parameter_value[idx,1]) 
        final_patch = Bm3d.BM3D_2nd_step(basic_patch, state[idx,:,:], parameter_value[idx,0], parameter_value[idx,1]) 
        current_patch [idx, :, :] = final_patch
    """
    org_patch = divide_patch(original_image)
    for  idx in range(Patch_num):
        final_patch = NLM.NLM(org_patch[idx,:,:], parameter_value[idx,0], parameter_value[idx,1],parameter_value[idx,2]) 
        current_patch [idx, :, :] = final_patch
    ############# NLM

    ### how to stitch  256x 256 patches into a single image fimg  after Bm3d
    next_img = np.reshape(current_patch[:, int(PATCH_SIZE[0]//2), int(PATCH_SIZE[0]//2)], (RESOLUTION, RESOLUTION), order='F')
    next_state = current_patch
    current_image = np.reshape(state[:, int(PATCH_SIZE[0]//2), int(PATCH_SIZE[0]//2)], (RESOLUTION, RESOLUTION), order='F')
     #next_state = divid_patch(fimg)

    #############   calculate reward and error
    #dist1 = np.reshape(original_image-GroundTruth,(Patch_num),order='F')

    dist1img = current_image - GroundTruth
    dist2img = next_img - GroundTruth  #######################################
    dist2 = np.reshape(dist2img, (Patch_num), order='F')

    dist1imgLarge = np.zeros((RESOLUTION+PATCH_reward-1,RESOLUTION+PATCH_reward-1))
    margin = int((PATCH_reward-1)/2)
    dist1imgLarge[margin:RESOLUTION+margin,margin:RESOLUTION+margin]=np.absolute(dist1img)

    dist2imgLarge = np.zeros((RESOLUTION + PATCH_reward-1, RESOLUTION + PATCH_reward-1))
    dist2imgLarge[margin:RESOLUTION + margin, margin:RESOLUTION + margin] = np.absolute(dist2img)

    rewardimg = np.zeros((RESOLUTION,RESOLUTION))
    reward = np.zeros((Patch_num))

    count=0
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            
            temp = np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) - np.sum(dist2imgLarge[j:j + PATCH_reward, i:i + PATCH_reward])

            ### when reach the ground truth 
            if np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])==0:
                if temp==0:
                    reward[count]=1
                else:
                    reward[count]=-1
                count += 1
            ### not yet get the best result
            else:
                factor = 0.005
                if temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])>=factor:
                    reward[count]=1
                if temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])<factor and temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])>=factor*0.1:
                    reward[count] = 0.5
                if temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])<factor*0.1 and temp>0:
                    reward[count] = 0.1
                if temp==0:
                    reward[count] = 0
                if temp / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) < 0:
                    reward[count] = -1
                count += 1
            
            rewardimg[i,j]= 1/(np.sum(dist2imgLarge[i:i+PATCH_reward,j:j+PATCH_reward])+0.001) - 1/(np.sum(dist1imgLarge[i:i+PATCH_reward,j:j+PATCH_reward])+0.001)
    reward = np.reshape(rewardimg,(Patch_num),order='F')
    error = np.sum(np.absolute(dist2))


    return next_state, reward, parameter_value, next_img ,error

def main():

    #train_data = skimage.external.tifffile.imread("C:/Users/ZhenjuYin/Downloads/a1.tif")
    TrueImgTrain = skimage.external.tifffile.imread("C:/Users/ZhenjuYin/Downloads/G.tif")
    datasize = TrueImgTrain.shape
    gaussian = np.random.normal(0.3, 0.9, (datasize[0],datasize[1],datasize[2])) 
    train_data = TrueImgTrain + gaussian
    
    # (None, channel, H, w, depth) for Volume
    mainDQN = RL.doubleDQN(PATCH_SIZE[0], PATCH_SIZE[1], len(Para),len(Actions))
    targetDQN = RL.doubleDQN(PATCH_SIZE[0], PATCH_SIZE[1], len(Para),len(Actions))

    state_sel = np.zeros((REPLAY_MEMORY, PATCH_SIZE[0], PATCH_SIZE[0]))
    next_state_sel = np.zeros((REPLAY_MEMORY, PATCH_SIZE[0], PATCH_SIZE[0]))
    action_sel = np.zeros((REPLAY_MEMORY))
    reward_sel = np.zeros((REPLAY_MEMORY))
    done_sel = np.zeros((REPLAY_MEMORY))
    para_sel = np.zeros((REPLAY_MEMORY))
    #value_sel = np.zeros((REPLAY_MEMORY, len(Para)))
    indicator = 0 
    if MAX_EPOCHS>0:  ########################## delete
                   
            State = np.zeros((datasize[0], Patch_num, PATCH_SIZE[0],PATCH_SIZE[0]))             ##  slabs  x  256**2 x 9 x 9
            parameter_value = np.zeros((Patch_num, len(Para)))
            parameter_value[:,0] = 1.5                                 # sigma at first step
            parameter_value[:,1] = 1.0                                 # Beta_kasier
            parameter_value[:,2] = 2.0 
            # for each image initializer
            for IMG in range(datasize[0]):
                state = divide_patch(  train_data[ IMG,:,:]  )  ##  256**2 x 9 x 9              
                GroundTruth = TrueImgTrain[IMG,:, : ]      
                ## initialize the 1st and 2nd paths 
                parameter = np.ones((Patch_num))
                action = 2 * np.ones((Patch_num))
                #next_state, reward, parameter_value, img, error = Denoise( state,  parameter, action, parameter_value , GroundTruth, train_data[ IMG,:,:] )
                State[ IMG, :, :, :] = state
                #psnr = PSNR.PSNR(train_data[IMG,:,:], img )
                #print(" current PSNR : {}".format(psnr))
                print(IMG)


            State_initial = State
            count_memory = 0

            for episode in range(MAX_EPOCHS-1):

                e = 0.999 / ((episode / 150) + 1)
                if e<0.1:
                    e=0.1
                step_count = 0
                State = State_initial

                for ITER_NUM in range(MAXSTEPS_BM3D):

                    for IMG_IDX in range(datasize[0]):
                       
                        state = State[ IMG_IDX, :, :, :]     ##  slabs x num of patches x 9 x 9
                        GroundTruth = TrueImgTrain[IMG_IDX,:, : ]                      
                        parameter = np.ones((Patch_num))
                        action = 2 * np.ones((Patch_num))

                        # random select patches and     action  for each image
                        flag = np.random.rand(Patch_num)
                        count_patch = 0
                        length_patch = 0
                        for idx in range(Patch_num):
                            if flag[idx]>=e:
                                length_patch += 1
                        
                        # yy  : patch samples
                        yy = torch.zeros(length_patch, 1, PATCH_SIZE[0],PATCH_SIZE[0])
                        for idx in range(Patch_num):
                            if flag[idx]<e:
                                action[idx] = np.random.randint(len(Actions), size=1)
                                parameter[idx] = np.random.randint(len(Para), size=1)
                            if flag[idx]>=e:
                                yy[count_patch,0,:, :] = torch.from_numpy(state[idx,:,:])
                                count_patch += 1
                        
                        y1, y2 = mainDQN(yy)
                        parameter_yy = torch.argmax(y1, axis=1)
                        action_yy = torch.argmax(y2, axis=1)
                        

                        avg_action = action_yy.type(torch.float64).mean()
                        avg_para = parameter_yy.type(torch.float64).mean()
                        print('average action taken is: {} average parameter taken is: {}'.format(avg_action,avg_para))
                        
                        #### action and paramter chosen
                        count_patch=0
                        for idx in range(Patch_num):
                            if flag[idx] >= e:
                                action[idx] = action_yy[count_patch]
                                parameter[idx] = parameter_yy[count_patch]
                                count_patch += 1

                        #original = np.reshape(state[:, int(PATCH_SIZE[0]//2), int(PATCH_SIZE[0]//2)], (RESOLUTION, RESOLUTION), order='F')
                        next_state, reward, parameter_value, img, error = Denoise( state,  parameter, action,parameter_value,  GroundTruth, train_data[ IMG_IDX,:,:] )
                        psnr = PSNR.PSNR(train_data[IMG_IDX,:,:], img )
                        print(" current PSNR : {}".format(psnr))
                        """
                        fig, axs = plt.subplots(1, 3)
                        fig.suptitle('current results Bm3d')
                        axs[ 0].imshow(train_data[ IMG_IDX,:,:], cmap='gray')
                        axs[ 1].imshow(np.reshape(next_state[:, int(PATCH_SIZE[0]//2), int(PATCH_SIZE[0]//2)], (RESOLUTION, RESOLUTION), order='F'), cmap='gray')
                        axs[ 1].imshow(GroundTruth, cmap='gray')

                        axs[ 0].set_title('original')
                        axs[ 1].set_title('Filtered')
                        axs[ 2].set_title('Truth')
                        #plt.show()
                        """
                        ###################  ? random replacement
                        sel_prob = 0.01
                        flag1 = np.random.rand(Patch_num)
                        flag2 = np.zeros([Patch_num])
                        for idx in range(Patch_num):
                            if flag1[idx]>=sel_prob:
                                flag2[idx] = 0
                            if flag1[idx]<sel_prob:  ##### chosen
                                flag2[idx] = 1

                        sel_num = int(np.sum(flag2))

                        ##### refresh the buffer randomly
                        if count_memory+sel_num<=REPLAY_MEMORY-2:
                            for idx in range(Patch_num):
                                if flag1[idx]<sel_prob:
                                    state_sel[count_memory,:,:] = state[idx,:,:]
                                    next_state_sel[count_memory,:,:] = next_state[idx,:,:]
                                    action_sel[count_memory]=action[idx]
                                    para_sel[count_memory] = parameter[idx]
                                    reward_sel[count_memory] = reward[idx]
                                    #value_sel[count_memory] = parameter_value[idx,:]
                                    if ITER_NUM >= MAXSTEPS_BM3D-1:
                                        done_sel[count_memory] = 0
                                    if ITER_NUM < MAXSTEPS_BM3D - 1:
                                        done_sel[count_memory] = 1
                                    count_memory += 1
                        else:
                            indicator = 1
                            for idx in range(Patch_num):
                                if flag1[idx]<sel_prob:
                                    state_sel[count_memory,:] = state[idx,:,:]
                                    next_state_sel[count_memory,:] = next_state[idx,:,:]
                                    action_sel[count_memory]=action[idx]
                                    para_sel[count_memory] = parameter[idx]
                                    reward_sel[count_memory] = reward[idx]
                                    #value_sel[count_memory] = parameter_value[idx,:]
                                    if ITER_NUM >= MAXSTEPS_BM3D-1:
                                        done_sel[count_memory] = 0
                                    if ITER_NUM < MAXSTEPS_BM3D - 1:
                                        done_sel[count_memory] = 1
                                    if count_memory == REPLAY_MEMORY - 1:
                                        count_memory = 0
                                        print('Replay Memory is full')
                                    else:
                                        count_memory += 1
                        if indicator == 0:
                            replay_size = count_memory +  1
                        else:
                            replay_size = REPLAY_MEMORY

                        if replay_size > BATCH_SIZE:
                            
                            for i in range(TUNING_STEP):
                                shuffle_order = np.arange(replay_size)
                                np.random.shuffle(shuffle_order)
                                minibatch_state = state_sel[shuffle_order[0:BATCH_SIZE], :, :]
                                minibatch_next_state = next_state_sel[shuffle_order[0:BATCH_SIZE],:,:]
                                minibatch_action = action_sel[shuffle_order[0:BATCH_SIZE]]
                                minibatch_parameter = para_sel[shuffle_order[0:BATCH_SIZE]]
                                minibatch_reward = reward_sel[shuffle_order[0:BATCH_SIZE]]
                                #minibatch_value = value_sel[shuffle_order[0:BATCH_SIZE]]
                                ##minibatch_done = done_sel[shuffle_order[0:BATCH_SIZE]]

                                #minibatch = random.sample(replay_buffer, BATCH_SIZE)
                                loss = replay_train(mainDQN, targetDQN, minibatch_state,minibatch_next_state,minibatch_action,minibatch_parameter, minibatch_reward)
                                if step_count % TARGET_UPDATE_STEP == 0:
                                    targetDQN = copy.deepcopy(mainDQN)
                                step_count += 1

                        State[IMG_IDX, :, :, :] = next_state
                        

                    print("Episode: {}  Iterations: {} Loss: {}".format(episode, ITER_NUM, loss))

                CHECK = episode+1

if __name__ == '__main__':
    main()