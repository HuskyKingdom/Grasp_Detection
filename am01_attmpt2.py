from math import radians
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plts
import os
import cv2
from torch import LongTensor, optim
from matplotlib import patches, pyplot as plt, transforms
from PIL import Image,ImageDraw
import math
import copy


class CNN_model(nn.Module): # an simple implementation of AlexNet

    def __init__(self) -> None:

        super(CNN_model,self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
            nn.Conv2d(64,128,3,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
            nn.Conv2d(128,128,3,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,6)
            
            
        )


    def forward(self, x):
        
        x = self.model(x)

        return x







def build_models():
    # Model
    data_CNN = CNN_model()
    data_optimizer = optim.Adam(data_CNN.parameters(), 0.001)
    data_loss = nn.MSELoss()

    return data_CNN,data_optimizer,data_loss



def sample_training_data(): # sample the training dataset and returns 5 images and it`s corresponding ground-truths

    list_dirs = os.walk("Data/training") # extract RGB image candidates
    candidates = []
    final_images = []
    final_labels = []
    for root,dirs,files in list_dirs:
        for f in files:
            file_path = os.path.join(root,f)
            if "RGB" in file_path:
                candidates.append(file_path)

    sample_index = np.random.choice(len(candidates), 5) # random sampling

    # final images
    for i in range(len(sample_index)):
        final_images.append(candidates[sample_index[i]])
    
    # final ground-truth values
    for i in range(len(final_images)):
        split_array = final_images[i].split("_")
        split_array[2] = "grasps.txt"
        temp_str = ""
        for i in range(len(split_array)):
            if i != len(split_array) - 1:
                temp_str += split_array[i] + "_"
            else:
                temp_str += split_array[i]
        final_labels.append(temp_str)

    
    return final_images,final_labels
    
def img_split(img): # split the img into 8 * 8 grid and return

    return_array = []
    r_s = 0
    r_e = 128
    
    
    for i in range(8):

        c_s = 0
        c_e = 128
        return_array.append([])
        for j in range(8):

            return_array[i].append(img[r_s : r_e , c_s : c_e , :])
            c_s = c_e
            c_e = c_s + 128

        r_s = r_e
        r_e = r_s + 128
          

            
    
    return return_array

def sample_ground_truth(file_path): # sample the ground truth and return the splited data
   
    return_val = []

    with open(file_path) as file_obj:
        lines = file_obj.readlines()
    
    for i in range(len(lines)): # dilme the /n notation
        lines[i] = lines[i].replace("\n","")

    sample_index = np.random.choice(len(lines), 5) # random sampling

    # final images
    for i in range(len(sample_index)):
        # split
        values = lines[sample_index[i]].split(";")
        return_val.append(values)

    return return_val



def draw_result(img,result):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = float(result[0]) - 0.5 * float(result[3])
    y = float(result[1]) - 0.5 * float(result[4])
    mid_point = plt.Circle((float(result[0]),float(result[1])),radius=5)
    
    

    rect = patches.Rectangle((x, y), float(result[3]), float(result[4]), fill=False, edgecolor = 'red',linewidth=1)

    

    ax.add_patch(rect)
    ax.add_patch(mid_point)
    plt.imshow(torch.LongTensor(img))
    plt.show()


def block_locator(x,y): # return the block index given the x&y coordinate

    # floor up
    x = int(float(x))
    y = int(float(y))
    x += 1
    y += 1

    return (int(x/128),int(y/128))

def arrary_locator(index): # return the array index given the flatten array index
    index += 1

    if index % 8 == 0:
        base = int(index / 8 - 1)
        increment = 7
        return (base,increment)

    base = int(index / 8)
    increment = index % 8 - 1
    return (base,increment)


def RGB_train_progess(data_CNN,data_optimizer,data_loss):

    # initiallization
    total_loss = 0.0
    

    # sampling 5 data
    images, groud_truth = sample_training_data()
    
    # Iterating each sampled data
    for i in range(len(images)):

        # spliting
        cur_img = cv2.imread(images[i])
        splited_imgs = np.array(img_split(cur_img))

        

        # viewing the splited data
        # num = 0
        # for row in range(8):
        #       for col in range(8):
        #         plt.subplot(8,8,num+1)
        #         plt.imshow(torch.LongTensor(splited_imgs[row][col]))
        #         num += 1
        # plt.show()  

        # sample ground-truth & processing to target
        splited_gt = sample_ground_truth(groud_truth[i])
        # convert str2float
        for a in range(len(splited_gt)):
            for b in range(len(splited_gt[a])):
                splited_gt[a][b] = float(splited_gt[a][b])



        #  data target array -- 
        data_targets = []
        for height in range(8):
            data_targets.append([])
            for weight in range(8):
                data_targets[height].append([0.0,0.0,0.0,0.0,0.0,0.0])

        for k in range(len(splited_gt)):
            block_index = block_locator(splited_gt[k][0],splited_gt[k][1])
            data_targets[block_index[0]][block_index[1]] = copy.deepcopy(splited_gt[k])
            data_targets[block_index[0]][block_index[1]].insert(0,1.0)
            
        
        
        

        # input 64*3*128*128
        tensor_data = torch.FloatTensor(splited_imgs).permute(0,1,4,3,2)
        tensor_data = torch.reshape(tensor_data,(-1,3,128,128))

        # label 64*6
        label = torch.FloatTensor(data_targets)
        label = torch.reshape(label,(-1,6))

        

        # forward prapogatioon.
        result = data_CNN(tensor_data)

        # backward prapogation.

        # heatmap error
        
        heatmap = label[:,0]
        heat_predicted = result[:,0]

        loss = data_loss(heat_predicted,heatmap)
        
        
        # cell error
        actual = torch.FloatTensor()
        predicted = torch.FloatTensor()
        for count in range(result.shape[0]):
            if float(label[count][0]) == 1.0: # non-empty data
                actual = torch.cat((actual,label[count][1:]),dim=0)
                predicted = torch.cat((predicted,result[count][1:]),dim=0)

        loss2 = data_loss(predicted,actual)
        loss = loss + loss2
        total_loss = loss
        data_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        data_optimizer.step()

        

            
    return total_loss
            




def main():

    usr_input = input("1.Train RGB imgs 2.Evaluation: ")

    if usr_input == "1":
        CNN,data_optimizer,data_loss = build_models()

        for i in range(2000):
            data_loss_t = RGB_train_progess(CNN,data_optimizer,data_loss)
            print("Episode {} is end,  Loss : {} ".format(i,data_loss_t))


        torch.save(CNN, "data_net.pkl")








main()