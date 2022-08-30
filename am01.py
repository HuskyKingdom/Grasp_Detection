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
from torch.utils.tensorboard import SummaryWriter, writer
from torchmetrics import RetrievalFallOut
import tifffile as tiff



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
            nn.Linear(215296,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,5)
            
            
        )


    def forward(self, x):
        
        x = self.model(x)

        return x


class d_CNN_model(nn.Module): # an simple implementation of AlexNet

    def __init__(self) -> None:

        super(d_CNN_model,self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(4,64,3,2),
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
            nn.Linear(215296,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,5)
            
            
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


def build_d_model():
    # Model
    data_CNN = d_CNN_model()
    data_optimizer = optim.Adam(data_CNN.parameters(), 0.001)
    data_loss = nn.MSELoss()

    return data_CNN,data_optimizer,data_loss


def sample_depth_training_data(): # sample the training dataset and returns 5 images and it`s corresponding ground-truths

    list_dirs = os.walk("Data/training") # extract RGB image candidates
    candidates = []
    final_images = []
    final_labels = []
    final_depth = []
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

    # final depth image 
    for i in range(len(final_images)):
        split_array = final_images[i].split("_")
        split_array[2] = "perfect_depth.tiff"
        temp_str = ""
        for i in range(len(split_array)):
            if i != len(split_array) - 1:
                temp_str += split_array[i] + "_"
            else:
                temp_str += split_array[i]
        final_depth.append(temp_str)

    
    return final_images,final_labels,final_depth



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


def get_testing_data(): # sample the training dataset and returns 5 images and it`s corresponding ground-truths

    list_dirs = os.walk("Data/testing") # extract RGB image candidates
    candidates = []
    labels = []
    for root,dirs,files in list_dirs:
        for f in files:
            file_path = os.path.join(root,f)
            if "RGB" in file_path:
                candidates.append(file_path)

    
    
    # final ground-truth values
    for i in range(len(candidates)):
        split_array = candidates[i].split("_")
        split_array[2] = "grasps.txt"
        temp_str = ""
        for i in range(len(split_array)):
            if i != len(split_array) - 1:
                temp_str += split_array[i] + "_"
            else:
                temp_str += split_array[i]
        labels.append(temp_str)

    
    return candidates,labels


def get_testing_data_depth(): # sample the training dataset and returns 5 images and it`s corresponding ground-truths

    list_dirs = os.walk("Data/testing") # extract RGB image candidates
    candidates = []
    labels = []
    final_depth = []
    for root,dirs,files in list_dirs:
        for f in files:
            file_path = os.path.join(root,f)
            if "RGB" in file_path:
                candidates.append(file_path)

    
    
    # final ground-truth values
    for i in range(len(candidates)):
        split_array = candidates[i].split("_")
        split_array[2] = "grasps.txt"
        temp_str = ""
        for i in range(len(split_array)):
            if i != len(split_array) - 1:
                temp_str += split_array[i] + "_"
            else:
                temp_str += split_array[i]
        labels.append(temp_str)


    # final depth image 
    for i in range(len(candidates)):
        split_array = candidates[i].split("_")
        split_array[2] = "perfect_depth.tiff"
        temp_str = ""
        for i in range(len(split_array)):
            if i != len(split_array) - 1:
                temp_str += split_array[i] + "_"
            else:
                temp_str += split_array[i]
        final_depth.append(temp_str)
    
    return candidates,labels,final_depth
    
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

def sample_ground_truth(file_path,CNN,img): # the ground tureth with the most confident prediction
   
    return_val = []

    with open(file_path) as file_obj:
        lines = file_obj.readlines()
    
    for i in range(len(lines)): # dilme the /n notation
        lines[i] = lines[i].replace("\n","")


    # final images
    for i in range(len(lines)):
        # split
        values = lines[i].split(";")
        return_val.append(values)

    # convert str2float
    for i in range(len(return_val)):
        for j in range(5):
            return_val[i][j] = float(return_val[i][j])


    result = CNN(img)

    max_jac = 0.0
    max_jac_index = 0
    for i in range(len(return_val)):
        cur = jaccard_value(return_val[i],result.detach()[0].tolist())
        if cur >= max_jac:
            max_jac = cur
            max_jac_index = i
    

    return return_val[max_jac_index]

def get_all_gt(file_path):

    return_val = []

    with open(file_path) as file_obj:
        lines = file_obj.readlines()
    
    for i in range(len(lines)): # dilme the /n notation
        lines[i] = lines[i].replace("\n","")


    # final images
    for i in range(len(lines)):
        # split
        values = lines[i].split(";")
        return_val.append(values)

    for i in range(len(return_val)):
        for j in range(5):
            return_val[i][j] = float(return_val[i][j])

    return return_val


def draw_result(img,result):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = float(result[0]) - 0.5 * float(result[3])
    y = float(result[1]) - 0.5 * float(result[4])
    mid_point = plt.Circle((float(result[0]),float(result[1])),radius=5)
    
    

    rect = patches.Rectangle((x, y), float(result[3]), float(result[4]), fill=False, edgecolor = 'red',linewidth=1,label="test")

    

    ax.add_patch(rect)
    ax.add_patch(mid_point)
    plt.imshow(torch.LongTensor(img))
    plt.text(512.0,100.0,"Rotation Angle {}".format(result[2]))
    # plt.plot([3,500],[1,500])
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


def RGBD_train_progess(data_CNN,data_optimizer,data_loss):

    # initiallization
    total_loss = 0.0
    

    # sampling 5 data
    images, groud_truth, depth = sample_depth_training_data()

    
    
    

    # depth input 5*4*1024*1024
    raw_img = cv2.imread(images[0])
    raw_depth = tiff.imread(depth[0])[:,:,np.newaxis] 
    raw_img = np.append(raw_img,raw_depth,axis=2)
    temp = np.array([raw_img])
    tensor_data = torch.FloatTensor(temp)

    for i in range(1,len(images)):
        raw_img = cv2.imread(images[i])
        raw_depth = tiff.imread(depth[i])[:,:,np.newaxis] 
        raw_img = np.append(raw_img,raw_depth,axis=2)
        temp = np.array([raw_img])
        temp_tensor = torch.FloatTensor(temp)
        tensor_data = torch.cat((tensor_data, temp_tensor),dim=0)
        

    tensor_data = tensor_data.permute(0,3,1,2)

    
    


    # label 5*5
    label = []

    for i in range(len(groud_truth)):
        label.append(sample_ground_truth(groud_truth[i],data_CNN,tensor_data[i].unsqueeze(0)))
    
    

    label = torch.FloatTensor(label)
    


    # forward prapogatioon.
    result = data_CNN(tensor_data)

    # backward prapogation.
    loss = data_loss(result,label)
    total_loss = loss
    data_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    data_optimizer.step()

        

            
    return total_loss


def RGB_train_progess(data_CNN,data_optimizer,data_loss):

    # initiallization
    total_loss = 0.0
    

    # sampling 5 data
    images, groud_truth = sample_training_data()
    

    # input 5*3*1024*1024
    tensor_data = torch.FloatTensor(np.array([cv2.imread(images[0])]))
    for i in range(1,len(images)):
        tensor_data = torch.cat((tensor_data,torch.FloatTensor(np.array([cv2.imread(images[i])]))),dim=0)

    tensor_data = tensor_data.permute(0,3,1,2)
    


    # label 5*5
    label = []

    for i in range(len(groud_truth)):
        label.append(sample_ground_truth(groud_truth[i],data_CNN,tensor_data[i].unsqueeze(0)))
    
    

    label = torch.FloatTensor(label)
    


    # forward prapogatioon.
    result = data_CNN(tensor_data)

    # backward prapogation.
    loss = data_loss(result,label)
    total_loss = loss
    data_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    data_optimizer.step()

        

            
    return total_loss
            
def jaccard_index(value1,value2): # returns true if the prediction fits value in shape [5,]

    if abs(value1[2] - value2[2]) >= 30:
        return False
    
    # IOU
    
    minx_1 = value1[0] - 0.5 * value1[3] # left bottom
    miny_1 = value1[1] - 0.5 * value1[4]
    maxx_1 = value1[0] + 0.5 * value1[3] # right top
    maxy_1 = value1[1] + 0.5 * value1[4]

    minx_2 = value2[0] - 0.5 * value2[3] # left bottom
    miny_2 = value2[1] - 0.5 * value2[4]
    maxx_2 = value2[0] + 0.5 * value2[3] # right top
    maxy_2 = value2[1] + 0.5 * value2[4]

    pred_box = [minx_1,miny_1,maxx_1,maxy_1]
    gt_box = [minx_2,miny_2,maxx_2,maxy_2]
   
    # 1.get the coordinate of inters
    ixmin = max(minx_1, minx_2)
    ixmax = min(maxx_1, maxx_2)
    iymin = max(miny_1, miny_2)
    iymax = min(maxy_1, maxy_2)

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    if iou <= 0.25:
        return False

    return True


def jaccard_value(value1,value2):

    
    # IOU
    
    minx_1 = value1[0] - 0.5 * value1[3] # left bottom
    miny_1 = value1[1] - 0.5 * value1[4]
    maxx_1 = value1[0] + 0.5 * value1[3] # right top
    maxy_1 = value1[1] + 0.5 * value1[4]

    minx_2 = value2[0] - 0.5 * value2[3] # left bottom
    miny_2 = value2[1] - 0.5 * value2[4]
    maxx_2 = value2[0] + 0.5 * value2[3] # right top
    maxy_2 = value2[1] + 0.5 * value2[4]

    pred_box = [minx_1,miny_1,maxx_1,maxy_1]
    gt_box = [minx_2,miny_2,maxx_2,maxy_2]
   
    # 1.get the coordinate of inters
    ixmin = max(minx_1, minx_2)
    ixmax = min(maxx_1, maxx_2)
    iymin = max(miny_1, miny_2)
    iymax = min(maxy_1, maxy_2)

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

   

    return iou


def RGB_evaluation(model): # returns accuracy

    # sampling 5 data
    images, groud_truth = get_testing_data()
    correct = 0

    # Iterating each  data
    for i in range(len(images)):

        
        cur_img = cv2.imread(images[i])
        

        # sample ground-truth & processing to target
        splited_gt = get_all_gt(groud_truth[i])

     
        
        tensor_data = torch.FloatTensor(cur_img).unsqueeze(0).permute(0,3,1,2)

        result = model(tensor_data)

        draw_result(cur_img,result.detach()[0].tolist())

        # jaccard index validation
        for gt_index in range(len(splited_gt)):
            y = torch.FloatTensor(splited_gt[gt_index]).unsqueeze(0)
            if jaccard_index(y.detach()[0].tolist(),result.detach()[0].tolist()) == True:
                correct += 1
                break



    return correct/len(images)


def RGBD_evaluation(model): # returns accuracy

    # sampling 5 data
    images, groud_truth, depth = get_testing_data_depth()
    correct = 0

    # Iterating each  data
    for i in range(len(images)):

    
        

        # sample ground-truth & processing to target
        splited_gt = get_all_gt(groud_truth[i])

        raw_img = cv2.imread(images[i])
        cur_img = copy.deepcopy(raw_img) 
        raw_depth = tiff.imread(depth[i])[:,:,np.newaxis] 
        raw_img = np.append(raw_img,raw_depth,axis=2)
        temp = np.array([raw_img])
        tensor_data = torch.FloatTensor(temp)
        
        tensor_data = tensor_data.permute(0,3,1,2)

        result = model(tensor_data)

        draw_result(cur_img,result.detach()[0].tolist())

        # jaccard index validation
        for gt_index in range(len(splited_gt)):
            y = torch.FloatTensor(splited_gt[gt_index]).unsqueeze(0)
            if jaccard_index(y.detach()[0].tolist(),result.detach()[0].tolist()) == True:
                correct += 1
                break



    return correct/len(images)




def main():



    writer = SummaryWriter("logs")

    usr_input = input("1.Train RGB imgs 2.RGB Evaluation 3. Train RGB-D imags 4.RGBD Evaluation 5. Wild: ")

    if usr_input == "1":

        CNN,data_optimizer,data_loss = build_models()

        for i in range(500):
            data_loss_t = RGB_train_progess(CNN,data_optimizer,data_loss)
            writer.add_scalar("Training3",data_loss_t,global_step=i)
            print("Episode {} is end,  Loss : {} ".format(i,data_loss_t))

        torch.save(CNN, "data_net.pkl")
        writer.close()

    elif usr_input == "2":

        model_dict=torch.load("data_net.pkl")
        accuracy = RGB_evaluation(model_dict)
        print("Arccuracy: {}".format(accuracy))

    elif usr_input == "3":

        d_CNN,data_optimizer,data_loss = build_d_model()

        for i in range(500):
            data_loss_t = RGBD_train_progess(d_CNN,data_optimizer,data_loss)
            writer.add_scalar("Training_d",data_loss_t,global_step=i)
            print("Episode {} is end,  Loss : {} ".format(i,data_loss_t))

        torch.save(d_CNN, "d_data_net.pkl")
        writer.close()

    elif usr_input == "4":

        model_dict=torch.load("d_data_net.pkl")
        accuracy = RGBD_evaluation(model_dict)
        print("Arccuracy: {}".format(accuracy))

    elif usr_input == "5":

        while True:
            usr_input = input("Enter num 1-10 to view the result (-1 to exit):")
            if usr_input == "-1":
                break
            model_dict=torch.load("data_net.pkl")
            path = "wild/" + usr_input + ".png"
            img = cv2.imread(path)
            img = torch.FloatTensor(img).unsqueeze(0).permute(0,3,1,2)
            result = model_dict(img)
            draw_result(cv2.imread(path),result.detach()[0].tolist())
            
        
        

    







main()