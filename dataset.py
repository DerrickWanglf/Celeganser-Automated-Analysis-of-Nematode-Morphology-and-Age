import os, random, time, copy
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import ndimage, signal
from scipy import misc
import math
import cv2 
import csv
import matplotlib.pyplot as plt
from io import BytesIO
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

        
class Dataset4Celeganser(Dataset):
   
    def __init__(self, data_dir,size=[960, 960],sets ="train", downsampleFactor=1, downsampleFactorOri=0.25, scale=(0,1,2,3), intersectThre=-1, sizeOri=[512, 512]):

        self.data_dir = data_dir
        self.downsampleFactor = downsampleFactor
        self.downsampleFactorOri = downsampleFactorOri
        self.originPaths = []
        self.imagePaths = []
        self.gtPaths = []
        self.straight = []
        self.rgtPaths = []
        self.ages = {}
        self.cropped_size = size   
        self.croppedSizeOri = sizeOri
        self.scale = scale   
        ifadded = {}
        
        
        imageFolder = path.join(self.data_dir, 'image')
        gtFolder = path.join(self.data_dir, 'GT')
        rgtFolder = path.join(self.data_dir, 'R_GT')
        originFolder = path.join(self.data_dir, 'origin')
        straightFolder = path.join(self.data_dir, 'straight')
        dataFile = path.join(self.data_dir, "data.tsv")
          
        with open(dataFile) as f:
            f_csv = csv.reader(f,delimiter='\t')
            for row in f_csv:   
                name = (row[0] + ".png")
                self.ages[name] = int(row[3])

        for _, _, files in os.walk(imageFolder):
            for f in files:
                if f.endswith(".png"):
                    self.gtPaths.append(path.join(gtFolder, f))
                    self.imagePaths.append(path.join(imageFolder, f))
                    self.originPaths.append(path.join(originFolder, f))
                    self.straight.append(path.join(straightFolder, f))
                    self.rgtPaths.append(path.join(rgtFolder, f))            
                
        #print the path and size
        print(sets,"image  \t\tsize", len(self.imagePaths))
        print(sets,"GT \t\tsize", len(self.gtPaths))
        self.current_set_len = len(self.imagePaths)   


    def __len__(self):        
        return self.current_set_len
    
    def getAllFiles(self):
        return self.imagePaths, self.gtPaths
        
    
    def __getitem__(self, idx):
        #1127, 41
        imgName = self.imagePaths[idx]
        gtName = self.gtPaths[idx]
        rgtName = self.rgtPaths[idx]
        originName = self.originPaths[idx]
        straightName = self.straight[idx]
        
        filename = imgName.split('/')[-1]  
        if filename[-6] == '-':
            filename = filename[:-6] + filename[-4:]
        age = self.ages[filename]

        image = cv2.imread(imgName, -1)
        gt = cv2.imread(gtName, -1)   
        rgt = cv2.imread(rgtName, -1)
        origin = cv2.imread(originName, -1)   
        straight = cv2.imread(straightName, -1)
        

        image = image.astype(np.float32) 
        gt = gt.astype(np.float32) 
        rgt = rgt.astype(np.float32) 
        origin = origin.astype(np.float32) 
        straight = straight.astype(np.float32) 
     
        
        
        #### DownSample the image 
        if self.downsampleFactorOri > 0:
            height, width = origin.shape[:2] 
            t_size = (int(width * self.downsampleFactorOri), int(height * self.downsampleFactorOri))  
            origin = cv2.resize(origin, t_size, interpolation=cv2.INTER_CUBIC) 
                
        if self.downsampleFactor > 0 and self.downsampleFactor != 1:#and self.set_name!='train':    
            height, width = image.shape[:2] 
            t_size = (int(width * self.downsampleFactor), int(height * self.downsampleFactor))  
            image = cv2.resize(image, t_size, interpolation=cv2.INTER_CUBIC) 
            gt = cv2.resize(gt, t_size, interpolation=cv2.INTER_NEAREST)
            rgt = cv2.resize(rgt, t_size, interpolation=cv2.INTER_NEAREST)
            
        
        
        height, width = image.shape[:2] 
        #### Crop the image image into given size
        new_hight = [int((height - self.cropped_size[0]) / 2), int((height + self.cropped_size[0]) / 2) ] 
        new_width = [int((width - self.cropped_size[1]) / 2), int((width + self.cropped_size[1]) / 2 )] 
        
        cropped_image = image[new_hight[0]:new_hight[1], new_width[0]:new_width[1]]
        cropped_gt = gt[new_hight[0]:new_hight[1], new_width[0]:new_width[1]]  
        cropped_rgt = rgt[new_hight[0]:new_hight[1], new_width[0]:new_width[1]]
        
        height, width = origin.shape[:2] 
        #### Crop the image image into given size
        new_hight = [int((height - self.croppedSizeOri[0]) / 2), int((height + self.croppedSizeOri[0]) / 2) ] 
        new_width = [int((width - self.croppedSizeOri[1]) / 2), int((width + self.croppedSizeOri[1]) / 2 )] 
        
        cropped_origin = origin[new_hight[0]:new_hight[1], new_width[0]:new_width[1]]
       
        
        
        cropped_image = np.expand_dims(cropped_image, axis=0)
        cropped_origin = (cropped_origin - 20000) / 20000
        cropped_origin = np.expand_dims(cropped_origin, axis=0)
        extend_ori = np.concatenate((cropped_origin,cropped_origin,cropped_origin),axis=0)
        
        ## scale the image pixel value into a trainable range
        cropped_image = (cropped_image - 20000) / 20000
        extend_img = np.concatenate((cropped_image,cropped_image,cropped_image),axis=0)
        mask = (cropped_gt[:,:,2]>0).astype(np.float32)

        outside_mask = (cropped_gt[:,:,0] == 0)
        mask = (cropped_gt[:,:,2]>0).astype(np.float32)
        
      
        maxx = np.max(cropped_gt[:,:,1])
        maxy = np.max((cropped_gt[:,:,0]))
        tem = cropped_gt[:,:,0]
        tem[tem==0] = maxy
        secMin = np.min(tem)
        
        if maxx != 0:
            xcd = cropped_gt[:,:,1] / 2000
            ycd = (cropped_gt[:,:,0] - secMin) / 6000 #/ 5000
            rycd = cropped_rgt[:,:,0] / 6000
            
            ycd = ycd * mask
        else:
            xcd = cropped_gt[:,:,1]
            ycd = cropped_gt[:,:,0]
            rycd = cropped_rgt[:,:,0]
            
            
            
            
        masks, xcoods, ycoods, rycoods = [], [], [], []

        for i in self.scale:
            s = 2 ** i
            xtmp = cv2.resize(xcd,(self.cropped_size[1]//s, self.cropped_size[0]//s), interpolation=cv2.INTER_NEAREST)
            xtmp = np.expand_dims(xtmp, axis=0)
            
            ytmp = cv2.resize(ycd,(self.cropped_size[1]//s, self.cropped_size[0]//s), interpolation=cv2.INTER_NEAREST) 
            ytmp = np.expand_dims(ytmp, axis=0)
            
            rytmp = cv2.resize(rycd,(self.cropped_size[1]//s, self.cropped_size[0]//s), interpolation=cv2.INTER_NEAREST) 
            rytmp = np.expand_dims(rytmp, axis=0)
            
            mtmp = cv2.resize(mask,(self.cropped_size[1]//s, self.cropped_size[0]//s), interpolation=cv2.INTER_NEAREST) 
            mtmp = np.expand_dims(mtmp, axis=0)
            
            xcoods.append(xtmp)
            ycoods.append(ytmp)
            rycoods.append(rytmp)
            masks.append(mtmp)
        
        return extend_img, masks, xcoods, ycoods, rycoods, extend_ori, age, straight
                 
                 
    
    