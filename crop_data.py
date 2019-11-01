#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:50:52 2019

@author: sora
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:57:46 2018

@author: hayashi
"""


import os
import numpy as np
import cv2
from PIL import Image
import scipy as sp

size = 800
step = 400
hight = 1200
width = 1600
path = "/home/sora"
trm = 0
for d_num in [1,2,3,4,5]:
    print('---trainmask start')
    os.chdir(path + "/data/ips/dataset")
    #trainmask
    f = open("train_mask%d.txt" %(d_num))
    lines = f.readlines()
    f.close()

    #改行を取り除く
    for r in range(len(lines)):
        lines[r] = lines[r].strip()

    for line in lines:
        path2 = os.path.join(path,line)
        im2 = cv2.imread(path2)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        for y in range(0, hight-size+1, step):
            for x in range(0, width-size+1, step):
                c_im2 = im2[y:size+y, x:size+x]
                #c_im2.resize(473,473,3)
                #imim = c_im2[:,:,0]+c_im2[:,:,1]+c_im2[:,:,2]
                #c_im2[:,:,0]=(imim==255)*c_im2[:,:,0]
                #c_im2[:,:,1]=(imim==255)*c_im2[:,:,1]
                #c_im2[:,:,2]=(imim==255)*c_im2[:,:,2]
                #c_im2 = np.array(c_im2)
                pil_img = Image.fromarray(np.uint8(c_im2))
                os.chdir("/home/sora/SRGAN-Keras/data_crop/ips/%d/train_mask%d"%(d_num,d_num))
                pil_img.save('%d.png' %(trm))
                trm += 1
