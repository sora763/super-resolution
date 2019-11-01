import scipy
#from glob import glob
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image

class DataLoader():
    def __init__(self, dataset_name, img_res=(400, 400)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        #path = glob('./data/%s/*' % (self.dataset_name))
        files = glob.glob("/home/sora/SRGAN-Keras/data_crop/ips/1/train_mask1/*.png", recursive=True)
        if data_type =="train":
            batch_images = random.sample(files, batch_size)
        else:
            files = glob.glob("/home/sora/keras-BoF/weights/valid_all/50_40_classifi/test/vis/*.png", recursive=True)
            batch_images = random.sample(files, 12)

        #batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #img = Image.open(img_path)
            img = Image.fromarray(img)
            h, w = self.img_res
            low_h, low_w = int(h / 8), int(w / 8)

            img_hr = img.resize((h, w))  #(64, 64)
            img_lr = img.resize((low_h, low_w))
            #print(img_hr.size)
            img_hr = np.array(img_hr)
            #img_hr = (img_hr - 127.5) / 127.5
            img_lr = np.array(img_lr)
            #img_lr = (img_lr - 127.5) / 127.5
            #img_hr = scipy.misc.imresize(img, self.img_res)
            #img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

"""
    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
"""
