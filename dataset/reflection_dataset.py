from numpy.core.shape_base import stack
from torch.utils.data import Dataset 
import numpy as np 
import os 
import cv2 
import albumentations as albu 

class ReflectionDataset(Dataset): 
    CLASSES = ['unlabelled', 'mirror']
    def __init__(self, data_root, augmentation=None):
        self.data_root = data_root
        self.ids = os.listdir(data_root)
        self.img_path = [os.path.join(data_root,idx) for idx in self.ids]
        self.augmentation = augmentation

    def stack_channels(self, idx): 
        path = self.img_path[idx]
        count = os.path.join(path,'count.png')
        close = os.path.join(path,'close.png')
        color = os.path.join(path,'color.jpg')
        if not os.path.isfile(color): 
            color = os.path.join(path,'color.png')
        far = os.path.join(path,'far.png')
        countimg = cv2.imread(count,cv2.IMREAD_GRAYSCALE)
        closeimg = cv2.imread(close,cv2.IMREAD_GRAYSCALE)
        colorimg = cv2.imread(color,cv2.IMREAD_COLOR)
        
        colorimg = np.transpose(colorimg,(2,0,1))
        farimg = cv2.imread(far,cv2.IMREAD_GRAYSCALE)

        # if only 3 channel
        stacked_img = np.stack([countimg, closeimg, farimg])
        # stacked_img = np.stack([countimg, closeimg])
        stacked_img = np.vstack([stacked_img, colorimg])
        return stacked_img

    def __getitem__(self, idx): 

        imgnum = self.ids[idx]
        target_imsize = (2016,576)
        image =  self.stack_channels(idx)
        mask = np.load(os.path.join(self.img_path[idx],'label.npy'))
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask,dsize=target_imsize)
        image = cv2.resize(np.transpose(image,(1,2,0)),dsize=target_imsize)
        
        if self.augmentation: 
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']

        image = image.astype(np.float32)
        mask =  mask.astype(np.float32)
        # mask =  mask.astype(np.int64)
        mask = np.expand_dims(mask, axis = 0)
        image = np.transpose(image,(2,0,1))
        
        return image, mask 

    
    def __len__(self):
        return len(self.ids)




def get_training_augmentation():
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            # albu.PadIfNeeded(min_height=576, min_width=2016, always_apply=True, border_mode=0),
            albu.RandomCrop(height=384, width=1152, always_apply=True),
            # albu.RandomCrop(height=576, width=576, always_apply=True),
        ]
        
        return albu.Compose(train_transform)
