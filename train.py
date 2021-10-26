from genericpath import isdir
import torch 
import numpy as np 
import segmentation_models_pytorch as smp 
import os 
from dataset.reflection_dataset import ReflectionDataset, get_training_augmentation
from torch.utils.data import DataLoader 
import time 
import cv2 
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--name',default=None)
parser.add_argument('--epoch',default=50,type=int)
args = parser.parse_args()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STOR_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = os.path.join(STOR_DIR,'data','unet_data','unet_traindata')
TDATA_DIR = os.path.join(STOR_DIR,'data','unet_data','unet_validdata')

def main(): 
    name = args.name
    epoch = args.epoch
    if name == None: 
        folder_name= time.strftime("%Y%m%d-%H%M%S")
    else: 
        folder_name = name

    # ENCODER = 'se_resnext50_32x4d'
    ENCODER = 'resnet50'
    # ENCODER_WEIGHTS = 'imagenet'
    ENCODER_WEIGHTS = None
    CLASSES = ['mirror']
    ACTIVATION = 'sigmoid'
    # ACTIVATION = 'softmax2d'
    # ACTIVATION = 'softmax'
    DEVICE = 'cuda'

    train_dataset = ReflectionDataset(DATA_DIR, augmentation = get_training_augmentation()) 
    valid_dataset = ReflectionDataset(TDATA_DIR)
    # valid_dataset = ReflectionDataset(DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers =1)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers =1)
    input_channel = 6

    model = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = len(CLASSES),
        activation = ACTIVATION,
        in_channels = input_channel,
    )

    # loss = smp.utils.losses.DiceLoss()
    loss = smp.utils.losses.BCELoss()
    # loss = smp.utils.losses.CrossEntropyLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.8),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    max_score = 0

    loss_for_vis_train = []
    loss_for_vis_valid = []

    # if not os.path.isdir('./result_image/%s'%folder_name): 
    #     os.mkdir('./result_image/%s'%folder_name)
    if not os.path.isdir('./model_result/%s'%folder_name): 
        os.mkdir('./model_result/%s'%folder_name)

    for i in range(0, epoch):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        k = list(train_logs.keys())
        loss_for_vis_train.append(float(train_logs[k[0]]))
        loss_for_vis_valid.append(float(valid_logs[k[0]]))

        
        
        # do something (save model, change lr, etc.)
        # if max_score < valid_logs['iou_score']:
        #     max_score = valid_logs['iou_score']
        #     torch.save(model, './model_result/%s/best_model.pth'%folder_name)
        #     print('Model saved!')
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './model_result/%s/best_model_%s.pth'%(folder_name,str(i)))
            print('Model saved!')


            
        if i == (epoch - (epoch//3)):
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        # if (i % 10) == 0: 
        #     with torch.no_grad(): 
        #         for i in range(len(valid_dataset)):
        #             image, mask = valid_dataset[i] 
        #             # for dice loss
        #             # input_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        #             # for cross entropy 
        #             input_tensor = torch.from_numpy(image).to(DEVICE)
        #             pr_mask = model.predict(input_tensor)
        #             print('min',np.min(pr_mask.squeeze().cpu().numpy()))
        #             print('max',np.max(pr_mask.squeeze().cpu().numpy()))
        #             # pr_mask = (pr_mask.squeeze().cpu().numpy().round()) * 255
        #             pr_mask = (pr_mask.squeeze().cpu().numpy()) * 255
        #             timestr = time.strftime("%Y%m%d-%H%M%S")
        #             name = timestr + valid_dataset.ids[i]
        #             cv2.imwrite('./result_image/{0}/{1}.png'.format(folder_name,name),pr_mask)
        if ((i%50) == 0 and i > 0): 
            plot_path = './model_result/%s/%s.jpg'%(folder_name,str(i))
            plt.plot(range(i+1),loss_for_vis_train,'r',label='train')
            plt.plot(range(i+1),loss_for_vis_valid,'b',label='valid')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(plot_path)
            plt.close()
    
        
if __name__ == '__main__' : 
    main()