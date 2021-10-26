import torch 
import numpy as np 
import segmentation_models_pytorch as smp 
import os 
from dataset.reflection_dataset import ReflectionDataset
from torch.utils.data import DataLoader 
import cv2
import time 
import argparse



CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STOR_DIR = os.path.dirname(CUR_DIR)
# DATA_DIR = os.path.join(STOR_DIR,'data','unet_data','unet_traindata')

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--dir',default=None)
parser.add_argument('--data_type',default = 'test')
parser.add_argument('--epoch_num',default = 0,type=str)
args = parser.parse_args()

def main(): 

    epnum = args.epoch_num
    data_type = args.data_type
    if data_type == 'test':
        DATA_DIR = os.path.join(STOR_DIR,'data','unet_data','unet_testdata')
    elif data_type == 'train':
        DATA_DIR = os.path.join(STOR_DIR,'data','unet_data','unet_traindata')

    DEVICE = 'cuda'
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    model_path = args.dir
    trained_model = torch.load('./model_result/{0}/best_model_{1}.pth'.format(model_path,epnum))
    test_dataset = ReflectionDataset(DATA_DIR)

    test_dataloader = DataLoader(test_dataset)
    test_epoch = smp.utils.train.ValidEpoch(
        model = trained_model,
        loss = loss, 
        metrics = metrics, 
        device = DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    folder_name= time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir('./result_image/%s'%folder_name): 
        os.mkdir('./result_image/%s'%folder_name)

    for i in range(len(test_dataset)): 
        image, mask = test_dataset[i]
        input_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = trained_model.predict(input_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round()) * 255
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # timestr = timestr + str(i)
        name = test_dataset.ids[i]
        cv2.imwrite('./result_image/{0}/{1}.png'.format(folder_name, name),pr_mask)

    


    
    








if __name__ == '__main__': 
    main()