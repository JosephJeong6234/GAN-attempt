# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 00:03:49 2025

@author: jumpj
"""
#defines

#import pandas as pd
import numpy as np
import torch
#import torch.nn.functional as F
from torch import nn
import os
import cv2 
#from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
#from PIL import Image
#import random

from torchvision.utils import save_image
#%% define discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1) #not sure if is 6
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1) #might work not sure as math odd
        self.conv6 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
        self.batchNorm1 = nn.BatchNorm2d(128)
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.batchNorm3 = nn.BatchNorm2d(512)
        self.batchNorm4 = nn.BatchNorm2d(512)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z1 = self.relu(self.conv1(x))
        Z2 = self.relu(self.batchNorm1(self.conv2(Z1)))
        Z3 = self.relu(self.batchNorm2(self.conv3(Z2)))
        Z4 = self.relu(self.batchNorm3(self.conv4(Z3)))
        Z5 = self.relu(self.batchNorm4(self.conv5(Z4)))
        patchOut = self.sigmoid(self.conv6(Z5))
        return patchOut 
#discrim = Discriminator() #Moved below
#criterion = nn.BCELoss()
#optimizer = torch.optim.Adam(discrim.parameters(), lr=0.0001, betas=(0.5, 0.999))
#note need a torch.cat([source, target], dim=1) when passing to discrim

#%% define encoder and decoder
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(DecoderBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.decode = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.decode(x)
        x = torch.cat([x, skip_input], dim=1) #on channel
        return x
    
#%% define generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.en1 = EncoderBlock(3, 64, batchnorm=False)
        self.en2 = EncoderBlock(64, 128)
        self.en3 = EncoderBlock(128, 256)
        self.en4 = EncoderBlock(256, 512)
        self.en5 = EncoderBlock(512, 512)
        self.en6 = EncoderBlock(512, 512)
        self.en7 = EncoderBlock(512, 512)
        
        self.convIn = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        self.de1 = DecoderBlock(512, 512)
        self.de2 = DecoderBlock(1024, 512)
        self.de3 = DecoderBlock(1024, 512)
        self.de4 = DecoderBlock(1024, 512, dropout=False)
        self.de5 = DecoderBlock(1024, 256, dropout=False)
        self.de6 = DecoderBlock(512, 128, dropout=False)
        self.de7 = DecoderBlock(256, 64, dropout=False)
        
        self.convOut = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        e7 = self.en7(e6)
        
        bottleneck = self.relu(self.convIn(e7))
        
        d1 = self.de1(bottleneck, e7)
        d2 = self.de2(d1, e6)
        d3 = self.de3(d2, e5)
        d4 = self.de4(d3, e4)
        d5 = self.de5(d4, e3)
        d6 = self.de6(d5, e2)
        d7 = self.de7(d6, e1)
        
        g = self.tanh(self.convOut(d7))
        return g
#gen = Generator() #moved to training loop

#%% steal the data loading functions from the original a to b file

#slightly changed to work with the weird nested folders 
def load_images_jitter(path):
    # load image
    img = cv2.imread(path)
    
    # split for map
#    img_A = img[:,:600,:]  
#    img_B = img[:,600:,:] 

    # split for facades
    img_A = img[:,:256,:] 
    img_B = img[:,256:,:]  
    
    # resize as 286 x 286
    img_A286 = cv2.resize(img_A,(286,286) )
    img_B286 = cv2.resize(img_B,(286,286) )
    
    # randomly crop to (256, 256)
    # select between 0 and 30 (286 - 256)
    [ix,iy] = np.random.randint(0, 29, 2)
    img_A256 = img_A286[ix:ix+256,iy:iy+256,:]
    img_B256 = img_B286[ix:ix+256,iy:iy+256,:]
    
    # randomly mirror
    if np.random.uniform()> 0.5:
        # random mirroring
        img_A256 = np.fliplr(img_A256)
        img_B256 = np.fliplr(img_B256)
    
    
    # BGR to RGB
    img_A1 = img_A256[...,::-1]    
    img_B1 = img_B256[...,::-1]  
    
    # normalize [-1,1]
    img_A2 = (img_A1 / 127.5) - 1
    img_B2 = (img_B1 / 127.5) - 1    
    return img_A2, img_B2

#%% figure out what the patch shape is
def getPatchSize(discriminator, input_shape = (3, 256, 256), device='cpu'):
    dummy_input = torch.randn(1, 6, input_shape[1], input_shape[2]).to(device)

    with torch.no_grad():
        output = discriminator(dummy_input)

    patch_h = output.shape[2]
    patch_w = output.shape[3]

    return patch_h, patch_w


#%% load data
basePath = os.path.dirname(__file__)  # folder containing the script
trainPath = os.path.join(basePath, "cityscapes", "leftImg8bit_trainvaltest", "leftImg8bit", "train")
valPath =  os.path.join(basePath, "cityscapes", "leftImg8bit_trainvaltest", "leftImg8bit", "val")
testPath =  os.path.join(basePath, "cityscapes", "leftImg8bit_trainvaltest", "leftImg8bit", "test")
#r"C:\Users\jumpj\assignment6\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\train" #alt way is this
def getFileList(image_root):
    filelist = []

    for root, _, files in os.walk(image_root):
        for file in files:
            if file.endswith('_leftImg8bit.png'):
                # Relative path from image_root to the file's folder
                relative_folder = os.path.relpath(root, image_root)
                filelist.append((relative_folder, file))

    return filelist
trainList = getFileList(trainPath)
valList = getFileList(valPath)
testList = getFileList(testPath)
m = len(trainList) #relative to trainList after all 

#%% dataloader

class CityscapesDataset(Dataset):
    def __init__(self, file_list, base_path, transform=None):
        self.file_list = file_list
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        folder, filename = self.file_list[idx]
        full_path = os.path.join(self.base_path, folder, filename)
        img_A_np, img_B_np = load_images_jitter(full_path)

        img_A = torch.from_numpy(img_A_np.transpose(2, 0, 1)).float()
        img_B = torch.from_numpy(img_B_np.transpose(2, 0, 1)).float()

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

batch_size = 256
trainData = CityscapesDataset(trainList, trainPath)
train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=0) #kept crashing if > 0

valData = CityscapesDataset(valList, valPath)
testData = CityscapesDataset(testList, testPath)
val_loader = DataLoader(valData, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=0)

#%% do loop as making into 1 model doesn't seem right as I would still need to update seperately
gen = Generator()
discrim = Discriminator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discrim = discrim.to(device)
gen = gen.to(device)

patch_h, patch_w = getPatchSize(discrim, input_shape=(3, 256, 256), device=device)
patch = (patch_h, patch_w)

#label_real = torch.full((m, 1, patch_h, patch_w), 0.9, device=device) #apparently preds better this way if not 1 and 0 exactly
#label_fake = torch.full((m, 1, patch_h, patch_w), 0.1, device=device)

genOpt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999)) #assuming is same paras from the actual
discOpt = torch.optim.Adam(discrim.parameters(), lr=1e-5, betas=(0.5, 0.999)) #was overpowering generator
bce_loss = torch.nn.BCELoss() #exists other type BCEWithLogitsLoss but already have discrim sigmoid 
l1_loss = torch.nn.L1Loss()

os.makedirs('./model', exist_ok=True)

epochs = 100
genAdd = 1

for epoch in range(1, epochs+1):
    d_loss1_sum = 0
    d_loss2_sum = 0
    g_loss_sum = 0

    counter = 1
    l1weight = max(50 * (1 - epoch/100), 10)
    
    for img_A, img_B in train_loader:
        img_A = img_A.to(device)
        img_B = img_B.to(device)

        #discrim
        gen.eval()
        discrim.train()

        with torch.no_grad():
            fake_B = gen(img_A).detach()

        discOpt.zero_grad()
        # input_real = torch.cat([img_A, img_B], dim=1)
        # input_fake = torch.cat([img_A, fake_B], dim=1)
        
        # d_real = discrim(input_real)
        # d_fake = discrim(input_fake)
        noise_std = max(0.05 * (1 - epoch / epochs), 0.01)  
        noisy_real = img_B + noise_std * torch.randn_like(img_B)
        noisy_fake = fake_B.detach() + noise_std * torch.randn_like(fake_B)
        
        #for normalization of tanh
        noisy_real = noisy_real.clamp(-1.0, 1.0)
        noisy_fake = noisy_fake.clamp(-1.0, 1.0)
        
        input_real = torch.cat([img_A, noisy_real], dim=1)
        input_fake = torch.cat([img_A, noisy_fake], dim=1)
        
        d_real = discrim(input_real)
        d_fake = discrim(input_fake)
        
        label_real = torch.full_like(d_real, 0.9) #apparently works ebtter if not 1 and 0 exactly
        label_fake = torch.full_like(d_fake, 0.1)
        
        flip_mask_real = torch.rand_like(label_real) < max(0.05 * (1 - epoch / epochs), 0.01) 
        flip_mask_fake = torch.rand_like(label_fake) < max(0.05 * (1 - epoch / epochs), 0.01) 
        label_real[flip_mask_real] = 0.1
        label_fake[flip_mask_fake] = 0.9
        
        d_loss1 = bce_loss(d_real, label_real)
        d_loss2 = bce_loss(d_fake, label_fake)
        d_loss = (d_loss1 + d_loss2) * 0.5

        d_real_prob = d_real.mean().item()
        d_fake_prob = d_fake.mean().item()
        pred_real = (d_real >= 0.5).float()
        pred_fake = (d_fake < 0.5).float()
        d_real_acc = (pred_real == 1).float().mean().item()
        d_fake_acc = (pred_fake == 1).float().mean().item()
        if counter % 10 == 0:
            print(
                f"D_real_prob={d_real_prob:.3f}, "
                f"D_fake_prob={d_fake_prob:.3f}, "
                f"D_real_acc={d_real_acc:.3f}, "
                f"D_fake_acc={d_fake_acc:.3f}"
            )
            counter = 1
        else:
            counter += 1

        d_acc = 0.5 * (d_real_acc + d_fake_acc)
        if d_acc < 0.85:
            d_loss.backward()
            discOpt.step()
        else:
            # skip update to prevent overpowering
            pass

        #gen
        for i in range(genAdd): #generator wasn't making good images so train more in 1 epoch
            gen.train()
            discrim.eval()
    
            genOpt.zero_grad()
            fake_B = gen(img_A)
            input_fake = torch.cat([img_A, fake_B], dim=1)
            pred_fake = discrim(input_fake)
    
            g_adv = bce_loss(pred_fake, label_real)
            g_l1 = l1_loss(fake_B, img_B) * l1weight
            g_loss = g_adv + g_l1
            g_loss.backward()
            genOpt.step()
    
            d_loss1_sum += d_loss1.item()
            d_loss2_sum += d_loss2.item()
            g_loss_sum += g_loss.item()

    print(f"Epoch {epoch}, d1: {d_loss1_sum/m:.3f}, d2: {d_loss2_sum/m:.3f}, g: {g_loss_sum/(m*genAdd):.3f}")

    #val check
    gen.eval()
    val_g_loss_sum = 0
    with torch.no_grad():
        for val_img_A, val_img_B in val_loader:
            val_img_A = val_img_A.to(device)
            val_img_B = val_img_B.to(device)
    
            fake_B = gen(val_img_A)
            input_fake = torch.cat([val_img_A, fake_B], dim=1)
            pred_fake = discrim(input_fake)
            
            g_adv = bce_loss(pred_fake, torch.ones_like(pred_fake))
            g_l1 = l1_loss(fake_B, val_img_B) * l1weight
            val_g_loss_sum += (g_adv + g_l1).item()
    print(f"Validation Generator Loss: {val_g_loss_sum / len(val_loader):.4f}")

    if epoch % 5 == 0:
        with torch.no_grad():
            gen.eval()
            fake_B = gen(img_A)
        #save visualization
        save_image((fake_B + 1) / 2, f'plot2_epoch_{epoch:03d}.png')
        torch.save(gen.state_dict(), f'./model/cityAtoB_epoch_{epoch:03d}.pt')
        print(f"Saved model and image at epoch {epoch}")

gen.eval()  # set generator to eval mode
test_loss_sum = 0
with torch.no_grad():
    for test_img_A, test_img_B in test_loader:
        test_img_A = test_img_A.to(device)
        test_img_B = test_img_B.to(device)

        fake_B = gen(test_img_A)

        #calc loss
        l1 = l1_loss(fake_B, test_img_B)
        test_loss_sum += l1.item()

    #save some output images
    image_counter = 0
    for batch_idx, (test_img_A, test_img_B) in enumerate(test_loader):
        test_img_A = test_img_A.to(device)
        fake_B = gen(test_img_A)

        for i in range(fake_B.size(0)):
            img = (fake_B[i] + 1) / 2  # denormalize tanh
            save_image(img, f'test_output_{image_counter}.png')
            image_counter += 1

            if image_counter >= 10:  # limit total saved images to 10
                break
        if image_counter >= 10:
            break

avg_test_loss = test_loss_sum / len(test_loader)
print(f"Average test L1 loss: {avg_test_loss:.4f}")


#final save
torch.save(gen.state_dict(), './model/temp_cityAtoB_final.pt')


        