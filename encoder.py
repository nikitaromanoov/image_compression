import os
from tqdm import tqdm
import numpy as np

import torch
import torchvision

from PIL import Image
import matplotlib

import argparse

from matplotlib import pyplot as plt



class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="DEFAULT")

        self.conv1 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out
   
    
      
transform = torchvision.transforms.Compose(
        [
         torchvision.transforms.ToTensor()
        ])      
      


parser = argparse.ArgumentParser(description='')
parser.add_argument("--path_encoder", type=str, default="./models/encoder.pth",  help= "path of encoder" )
parser.add_argument("--path_image", type=str, default="./images/baboon.png",  help= "path of image" )
parser.add_argument("--path_result", type=str, default="compressed.json",  help= "path of image" )


args = parser.parse_args()


encoder = Encoder()
encoder.load_state_dict(torch.load(args.path_encoder))

image = Image.open(args.path_image)
img_tensor = transform(image) 
dec_img = encoder(img_tensor.reshape(1,3,512,512))



with open(args.path_result, "w") as w:
    w.write(json.dumps(dec_img))
