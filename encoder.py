import os
from tqdm import tqdm
import numpy as np

import torch
import torchvision

from PIL import Image
import matplotlib

import argparse



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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class type1(torch.nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv2d_1 = torch.nn.Sequential(
                        torch.nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(inp))
        self.conv2d_2 = torch.nn.Sequential(
                        torch.nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(out))
        #self.shortcut = torch.nn.Sequential()
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        #out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class type2(torch.nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv2d_1 = torch.nn.Sequential(
                        torch.nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(inp))
        self.conv2d_2 = torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2),
                            torch.nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1),
                            torch.nn.BatchNorm2d(out))
        self.upsample = torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2),
                            torch.nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1))
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        out += self.upsample(x)
        out = self.relu(out)
        return out    
    
    
    
    
    
    
    
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        
        
        self.layer1 = torch.nn.Sequential(*[type2(512, 256),
                            type1(256, 256)])

        self.layer2 = torch.nn.Sequential(*[type2(256, 128),
                            type1(128, 128)])

        self.layer3 = torch.nn.Sequential(*[type2(128, 64),
                            type1(64, 64)])

        self.layer4 = torch.nn.Sequential(*[type1(64, 64),
                            type1(64, 64)])
        
        self.resize = torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=4),
                            torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))
        

    
    
    def forward(self, x):
        out = x.view(-1, 512, 8, 8)
        out = self.up(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.resize(out)
        return out    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

      
      
      
transform = torchvision.transforms.Compose(
        [
         torchvision.transforms.ToTensor()
        ])      
      


parser = argparse.ArgumentParser(description='')
parser.add_argument("--path_encoder", type=str, default="./models/encoder.pth",  help= "path of encoder" )
parser.add_argument("--path_decoder", type=str, default="./models/decoder.pth",  help= "path of decoder" )
parser.add_argument("--path_image", type=str, default="./images/baboon.png",  help= "path of image" )


args = parser.parse_args()


encoder = Encoder()
encoder.load_state_dict(torch.load(args.path_encoder))

image = Image.open(args.path_image)
img_tensor = transform(image) 
dec_img = encoder(img_tensor.reshape(1,3,512,512))
print(dec_img)

decoder = Decoder()
decoder.load_state_dict(torch.load(args.path_decoder))
dec_img = decoder(encoder(img_tensor.reshape(1,3,512,512)))
matplotlib.pyplot.imshow( dec_img.squeeze().permute(1,2,0).detach().numpy() )
matplotlib.pyplot.show()
