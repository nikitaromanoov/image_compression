import os
from tqdm import tqdm
import numpy as np

import json

import torch
import torchvision

from PIL import Image
import matplotlib

import argparse

from matplotlib import pyplot as plt

from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import\
   BaseFrequencyTable,\
   SimpleAdaptiveModel
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def dequantification(l, B=2):
    return [i/(2** B) for i in l]

def arithmetic_decoding(q, l , B=2):
    a_c = SimpleAdaptiveModel({k: 1. / (2 ** B) for k in [i for i in range(0, 2 ** B + 1)]})
    coder = AECompressor(a_c)

    return coder.decompress(q, l)  
    
    
    

      
      
      
transform = torchvision.transforms.Compose(
        [
         torchvision.transforms.ToTensor()
        ])      
      


parser = argparse.ArgumentParser(description='')
parser.add_argument("--path_decoder", type=str, default="./models/decoder.pth",  help= "path of decoder" )
parser.add_argument("--path_compressed", type=str, default="compressed.json",  help= "path of image" )
parser.add_argument("--path_result", type=str, default="result.png",  help= "path of image" )
parser.add_argument("--B", type=int, default=2,  help= "B" )


args = parser.parse_args()

with open(args.path_compressed) as f:
    cod = json.loads(f.read())
    
e = dequantification(arithmetic_decoding(cod[0], cod[1], args.B), args.B)

decoder = Decoder()
decoder.load_state_dict(torch.load(args.path_decoder))

qwer = decoder(torch.Tensor(e))

torchvision.utils.save_image(qwer, args.path_result)

