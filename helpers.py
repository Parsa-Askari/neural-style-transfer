import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.optim import LBFGS
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]



def returnTransforms(img,resize=True):
    (H,W)=img.size
    if(resize==False):
        (W,H)=(224,224)

    transform=transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Lambda(lambda x : x.unsqueeze(0))
    ])
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda x : x.squeeze(0)),
        transforms.ToPILImage(),
        transforms.Resize((W,H))
    ])

    return transform,reverse_transform

def showImage(images,count=3):
    fig,axes=plt.subplots(1,count)
    i=0
    for name,img in images.items():
        axes[i].imshow(img)
        axes[i].set_title(name)
        i+=1
    plt.show()

def gramMatrix(mat):
    (b_size,channel,H,W)=mat.shape
    new_mat=torch.reshape(mat,(b_size*channel,H*W))
    return torch.matmul(new_mat,new_mat.T)
def featureMaps(mat):
    (b_size,channel,H,W)=mat.shape
    new_mat=torch.reshape(mat,(b_size*channel,H*W))
    return new_mat

class Loss:
    def __init__(self):
        self.mse=F.mse_loss
    def forward(self,input,target):
        return self.mse(input,target)

def buildOptimizer(parameter,max_iter=1000):
    return LBFGS((parameter,),max_iter=max_iter, line_search_fn='strong_wolfe')
