import torch
from helpers import *
from model import *
from tqdm.auto import tqdm
from torch.autograd import Variable
context_folder="./data/content-images"
style_folder="./data/style-images"
style_name="edtaonisl.jpg"
context_name="figures.jpg"

style=["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]
context=["conv4_2"]

style_img=Image.open(os.path.join(style_folder,style_name))
context_img=Image.open(os.path.join(context_folder,context_name))

transform,reverse_transform=returnTransforms(style_img,resize=False)

style_img=transform(style_img)
context_img=transform(context_img)
noise_img=Variable(torch.clone(context_img),requires_grad=True)

vgg_model=VggNewModel(style=style,context=context)
vgg_model.createModel()
vgg_model.set_layers(style=style_img,context=context_img)

def train(noise_img,vgg_model,epochs=5):
    optimizer=buildOptimizer(noise_img)
    for ep in tqdm(range(epochs)):
        def closure():
            optimizer.zero_grad()
            loss=vgg_model.forward(noise_img)
            loss.backward()
            return loss
        optimizer.step(closure)

train(noise_img=noise_img,vgg_model=vgg_model)





