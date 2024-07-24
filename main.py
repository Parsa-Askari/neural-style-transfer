import torch
from helpers import *
from model import *
from tqdm.auto import tqdm
from torch.autograd import Variable

device="cuda" if torch.cuda.is_available() else "cpu"

context_folder="/kaggle/input/nst2-dataset/data/content-images"
style_folder="/kaggle/input/nst2-dataset/data/style-images"
style_name="edtaonisl.jpg"
context_name="figures.jpg"

style=["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]
context=["conv4_2"]

style_img=Image.open(os.path.join(style_folder,style_name))
context_img=Image.open(os.path.join(context_folder,context_name))

transform,reverse_transform=returnTransforms(style_img,resize=False)

style_img=transform(style_img).to(device)
context_img=transform(context_img).to(device)
noise_img=Variable(torch.clone(context_img),requires_grad=True)
# noise_img._to(device)
vgg_model=VggNewModel(style=style,context=context,device=device)
vgg_model.createModel()
vgg_model.set_layers(style=style_img,context=context_img)

# print(vgg_model.contextModel)
# print("===")
# print(vgg_model.styleModel)
loss_recorder={
    "sloss":[],
    "closs":[],
    "total loss":[]
}

def train(noise_img,vgg_model,epochs=20):
    optimizer=buildOptimizer(noise_img,max_iter=1000)
    with tqdm(total=epochs) as pbar:
        for ep in range(epochs):
            def closure():
                optimizer.zero_grad()
                loss=vgg_model.forward(noise_img)
                loss.backward()
                return loss
            optimizer.step(closure)
            total_loss=vgg_model.total_loss
            closs=vgg_model.closs_recorder
            sloss=vgg_model.sloss_recorder
            loss_recorder["sloss"].append(sloss)
            loss_recorder["closs"].append(closs)
            loss_recorder["total loss"].append(total_loss)
            pbar.set_postfix({"sloss":sloss,"closs":closs,"total loss":total_loss})
            pbar.update(1)
    return noise_img.clamp(0,1)

noise_img=train(noise_img=noise_img,vgg_model=vgg_model)


"""
for showing plots and images
"""
showImage({
   "style" : reverse_transform(style_img.cpu()),
   "context" :reverse_transform(context_img.cpu()),
   "converted" : reverse_transform(noise_img.cpu())
},3)

drawPlots(loss_recorder)