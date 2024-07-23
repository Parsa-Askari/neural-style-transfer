from torchvision.models.vgg import vgg19
from torchvision.models import VGG19_Weights
import torch.nn as nn
import torch
from helpers import gramMatrix , featureMaps,Loss
class VggNewModel:
    def __init__(self,style,context,a=10,b=10000,device="cpu"):
        # super(VggNewModel,self).__init__()
        self.context=context
        self.style=style
        self.model=vgg19(weights=VGG19_Weights.DEFAULT)
        # self.model.eval()
        self.contextModel=[]
        self.styleModel=[]
        self.contextFeatures=[]
        self.styleFeatures=[]
        self.a=a
        self.b=b
        self.lamda=0.2
        self.loss_fn=Loss()
        self.device=device
    def forward(self,noise_img):
        context_loss=self.contextForward(noise_img)
        style_loss=self.styleForward(noise_img)
        loss=(self.a*context_loss)+(self.b*style_loss)
        self.total_loss=loss.detach().cpu().item()
        return loss
    def contextForward(self,noise):
        loss=0
        inp_noise=noise
        for i,model in enumerate(self.contextModel):
            res=model(inp_noise)
            target=self.contextFeatures[i]
            loss+=self.loss_fn.forward(featureMaps(res),target)
            inp_noise=res
        self.closs_recorder=loss.detach().cpu().item()
        return loss
    def styleForward(self,noise):
        loss=0
        inp_noise=noise
        for i,model in enumerate(self.styleModel):
            res=model(inp_noise)
            target=self.styleFeatures[i]
            scale=((target.shape[0])*(target.shape[1]))
            loss+=self.lamda*((self.loss_fn.forward(gramMatrix(res),target))/scale)
            inp_noise=res
        self.sloss_recorder=loss.detach().cpu().item()
        return loss
    @torch.no_grad()
    def set_layers(self,style,context):
        inp_context=context
        for model in self.contextModel:
            res=model(inp_context)
            self.contextFeatures.append(featureMaps(res))
            inp_context=res

        inp_style=style
        for model in self.styleModel:
            res=model(inp_style)
            self.styleFeatures.append(gramMatrix(res))
            inp_style=res

    def createModel(self):
        self.createContextModel()
        self.createStyleModel()
    def createContextModel(self):
        layer_number=1
        sublayer_numnber=1
        layers=self.model.features
        model=nn.Sequential()
        for layer in layers:
            layer_name=""
            if(self.context==[]):
                break
            if(isinstance(layer,nn.Conv2d)):
                layer_name=f"conv{layer_number}_{sublayer_numnber}"
            elif(isinstance(layer,nn.ReLU)):
                layer_name=f"relu{layer_number}_{sublayer_numnber}"
                sublayer_numnber+=1
                layer.inplace = False
            elif(isinstance(layer,nn.MaxPool2d)):
                layer_name=f"maxpool{layer_number}_{sublayer_numnber}"
                layer_number+=1
                sublayer_numnber=1

            model.add_module(layer_name,layer)
            if(layer_name==self.context[0]):
                # model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                self.contextModel.append(model.to(self.device))
                self.context=self.context[1:]
                model=nn.Sequential()
                
    def createStyleModel(self):
        layer_number=1
        sublayer_numnber=1
        layers=self.model.features
        model=nn.Sequential()
        for layer in layers:
            layer_name=""
            if(self.style==[]):
                break
            if(isinstance(layer,nn.Conv2d)):
                layer_name=f"conv{layer_number}_{sublayer_numnber}"
            elif(isinstance(layer,nn.ReLU)):
                layer_name=f"relu{layer_number}_{sublayer_numnber}"
                sublayer_numnber+=1
                layer.inplace = False
            elif(isinstance(layer,nn.MaxPool2d)):
                layer_name=f"maxpool{layer_number}_{sublayer_numnber}"
                layer_number+=1
                sublayer_numnber=1
            
            model.add_module(layer_name,layer)
            if(layer_name==self.style[0]):
                # model.eval()
                for param in model.parameters():
                    param.requires_grad = False

                self.styleModel.append(model.to(self.device))
                model=nn.Sequential()
                self.style=self.style[1:]
                