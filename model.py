import torch 
from torch import nn 
from torchvision.models import resnet18

class AudioClassify_model(nn.Module):
  def __init__(self, input_shape, num_classes): 
    super(AudioClassify_model, self).__init__()
    self.model = resnet18(weights="IMAGENET1K_V1")

    self.model.fc=  embedding= nn.Linear(512, 512, bias= False)
    self.model.conv1= nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.classifier= nn.Sequential(
        nn.Dropout(0.3), 
        nn.Linear(512, num_classes)
    )

    nn.init.xavier_uniform_(self.model.fc.weight)

  def forward(self, x):
    x= self.model(x)
    x= self.classifier(x)
    
    return x 
