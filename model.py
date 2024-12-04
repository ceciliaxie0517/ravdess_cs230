from torchvision.models import resnet18
import torch.nn as nn
import torch

__all__ = ['ResNet18']

class classifier(nn.Module):
    def __init__(self, in_feat, num_classes):
        super(classifier, self).__init__()
        self.fc = nn.Linear(in_feat, num_classes)
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x):
        '''多样本Dropout'''
        x = torch.stack([x]*10, dim=0)
        x = self.drop(x)
        x = self.fc(x)
        return x.mean(dim=0)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=True)
        # 由于mel频谱输入只有一个通道，所以需要将resnet18的输入通道数设置为1
        conv1_weight = self.model.conv1.weight.data
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.conv1.weight.data = conv1_weight.sum(dim=1, keepdim=True)
       
        self.model.fc = classifier(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    

if __name__ == '__main__':
    from torchkeras import summary
    summary(ResNet18(num_classes=6), input_shape=(1, 40, 173))
