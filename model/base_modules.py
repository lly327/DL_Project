import torch.nn as nn



class ConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch', act_type='relu', kernel_size=3, stride=1, padding=0, dilation=1):
        super(ConvBlocks, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = None
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x