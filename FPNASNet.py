import torch.nn as nn
import math
import torch
import numpy as np

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def block1(in_channels,out_channels,ratio=1,track=False):
    block=nn.Sequential(
            nn.Conv2d(in_channels*ratio,in_channels*ratio,kernel_size=3,padding=1,stride=1,bias=False,groups=in_channels*ratio),
            nn.BatchNorm2d(in_channels*ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*ratio,out_channels,kernel_size=1,padding=0,stride=1,bias=False),
            nn.BatchNorm2d(out_channels))
    return block

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, shortcut, split):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.split=split
        self.shortcut=shortcut
        assert stride in [1, 2]

        if split:
             self.in_channels_l=inp//2
             inp=inp- self.in_channels_l
             oup=oup- self.in_channels_l
             if stride!=1:
                self.splitSeq=nn.Sequential(
                     nn.Conv2d(self.in_channels_l, self.in_channels_l, 3, stride, 1, groups=self.in_channels_l, bias=False),
                     nn.BatchNorm2d(self.in_channels_l),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(self.in_channels_l, self.in_channels_l, 1, 1, 0, bias=False),
                     nn.BatchNorm2d(self.in_channels_l)
                )
             else:
                self.splitSeq=nn.Sequential()

        if shortcut:
           if stride!=1 or inp!=oup:
              self.shortcutSeq=nn.Sequential(
                   nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
                   nn.BatchNorm2d(oup)

              )
           else:
              self.shortcutSeq=nn.Sequential()


        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def _shuffleChannels(self,x,groups=2):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % groups == 0)
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        if self.split:
             x1,x2=x[:,:self.in_channels_l,:,:], x[:,self.in_channels_l:,:,:]
             x1=self.splitSeq(x1)
        else:
             x2=x
        conv=self.conv(x2)
        if self.shortcut:
           x2=conv+self.shortcutSeq(x2)
        else:
           x2=conv

        x=self._shuffleChannels(torch.cat([x1,x2],dim=1)) if self.split else x2
        return x


class FPNASNet(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,channels=None):
        super(FPNASNet, self).__init__()
        block = InvertedResidual
        input_channel = 34
        block_channel= 24
        last_channel = 1280

        CHANNELS=[24,24, 40,40,40, 80,80,80,80, 112,112,112, 192,192,192, 320] if channels is None else channels
        RATIOS=  [4,6, 8,6,8, 6,8,8,8, 6,6,6, 4,8,4, 6]
        STRIDES =[2,1, 2,1,1, 2,1,1,1, 1,1,1, 2,1,1, 1]
        SHORTCUT=[1,1, 0,1,1, 1,0,1,1, 0,1,0, 1,0,0, 0]
        SPLIT   =[1,0, 0,1,0, 0,1,0,1, 1,0,0, 1,1,0, 0]

        input_channel  = int(input_channel * width_mult)
        block1_channel = int(block_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2), block1(input_channel,block_channel)]

        input_channel=block_channel
        for i in range(16):
            c=CHANNELS[i]
            s=STRIDES[i]
            t=RATIOS[i]
            shortcut=SHORTCUT[i]==1
            split=SPLIT[i]==1
            output_channel = int(c * width_mult)
            self.features.append(block(input_channel, output_channel, s, expand_ratio=t, shortcut=shortcut, split=split))
            input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__=='__main__':

  model=FPNASNet()
  #model.load_state_dict(torch.load("transfer_model.pth")["state_dict"])
  #input=torch.ones(1,3,224,224)
  #model.eval()
  #out=model(input)
  #print(out)

  for key in model.state_dict().keys():
      open("FPNAS.txt","a+").write(key+"\n")

