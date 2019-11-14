import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
#引入了1x1的卷积块来减少参数量 
def conv3x3(in_channel, out_channel, stride=1):
     return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
def conv2x2(in_channel,out_channel):
     return nn.Conv2d(in_channel,out_channel,(2,3),(1,1),(0,1),bias=False)
def conv1x1(in_channel,out_channel,stride=1):
     return nn.Conv2d(in_channel,out_channel,1,stride=stride,padding=0,bias=False)

     

class residual_block(nn.Module):
     def __init__(self, in_channel, out_channel,  same_shape=True,same_channel=True):
          super(residual_block, self).__init__()
          self.same_shape=same_shape
         
          self.same_channel=same_channel
          stride = 1 if self.same_shape else 2
          #stride=2则conv1x1为空洞卷积
          self.conv1 = conv1x1(in_channel, in_channel)
          self.bn1 = nn.BatchNorm2d(in_channel)
          
          self.conv2 = conv3x3(in_channel, out_channel,stride=stride)#卷积核为3x3，步长为2，padding为1，则尺寸等于(x+2-3)/2的上取整的值
          self.bn2 = nn.BatchNorm2d(in_channel)#如果是第一代resnet这里修改为out_channel
          self.conv3 = conv1x1(out_channel, out_channel)
          self.bn3 = nn.BatchNorm2d(out_channel)
          if not self.same_shape:
                self.conv4 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
          if not self.same_channel:
            self.conv4=nn.Conv2d(in_channel,out_channel,1,1)
          
     def forward(self, x):
          out = F.relu(self.bn1(x), True)
          out = self.conv1(out)
          #print("1")
          out = F.relu(self.bn2(out),True)
          #print("2")
          out = self.conv2(out)
          #print("3")
          out= F.relu(self.bn3(out),True)
          #print("4")
          out = self.conv3(out)
          #print("5")
          
          if not self.same_shape:#通道和大小都变
               x = self.conv4(x)#为了让x和out具有相同大小和通道数
               #print("1")               
          if not self.same_channel:#只改变通道
               x=self.conv4(x)
               #print("2")
          return x+out
          
 
#下面我们尝试实现一个 ResNet，它就是 residual block 模块的堆叠
class CRNN(nn.Module):
     def __init__(self, in_channel, num_classes, verbose = False):
          super(CRNN, self).__init__()
          self.verbose = verbose
          #只用了1x1的卷积增加通道数
          self.block1 = nn.Sequential(nn.Conv2d(in_channel, 32,1,1,1),nn.BatchNorm2d(32),nn.ReLU(True))
          
          self.block2 = nn.Sequential(
               
               residual_block(32, 32),
               residual_block(32, 32, False,True),#只改变尺寸
               residual_block(32, 32),
               residual_block(32,64,True,False),#只改变通道数
               residual_block(64,64),
               residual_block(64, 64, False,True),#只改变尺寸
               residual_block(64,64)
          )
          self.block3 = nn.Sequential(
               #nn.MaxPool2d(2, 2)
               residual_block(64, 128,True,False),#只改变通道
               
               residual_block(128, 128),
               residual_block(128, 128, False,True),#只改变尺寸
               residual_block(128, 128)
			   
          )
          self.block4 = nn.Sequential(
               nn.MaxPool2d((3,3),(2,1),(0,1)),
               residual_block(128, 256, True,False),#都改变
               residual_block(256, 256),
          )
          self.block5 =  nn.Sequential(nn.Conv2d(256, 512, 3, 1,1),nn.BatchNorm2d(512),nn.ReLU(True))
          self.rnn = nn.Sequential(
            BidirectionalLSTM(2048, 256, 256),
            BidirectionalLSTM(256, 256, num_classes))
               
          self.conv2=conv2x2(128,128)
               
          
          
          
     def forward(self, x):
          x = self.block1(x)
          if self.verbose:
               print('block 1 output: {}'.format(x.shape))#32x34x258
          x = self.block2(x)
          if self.verbose:
               print('block 2 output: {}'.format(x.shape))#64x9x65
          #print(3)
          x1 = self.block3(x)
          #print(4)
          if self.verbose:
               print('block 3 output: {}'.format(x1.shape))#128x5x33 结一个2x2的卷积成为128x4x33,然后reshape为256x2x33
          x2 = self.block4(x1)
          if self.verbose:
               print('block 4 output: {}'.format(x2.shape))#256x2x33
          x3 = self.block5(x2)
          if self.verbose:
               print('block 5 output: {}'.format(x3.shape))#512x2x33
          x1= self.conv2(x1)
          #print(x1.shape)
          k=x1.size()[0]
          k1=x1.size()[3]
          x1=x1.view(k,256,2,k1)
          #print(x1.shape) 
          x3= torch.cat((x1,x2,x3),1) 
          #print(x3.shape)#1x1024x2x33
          b, c, h, w = x3.size()
          assert h==2
          #x=x.squeeze(2)
          x3=x3.view(b,c*h,w)
          
          x3=x3.permute(2,0,1)
          x3 = self.rnn(x3)
          if self.verbose:
               print('LSTM output: {}'.format(x3.shape))
          #torch.cat((x1,x2,x3),0)在第一个维度上拼接x1.x2.x3   
          return x3
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output =self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
test_net = CRNN(1, 10, True)
test_x = Variable(torch.zeros(1, 1, 32, 256))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))    