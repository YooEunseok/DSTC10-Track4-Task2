import torch
from torch import nn

class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE #4096 #2048
        hidden_size = cfg.HIDDEN_SIZE #512
        kernel_size = cfg.KERNEL_SIZE #16
        stride = cfg.STRIDE #16

        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1) #여기를 바꿔주는게 말이 되나
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, visual_input): #torch.Size([batch_size, 4096, 256]) #Batch, Feature dimension, Time_step
        vis_h = torch.relu(self.vis_conv(visual_input)) #torch.Size([batch_size, 512, 256]) 
        vis_h = self.avg_pool(vis_h) #torch.Size([batch_size, 512, 16]) #동영상 길이도 총 16개로 나누니까
        return vis_h

class FrameMaxPool(nn.Module):

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(vis_h)
        return vis_h
