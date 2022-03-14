import torch 
import torch.nn as nn
import torch.nn.functional as F

class DFSMN_CELL(nn.Module):

    def __init__(
                    self,
                    input_dim,
                    hidden_dim,
                    output_dim,
                    left_frames=1,
                    left_dilation=1, 
                    right_frames=1,
                    right_dilation=1, 
        ): 
        super(DFSMN_CELL, self).__init__()  
        self.left_frames = left_frames
        self.right_frames = right_frames 
        
        self.in_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.left_conv = nn.Sequential(
                    nn.ConstantPad1d([left_dilation*left_frames,0],0),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=left_frames+1,dilation=left_dilation, bias=False, groups=hidden_dim)
            )
        self.right_conv = nn.Sequential(
                    nn.ConstantPad1d([-right_dilation,right_frames*right_dilation],0),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=right_frames,dilation=right_dilation, bias=False,groups=hidden_dim)
            )

        self.out_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
        self.weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, inputs, hidden=None):
        out = self.in_conv(inputs) 
        if self.left_frames > 0:
            left = self.left_conv(out)
        else:
            left = 0
        if self.right_frames > 0:
            right = self.right_conv(out)
        else:
            right = 0.
        out_p = (out+left+right) / 3 
        if hidden is not None:
            out_p = (hidden + F.relu(out_p)*self.weight)/2
            #out_p = hidden + out_p
        out = self.out_conv(out_p)
        return out, out_p