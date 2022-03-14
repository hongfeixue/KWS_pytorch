import torch
import torch.nn as nn
import torch.nn.functional as F
from dfsmn_cell import DFSMN_CELL

class CNN(nn.Module):
    # CNN
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (100,161,101)
            nn.Conv1d(in_channels=161, #input height 
                    out_channels=128, #n_filter
                    kernel_size=5, #filter size
                    stride=1, #filter step
                    padding=2 #con2d出来的图片大小不变
                    ), #output shape (64,101)
            nn.BatchNorm1d(128,0.15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) #1x2采样，output shape (32,50)
         )
        self.conv2 = nn.Sequential(nn.Conv1d(128, 64, 5, 1, 2), #output shape (32,50)
                                nn.BatchNorm1d(64,0.15),
                                nn.ReLU(),
                                nn.MaxPool1d(2)) # output shape (32, 25)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 16, 3, 1, 1), #output shape (32,50)
                                nn.BatchNorm1d(16,0.15),
                                nn.ReLU(),
                                nn.MaxPool1d(2)) # output shape (16, 12)
        self.out = nn.Linear(16*12,6)

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.conv1(x)
        x = self.conv2(x)   
        x = self.conv3(x)   # 100 16 25
        x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        output = self.out(x)
        return output

class DNN(nn.Module):
    # DNN
    def __init__(self):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(101, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 64)
        self.l4 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = torch.mean(x, dim=2)
        x = torch.squeeze(x)
        x = self.l4(x)
        return x

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.l1 = nn.Linear(101, 256)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,  # RNN隐藏神经元个数 
            num_layers=3,  # RNN隐藏层个数
            batch_first = False
        )
        self.bn = nn.BatchNorm1d(161,0.15)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 6)
    def forward(self, x, hidden):
        x = torch.squeeze(x)
        x = self.l1(x)
        x = x.transpose(0,1)
        out, (h, c) = self.rnn(x, hidden)
        out = out.transpose(0,1)
        out = self.relu(self.bn(self.out(out)))
        out = torch.mean(out, dim=1)
        return out, (h, c)

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=101, 
                    output_dim=6,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.3
                ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )
        x = x.unsqueeze(1)
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )
        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)
        x = torch.mean(x, dim=1)

        return x
class DFSMN(nn.Module):
    def __init__(self):
        super(DFSMN, self).__init__()
        in_dim=162
        out_dim=6
        rnn_units=256
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(in_dim, affine=False)
        self.bn2 = nn.BatchNorm1d(out_dim, affine=False)

        self.fsmn1 = DFSMN_CELL(in_dim, rnn_units, in_dim, 3,3,1,1)
        self.fsmn2 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn3 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn4 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn5 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn6 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn7 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn8 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn9 = DFSMN_CELL(in_dim, rnn_units, in_dim, 7,3,1,1)
        self.fsmn10 = DFSMN_CELL(in_dim, rnn_units, in_dim, 3,3,1,1)
    
    def forward(self, x, hidden):
        x = x.squeeze(1)
        y = torch.zeros(x.shape[0],1,x.shape[2]).cuda()
        x = torch.cat([x,y], 1)
        r_specs, i_specs = torch.chunk(x, 2, dim=1)

        x = self.bn1(x)
        out, outh = self.fsmn1(x, hidden)
        out, outh = self.fsmn2(out, outh)
        out, outh = self.fsmn3(out, outh)
        out, outh = self.fsmn4(out, outh)
        out, outh = self.fsmn5(out, outh)
        out, outh = self.fsmn6(out, outh)
        out, outh = self.fsmn7(out, outh)
        out, outh = self.fsmn8(out, outh)
        out, outh = self.fsmn9(out, outh)
        out, outh = self.fsmn10(out, outh)
        
        r_mask, i_mask = torch.chunk(out, 2, dim=1)
        r_out_spec = r_mask*r_specs - i_mask*i_specs
        i_out_spec = r_mask*i_specs + i_mask*r_specs
        out_spec = torch.cat([r_out_spec,i_out_spec], 1)
        out_spec = out_spec.transpose(1,2)
        out_spec = self.linear(out_spec)
        out_spec = out_spec.transpose(1,2)
        out_spec = F.relu(self.bn2(out_spec))
        out_spec = torch.mean(out_spec, dim=2)
        return out_spec, outh
