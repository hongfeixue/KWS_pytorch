import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(1640, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 64)
        self.l4 = nn.Linear(64, 8)
        self.l5 = nn.Linear(8, 2)

    def forward(self, x):
        # print(x.shape) 50 98 1640
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = torch.mean(x, dim=1)
        x = self.l5(x)
        x = F.log_softmax(x, dim=1)
        # print(x.shape) 50 2
        return x

class CNN(nn.Module):
    # CNN
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,161,101)
              nn.Conv2d(in_channels=1, #input height 
                        out_channels=16, #n_filter
                       kernel_size=5, #filter size
                       stride=1, #filter step
                       padding=2 #con2d出来的图片大小不变
                      ), #output shape (16,161,101)
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2) #2x2采样，output shape (16,80,50)
               
         )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), #output shape (32,80,50)
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.out = nn.Linear(32*40*25,6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # 32 * 40 *25
        x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        output = self.out(x)
        output = F.log_softmax(output,dim=1)
        return output


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.l1 = nn.Linear(101, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 64)
        self.rnn = nn.RNN(
            input_size=64,
            hidden_size=32,  # RNN隐藏神经元个数
            num_layers=1,  # RNN隐藏层个数
            batch_first = True
        )
        self.out = nn.Linear(32, 6)
    def forward(self, x, h):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        x = torch.squeeze(x)
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = F.sigmoid(self.l3(x))
        out, h = self.rnn(x, h)
        out = torch.mean(out, dim=1)
        out = torch.squeeze(out)
        prediction = self.out(out)
        prediction = F.log_softmax(prediction,dim=1)
        return prediction, h

class DFSMN(nn.Module):

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
        ''' 
            input_dim as it's name ....
            hidden_dim means the dimension or channles num of the memory 
            left means history
            right means future
        '''
        super(DFSMN, self).__init__()  
        self.left_frames = left_frames
        self.right_frames = right_frames 
        
        self.in_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        #self.norm = nn.InstanceNorm1d(hidden_dim)
        #nn.init.normal_(self.in_conv.weight.data,std=0.05)
        if left_frames > 0:
            self.left_conv = nn.Sequential(
                        #nn.ConstantPad1d([left_dilation*left_frames,-left_dilation],0),
                        nn.ConstantPad1d([left_dilation*left_frames,0],0),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=left_frames+1,dilation=left_dilation, bias=False, groups=hidden_dim)
                )
        #    nn.init.normal_(self.left_conv[1].weight.data,std=0.05)
        if right_frames > 0:
            self.right_conv = nn.Sequential(
                        nn.ConstantPad1d([-right_dilation,right_frames*right_dilation],0),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=right_frames,dilation=right_dilation, bias=False,groups=hidden_dim)
                )
        #    nn.init.normal_(self.right_conv[1].weight.data,std=0.05)

        self.out_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        #nn.init.normal_(self.out_conv.weight.data,std=0.05)
        
        self.weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, inputs, hidden=None):
        out = self.in_conv(inputs) 
        #out = F.relu(out)
        #out = self.norm(out)
        if self.left_frames > 0:
            left = self.left_conv(out)
        else:
            left = 0
        if self.right_frames > 0:
            right = self.right_conv(out)
        else:
            right = 0.
        out_p = out+left+right 
        if hidden is not None:
            out_p = hidden + F.relu(out_p)*self.weight
            #out_p = hidden + out_p
        out = self.out_conv(out_p)
        return out, out_p

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=1640, 
                    output_dim=2,
                    context_size=3,
                    stride=1,
                    dilation=2,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel1 = nn.Linear(input_dim*context_size, 164)
        self.kernel2 = nn.Linear(164, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        # print(x.size()) 50 98 1640
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1) #50 1 98 1640  (W − F + 2P )/S+1 
        # print(x.shape)
        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(2,self.input_dim), 
                        dilation=(self.dilation,1)
                    )
        x = x.unsqueeze(1) # 50 1620 * 3 94
        x = x.transpose(2,3)
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(2,self.input_dim), 
                        dilation=(self.dilation,1)
                    )
        # x = x.unsqueeze(1)
        # x = F.unfold(
        #                 x, 
        #                 (self.context_size - 5, self.input_dim), 
        #                 stride=(1,self.input_dim), 
        #                 dilation=(self.dilation,1)
        #             )
        # x = x.unsqueeze(1)
        # x = F.unfold(
        #                 x, 
        #                 (self.context_size - 8, self.input_dim), 
        #                 stride=(1,self.input_dim), 
        #                 dilation=(self.dilation,1)
        #             )

        # N, output_dim*context_size, new_t = x.shape
        # print(x.shape) # 50 1640*3 94=1*(98-3)
        x = x.transpose(1,2)
        x = self.kernel1(x)
        x = self.nonlinearity(x)

        x = self.kernel2(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = F.log_softmax(x,dim=1)

        return x