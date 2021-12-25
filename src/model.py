import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

class Model(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Model, self).__init__()
    self.num_layers = 1
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
    self.dense = nn.Linear(hidden_size,output_size)
  def forward(self, input):
    # input [1,80,2]
    h_0 = Variable(torch.zeros(
        self.num_layers, 1, self.hidden_size)) 
    c_0 = Variable(torch.zeros(
        self.num_layers, 1, self.hidden_size))
    
    # x [1,80,20]
    x, (h_0, c_0) = self.lstm(input, (h_0, c_0))
    
    output = self.dense(x)
    
    return output
