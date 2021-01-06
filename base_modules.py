import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units, \
                 activation='relu', l2_reg=None, dropout=None, BN=None):
        super(DNN, self).__init__() #load super class for training data
        if len(hidden_units) < 1: 
            pass
        elif len(hidden_units) == 1:
            self.all_modules = nn.ModuleList( [nn.Linear(input_dim, hidden_units[0])] )
        else:
            self.all_modules = nn.ModuleList([ nn.Linear(hidden_units[i], hidden_units[i+1]) \
                                             for i in range(len(hidden_units)-1)])
            self.all_modules.insert(0, nn.Linear(input_dim, hidden_units[0]))
        # init nn weights
        for layer in self.all_modules:
            torch.nn.init.xavier_uniform(layer.weight)
        
        # activation function
        if activation== 'relu':
            self.activation = nn.ReLU() 
        if activation== 'tanh':
            self.activation = nn.Tanh()
        # dropout
        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

    def forward(self, x): #feed forward
        for layer in self.all_modules:
            x = self.activation(layer(x))
        if self.dropout_layer:
            x =self.dropout_layer(x)
        return x