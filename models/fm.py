from base_modules import DNN
import torch
import torch.nn as nn

class FM(nn.Module):
    def __init__(self, dim, k, task_type='regression'):
        """
        param dim: feature dimension
        param k: dim of matrix V
        """
        super(FM, self).__init__()
        self.dim = dim
        self.k = k
        self.w = nn.Linear(self.dim, 1, bias=True)
        # 初始化V矩阵
        self.v = nn.Parameter(torch.rand(self.dim, self.k) / 100)
        # xavier init
        torch.nn.init.xavier_uniform(self.v)

        self.task = task_type
        
    def forward(self, x):
        """
        x: [batch_size, dim]
        """
        linear = self.w(x)
        # 二次项
        quadradic = 0.5 * torch.sum(torch.pow(torch.mm(x, self.v), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.v, 2)))
        #[batch_size, ]
        # 套一层sigmoid转成分类模型，也可以不加，就是回归模型
        if self.task == 'regression':
            return linear + quadradic
        else:
            return torch.sigmoid(linear + quadradic)