import torch
import torch.nn as nn


class FeatureMask(nn.Module):
    def __init__(self, ratio,feature_dim,threshold=0.3, use_rnn=False):
        super(FeatureMask, self).__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        if use_rnn:
            self.rnn = nn.GRUCell(feature_dim, feature_dim)
        else:
            self.rnn = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim
        self.use_rnn = use_rnn
        self.threshold = threshold
        self.mask_ratio = ratio
        self.maskout_1 = None
        self.maskout_0 = None

    # create a init hidden tensor with shape (1,hidden_dim)
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, feature, hidden_state=None):
        x = torch.relu(self.fc1(feature))
        if self.use_rnn:
            h_in = hidden_state.reshape(-1, self.feature_dim)
            h = self.rnn(x, h_in)
        else:
            h = self.rnn(x)

        x = torch.sigmoid(self.fc2(h))
        # x = torch.sigmoid(self.fc2(h)) - self.threshold
        # x = torch.sign(x)
        mask = torch.relu(x)
        # print(mask.size())
        masknum = int(mask.shape[1] * self.mask_ratio)
        _,mask_indices = torch.topk(x, k=masknum,dim=1, largest=False)
        
        self.maskout_1 = mask.clone()  # 
        self.maskout_0 = mask.clone()  # 
        self.maskout_0[torch.arange(mask.size(0)).unsqueeze(1), mask_indices] = 0

        return self.maskout_0

    def get_maskout_1(self):
        return self.maskout_1

    def get_maskout_0(self):
        return self.maskout_0
    

