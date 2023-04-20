import torch.nn as nn

class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.linear(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

