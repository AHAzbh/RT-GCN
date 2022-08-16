import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.nn.init as init


class layer(nn.Module):
    """
    GCN layer
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(layer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        # print("inputs device {}, weight device {}".format(inputs.device, self.weight.device))
        output = torch.matmul(inputs.float(), self.weight.float())
        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        return output
