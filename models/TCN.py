import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

import utils.global_var

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.name = 'TCN'

        self.task_name = configs.task_name
        num_inputs = configs.seq_len
        self.pred_len = configs.pred_len
        num_channels = configs.d_model
        kernel_size = 10
        dropout = configs.dropout

        layers = []
        num_levels = 4
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        if self.task_name == 'fault_prediction':
            self.fc1 = nn.Linear(configs.d_model * configs.enc_in, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, len(utils.global_var.get_value('class_names')))

            self.act = self._get_activation_fn(configs.activation)

        elif self.task_name == 'rul':
            self.fc1 = nn.Linear(configs.d_model * configs.enc_in, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, configs.seq_len)

            self.act = self._get_activation_fn(configs.activation)

        elif self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.fc1 = nn.Linear(configs.d_model, configs.pred_len)

    def forecast(self, x):

        output = self.network(x)

        output = output.permute(0,2,1)
        output = self.fc1(output)
        output = output.permute(0,2,1)
        return output

    def fault_prediction(self, x):

        output = self.network(x)

        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        output = self.act(self.fc1(output))
        output = self.act(self.fc2(output))
        output = self.fc3(output)

        return output
    
    def rul(self, x):

        output = self.network(x)

        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        output = self.act(self.fc1(output))
        output = self.act(self.fc2(output))
        output = self.fc3(output)

        return output

    def _get_activation_fn(self, activation):

        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise ValueError("activation should be relu/gelu, not {}".format(activation))
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'fault_prediction':
            dec_out = self.fault_prediction(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'rul':
            dec_out = self.rul(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None