import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.global_var

class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.name = 'GRU'

        self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.task_name = configs.task_name

        self.n_feats = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers


        self.lstm = nn.GRU(input_size = self.n_feats, hidden_size= self.d_model, num_layers= self.n_layers, batch_first=True)

        if self.task_name == 'fault_prediction':
            self.fc1 = nn.Linear(configs.d_model * configs.seq_len, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, len(utils.global_var.get_value('class_names')))

            self.act = self._get_activation_fn(configs.activation)

    def fault_prediction(self, x_enc):

        batch = x_enc.shape[0]
        h0 = torch.randn(self.n_layers, batch, self.d_model).to(self.device)

        output, hn = self.lstm(x_enc, h0)


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
        return None
