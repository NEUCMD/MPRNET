import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

import utils.global_var


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3



class Model(nn.Module):
    def __init__(self, configs, individual=False):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.adjacency_matrix = torch.ones((configs.enc_in, configs.enc_in))

        # Embedding
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.block1 = STGCNBlock(in_channels=configs.d_model, out_channels=64,
                                 spatial_channels=16, num_nodes=configs.enc_in)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=configs.enc_in)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.fully = nn.Linear((configs.seq_len - 2 * 5) * 64,
                               configs.pred_len)
        elif self.task_name == 'fault_prediction':
            self.fully = nn.Linear((configs.seq_len - 2 * 5) * 64 * configs.enc_in,
                               len(utils.global_var.get_value('class_names')))
        elif self.task_name == 'rul':
            self.fully = nn.Linear((configs.seq_len - 2 * 5) * 64 * configs.enc_in, configs.seq_len)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        A_hat = self.normalize_adjacency(self.adjacency_matrix).to(self.device)
        X = torch.unsqueeze(x_enc, dim=3)
        x_enc = X.reshape((X.shape[0]*X.shape[2], X.shape[1], -1))
        x_mark_enc = x_mark_enc.repeat(self.enc_in, 1, 1)
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        x_enc = x_enc.reshape((X.shape[0], X.shape[2], X.shape[1], -1))
        out1 = self.block1(x_enc, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out3 = out3.reshape((out3.shape[0], -1, out3.shape[1]))
        out3 = out3.permute(0, 2, 1)
        out4 = self.fully(out3)
        out4 = out4.permute(0, 2, 1)
        return out4

    def fault_prediction(self, X, x_mark_enc):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        A_hat = self.normalize_adjacency(self.adjacency_matrix).to(self.device)
        X = torch.unsqueeze(X, dim=3)
        x_enc = X.reshape((X.shape[0]*X.shape[2], X.shape[1], -1))
        x_mark_enc = x_mark_enc.repeat(self.enc_in, 1, 1)
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        x_enc = x_enc.reshape((X.shape[0], X.shape[2], X.shape[1], -1))
        out1 = self.block1(x_enc, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], -1)))
        return out4
    
    def rul(self, X, x_mark_enc):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        A_hat = self.normalize_adjacency(self.adjacency_matrix).to(self.device)
        X = torch.unsqueeze(X, dim=3)
        x_enc = X.reshape((X.shape[0]*X.shape[2], X.shape[1], -1))
        x_enc = self.enc_embedding(x_enc, None)
        x_enc = x_enc.reshape((X.shape[0], X.shape[2], X.shape[1], -1))
        out1 = self.block1(x_enc, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], -1)))
        return out4
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'fault_prediction':
            dec_out = self.fault_prediction(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'rul':
            dec_out = self.rul(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
    def normalize_adjacency(self, adjacency_matrix):
        # 计算每个节点的度数
        degree = torch.sum(adjacency_matrix, dim=1)
        
        # 计算度数的倒数
        degree_inv_sqrt = torch.pow(degree, -0.5)
        
        # 将度数的倒数矩阵转换为对角矩阵
        degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt)
        
        # 计算标准化邻接矩阵
        normalized_adjacency = torch.matmul(torch.matmul(degree_inv_sqrt_matrix, adjacency_matrix), degree_inv_sqrt_matrix)
    
        return normalized_adjacency