import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

import utils.global_var
from utils.gat import GAT


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=4, stride=1):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
                # patching and embedding
        self.patch_embedding_1 = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.patch_embedding_2 = PatchEmbedding(
            configs.d_model, patch_len*2, stride*2, padding*2, configs.dropout)
        self.patch_embedding_3 = PatchEmbedding(
            configs.d_model, patch_len*4, stride*4, padding*4, configs.dropout)
        self.patch_embedding_4 = PatchEmbedding(
            configs.d_model, patch_len*8, stride*8, padding*8, configs.dropout)
        
        self.gat = GAT(configs.d_model, configs.d_ff, dropout=configs.dropout, alpha=1,  n_heads=8)

        # Encoder
        self.encoder_1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

         # Encoder
        self.encoder_2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Encoder
        self.encoder_3 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Encoder
        self.encoder_4 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf_1 = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf_2 = configs.d_model * \
                int((configs.seq_len - patch_len*2) / (stride*2) + 2)
        self.head_nf_3 = configs.d_model * \
                int((configs.seq_len - patch_len*4) / (stride*4) + 2)
        self.head_nf_4 = configs.d_model * \
                int((configs.seq_len - patch_len*8) / (stride*8) + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)
        elif self.task_name == 'fault_prediction':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection_1 = nn.Linear(
                (self.head_nf_1 + self.head_nf_2 + self.head_nf_3) * configs.enc_in, configs.d_model)
            self.projection_2 = nn.Linear(
                configs.d_model, int(configs.d_model/4))
            self.projection_3 = nn.Linear(
                int(configs.d_model/4), len(utils.global_var.get_value('class_names')))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def fault_prediction(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)

        ##TODO 多尺度
        # u: [bs * nvars x patch_num x d_model]
        enc_out_1, n_vars_1 = self.patch_embedding_1(x_enc)
        enc_out_2, n_vars_2 = self.patch_embedding_2(x_enc)
        enc_out_3, n_vars_3 = self.patch_embedding_3(x_enc)

        ## 构建邻接矩阵
        # input: [bs x nvars * patch_num x d_model]
        # output: [bs x nvars * patch_num x d_model]
        gat_inp_1 = torch.reshape(
            enc_out_1, (-1, n_vars_1 * enc_out_1.shape[-2], enc_out_1.shape[-1]))
        gat_inp_2 = torch.reshape(
            enc_out_2, (-1, n_vars_2 * enc_out_2.shape[-2], enc_out_2.shape[-1]))
        gat_inp_3 = torch.reshape(
            enc_out_3, (-1, n_vars_3 * enc_out_3.shape[-2], enc_out_3.shape[-1]))


        ## 构建邻接矩阵
        # 邻接矩阵: [bs x nvars * patch_num x nvars * patch_num]
        adj_matrix_1 = np.eye(n_vars_1 * enc_out_1.shape[-2], k = 1)
        for i in range(0, n_vars_1):
            adj_matrix_1[i*enc_out_1.shape[-2], i*enc_out_1.shape[-2]+1] = 0
            adj_matrix_1[i*enc_out_1.shape[-2]:(i+1)*enc_out_1.shape[-2], i*enc_out_1.shape[-2]:(i+1)*enc_out_1.shape[-2]] = np.eye(enc_out_1.shape[-2])

        adj_matrix_2 = np.eye(n_vars_2 * enc_out_2.shape[-2], k = 1)
        for i in range(0, n_vars_2):
            adj_matrix_2[i*enc_out_2.shape[-2], i*enc_out_2.shape[-2]+1] = 0
            adj_matrix_2[i*enc_out_2.shape[-2]:(i+1)*enc_out_2.shape[-2], i*enc_out_2.shape[-2]:(i+1)*enc_out_2.shape[-2]] = np.eye(enc_out_2.shape[-2])

        adj_matrix_3 = np.eye(n_vars_3 * enc_out_3.shape[-2], k = 1)
        for i in range(0, n_vars_3):
            adj_matrix_3[i*enc_out_3.shape[-2], i*enc_out_3.shape[-2]+1] = 0
            adj_matrix_3[i*enc_out_3.shape[-2]:(i+1)*enc_out_3.shape[-2], i*enc_out_3.shape[-2]:(i+1)*enc_out_3.shape[-2]] = np.eye(enc_out_3.shape[-2])

        # adj  = sp.coo_matrix(adj_matrix) #转换成coo_matrix矩阵
        # indices = np.vstack((adj.row,adj.col)) # 我们需要的coo形式的edge_index
        # edge_index = torch.LongTensor(indices)
        
        gat_out_1 = self.gat(gat_inp_1, torch.tensor(adj_matrix_1).cuda()).relu()
        gat_out_2 = self.gat(gat_inp_2, torch.tensor(adj_matrix_2).cuda()).relu()
        gat_out_3 = self.gat(gat_inp_3, torch.tensor(adj_matrix_3).cuda()).relu()

        gat_out_1 = torch.reshape(
            gat_out_1, (enc_out_1.shape[0], enc_out_1.shape[1], enc_out_1.shape[2]))
        gat_out_2 = torch.reshape(
            gat_out_2, (enc_out_2.shape[0], enc_out_2.shape[1], enc_out_2.shape[2]))
        gat_out_3 = torch.reshape(
            gat_out_3, (enc_out_3.shape[0], enc_out_3.shape[1], enc_out_3.shape[2]))

        enc_out_1 = enc_out_1 + gat_out_1
        enc_out_2 = enc_out_2 + gat_out_2
        enc_out_3 = enc_out_3 + gat_out_3

        ##TODO 共享一个encoder，还是分别设置
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out_1, attns_1 = self.encoder_1(enc_out_1)
        enc_out_2, attns_2 = self.encoder_2(enc_out_2)
        enc_out_3, attns_3 = self.encoder_3(enc_out_3)
        # enc_out_4, attns_4 = self.encoder_4(enc_out_4)
        
        # z: [bs x nvars x patch_num x d_model]
        enc_out_1 = torch.reshape(
            enc_out_1, (-1, n_vars_1, enc_out_1.shape[-2], enc_out_1.shape[-1]))
        enc_out_2 = torch.reshape(
            enc_out_2, (-1, n_vars_2, enc_out_2.shape[-2], enc_out_2.shape[-1]))
        enc_out_3 = torch.reshape(
            enc_out_3, (-1, n_vars_3, enc_out_3.shape[-2], enc_out_3.shape[-1]))
        # enc_out_4 = torch.reshape(
        #     enc_out_4, (-1, n_vars_4, enc_out_4.shape[-2], enc_out_4.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out_1 = enc_out_1.permute(0, 1, 3, 2)
        enc_out_2 = enc_out_2.permute(0, 1, 3, 2)
        enc_out_3 = enc_out_3.permute(0, 1, 3, 2)
        # enc_out_4 = enc_out_4.permute(0, 1, 3, 2)

        enc_out_l1 = torch.cat((enc_out_1, enc_out_2, enc_out_3), 3)

        # Decoder
        output = self.flatten(enc_out_l1)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection_1(output).relu()  # (batch_size, num_classes)
        output = self.dropout(output)
        output = self.projection_2(output).relu()  # (batch_size, num_classes)
        output = self.dropout(output)
        output = self.projection_3(output).relu()  # (batch_size, num_classes)
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, attributes, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'fault_prediction':
            dec_out = self.fault_prediction(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
