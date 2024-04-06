import torch
from torch import nn
import numpy as np
from typing import List, Optional, Tuple, Union


from layers.Embed import DataEmbedding_wo_pos
from utils.xpos_relative_position import XPOS
from utils.gat import GAT
import utils.global_var

# helper functions
def split_chunks(*tensors, size, dim=0):
    return [torch.split(x, size, dim=dim) for x in tensors]


def split_heads(tensors, bsz, seqlen, n_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, n_heads, -1).transpose(1, 2) for x in tensors]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiScaleRetention(nn.Module):
    # TODO: normalization to decay in the paper
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gat = GAT(int(config.qk_dim*config.v_dim/(config.n_heads*config.n_heads)), config.d_ff, dropout=config.dropout, alpha=1, n_heads=16)
        self.qkv = nn.Linear(config.d_model,
                             config.qk_dim * 2 + config.v_dim,
                             bias=False)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(config.d_model, config.v_dim, bias=False)
        self.proj = nn.Linear(config.v_dim, config.d_model, bias=False)
        self.gn = nn.GroupNorm(num_groups=config.n_heads, num_channels=config.v_dim, affine=False)
        self.xpos = XPOS(config.qk_dim)


        # initialize gamma
        # if config.use_default_gamma:
        #     gamma = 1 - 2**(-5 - torch.arange(0, config.n_heads, dtype=torch.float))
        # else:
        s = torch.log(torch.tensor(1 / 32))
        e = torch.log(torch.tensor(1 / 512))
        # gamma = 1 - torch.exp(torch.linspace(s, e, int(self.config.d_ff/self.config.d_model)))  # [h,]
        gamma = 1 - torch.exp(torch.linspace(s, e, config.n_heads))  # [h,]
        self.decay = nn.Parameter(gamma, requires_grad=False)

    def get_parallel_decay_mask(self, length, retention_mask=None):
        range_tensor = torch.arange(length, device=self.decay.device)
        # range_tensor = range_tensor[None, :, None].expand(int(self.config.d_ff/self.config.d_model), length, 1)
        range_tensor = range_tensor[None, :, None].expand(self.config.n_heads, length, 1)
        exponent = range_tensor - range_tensor.transpose(-1, -2)
        decay_mask = self.decay.view(-1, 1, 1)**exponent
        decay_mask = torch.tril(decay_mask, diagonal=0)  # [h, t, t]
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, length)
            decay_mask = decay_mask.unsqueeze(0) * retention_mask
        else:
            decay_mask = decay_mask.unsqueeze(0)
        return decay_mask

    def get_recurrent_decay(self):
        decay = self.decay.view(1, self.config.n_heads, 1, 1)
        return decay

    def get_chunkwise_decay(self, chunk_size, retention_mask=None):
        # within chunk decay
        decay_mask = self.get_parallel_decay_mask(chunk_size, retention_mask=retention_mask)
        # decay of the chunk
        chunk_decay = self.decay.view(1, self.config.n_heads, 1, 1)**chunk_size
        # chunk_decay = self.decay.view(1, int(self.config.d_ff/self.config.d_model), 1, 1)**chunk_size
        # cross-chunk decay
        exponent = torch.arange(chunk_size, dtype=torch.float,
                                device=decay_mask.device).unsqueeze(0) + 1
        # inner_decay = (self.decay.unsqueeze(-1)**exponent).view(1, int(self.config.d_ff/self.config.d_model), chunk_size, 1)
        inner_decay = (self.decay.unsqueeze(-1)**exponent).view(1, self.config.n_heads, chunk_size, 1)
        return decay_mask, chunk_decay, inner_decay

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # (1 or bsz) * num_head * len * len
        """
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5  # (scaled dot-product)
        retention = retention * decay_mask
        output = retention @ v

        # kv cache
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # [bsz, num_head, qk_dim, v_dim]
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        return output, current_kv, retention

    def recurrent_retention(self, q, k, v, past_key_value=None, decay=None, retention_mask=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        past_key_value, # bsz * num_head * qk_dim * v_dim
        decay # num_head * 1 * 1
        retention_mask # bsz * 1
        """
        past_key_value = past_key_value if past_key_value is not None else 0
        decay = decay if decay is not None else 0
        retention_mask = retention_mask.view(-1, 1, 1, 1) if retention_mask is not None else 1
        # (b, h, d_k, d_v)
        current_kv = decay * past_key_value + retention_mask * (k.transpose(-1, -2) @ v)
        output = q @ current_kv * k.size(-1)**-0.5  # (b, h, 1, d_v)
        return output, current_kv

    def chunkwise_retention(self,
                            q,
                            k,
                            v,
                            decay_mask,
                            past_key_value=None,
                            chunk_decay=None,
                            inner_decay=None):
        """
        q, k, v,  # bsz * num_head * chunk_size * qkv_dim
        past_key_value,  # bsz * num_head * qk_dim * v_dim
        decay_mask,  # 1 * num_head * chunk_size * chunk_size
        chunk_decay,  # 1 * num_head * 1 * 1
        inner_decay,  # 1 * num_head * chunk_size * 1
        """
        # [bsz, num_head, chunk_size, chunk_size]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5
        retention = retention * decay_mask
        inner_retention = retention @ v  # [bsz, num_head, chunk_size, v_dim]

        if past_key_value is None:
            cross_retention = 0
            past_chunk = 0
        else:
            cross_retention = (q @ past_key_value) * inner_decay * k.size(-1)**-0.5
            past_chunk = chunk_decay * past_key_value

        # [bsz, num_head, chunk_size, v_dim]
        retention = inner_retention + cross_retention
        # [bsz, num_head, chunk_size, qk_dim, v_dim]
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # NOTE: intra_decay is omitted in the paper; but this detail is important
        # [bsz, num_head, qk_dim, v_dim]
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
           
        current_gat = torch.reshape(current_kv,(self.batchsize, self.config.enc_in, self.config.n_heads, -1))
        # current_gat = torch.reshape(current_gat,(self.batchsize, self.config.enc_in * self.config.n_heads, -1))
        current_gat = torch.reshape(current_gat,(self.batchsize * self.config.n_heads, self.config.enc_in , -1))

        ## 构建邻接矩阵
        # input: [bs x nvars * n_heads x qk_dim * v_dim]
        # output: [bs x nvars * n_heads x qk_dim * v_dim]

        # # 邻接矩阵: [bs x nvars * n_heads x nvars * n_heads]
        # adj_matrix = np.zeros((self.config.enc_in * self.config.n_heads,self.config.enc_in * self.config.n_heads))
        # for i in range(0,  self.config.n_heads):
        #     adj_matrix[i*self.config.enc_in:(i+1)*self.config.enc_in, i*self.config.enc_in:(i+1)*self.config.enc_in] = np.ones((self.config.enc_in,self.config.enc_in))

        
        # 邻接矩阵: [bs  * n_heads x nvars x nvars]
        adj_matrix = np.ones((self.config.enc_in,self.config.enc_in))
        # adj  = sp.coo_matrix(adj_matrix) #转换成coo_matrix矩阵
        # indices = np.vstack((adj.row,adj.col)) # 我们需要的coo形式的edge_index
        # edge_index = torch.LongTensor(indices)

        gat_out = self.gat(current_gat, torch.tensor(adj_matrix).cuda()).relu()

        gat_out = torch.reshape(gat_out,(self.batchsize, self.config.enc_in, self.config.n_heads, -1))
        gat_out = torch.reshape(
            gat_out, (current_kv.shape[0], current_kv.shape[1], current_kv.shape[2],current_kv.shape[3])) 

        current_kv = past_chunk + current_kv + gat_out
        return retention, current_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'chunkwise',
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        batchsize: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        self.batchsize = batchsize
        B, T, H = hidden_states.size()
        q, k, v = self.qkv(hidden_states).split(
            [self.config.qk_dim, self.config.qk_dim, self.config.v_dim], dim=-1)
        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset)
        q, k, v = split_heads((q, k, v), B, T, self.config.n_heads)
        # q, k, v = split_heads((q, k, v), B, T, int(self.config.d_ff/self.config.d_model))
        # retention
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(T, retention_mask=retention_mask)
            retention_out, curr_kv, retention_weights = self.parallel_retention(q, k, v, decay_mask)
        elif forward_impl == 'recurrent':
            decay = self.get_recurrent_decay()
            retention_out, curr_kv = self.recurrent_retention(q,
                                                              k,
                                                              v,
                                                              past_key_value=past_key_value,
                                                              decay=decay,
                                                              retention_mask=retention_mask)
        elif forward_impl == 'chunkwise':
            assert chunk_size is not None
            q_chunks_1, k_chunks_1, v_chunks_1 = split_chunks(q, k, v, size=chunk_size, dim=2)
            q_chunks_2, k_chunks_2, v_chunks_2 = split_chunks(q, k, v, size=chunk_size, dim=2)
            q_chunks_3, k_chunks_3, v_chunks_3 = split_chunks(q, k, v, size=chunk_size, dim=2)
            if retention_mask is not None:
                retention_mask_chunks = split_chunks(retention_mask, size=chunk_size, dim=1)[0]
            ret_chunks_1 = []
            for i, (_q, _k, _v) in enumerate(zip(q_chunks_1, k_chunks_1, v_chunks_1)):
                csz = _q.size(2)
                ret_mask = retention_mask_chunks[i] if retention_mask is not None else None
                decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(
                    csz, retention_mask=ret_mask)
                out_chunk, past_key_value = self.chunkwise_retention(_q,
                                                                     _k,
                                                                     _v,
                                                                     decay_mask,
                                                                     past_key_value=past_key_value,
                                                                     chunk_decay=chunk_decay,
                                                                     inner_decay=inner_decay)
                ret_chunks_1.append(out_chunk)
            ret_chunks_2 = []
            for i, (_q, _k, _v) in enumerate(zip(q_chunks_2, k_chunks_2, v_chunks_2)):
                csz = _q.size(2)
                ret_mask = retention_mask_chunks[i] if retention_mask is not None else None
                decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(
                    csz, retention_mask=ret_mask)
                out_chunk, past_key_value = self.chunkwise_retention(_q,
                                                                     _k,
                                                                     _v,
                                                                     decay_mask,
                                                                     past_key_value=past_key_value,
                                                                     chunk_decay=chunk_decay,
                                                                     inner_decay=inner_decay)
                ret_chunks_2.append(out_chunk)
            ret_chunks_3 = []
            for i, (_q, _k, _v) in enumerate(zip(q_chunks_3, k_chunks_3, v_chunks_3)):
                csz = _q.size(2)
                ret_mask = retention_mask_chunks[i] if retention_mask is not None else None
                decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(
                    csz, retention_mask=ret_mask)
                out_chunk, past_key_value = self.chunkwise_retention(_q,
                                                                     _k,
                                                                     _v,
                                                                     decay_mask,
                                                                     past_key_value=past_key_value,
                                                                     chunk_decay=chunk_decay,
                                                                     inner_decay=inner_decay)
                ret_chunks_3.append(out_chunk)
            # [bsz, num_head, seqlen, v_dim]
            ##TODO确认该数据维度，好决定如何拼接
            print('ret_chunks_1:',ret_chunks_1.shape)
            print('ret_chunks_2:',ret_chunks_2.shape)
            print('ret_chunks_3:',ret_chunks_3.shape)
            retention_out = torch.cat([ret_chunks_1,ret_chunks_2,ret_chunks_3], dim=2)
            print('retention_out:',retention_out.shape)
            curr_kv = past_key_value
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        # concaat heads
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.config.v_dim)
        print('retention_out:',retention_out.shape)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.config.v_dim))
        normed = normed.view(B, T, self.config.v_dim)
        # out gate & proj
        out = self.silu(self.gated(hidden_states)) * normed
        print(out.shape)
        print(curr_kv.shape)

        outputs = (self.proj(out), curr_kv)
        if output_retentions:
            outputs += (retention_weights,) if forward_impl == 'parallel' else (None,)
        return outputs

class RetNetBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.msr = MultiScaleRetention(config)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'chunkwise',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        batchsize: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:

        msr_outs = self.msr(self.ln1(hidden_states),
                            retention_mask=retention_mask,
                            past_key_value=past_key_value,
                            forward_impl=forward_impl,
                            sequence_offset=sequence_offset,
                            chunk_size=chunk_size,
                            batchsize=batchsize,
                            output_retentions=output_retentions)
        msr = msr_outs[0]
        curr_kv = msr_outs[1]
        y = hidden_states + msr
        y = y + self.ffn(self.ln2(y))

        outputs = (y, curr_kv)

        if output_retentions:
            outputs += (msr_outs[2],)
        return outputs

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=60, stride=10):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.enc_in = configs.enc_in
        padding = stride

        # embedding
        self.embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.blocks = nn.ModuleList([RetNetBlock(configs) for _ in range(configs.e_layers)])

       
        if self.task_name == 'fault_prediction':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection_1 = nn.Linear(
               self.enc_in*self.seq_len*configs.d_model, configs.d_model)
            self.relu = nn.ReLU()#隐藏层，使用ReLU函数
            self.projection_2 = nn.Linear(
            configs.d_model, len(utils.global_var.get_value('class_names')))
          

    def fault_prediction(self, x_enc, x_mark_enc,  
                        past_key_values:Optional[List[torch.FloatTensor]] = None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        batchsize = x_enc.shape[0]

        x_enc = x_enc.permute(0, 2, 1)
        x_enc = torch.unsqueeze(x_enc, 3)
        x_enc = torch.reshape(x_enc, (x_enc.shape[0] * x_enc.shape[1], x_enc.shape[2], x_enc.shape[3]))

        x_mark_enc = x_mark_enc.repeat(self.enc_in,1,1)

        x_enc = self.embedding(x_enc, x_mark_enc)

        #TODO might be updated
        retention_mask = torch.ones((x_enc.shape[0], self.seq_len),
                                            dtype=torch.bool,
                                            device=x_enc.device)
        
        hidden_states = x_enc
        
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            block_outputs = block(hidden_states,
                        retention_mask=retention_mask,
                        forward_impl='chunkwise',
                        past_key_value=past_key_value,
                        chunk_size=self.patch_len,
                        batchsize = batchsize)

            hidden_states = block_outputs[0]   
       
        # Decoder
        output = self.flatten(hidden_states)
        output = self.dropout(output)
        output = output.reshape(batchsize, -1)
        output = self.projection_1(output) 
        output =self.relu(output)
        output = self.projection_2(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, attributes, mask=None):
        if self.task_name == 'fault_prediction':
            dec_out = self.fault_prediction(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None