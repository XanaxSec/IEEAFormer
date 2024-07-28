import torch.nn as nn
import torch
from torchinfo import summary
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Trend_aware_attention(nn.Module):
    '''
    Trend_aware_attention  mechanism
    X:      [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, kernel_size):
        super(Trend_aware_attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D, D)
        self.FC = nn.Linear(D, D)
        self.kernel_size = kernel_size
        self.padding = self.kernel_size - 1
        self.cnn_q = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)

    def forward(self, X):
        # (B,T,N,D)
        batch_size = X.shape[0]

        X_ = X.permute(0, 3, 2, 1)  # 对x进行变换, 便于后续执行CNN: (B,T,N,D) --> (B,D,N,T)

        # key: (B,T,N,D)  value: (B,T,N,D)
        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2,
                                                                             1)  # 通过1×k的因果卷积来生成query表示: (B,D,N,T)--permute-->(B,T,N,D)
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2,
                                                                           1)  # 通过1×k的因果卷积来生成key表示: (B,D,N,T)--permute-->(B,T,N,D)
        value = self.FC_v(X)  # 通过简单的线性层生成value: (B,T,N,D)-->(B,T,N,D)

        query = torch.cat(torch.split(query, self.d, dim=-1),
                          dim=0)  # 将query划分为多头: (B,T,N,D)-->(B*k,T,N,d);  D=k*d; h:注意力头的个数; d:每个注意力头的通道上
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  # 将key划分为多头: (B,T,N,D)-->(B*k,T,N,d);
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)  # 将value划分为多头: (B,T,N,D)-->(B*k,T,N,d);

        query = query.permute(0, 2, 1, 3)  # query: (B*k,T,N,d) --> (B*k,N,T,d)
        key = key.permute(0, 2, 3, 1)  # key: (B*k,T,N,d) --> (B*k,N,d,T)
        value = value.permute(0, 2, 1, 3)  # key: (B*k,T,N,d) --> (B*k,N,T,d)

        attention = (query @ key) * (self.d ** -0.5)  # 以上下文的方式计算注意力矩阵: (B*k,N,T,d) @ (B*k,N,d,T) = (B*k,N,T,T)
        attention = F.softmax(attention, dim=-1)  # 通过softmax进行归一化

        X = (attention @ value)  # 通过上下文化的注意力矩阵对value进行加权: (B*k,N,T,T) @ (B*k,N,T,d) = (B*k,N,T,d)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # (B*k,N,T,d)-->(B,N,T,d*k)==(B,N,T,D)
        X = self.FC(X)  # 融合多个子空间的特征进行输出
        return X.permute(0, 2, 1, 3)  # (B,N,T,D)-->(B,T,N,D)


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim=256, num_heads=8, mask=False, geo_mask=False, sem_mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.geo_mask = geo_mask
        self.sem_mask = sem_mask

        self.head_dim = model_dim // num_heads

        self.FC_Q_s = nn.Linear(model_dim, model_dim)
        self.FC_K_s = nn.Linear(model_dim, model_dim)
        self.FC_V_s = nn.Linear(model_dim, model_dim)
        # self.FC_Q_t = nn.Linear(12, 12)
        # self.FC_K_t = nn.Linear(12, 12)
        # self.FC_V_t = nn.Linear(12, 12)
        self.out_proj = nn.Linear(model_dim, model_dim)

    # self.out_proj_t = nn.Linear(12, 12)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)

        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        geo_data = np.load('geo_mask_pems04.npz')

        geo_mask_array = geo_data['arr_0']
        geo_mask_array = torch.tensor(geo_mask_array)

        sem_data = np.load('sem_mask_pems04.npz')
        sem_mask_array = sem_data['arr_0']
        sem_mask_array = torch.tensor(sem_mask_array)
        sem_mask_array = sem_mask_array.bool()
        geo_mask_array = geo_mask_array.bool()
        geo_mask_array = geo_mask_array.to('cuda:0')
        sem_mask_array = sem_mask_array.to('cuda:0')
        '''
        if query.size(-2) == 12:
            query = torch.transpose(query, 2, 3)
            key = torch.transpose(key, 2, 3)
            value = torch.transpose(value, 2, 3)
            query = self.FC_Q_t(query)
            key = self.FC_K_t(key)
            value = self.FC_V_t(value)
            key = key.transpose(-1, -2)
            attn_score = (query @ key) / 96 ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
            attn_score = torch.softmax(attn_score, dim=-1)
            out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

            out = self.out_proj_t(out)
            out = torch.transpose(out, 2, 3)


        else:
        '''
        query = self.FC_Q_s(query)
        key = self.FC_K_s(key)
        value = self.FC_V_s(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.geo_mask:
            attn_score.masked_fill_(geo_mask_array, -torch.inf)
        if self.sem_mask:
            attn_score.masked_fill_(sem_mask_array, -torch.inf)

            '''
            if self.mask:
                mask = torch.ones(
                    tgt_length, src_length, dtype=torch.bool, device=query.device
                ).tril()  # lower triangular part of the matrix
                attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
            '''
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class Time_SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False, geo_mask=False, sem_mask=False
    ):
        super().__init__()

        self.time_attn = Trend_aware_attention(K=8, d=19, kernel_size=3)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, 152)
        residual = x

        out = self.time_attn(x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False, geo_mask=False, sem_mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, 152)
        residual = x

        out = self.attn(x, x, x, )  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,

    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.embedding_lap_pos_enc = nn.Linear(8, 152)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.proj = nn.Linear(304, self.model_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                Time_SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_geo = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout, geo_mask=True)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_sem = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout, sem_mask=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)
        # print(x.shape)# (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)

        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x_t = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        lap_mx = np.load('lap_pems04.npz')['arr_0']

        lap_mx = torch.tensor(lap_mx).to('cuda:0')
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        x_t = x_t.to('cuda:0')  # 将 x_t 移动到 CUDA 设备上
        lap_pos_enc = lap_pos_enc.to('cuda:0')

        for time_attn in self.attn_layers_t:
            x_t = time_attn(x_t)
        x_t += lap_pos_enc
        x_geo = x_t
        x_sem = x_t
        for attn in self.attn_layers_geo:
            x_geo = attn(x_geo, dim=2)

        for attn in self.attn_layers_sem:
            x_sem = attn(x_sem, dim=2)
        x = self.proj(torch.cat([x_geo, x_sem], dim=-1))

        x = self.proj_drop(x)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out



