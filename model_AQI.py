import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, batch_first: bool = False):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        self.register_buffer("pe", pe) # no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first == False:
            return x + self.pe[: x.size(0)].unsqueeze(1)
        else:
            return x + self.pe[: x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        batch_first: bool = False,
        max_len: int = 1000,
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=hid_dim * 4,
            dropout=dropout,
            batch_first=batch_first,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.pos_enc = PositionalEncoding(hid_dim, max_len, batch_first=batch_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1) # [T, B*N, hid]
        x = self.pos_enc(x)
        out = self.transformer(x) # [T, B*N, hid]
        return out[-1] # last step, [B*N, hid]


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, drop):
        super().__init__()
        padding = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=padding, dilation=dilation)
        self.chomp = lambda t: t[..., :-padding] # cut right padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.res  = nn.Conv1d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()

    def forward(self, x):
        out = self.conv(x) # [B*N, C_out, T+2p]
        out = self.chomp(out) # [B*N, C_out, T]
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + self.res(x))


class GLUTemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, drop):
        super().__init__()
        padding = (k - 1) * dilation
        self.conv_f = nn.Conv1d(c_in, c_out, k, padding=padding, dilation=dilation)
        self.conv_g = nn.Conv1d(c_in, c_out, k, padding=padding, dilation=dilation)
        self.chomp  = lambda t: t[..., :-padding]
        self.drop   = nn.Dropout(drop)
        self.res    = nn.Conv1d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()

    def forward(self, x):
        f = torch.tanh(self.chomp(self.conv_f(x)))
        g = torch.sigmoid(self.chomp(self.conv_g(x)))
        y = self.drop(f * g)
        return torch.relu(y + self.res(x))


class SeparableTemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, drop):
        super().__init__()
        padding = (k - 1) * dilation
        self.depth = nn.Conv1d(c_in, c_in, k, groups=c_in, padding=padding, dilation=dilation)
        self.point = nn.Conv1d(c_in, c_out, 1)
        self.chomp = lambda t: t[..., :-padding]
        self.drop  = nn.Dropout(drop)
        self.res   = nn.Conv1d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()

    def forward(self, x):
        y = self.point(self.chomp(self.depth(x)))
        y = self.drop(torch.relu(y))
        return torch.relu(y + self.res(x))


class InceptionTemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, ks_set=(3,5,7,9), dilation=1, drop=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            TemporalBlock(c_in, c_out//len(ks_set), k, dilation, drop)
            for k in ks_set
        ])
        self.res = nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return torch.relu(y + self.res(x))


class TemporalTCN(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        t_block: str = "basic", # ['basic', 'glu', 'separable', 'inception']
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    hid_dim, hid_dim, kernel_size, dilation, dropout
                ) if t_block == "basic"
                
                else GLUTemporalBlock(
                    hid_dim, hid_dim, kernel_size, dilation, dropout
                ) if t_block == "glu"
                
                else SeparableTemporalBlock(
                    hid_dim, hid_dim, kernel_size, dilation, dropout
                ) if t_block == "separable"
                
                else InceptionTemporalBlock(
                    hid_dim, hid_dim, dilation=dilation, drop=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2) # [B*N, hid, T]
        y = self.network(x) # [B*N, hid, T]
        return y[:, :, -1] # last step, [B*N, hid]


class SpatialNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        train_idx: torch.Tensor,
        dropout: float = 0.1,
        alpha: float = 0.2,
        hop: int = 1,
    ):
        super().__init__()
        self.hop = hop
        self.train_idx = train_idx
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.w_lot = nn.ModuleList()
        for i in range(hop):
            dim_in = in_features if i == 0 else out_features
            self.w_lot.append(nn.Linear(dim_in, out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B*T, N_train, F]
        adj: [N, N]
        """
        adj_train = adj[:, self.train_idx] # [N, N_train]
        emb = x
        for i in range(self.hop):
            if i == 0:
                emb = torch.einsum('bwf,vw->bvf', emb, adj_train) # [B*T, N, out_features]
            else:
                emb = torch.einsum('bwf,vw->bvf', emb, adj)
            emb = self.leakyrelu(self.w_lot[i](emb))
            emb = self.dropout(emb)
        return emb # [B*T, N, out_features]


def generate_time_features(T: int, mode: str, device='cpu'):
    """
    mode: 'days', 'weeks', 'months'
    return: [T, 2] tensor with sin/cos time features
    """
    periods = {
        'days': 24,
        'weeks': 168,
        'months': 720
    }

    if mode not in periods:
        raise ValueError(f"Unsupported mode: {mode}")

    period = periods[mode]
    t = torch.arange(T, device=device).float() # [T]
    sin_feat = torch.sin(2 * np.pi * t / period)
    cos_feat = torch.cos(2 * np.pi * t / period)

    time_feat = torch.stack([sin_feat, cos_feat], dim=1) # [T, 2]
    return time_feat


def add_time_features(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x: [B, N, T, F]
    return: [B, N, T, F+2]
    """
    B, N, T, F = x.shape
    time_feat = generate_time_features(T, mode, device=x.device) # [T, 2]
    time_feat = time_feat.unsqueeze(0).unsqueeze(0).expand(B, N, T, 2) # [B, N, T, 2]
    x_aug = torch.cat([x, time_feat], dim=-1) # [B, N, T, F+2]
    return x_aug


class GGTN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        train_idx: torch.Tensor,
        in_dim: int = 7,
        time_mode='none', # ['none', 'days', 'weeks', 'months']
        hid_dim: int = 32,
        hop: int = 1,
        horizon: int = 12,
        dropout: float = 0.1,
        alpha: float = 0.2, # LeakyReLU
        use_relu_out: bool = False, # ReLU after output proj
        temporal_model: str = "transformer", # ['gru', 'transformer', 'tcn']
        
        # for transformer
        n_heads: int = 4,
        trans_layers: int = 2,
        
        # for tcn
        tcn_layers: int = 4,
        t_block: str = "basic", # ['basic', 'glu', 'separable', 'inception']
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.train_idx = train_idx
        self.time_mode = time_mode
        self.temporal_model = temporal_model.lower()
        self.use_relu_out = use_relu_out

        if self.time_mode != 'none':
            in_dim += 2
        
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        self.gcn = SpatialNN(hid_dim, hid_dim, train_idx, dropout=dropout, alpha=alpha, hop=hop)

        if self.temporal_model == "gru":
            self.temporal = nn.GRU(hid_dim, hid_dim, batch_first=True)
        elif self.temporal_model == "transformer":
            self.temporal = TemporalTransformer(
                hid_dim, n_heads=n_heads, n_layers=trans_layers, dropout=dropout, batch_first=False
            )
        elif self.temporal_model == "tcn":
            self.temporal = TemporalTCN(
                hid_dim, n_layers=tcn_layers, dropout=dropout, t_block=t_block
            )
        else:
            raise ValueError("Supported model choice: ['gru', 'transformer', 'tcn']")

        self.output_proj = nn.Linear(hid_dim, horizon)
        
        if use_relu_out:
            self.out_act = nn.ReLU() # AQI >= 0
        else:
            self.out_act = nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N_train, T, F]
        adj: [N, N]
        Return pred: [B, N, horizon]
        """
        B, N_train, T, F = x.shape

        if self.time_mode != 'none':
            x = add_time_features(x, mode=self.time_mode)
        
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.leakyrelu(self.fc2(x))
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3).reshape(B * T, N_train, -1) # [B*T, N_train, hid]

        x = self.gcn(x, adj) # [B*T, N, hid]

        x = x.reshape(B, T, self.num_nodes, -1).permute(0, 2, 1, 3) # [B, N, T, hid]

        x = x.reshape(B * self.num_nodes, T, -1) # [B*N, T, hid]
        
        if self.temporal_model == "gru":
            out, _ = self.temporal(x) # [B*N, T, hid]
            emb = out[:, -1] # last ts in T, [B*N, hid]
        else:
            emb = self.temporal(x) # [B*N, hid]

        emb = emb.view(B, self.num_nodes, -1) # [B, N, hid]

        pred = self.output_proj(emb) # [B, N, horizon]
        output = self.out_act(pred) # optional ReLU
        
        return output # [B, N, horizon]
