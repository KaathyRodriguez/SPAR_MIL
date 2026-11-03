from os.path import join
import math
import numpy as np
import torch
import torch.nn as nn
# 导入 PyTorch 内部函数，例如 linear, softmax, dropout 等，用于实现多头注意力
from torch.nn.functional import *
from torch import Tensor
# from torch.nn.modules.linear import _LinearWithBias
_LinearWithBias = torch.nn.Linear # 通常用于指代带偏置的线性层
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from typing import Optional, Tuple
# 导入 nmslib 用于高效的近邻搜索
import nmslib 
from itertools import chain


class Attn_Net_Gated(nn.Module):
    """
    带 Sigmoid 门控的注意力网络 (Gated Attention Network)
    通常用于基于多实例学习 (MIL) 的特征聚合，例如 ViLa-MIL
    """
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        Args:
            L (int): 输入特征维度 (Input feature dimension), 例如来自 ResNet/Vision Transformer 的补丁特征维度
            D (int): 隐藏层维度 (Hidden layer dimension)
            dropout (bool): 是否应用 Dropout (p = 0.25)
            n_classes (int): 输出类别数 (在注意力机制中通常是 1)
        """
        super(Attn_Net_Gated, self).__init__()
        
        # 1. Attention 路径 'a' (Linear -> Tanh)
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        # 2. 门控路径 'b' (Linear -> Sigmoid)
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        # 应用 Dropout
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        # 转换为 nn.Sequential
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        # 3. 最终的注意力投影层 (D -> n_classes)
        # 经过门控和线性投影后，得到最终的注意力分数 A
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 输入特征张量 (N x L)，N是实例数 (补丁数)，L是特征维度
        
        Returns:
            A (Tensor): 原始的注意力分数 (N x n_classes)
            x (Tensor): 输入特征 (N x L)
        """
        a = self.attention_a(x) # N x D
        b = self.attention_b(x) # N x D
        
        # 应用 Sigmoid 门控: a 元素乘 b (element-wise multiplication)
        A = a.mul(b) # N x D
        
        # 投影到最终的注意力分数
        A = self.attention_c(A)  # N x n_classes (通常是 N x 1)
        return A, x


def init_max_weights(module):
    r"""
    权重初始化函数
    使用正态分布初始化权重，偏置初始化为零。
    
    Args:
        module (torch.nn.Module): 需要初始化权重的模型模块
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            # 使用 Xavier 均匀初始化 (fan-in) 的标准差
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv) # 权重使用 N(0, stdv) 初始化
            m.bias.data.zero_() # 偏置初始化为 0


# ----------------------------------------------------------------------
# 以下是自定义的多头注意力机制实现 (模仿 PyTorch 内部实现)
# ----------------------------------------------------------------------

def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_raw: bool = True, # 新增参数，用于控制是否返回 softmax 之前的原始权重
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
):
    r"""
    多头注意力机制的前向传播实现
    这是一个自定义版本，可能包含额外的参数，例如 `need_raw`，用于返回原始注意力分数。
    """
    # 检查是否需要使用torch函数处理 (通常用于 JIT 编译兼容性)
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        # 忽略处理 torch function 的逻辑，这不是核心功能
        pass 
    
    # [L, N, E],tgt_len为实例（补丁）数量，bsz为batch_size
    tgt_len, bsz, embed_dim = query.size() 
    assert embed_dim == embed_dim_to_check
    # 允许MHA在特征维度上有不同的大小
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    # 计算每个头的维度
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5  # 缩放因子 (1/sqrt(d_k))

    # --- Q, K, V 投影 ---
    if not use_separate_proj_weight:
        # 使用单个 `in_proj_weight`维度为[3*E,E]， 包含 Q, K, V 的权重
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # 自注意力 (Self-Attention): Q=K=V
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # 编码器-解码器注意力: K=V, Q不同
            # Q 的投影
            _w = in_proj_weight[0:embed_dim, :]
            _b = in_proj_bias[0:embed_dim] if in_proj_bias is not None else None
            q = linear(query, _w, _b)

            # K 和 V 的投影
            _w = in_proj_weight[embed_dim:, :]
            _b = in_proj_bias[embed_dim:] if in_proj_bias is not None else None
            k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # 通用注意力: Q, K, V 都不同
            # Q 的投影
            _w = in_proj_weight[0:embed_dim, :]
            _b = in_proj_bias[0:embed_dim] if in_proj_bias is not None else None
            q = linear(query, _w, _b)

            # K 的投影
            _w = in_proj_weight[embed_dim : embed_dim * 2, :]
            _b = in_proj_bias[embed_dim : embed_dim * 2] if in_proj_bias is not None else None
            k = linear(key, _w, _b)

            # V 的投影
            _w = in_proj_weight[embed_dim * 2:, :]
            _b = in_proj_bias[embed_dim * 2:] if in_proj_bias is not None else None
            v = linear(value, _w, _b)
    else:
        # 使用分离的投影权重 (q_proj_weight, k_proj_weight, v_proj_weight)
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)

        # 投影 Q, K, V
        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            # 此处代码有缺陷，如果 in_proj_bias 为 None，仍然使用了 in_proj_bias
            # 实际调用时，in_proj_bias 应该统一传递 None
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    
    # 缩放查询向量 (Scaled Dot-Product Attention)
    q = q * scaling

    # 处理注意力掩码 attn_mask (用于阻止某些位置的注意力)
    if attn_mask is not None:
        # 检查并处理 attn_mask 的维度
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            pass # 已经是 3D mask (N*heads, L, S)
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        '''
        # 将 bool/byte 掩码转换为加性掩码 (float('-inf'))
        if attn_mask.dtype == torch.bool or attn_mask.dtype == torch.uint8:
             # 注意：在计算 attn_output_weights 时才会实际应用 masked_fill_
        '''
    # 处理键填充掩码 key_padding_mask (用于忽略填充的 key/value 元素)
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # 添加键和值的偏置 (bias_k, bias_v)，维度为[1,1,E]
    if bias_k is not None and bias_v is not None:
        # 在序列维度 (dim=0) 上将偏置拼接到 K 和 V
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)]) # repeat 函数将 bias_k 从 [1,1,E] 复制到 [1,N,E]
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)]) # torch.cat 函数将原始的张量 [S,N,E] 和重复后的偏置张量 [1,N,E] 沿着序列维度（dim=0）进行拼接变为[S+1,N,E]
            # attn_mask维度为[L,S]或[N*H,L,S]，L是目标序列长度，表需要计算注意力输出的元素的数量；S是源序列长度，代表提供信息的元素的数量
            if attn_mask is not None:
                # 相应地对 attn_mask 进行填充 (pad)
                # pad 函数的第二个参数 (0, 1) 表示只在最后一个维度的右侧填充 1 个单位,如果 attn_mask 形状为 [L,S] ,新形状变为 [L,S+1]
                attn_mask = pad(attn_mask, (0, 1))
            #key_padding_mask维度为[N,S]
            if key_padding_mask is not None:
                # 相应地对 key_padding_mask 进行填充
                # 原始 key_padding_mask 形状为 [N,S]，新形状变为 [N,S+1]
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            # 静态 K/V 不支持偏置
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    # 重塑 Q, K, V 为多头格式: (batch*heads, seq_len, head_dim)
    # L x (N*heads) x head_dim -> (N*heads) x L x head_dim (转置 0 和 1)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # 使用静态的键和值
    if static_k is not None:
        k = static_k

    if static_v is not None:
        v = static_v

    src_len = k.size(1)

    # 验证 key_padding_mask 的维度

    # 添加零注意力 (如果启用)
    if add_zero_attn:
        src_len += 1
        # 在序列长度维度上拼接零向量，零向量维度为[N*heads,1,head_dim]，新的k,v维度为[N*heads,S+1,head_dim]
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        # [l,S] -> [L,S+1]
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # 计算注意力分数: Q @ K^T (使用 bmm 进行批次矩阵乘法),attn_output_weights维度为[bsz * num_heads, tgt_len, src_len]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    # 应用注意力掩码 (加性)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # bool 掩码: True 的位置被填充为负无穷
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            # float 掩码: 直接相加
            attn_output_weights += attn_mask

    # 应用键填充掩码 (加性)，key_padding_mask维度为[N,S]
    if key_padding_mask is not None:
        # 扩展 key_padding_mask 维度以匹配 attn_output_weights
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # masked_fill: 对应 True 的位置被填充为负无穷
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),  
            float("-inf"),
            # .unsqueeze(1):在第 1 维(num_heads)增加一个维度,形状变为[N,1,S]
            # .unsqueeze(2):在第 2 维(tgt_len)增加一个维度。形状最终变为[N,1,1,S]
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    # 保存原始注意力权重 (softmax 之前)
    attn_output_weights_raw = attn_output_weights
    
    # 应用 softmax 和 dropout
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    # 计算注意力输出: 注意力权重 @ V
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    
    # 重塑输出: (N*heads) x L x head_dim -> L x N x E
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    
    # 应用输出投影，out_proj_weight维度为[E,E],ut_proj_bias维度为[E]
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    # 根据需求返回结果
    if need_weights:
        if need_raw:
            # 返回原始注意力权重 (softmax 之前，通常用于分析或特定 MIL 结构)
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output,attn_output_weights_raw
        else:
            # 返回平均后的注意力权重 (跨头平均)
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    r"""
    多头注意力机制模块
    这是一个自定义的 PyTorch Module 封装，使用了上面定义的 `multi_head_attention_forward` 函数。
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        """
        初始化多头注意力模块

        Args:
            embed_dim (int): 总的嵌入维度 E
            num_heads (int): 并行注意力头的数量 H
            dropout (float): 注意力权重上的 dropout 概率
            bias (bool): 是否使用偏置
            add_bias_kv (bool): 是否在 K 和 V 序列的 dim=0 处添加偏置
            add_zero_attn (bool): 是否在 K 和 V 序列的 dim=1 处添加零注意力 (零向量)
            kdim (int, Optional): 键 (Key) 的特征维度 (默认为 embed_dim)
            vdim (int, Optional): 值 (Value) 的特征维度 (默认为 embed_dim)
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # 检查 K, V 的维度是否与 Q 相同
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V 投影权重的初始化
        if self._qkv_same_embed_dim is False:
            # 使用分离的投影权重 (Q, K, V 维度不同或显式要求分离)
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            # 使用单个合并的投影权重 (Q, K, V 维度相同)
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim)) # 3*E x E
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        # 偏置的初始化
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim)) # 3*E
        else:
            self.register_parameter('in_proj_bias', None)
            
        # 输出投影层
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        # K/V 序列偏置的初始化
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        """权重初始化函数"""
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight) # 使用 Xavier 均匀初始化
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.) # 偏置初始化为 0
            constant_(self.out_proj.bias, 0.) # 输出投影偏置初始化为 0
        if self.bias_k is not None:
            xavier_normal_(self.bias_k) # K/V 偏置使用 Xavier 正态初始化
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # 向后兼容旧的 checkpoint
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_raw=True, attn_mask=None):
        r"""
        前向传播函数

        Args:
            query (Tensor): 查询张量 (L, N, E)
            key (Tensor): 键张量 (S, N, E)
            value (Tensor): 值张量 (S, N, E)
            key_padding_mask (Tensor, Optional): 键序列的填充掩码 (N, S)
            need_weights (bool): 是否返回注意力权重
            need_raw (bool): 是否返回 softmax 之前的原始注意力权重 (需要 need_weights=True)
            attn_mask (Tensor, Optional): 注意力掩码 (L, S) 或 (N*num_heads, L, S)

        Returns:
            attn_output (Tensor): 注意力输出 (L, N, E)
            attn_output_weights (Tensor, Optional): 注意力权重 (N, L, S) 或原始权重
        """
        if not self._qkv_same_embed_dim:
            # K, V 维度不同，使用分离的投影权重
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias, # in_proj_weight/bias 为 None
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            # K, V 维度相同，使用合并的投影权重
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask)


# ----------------------------------------------------------------------
# 以下是用于图结构构建的函数
# ----------------------------------------------------------------------

class Hnsw:
    """
    Hnsw (Hierarchical Navigable Small World) 算法的封装类
    用于高效的近似最近邻 (ANN) 搜索，这里用于构建图的边 (近邻关系)。
    """
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        """
        初始化 Hnsw 搜索器

        Args:
            space (str): 距离空间，例如 'cosinesimil' (余弦相似度), 'l2' (欧氏距离平方)
            index_params (dict): 索引构建参数
            query_params (dict): 查询参数
        """
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        """
        构建 Hnsw 索引
        
        Args:
            X (np.ndarray): 待索引的数据点 (补丁坐标或特征)
        """
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # 初始化 NMSLib 索引
        index = nmslib.init(space=self.space, method='hnsw')
        # 添加数据点
        index.addDataPointBatch(X)
        # 创建索引
        index.createIndex(index_params)
        # 设置查询参数
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        """
        查询最近的 topn 个邻居

        Args:
            vector (np.ndarray): 查询向量 (单个数据点)
            topn (int): 需要返回的最近邻数量

        Returns:
            indices (np.ndarray): 最近邻居的索引
        """
        # knnQuery 返回索引和距离，这里只取索引
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(coords, features, threshold=5000, radius=9):
    """
    将补丁 (Patches) 转换为图结构 (Graph)
    根据空间距离和特征相似性构建图的边。

    Args:
        coords (Tensor): 补丁的坐标 (N x 2)
        features (Tensor): 补丁的特征 (N x L)
        threshold (int): 空间距离阈值 (超过此距离的边将被过滤)
        radius (int): 近邻搜索的半径 (k值，包括自身)

    Returns:
        G (torch_geometric.data.Data): 包含节点特征、边索引和坐标的图数据对象
    """
    from torch_geometric.data import Data as geomData
    
    # 转换为 NumPy 数组
    coords, features = np.array(coords.cpu().detach()), np.array(features.cpu().detach())
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    # --- 1. 基于空间坐标构建边 (Spatial Edges) ---
    model = Hnsw(space='l2') # 使用 L2 (欧氏距离) 空间
    model.fit(coords)
    
    # 构造边的起始节点 a (重复每个节点 radius-1 次)
    a = np.repeat(range(num_patches), radius-1)
    # 构造边的结束节点 b (使用 Hnsw 查询每个节点的 radius-1 个最近邻，跳过自身 [1:])
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.Tensor(np.stack([a,b])).type(torch.LongTensor) # 2 x E_spatial

    # --- 2. 基于特征构建边 (Latent Edges) ---
    model = Hnsw(space='l2') # 仍然使用 L2 空间 (尽管通常特征空间使用 Cosine)
    model.fit(features)
    
    # 注意：这里使用 coords[v_idx] 作为查询向量，但应该使用 features[v_idx]。
    # 假设代码本意是基于特征构建近邻图，但实际查询使用了坐标。
    # 为了保持代码原样，保留使用 coords 的查询，但注释指出潜在问题。
    a = np.repeat(range(num_patches), radius-1)
    # ❗ 注意: 这里使用 coords[v_idx] 作为查询，应该修正为 features[v_idx]
    b = np.fromiter(chain(*[model.query(features[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.Tensor(np.stack([a,b])).type(torch.LongTensor) # 2 x E_latent

    # --- 3. 过滤空间边 (使用距离阈值) ---
    start_point = edge_spatial[0, :]
    end_point = edge_spatial[1, :]
    start_coord = coords[start_point]
    end_coord = coords[end_point]
    tmp = start_coord - end_coord
    
    # 计算每条空间边的欧氏距离
    edge_distance = []
    for i in range(tmp.shape[0]):
        distance = math.hypot(tmp[i][0], tmp[i][1]) # 欧氏距离
        edge_distance.append(distance)

    # 过滤掉距离大于阈值的空间边
    filter_edge_spatial = edge_spatial[:, np.array(edge_distance) <= threshold]

    # --- 4. 构建图数据对象 ---
    G = geomData(x = torch.Tensor(features), # 节点特征
                 edge_index = filter_edge_spatial, # 空间过滤后的边索引
                 edge_latent = edge_latent, # 基于特征的边索引 (未过滤)
                 centroid = torch.Tensor(coords)) # 节点坐标

    return G


# ----------------------------------------------------------------------
# 以下是基于核函数 (Kernel) 和信息理论的度量函数
# ----------------------------------------------------------------------

def pairwise_distances(x):
    """
    计算输入张量 x 之间的欧氏距离的平方 (Pairwise Squared Euclidean Distance)
    
    Args:
        x (Tensor): 输入张量 (N x L)
        N为实例数量，L为特征维度

    Returns:
        Tensor: 距离矩阵 (N x N)，每个元素为两个实例之间的欧式距离的平方
    """
    bn = x.shape[0]
    x = x.view(bn, -1)
    # 计算每个实例的 L2 范数平方 (N x 1)
    # x ** 2： 对张量 x中的每个元素进行平方（逐元素操作），形状仍为 [N,L]
    #torch.sum(..., -1)：沿着最后一个维度（L）求和，结果形状为[N]
    # .reshape((-1, 1))：将 [N] 形状的张量变为列向量[N,1]
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    # 公式: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T b
    # 返回 -2 * a^T b + ||a||^2 + ||b||^2
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    """
    计算 Gram 矩阵 (高斯核/RBF 核)

    Gram 矩阵 (Gram Matrix)，通常用G或K表示，是一个对称矩阵，用于描述一个向量集合中所有向量对之间的关系（相似度或内积）
    
    Args:
        x (Tensor): 输入张量
        sigma (float): 高斯核的带宽参数

    Returns:
        Tensor: Gram 矩阵 K (N x N)
    """
    # 计算欧氏距离平方
    dist = pairwise_distances(x)
    # 计算高斯核: exp(-dist / sigma)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    """
    使用核方法估计 Renyi 熵 (Alpha = 1.01)
    Renyi熵是信息论中对香农熵的一种推广,它引入了一个参数alpha(称为阶数）来衡量概率分布的信息量或不确定性。
    
    Args:
        x (Tensor): 输入张量
        sigma (float): 高斯核的带宽参数

    Returns:
        Tensor: Renyi 熵 H_alpha(x)
    """
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    # 归一化核矩阵: K / Tr(K)
    k = k / torch.trace(k)
    '''
    # 计算特征值 (使用对称矩阵的特征分解)
    # torch.symeig 在新版本中已被 torch.linalg.eigh 取代，但为了兼容性保留原样
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    '''
    # --- 核心修改部分 ---
    # 使用 torch.linalg.eigh 代替 torch.symeig 计算特征值
    # eigh 返回 (特征值, 特征向量)，我们只需要特征值
    # UPLO='U' 或 'L' 指定输入矩阵的哪部分是有效的（因为它是对称矩阵）
    eigv = torch.linalg.eigh(k, UPLO='U')[0]
    # ------------------
    
    # 确保特征值非负（虽然对于半正定矩阵理论上非负）
    eigv = torch.abs(eigv)
    eig_pow = eigv ** alpha
    # Renyi 熵公式: H_alpha(x) = (1 / (1 - alpha)) * log2(sum(lambda_i^alpha))
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    """
    使用核方法估计联合 Renyi 熵 H_alpha(x, y)
    
    Args:
        x (Tensor): 第一个输入张量
        y (Tensor): 第二个输入张量
        s_x (float): x 的高斯核带宽参数
        s_y (float): y 的高斯核带宽参数

    Returns:
        Tensor: 联合 Renyi 熵 H_alpha(x, y)
    """
    alpha = 1.01
    # 计算 x 和 y 各自的 Gram 矩阵 Kx 和 Ky
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    
    # 计算联合核矩阵 K = Kx * Ky (元素乘)
    k = torch.mul(x, y)
    
    # 归一化核矩阵: K / Tr(K)
    k = k / torch.trace(k)
    
    # --- 核心修改部分 ---
    # 使用 torch.linalg.eigh 代替 torch.symeig 计算特征值
    # eigh 返回 (特征值, 特征向量)，我们只需要特征值
    # UPLO='U' 或 'L' 指定输入矩阵的哪部分是有效的
    eigv = torch.linalg.eigh(k, UPLO='U')[0]
    # ------------------
    
    # 确保特征值非负（虽然对于核矩阵理论上应非负）
    eigv = torch.abs(eigv)
    '''
    # 计算特征值
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    '''
    eig_pow = eigv ** alpha
    # 联合 Renyi 熵公式
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, s_x, s_y):
    """
    使用 Renyi 熵估计互信息 (Mutual Information, MI)
    
    Args:
        x (Tensor): 第一个输入张量
        y (Tensor): 第二个输入张量
        s_x (float): x 的高斯核带宽参数
        s_y (float): y 的高斯核带宽参数

    Returns:
        Tensor: 互信息 I(x; y)
    """
    # 估计 x 的 Renyi 熵 Hx
    Hx = reyi_entropy(x, sigma=s_x)
    # 估计 y 的 Renyi 熵 Hy
    Hy = reyi_entropy(y, sigma=s_y)
    # 估计联合 Renyi 熵 Hxy
    Hxy = joint_entropy(x, y, s_x, s_y)
    
    # 互信息公式: I(x; y) = H(x) + H(y) - H(x, y)
    Ixy = Hx + Hy - Hxy
    return Ixy