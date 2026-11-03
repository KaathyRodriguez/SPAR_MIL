# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vision Transformer (ViT) 实现
主要用于DINO等自监督学习模型
大部分代码来自timm库
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    截断正态分布初始化（无梯度版本）
    从PyTorch官方master分支复制，直到正式发布
    方法基于：https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    
    参数:
        tensor: 要初始化的张量
        mean: 均值
        std: 标准差
        a: 下界
        b: 上界
    """
    def norm_cdf(x):
        """计算标准正态分布的累积分布函数"""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # 使用截断均匀分布生成值，然后使用正态分布的逆CDF变换
        # 获取上下界CDF值
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 用[l, u]范围内的值均匀填充张量，然后转换到[2l-1, 2u-1]
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # 使用正态分布的逆CDF变换得到截断标准正态分布
        tensor.erfinv_()

        # 变换到正确的均值和标准差
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # 截断以确保在正确范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    截断正态分布初始化
    
    参数:
        tensor: 要初始化的张量
        mean: 均值，默认0
        std: 标准差，默认1
        a: 下界，默认-2
        b: 上界，默认2
    
    返回:
        初始化后的张量
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop Path（随机深度）实现
    在训练时随机丢弃样本路径，用于正则化
    
    参数:
        x: 输入张量
        drop_prob: 丢弃概率
        training: 是否处于训练模式
    
    返回:
        处理后的张量
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于不同维度的张量，不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    output = x.div(keep_prob) * random_tensor  # 缩放以保持期望值
    return output


class DropPath(nn.Module):
    """
    Drop Path（随机深度）模块
    在残差块的主路径中按样本应用随机深度
    """
    def __init__(self, drop_prob=None):
        """
        初始化DropPath模块
        
        参数:
            drop_prob: 丢弃概率
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """前向传播"""
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    多层感知机（MLP）模块
    ViT Transformer块中的前馈网络
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化MLP模块
        
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度（默认等于in_features）
            out_features: 输出特征维度（默认等于in_features）
            act_layer: 激活函数层（默认GELU）
            drop: Dropout概率
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力模块
    实现Vision Transformer中的自注意力机制
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        初始化注意力模块
        
        参数:
            dim: 特征维度
            num_heads: 注意力头数
            qkv_bias: 是否在QKV投影中使用偏置
            qk_scale: QK缩放因子（默认head_dim^-0.5）
            attn_drop: 注意力dropout概率
            proj_drop: 投影dropout概率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # QKV投影：将输入投影到Q、K、V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 (B, N, C)，B是批次大小，N是序列长度，C是特征维度
        
        返回:
            x: 输出特征 (B, N, C)
            attn: 注意力权重 (B, num_heads, N, N)
        """
        B, N, C = x.shape
        # 计算Q、K、V并重塑为多头格式
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数：Q @ K^T / sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # Softmax归一化
        attn = self.attn_drop(attn)

        # 应用注意力权重到V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 输出投影
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """
    Transformer块
    包含自注意力层和前馈网络（MLP），使用残差连接和层归一化
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        初始化Transformer块
        
        参数:
            dim: 特征维度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度与输入维度的比例
            qkv_bias: 是否在QKV投影中使用偏置
            qk_scale: QK缩放因子
            drop: Dropout概率
            attn_drop: 注意力dropout概率
            drop_path: Drop Path概率
            act_layer: 激活函数层
            norm_layer: 归一化层
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        """
        前向传播
        
        参数:
            x: 输入特征
            return_attention: 是否返回注意力权重
        
        返回:
            输出特征（或注意力权重）
        """
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        # 残差连接 + Drop Path
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    图像到补丁嵌入模块
    将图像分割成补丁并投影到嵌入空间
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        初始化补丁嵌入模块
        
        参数:
            img_size: 输入图像大小
            patch_size: 补丁大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
        """
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用卷积层实现补丁嵌入（等价于线性投影）
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 (B, C, H, W)
        
        返回:
            补丁嵌入 (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        # 卷积提取补丁并展平
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模型
    将图像分割成补丁序列，使用Transformer编码器处理
    """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        """
        初始化Vision Transformer
        
        参数:
            img_size: 输入图像大小
            patch_size: 补丁大小
            in_chans: 输入通道数
            num_classes: 分类类别数（0表示不使用分类头）
            embed_dim: 嵌入维度
            depth: Transformer块的数量
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度比例
            qkv_bias: 是否在QKV投影中使用偏置
            qk_scale: QK缩放因子
            drop_rate: Dropout概率
            attn_drop_rate: 注意力dropout概率
            drop_path_rate: Drop Path概率
            norm_layer: 归一化层
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # 补丁嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token：用于分类的特殊token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码：为每个补丁和CLS token添加位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则：每个块的drop_path概率逐渐增加
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Transformer编码器块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 分类头（如果num_classes > 0）
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        参数:
            m: 模块
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        插值位置编码以适应不同大小的输入图像
        
        参数:
            x: 输入特征
            w: 图像宽度
            h: 图像高度
        
        返回:
            插值后的位置编码
        """
        npatch = x.shape[1] - 1  # 补丁数量（排除CLS token）
        N = self.pos_embed.shape[1] - 1  # 原始补丁数量
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]  # CLS token的位置编码
        patch_pos_embed = self.pos_embed[:, 1:]  # 补丁的位置编码
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # 添加小数以避免插值时的浮点误差
        # 参见：https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        # 将位置编码重塑为2D并插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """
        准备输入token：补丁嵌入 + CLS token + 位置编码
        
        参数:
            x: 输入图像 (B, C, H, W)
        
        返回:
            准备好的token序列 (B, num_patches+1, embed_dim)
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # 补丁线性嵌入

        # 添加[CLS] token到补丁token序列
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 为每个token添加位置编码
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 (B, C, H, W)
        
        返回:
            CLS token的特征 (B, embed_dim)
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # 返回CLS token的特征

    def get_last_selfattention(self, x):
        """
        获取最后一层的自注意力权重
        
        参数:
            x: 输入图像
        
        返回:
            最后一层的注意力权重
        """
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # 返回最后一层的注意力
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        """
        获取中间层的输出
        
        参数:
            x: 输入图像
            n: 返回最后n层的输出
        
        返回:
            最后n层的输出列表
        """
        x = self.prepare_tokens(x)
        # 返回最后n个块的输出token
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    """
    构建ViT-Tiny模型
    
    参数:
        patch_size: 补丁大小
        **kwargs: 其他参数
    
    返回:
        ViT-Tiny模型实例
    """
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    """
    构建ViT-Small模型
    
    参数:
        patch_size: 补丁大小
        **kwargs: 其他参数
    
    返回:
        ViT-Small模型实例
    """
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    """
    构建ViT-Base模型
    
    参数:
        patch_size: 补丁大小
        **kwargs: 其他参数
    
    返回:
        ViT-Base模型实例
    """
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    """
    DINO模型的投影头
    用于自监督学习的特征投影
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        """
        初始化DINO投影头
        
        参数:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            use_bn: 是否使用批归一化
            norm_last_layer: 是否归一化最后一层
            nlayers: MLP层数
            hidden_dim: 隐藏层维度
            bottleneck_dim: 瓶颈层维度
        """
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # 最后一层使用权重归一化
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征
        
        返回:
            投影后的特征
        """
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)  # L2归一化
        x = self.last_layer(x)
        return x
