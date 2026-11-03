"""
截断的ResNet架构实现
修改自PyTorch官方resnet.py
用于特征提取，只保留到layer3，不包含layer4和全连接层
"""
# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchsummary import summary
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

# 预训练模型URL
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck_Baseline(nn.Module):
    """
    ResNet瓶颈块（Bottleneck Block）
    使用1x1-3x3-1x1卷积结构，用于ResNet50及更深的网络
    """
    expansion = 4  # 输出通道数是输入planes的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        初始化瓶颈块
        
        参数:
            inplanes: 输入通道数
            planes: 中间层通道数
            stride: 卷积步长
            downsample: 下采样层（用于残差连接）
        """
        super(Bottleneck_Baseline, self).__init__()
        # 1x1卷积：降维
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3卷积：主要特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1卷积：升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图
        
        返回:
            输出特征图
        """
        residual = x  # 保存残差连接

        # 主路径：三个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要进行下采样，调整残差连接的维度
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接
        out += residual
        out = self.relu(out)

        return out

class ResNet_Trunc_Baseline(nn.Module):
    """
    截断的ResNet基线模型
    只包含到layer3，不包含layer4和全连接层
    用于特征提取任务
    """

    def __init__(self, block, layers):
        """
        初始化截断ResNet
        
        参数:
            block: 基础块类型（Bottleneck_Baseline）
            layers: 每个layer的块数量列表，例如[3, 4, 6, 3]对应ResNet50
        """
        self.inplanes = 64  # 初始通道数
        super(ResNet_Trunc_Baseline, self).__init__()
        # 初始卷积层：7x7卷积，stride=2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化：3x3，stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建三个残差层（不包含layer4）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 自适应平均池化：输出1x1特征图
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建一个残差层
        
        参数:
            block: 基础块类型
            planes: 该层的通道数
            blocks: 该层包含的块数量
            stride: 第一个块的步长
        
        返回:
            残差层（Sequential）
        """
        downsample = None
        # 如果需要下采样或改变通道数，创建下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个块可能需要下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 后续块不需要下采样
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 (batch_size, 3, H, W)
        
        返回:
            特征向量 (batch_size, feature_dim)
        """
        # 初始卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 三个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局平均池化并展平
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def resnet50_trunc_baseline(pretrained=False):
    """
    构建截断的ResNet-50模型
    
    参数:
        pretrained (bool): 如果为True，返回在ImageNet上预训练的模型
    
    返回:
        截断的ResNet-50模型
    """
    # ResNet50的层配置：[3, 4, 6, 3]，但这里只使用前三个
    model = ResNet_Trunc_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model


def load_pretrained_weights(model, name):
    """
    加载预训练权重
    
    参数:
        model: 模型实例
        name: 模型名称（如'resnet50'）
    
    返回:
        加载了预训练权重的模型
    """
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)  # strict=False允许部分匹配
    return model


