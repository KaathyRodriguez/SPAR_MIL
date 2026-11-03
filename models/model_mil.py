"""
多实例学习（MIL）模型实现
用于全片扫描图像（WSI）分类任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights

class MIL_fc(nn.Module):
    """
    基于全连接层的多实例学习模型（二分类）
    使用top-k实例选择策略
    """
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        """
        初始化MIL全连接模型
        
        参数:
            gate: 是否使用门控机制（未使用）
            size_arg: 模型大小参数，"small"表示使用[1024, 512]的隐藏层维度
            dropout: 是否使用dropout
            n_classes: 类别数量（必须为2）
            top_k: 选择的top-k实例数量
        """
        super(MIL_fc, self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [1024, 512]}  # 定义不同大小的隐藏层维度
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]  # 第一层全连接+ReLU
        if dropout:
            fc.append(nn.Dropout(0.25))  # 可选的dropout层

        fc.append(nn.Linear(size[1], n_classes))  # 输出层
        self.classifier= nn.Sequential(*fc)
        initialize_weights(self)  # 初始化权重
        self.top_k=top_k  # top-k实例选择数量

    def relocate(self):
        """
        将模型移动到GPU或CPU
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h, return_features=False):
        """
        前向传播
        
        参数:
            h: 输入特征 (num_instances, feature_dim)
            return_features: 是否返回特征
        
        返回:
            top_instance: top-k实例的logits
            Y_prob: top-k实例的预测概率
            Y_hat: 预测类别
            y_probs: 所有实例的概率
            results_dict: 结果字典（可能包含特征）
        """
        if return_features:
            # 如果需要返回特征，分别计算特征和logits
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1
        
        # 计算所有实例的概率
        y_probs = F.softmax(logits, dim = 1)
        # 选择top-k个实例（基于正类概率）
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        # 预测类别和概率
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            # 如果需要，返回top-k实例的特征
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    """
    基于全连接层的多实例学习模型（多分类）
    为每个类别使用独立的分类器
    """
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        """
        初始化MIL多分类模型
        
        参数:
            gate: 是否使用门控机制（未使用）
            size_arg: 模型大小参数
            dropout: 是否使用dropout
            n_classes: 类别数量（必须大于2）
            top_k: 选择的top-k实例数量（必须为1）
        """
        super(MIL_fc_mc, self).__init__()
        assert n_classes > 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)  # 共享的特征提取层

        # 为每个类别创建独立的分类器
        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1  # 多分类只支持top_k=1

    def relocate(self):
        """
        将模型移动到GPU或CPU
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)
    
    def forward(self, h, return_features=False):
        """
        前向传播
        
        参数:
            h: 输入特征 (num_instances, feature_dim)
            return_features: 是否返回特征
        
        返回:
            top_instance: top实例的logits
            Y_prob: top实例的预测概率
            Y_hat: 预测类别
            y_probs: 所有实例的概率
            results_dict: 结果字典（可能包含特征）
        """
        device = h.device
       
        # 通过共享的特征提取层
        h = self.fc(h)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)

        # 使用每个类别的独立分类器计算logits
        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)  # self.classifiers[c](h)维度为[num_instances,1]，.squeeze(1)维度为[num_instances]，logits[:, c]维度为[num_instances,n_classes]
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)        

        # 计算概率并找到最大概率的实例和类别
        y_probs = F.softmax(logits, dim = 1)  #类别维度 (dim=1) 应用 Softmax，y_probs维度为[num_instances, n_classes]
        m = y_probs.view(1, -1).argmax(1)  # 找到全局最大概率的索引位置，概率矩阵展平为一个行向量，.view维度为[1,num_instances*n_classes]，.argmax维度为[1]
        # 将一维索引转换为(实例索引, 类别索引)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]  # 预测类别
        Y_prob = y_probs[top_indices[0]]  # 预测概率
        
        results_dict = {}

        if return_features:
            # 如果需要，返回top实例的特征
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
