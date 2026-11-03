import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):
    """
    Focal Loss (焦点损失) 的实现
    
    主要用于解决类别不平衡问题，通过降低易分类样本的权重，
    使模型更关注难以分类的少数样本。
    Focal Loss = - (1 - pt)^gamma * log(pt)
    其中 pt = P(y|x) 是模型预测正确的概率，通常通过 Cross Entropy Loss 导出。
    """
    
    def __init__(self, weight=None, gamma=2.0, alpha=None, reduction='mean'):
        """
        初始化 Focal Loss
        
        Args:
            weight (Tensor, 可选): 保留的 PyTorch 风格类别权重。
            alpha (float or Tensor, 可选): Focal Loss 的 alpha 参数。若为标量，默认把 1-alpha 赋给负类。
            gamma (float): 调节因子 (1 - pt)^gamma 的指数。gamma > 0 会降低易分类样本的损失。
                           通常设置为 2。
            reduction (str): 损失的聚合方式 ('mean', 'sum', 'none')。
        """
        # 调用父类初始化方法，设置 weight 和 reduction
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, input, target):
        """
        前向传播计算 Focal Loss
        
        Args:
            input (Tensor): 模型输出的 logits (未经 Softmax/Sigmoid 处理的原始预测分数)。
            target (Tensor): 真实类别标签。
            
        Returns:
            Tensor: 计算得到的 Focal Loss 值。
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)

        target = target.long().view(-1, 1)
        logpt = logpt.gather(1, target)
        pt = pt.gather(1, target)

        if self.alpha is not None:
            if torch.is_tensor(self.alpha):
                alpha = self.alpha.to(input.device)
                if alpha.numel() == 1:
                    alpha_t = alpha.view(1, 1)
                else:
                    alpha_t = alpha.gather(0, target.view(-1)).view(-1, 1)
            else:
                alpha = float(self.alpha)
                if input.size(1) == 2:
                    alpha_vec = torch.tensor([1.0 - alpha, alpha], device=input.device, dtype=input.dtype)
                    alpha_t = alpha_vec.gather(0, target.view(-1)).view(-1, 1)
                else:
                    alpha_t = torch.full_like(pt, alpha)
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss.squeeze(1)
        return loss.mean()