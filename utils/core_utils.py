"""
核心训练和评估工具函数
包含训练循环、验证、评估等功能
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from utils.loss_utils import FocalLoss


def _rag_dspnet_class_names(task_name, n_classes):
    task_name = task_name.strip().lower() if task_name is not None else None
    task_map = {
        'task_pcr_vs_nonpcr': ['non-pCR', 'pCR'],
        'task_tcga_brca_subtyping': ['ILC', 'IDC'],
        'task_camelyon16_subtyping': ['Normal', 'Tumor'],
        'task_tcga_lung_subtyping': ['LUAD', 'LUSC'],
        'task_tcga_rcc_subtyping': ['CCRCC', 'PRCC', 'CRCC'],
    }
    if task_name in task_map:
        return task_map[task_name]
    return [f'class_{i}' for i in range(n_classes)]


class Accuracy_Logger(object):
    """
    准确率记录器
    用于记录每个类别的准确率统计信息
    """
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes  # 类别数量
        self.initialize()

    def initialize(self):
        """初始化统计数据结构"""
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        """
        记录单个样本的预测结果
        Args:
            Y_hat: 预测标签
            Y: 真实标签
        """
        # 如果预测为空，跳过记录（某些模型在训练时可能不返回预测）
        if Y_hat is None:
            return

        def _to_int(val):
            # Convert torch.Tensor, numpy arrays or scalars to Python int safely
            try:
                if isinstance(val, torch.Tensor):
                    return int(val.detach().cpu().item())
                if isinstance(val, np.ndarray):
                    return int(val.astype(int).ravel()[0])
                # handle lists/tuples or single numeric values
                if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                    arr = np.array(val)
                    return int(arr.astype(int).ravel()[0])
                return int(val)
            except Exception:
                # If conversion fails, raise to let caller decide
                raise

        try:
            Y_hat_i = _to_int(Y_hat)
            Y_i = _to_int(Y)
        except Exception:
            # 无法解析标签时跳过记录
            return

        self.data[Y_i]["count"] += 1  # 增加该类别样本计数
        self.data[Y_i]["correct"] += (Y_hat_i == Y_i)  # 如果预测正确，增加正确计数
    
    def log_batch(self, Y_hat, Y):
        """
        记录批量样本的预测结果
        Args:
            Y_hat: 预测标签数组
            Y: 真实标签数组
        """
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class  # 该类别的掩码
            self.data[label_class]["count"] += cls_mask.sum()  # 增加计数
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()  # 增加正确计数
    
    def get_summary(self, c):
        """
        获取指定类别的统计摘要
        Args:
            c: 类别索引
        Returns:
            acc: 准确率
            correct: 正确数量
            count: 总数量
        """
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """
    早停机制
    当验证损失在指定轮数内没有改善时，提前停止训练
    """
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): 在验证损失不再改善后等待的轮数，默认20
            stop_epoch (int): 最早可以停止的轮数，默认50
            verbose (bool): 是否打印详细信息，默认False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0  # 计数器
        self.best_score = None  # 最佳分数
        self.early_stop = False  # 是否早停标志
        self.val_loss_min = np.Inf  # 最小验证损失

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """
    单折训练函数
    Args:
        datasets: 数据集元组 (train_dataset, val_dataset, test_dataset)
        cur: 当前折数
        args: 训练参数
    Returns:
        results_dict: 结果字典
        test_auc: 测试集AUC
        val_auc: 验证集AUC
        test_acc: 测试集准确率
        val_acc: 验证集准确率
        each_class_acc: 每个类别的准确率
        test_f1: 测试集F1分数
    """
    print('\nTraining Fold {}!'.format(cur))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    task_name = args.task.strip().lower() if getattr(args, 'task', None) is not None else None

    if task_name == 'task_tcga_brca_subtyping':
        # TCGA-BRCA 固定使用原始 CrossEntropyLoss，不走 focal loss。
        loss_fn = nn.CrossEntropyLoss()
    elif args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = FocalLoss()
        loss_fn = loss_fn.cuda() if device.type == 'cuda' else loss_fn
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        # ViLa-MIL 内部使用 512 维表示，但当前特征提取为 1024 维，
        # 通过 raw_input_size 指示模型在前向时做一次 1024->512 线性投影
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.prototype_contrastive_weight = getattr(args, 'prototype_contrastive_weight', 0.1)
        config.prototype_contrastive_temperature = getattr(args, 'prototype_contrastive_temperature', 0.07)
        config.aux_bag_loss_weight = getattr(args, 'aux_bag_loss_weight', 0.2)
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'ViLa_MIL_Optimized':
        import ml_collections
        from models.model_ViLa_MIL_Optimized import ViLa_MIL__Optimized_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.prototype_contrastive_weight = getattr(args, 'prototype_contrastive_weight', 0.1)
        config.prototype_contrastive_temperature = getattr(args, 'prototype_contrastive_temperature', 0.07)
        config.aux_bag_loss_weight = getattr(args, 'aux_bag_loss_weight', 0.2)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = ViLa_MIL__Optimized_Model(**model_dict)

    elif args.model_type == 'ViLa_MIL_Semantic_Prototype_Guided':
        import ml_collections
        from models.model_Semantic_Prototype_guided_ViLa_MIL import Semantic_Prototype_guided_ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.prototype_prompt = getattr(args, 'prototype_prompt', None)
        config.prototype_prompt_csv = config.prototype_prompt or getattr(args, 'prototype_prompt_csv', None)
        config.prototype_contrastive_weight = getattr(args, 'prototype_contrastive_weight', 0.1)
        config.prototype_contrastive_temperature = getattr(args, 'prototype_contrastive_temperature', 0.07)
        config.aux_bag_loss_weight = getattr(args, 'aux_bag_loss_weight', 0.2)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = Semantic_Prototype_guided_ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'DSP_ViLa_MIL':
        import ml_collections
        from models.model_DSP_ViLa_MIL import DSP_ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.prototype_prompt = getattr(args, 'prototype_prompt', None)
        config.prototype_prompt_csv = config.prototype_prompt or getattr(args, 'prototype_prompt_csv', None)
        config.text_backend = getattr(args, 'text_backend', 'open_clip')
        config.conch_model_cfg = getattr(args, 'conch_model_cfg', 'conch_ViT-B-16')
        config.conch_checkpoint_path = getattr(args, 'conch_checkpoint_path', None)
        config.prototype_contrastive_weight = getattr(args, 'prototype_contrastive_weight', 0.1)
        config.prototype_contrastive_temperature = getattr(args, 'prototype_contrastive_temperature', 0.07)
        config.aux_bag_loss_weight = getattr(args, 'aux_bag_loss_weight', 0.2)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = DSP_ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'DSPG_ViLa_MIL':
        import ml_collections
        from models.model_DSPG_ViLa_MIL import DSPG_ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.prototype_prompt = getattr(args, 'prototype_prompt', None)
        config.prototype_prompt_csv = config.prototype_prompt or getattr(args, 'prototype_prompt_csv', None)
        config.text_backend = getattr(args, 'text_backend', 'open_clip')
        config.conch_model_cfg = getattr(args, 'conch_model_cfg', 'conch_ViT-B-16')
        config.conch_checkpoint_path = getattr(args, 'conch_checkpoint_path', None)
        config.prototype_contrastive_weight = getattr(args, 'prototype_contrastive_weight', 0.1)
        config.prototype_contrastive_temperature = getattr(args, 'prototype_contrastive_temperature', 0.07)
        config.aux_bag_loss_weight = getattr(args, 'aux_bag_loss_weight', 0.2)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = DSPG_ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'ViLa_MIL_Mammoth_Optimized':
        import ml_collections
        from models.model_ViLa_MIL_Mammoth_Optimized import ViLa_MIL_Mammoth_Optimized_Model
        config = ml_collections.ConfigDict()
        # 与 ViLa-MIL / ViLa_MIL_Mammoth 保持一致：内部 512 维，原始 1024 维
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        # MAMMOTH rank 配置（训练侧）
        config.mammoth_auto_rank = getattr(args, 'mammoth_auto_rank', False)
        config.mammoth_lora_rank = getattr(args, 'mammoth_lora_rank', 4)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = ViLa_MIL_Mammoth_Optimized_Model(**model_dict)

    elif args.model_type == 'ViLa_MIL_Mammoth_DynaMoE':
        import ml_collections
        from models.model_ViLa_MIL_Mammoth_DynaMoE import ViLa_MIL_Mammoth_DynaMoE
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.mammoth_auto_rank = getattr(args, 'mammoth_auto_rank', True)
        config.mammoth_lora_rank = getattr(args, 'mammoth_lora_rank', 4)
        config.mammoth_num_experts = getattr(args, 'mammoth_num_experts', 16)
        config.mammoth_num_slots = getattr(args, 'mammoth_num_slots', 8)
        config.mammoth_num_heads = getattr(args, 'mammoth_num_heads', 8)
        config.mammoth_slot_dim = getattr(args, 'mammoth_slot_dim', 256)
        config.mammoth_dropout = getattr(args, 'mammoth_dropout', 0.1)
        config.mammoth_slot_dropout = getattr(args, 'mammoth_slot_dropout', 0.0)
        config.mammoth_share_lora_weights = getattr(args, 'mammoth_share_lora_weights', True)
        config.mammoth_use_layernorm = getattr(args, 'mammoth_use_layernorm', True)
        config.mammoth_routing_temperature = getattr(args, 'mammoth_routing_temperature', 1.0)
        config.mammoth_routing_balance_weight = getattr(args, 'mammoth_routing_balance_weight', 0.0)
        config.mammoth_routing_entropy_weight = getattr(args, 'mammoth_routing_entropy_weight', 0.0)
        config.mammoth_routing_aux_weight = getattr(args, 'mammoth_routing_aux_weight', 0.01)
        config.mammoth_residual_dropout = getattr(args, 'mammoth_residual_dropout', 0.1)
        config.mammoth_noise_epsilon = getattr(args, 'mammoth_noise_epsilon', 0.1)
        config.mammoth_noise_epochs = getattr(args, 'mammoth_noise_epochs', 5)
        config.mammoth_warmup_ratio = getattr(args, 'mammoth_warmup_ratio', 0.2)
        model_dict = {'config': config, 'num_classes': args.n_classes, 'max_epochs': args.max_epochs}
        model = ViLa_MIL_Mammoth_DynaMoE(**model_dict)

    elif args.model_type == 'ViLa_MIL_TextGammaBeta':
        import ml_collections
        from models.model_ViLa_MIL_TextGammaBeta import ViLa_MIL_TextGammaBeta_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.text_backend = getattr(args, 'text_backend', 'open_clip')
        config.text_gamma_beta_position = getattr(args, 'text_gamma_beta_position', 'post_ln_final')
        config.text_gamma_init_mean = getattr(args, 'text_gamma_init_mean', 1.0)
        config.text_gamma_init_std = getattr(args, 'text_gamma_init_std', 0.02)
        config.text_beta_init_mean = getattr(args, 'text_beta_init_mean', 0.0)
        config.text_beta_init_std = getattr(args, 'text_beta_init_std', 0.02)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = ViLa_MIL_TextGammaBeta_Model(**model_dict)

    elif args.model_type == 'RAG_DSPNet':
        import ml_collections
        from models.AgentViLa import RAGDSPNet, RAGDSPNetConfig
        config = ml_collections.ConfigDict()
        config.input_dim = 1024
        config.feature_dim = getattr(args, 'rag_feature_dim', 512)
        config.hidden_dim = getattr(args, 'rag_hidden_dim', 256)
        config.num_classes = args.n_classes
        config.navigation_max_steps = getattr(args, 'rag_navigation_max_steps', 4)
        config.navigation_use_coords = getattr(args, 'rag_navigation_use_coords', True)
        config.filter_target_keep_ratio = getattr(args, 'rag_filter_target_keep_ratio', 0.6)
        config.filter_min_keep = getattr(args, 'rag_filter_min_keep', 1)
        config.prototype_max_count = getattr(args, 'rag_prototype_max_count', 8)
        config.prototype_min_count = getattr(args, 'rag_prototype_min_count', 4)
        config.class_names = _rag_dspnet_class_names(task_name, args.n_classes)
        config.llm_model_name = getattr(args, 'rag_llm_model_name', None)
        config.prototype_llm_model_name = getattr(args, 'rag_prototype_llm_model_name', None)
        config.diagnosis_llm_model_name = getattr(args, 'rag_diagnosis_llm_model_name', None)
        config.llm_backend = getattr(args, 'rag_llm_backend', 'auto')
        config.ollama_base_url = getattr(args, 'rag_ollama_base_url', 'http://127.0.0.1:11434')
        config.ollama_timeout = getattr(args, 'rag_ollama_timeout', 120.0)
        config.ollama_num_predict = getattr(args, 'rag_ollama_num_predict', 256)
        config.diagnosis_use_llm = getattr(args, 'rag_diagnosis_use_llm', False)
        config.prototype_use_llm = getattr(args, 'rag_prototype_use_llm', False)
        config.report_loss_weight = getattr(args, 'rag_report_loss_weight', 0.0)
        config.low_to_high_k = getattr(args, 'rag_low_to_high_k', 8)
        config.nav_entropy_weight = getattr(args, 'rag_nav_entropy_weight', 0.0)
        config.filter_entropy_weight = getattr(args, 'rag_filter_entropy_weight', 0.0)
        config.filter_aux_weight = getattr(args, 'rag_filter_aux_weight', 0.0)
        config.bag_dropout = getattr(args, 'rag_bag_dropout', 0.0)
        model = RAGDSPNet(RAGDSPNetConfig(**config.to_dict()))

    elif args.model_type == 'ViLa_MIL_Mammoth_LinearReplace':
        import ml_collections
        from models.model_ViLa_MIL_Mammoth_LinearReplace import ViLa_MIL_Mammoth_LinearReplace_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.raw_input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.mammoth_num_experts = getattr(args, 'mammoth_num_experts', 16)
        config.mammoth_num_slots = getattr(args, 'mammoth_num_slots', 8)
        config.mammoth_num_heads = getattr(args, 'mammoth_num_heads', 8)
        config.mammoth_dropout = getattr(args, 'mammoth_dropout', 0.1)
        config.mammoth_slot_dropout = getattr(args, 'mammoth_slot_dropout', 0.0)
        config.mammoth_lora_rank = getattr(args, 'mammoth_lora_rank', 4)
        config.mammoth_auto_rank = getattr(args, 'mammoth_auto_rank', True)
        config.mammoth_share_lora_weights = getattr(args, 'mammoth_share_lora_weights', True)
        config.mammoth_use_layernorm = getattr(args, 'mammoth_use_layernorm', True)
        config.mammoth_routing_temperature = getattr(args, 'mammoth_routing_temperature', 1.0)
        config.mammoth_routing_balance_weight = getattr(args, 'mammoth_routing_balance_weight', 0.0)
        config.mammoth_routing_entropy_weight = getattr(args, 'mammoth_routing_entropy_weight', 0.0)
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = ViLa_MIL_Mammoth_LinearReplace_Model(**model_dict)


    elif args.model_type == 'maple':
        # MAPLE: 双尺度 transformer 视觉-语言 MIL 模型
        from models.maple.model import MAPLE
        # 将命令行参数映射到 MAPLE 期望的属性名
        # 直接在 args 上挂所需字段，便于 MAPLE 使用
        args.base_model = getattr(args, 'maple_base_model', 'CLIP')
        args.text_path = args.maple_text_path
        args.weight = getattr(args, 'maple_weight', 0.5)
        args.pos_ratio = getattr(args, 'maple_pos_ratio', 0.25)
        args.neg_ratio = getattr(args, 'maple_neg_ratio', 0.25)
        args.attr_n_ctx = getattr(args, 'maple_attr_n_ctx', 4)
        args.bagLevel_n_ctx = getattr(args, 'maple_bag_n_ctx', 4)
        args.attr_edge_topk = getattr(args, 'maple_attr_edge_topk', 3)
        args.all_ctx_trainable = getattr(args, 'maple_all_ctx_trainable', True)
        args.csc = getattr(args, 'maple_csc', True)
        args.p_drop_out = getattr(args, 'maple_p_drop_out', 0.0)
        args.p_bag_drop_out = getattr(args, 'maple_p_bag_drop_out', 0.0)
        # 原始 patch 特征为 1024 维, MAPLE 内部在 CLIP 空间 512 维工作
        # 通过 raw_input_dim 指示模型在前向时做一次 1024->512 投影
        args.raw_input_dim = 1024
        model = MAPLE(args)

    elif args.model_type == 'FOCUS':
        # FOCUS: 只使用大尺度特征的 transformer 模型
        import ml_collections
        from models.FOCUS import FOCUS
        config = ml_collections.ConfigDict()
        config.window_size = getattr(args, 'focus_window_size', 8)
        config.sim_threshold = getattr(args, 'focus_sim_threshold', 0.5)
        # FOCUS 内部在 512 维空间中工作, 但当前大尺度特征为 1024 维,
        # 通过 raw_input_size 指示模型在前向时做一次 1024->512 线性投影
        config.input_size = getattr(args, 'focus_input_size', 512)
        config.raw_input_size = 1024
        config.max_context_length = getattr(args, 'focus_max_context_length', 256)
        config.text_prompt = args.text_prompt
        model = FOCUS(config=config, num_classes=args.n_classes)

    elif args.model_type == 'TOP_MIL':
        # TOP-MIL: 基于 CLIP + learnable prompt 的单尺度 MIL 模型
        from models.top.TOP_MIL import build_topmil_model
        # 根据类别数自动生成类名，主要用于文本提示，名称本身不影响标签编码
        bag_classnames = [f'class_{i}' for i in range(args.n_classes)]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = build_topmil_model(bag_classnames=bag_classnames, device=device)

    elif args.model_type == 'abmil':
        # 新增ABMIL模型
        from models.abmil import Attention, GatedAttention
        abmil_type = getattr(args, 'abmil_type', 'attention')
        if abmil_type == 'gated_attention':
            # ABMIL 使用单尺度 1024 维特征（与 CLAM 等模型一致）
            model = GatedAttention(feature_dim=1024)
        else:
            model = Attention(feature_dim=1024)
        # ABMIL是二分类模型，调整为多分类如果需要
        if args.n_classes > 2:
        # 如果需要多分类，可能需要修改ABMIL模型
            raise NotImplementedError("ABMIL currently supports binary classification")

    elif args.model_type == 'dsmil':
        # DSMIL模型
        from models.dsmil import MILNet, IClassifier, BClassifier
        # DSMIL 同样基于 1024 维实例特征
        i_classifier = IClassifier(nn.Identity(), 1024, args.n_classes)
        b_classifier = BClassifier(1024, args.n_classes)
        model = MILNet(i_classifier, b_classifier)
        if args.n_classes > 2:
            raise NotImplementedError("DSMIL currently supports binary classification")

    elif args.model_type == 'transmil':
        # 新增TransMIL模型
        from models.transmil import TransMIL
        model = TransMIL(n_classes=args.n_classes)
        if args.n_classes > 2:
            raise NotImplementedError("TransMIL currently supports binary classification")


    elif args.model_type == 'clam_sb':
        from models.clam import CLAM_SB
        # CLAM 默认使用 1024 维输入特征
        model = CLAM_SB(size_arg=args.model_size, n_classes=args.n_classes, embed_dim=1024)

    elif args.model_type == 'clam_mb':
        from models.clam import CLAM_MB
        model = CLAM_MB(size_arg=args.model_size, n_classes=args.n_classes, embed_dim=1024)

    elif args.model_type == 'maxpooling':
        from models.maxpooling import MIL_MaxPooling
        # Max/Mean pooling 也基于 1024 维实例特征
        model = MIL_MaxPooling(input_dim=1024, n_classes=args.n_classes)

    elif args.model_type == 'meanpooling':
        from models.meanpooling import MIL_MeanPooling
        model = MIL_MeanPooling(input_dim=1024, n_classes=args.n_classes)

    if model is None:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=80, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    # 初始化混合精度训练的GradScaler（可选）
    if getattr(args, 'fp16', False):
        scaler = GradScaler()
        print('Mixed Precision Training: Enabled')
    else:
        scaler = None
        print('Mixed Precision Training: Disabled')

    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, scaler)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir, args.mode, args.model_type)
        if stop: 
            break

    if args.early_stopping: 
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader, args.n_classes, args.model_type)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(val_error, val_auc, val_f1))

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(args.mode, model, test_loader, args.n_classes, args.model_type)
    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_f1))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        acc_str = 'N/A' if acc is None else f"{acc:.4f}"
        print('class {}: acc {}, correct {}/{}'.format(i, acc_str, correct, count))

        if writer and acc is not None:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, each_class_acc, test_f1


def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, scaler = None):
    """
    训练循环（支持可选的混合精度训练）
    Args:
        args: 训练参数
        epoch: 当前轮数
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        n_classes: 类别数量
        writer: TensorBoard写入器（可选）
        loss_fn: 损失函数（可选）
        scaler: GradScaler用于混合精度训练（可选，None表示不使用FP16）
    """

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    train_cls_loss = 0.
    
    # 是否使用混合精度训练
    use_amp = scaler is not None

    print('\n')
    if args.mode == 'mil':
        for batch_idx, (features, label) in enumerate(loader):
            
            if features is None or not isinstance(features, torch.Tensor):
            # 打印警告或跳过当前批次
                print(f"Warning: Skipping empty or invalid batch {batch_idx} in epoch {epoch}.")
                continue
            features, label = features.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            if args.model_type == 'abmil':
                # For ABMIL, use calculate_objective for loss
                if use_amp:
                    with autocast():
                        loss, A = model.calculate_objective(features, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, A = model.calculate_objective(features, label)
                    loss.backward()
                    optimizer.step()

                # Get predictions
                with torch.no_grad():
                    Y_prob, Y_hat, _ = model(features)
                    Y_prob = Y_prob.squeeze()
                    
            elif args.model_type == 'dsmil':
                if use_amp:
                    with autocast():
                        _, prediction_bag, _, _ = model(features)
                        logits = prediction_bag
                        label_processed = label.long()
                        if label.numel() == 0:
                            label_processed = torch.tensor([0], dtype=torch.long)
                        loss = loss_fn(logits, label_processed)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _, prediction_bag, _, _ = model(features)
                    logits = prediction_bag
                    label_processed = label.long()
                    if label.numel() == 0:
                        label_processed = torch.tensor([0], dtype=torch.long)
                    loss = loss_fn(logits, label_processed)
                    loss.backward()
                    optimizer.step()

                Y_prob = F.softmax(prediction_bag.squeeze(0), dim=0)
                Y_hat = torch.argmax(Y_prob)
                Y_hat = Y_hat.unsqueeze(0)

            elif args.model_type == 'transmil':
                if use_amp:
                    with autocast():
                        results = model(data=features)
                        logits = results['logits']
                        label_processed = label.long()
                        if label.numel() == 0:
                            label_processed = torch.tensor([0], dtype=torch.long)
                        loss = loss_fn(logits, label_processed)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    results = model(data=features)
                    logits = results['logits']
                    label_processed = label.long()
                    if label.numel() == 0:
                        label_processed = torch.tensor([0], dtype=torch.long)
                    loss = loss_fn(logits, label_processed)
                    loss.backward()
                    optimizer.step()

                Y_prob = results['Y_prob']
                Y_hat = results['Y_hat']
                Y_hat = Y_hat.unsqueeze(0)

            elif args.model_type == 'clam_sb' or args.model_type == 'clam_mb':
                if use_amp:
                    with autocast():
                        logits, Y_prob, Y_hat, A_raw, results_dict = model(features, label=label, instance_eval=True)
                        Y_hat = Y_hat.squeeze(0)
                        bag_loss = loss_fn(logits, label)
                        instance_loss = results_dict.get('instance_loss', 0.0)
                        loss = bag_loss + instance_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, Y_prob, Y_hat, A_raw, results_dict = model(features, label=label, instance_eval=True)
                    Y_hat = Y_hat.squeeze(0)
                    bag_loss = loss_fn(logits, label)
                    instance_loss = results_dict.get('instance_loss', 0.0)
                    loss = bag_loss + instance_loss
                    loss.backward()
                    optimizer.step()

            elif args.model_type == 'maxpooling' or args.model_type == 'meanpooling':
                if use_amp:
                    with autocast():
                        logits, Y_prob, Y_hat, A = model(features)
                        Y_prob = Y_prob.squeeze()
                        loss = loss_fn(logits, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, Y_prob, Y_hat, A = model(features)
                    Y_prob = Y_prob.squeeze()
                    loss = loss_fn(logits, label)
                    loss.backward()
                    optimizer.step()

            elif args.model_type == 'ViLa_MIL_Mammoth_DynaMoE':
                if use_amp:
                    with autocast():
                        Y_prob, Y_hat, loss, aux_loss = model(
                            data_s, coord_s, data_l, coords_l, label, current_epoch=epoch
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    Y_prob, Y_hat, loss, aux_loss = model(
                        data_s, coord_s, data_l, coords_l, label, current_epoch=epoch
                    )
                    loss.backward()
                    optimizer.step()

                train_cls_loss += aux_loss

            elif args.model_type == 'TOP_MIL':
                # TOP-MIL: MIL_CLIP 前向返回 (logits, A)，在这里计算 Y_prob 与 Y_hat
                if use_amp:
                    with autocast():
                        logits, A = model(features)
                        Y_prob = F.softmax(logits, dim=1)
                        Y_hat = torch.argmax(Y_prob, dim=1)
                        Y_prob = Y_prob.squeeze()
                        loss = loss_fn(logits, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, A = model(features)
                    Y_prob = F.softmax(logits, dim=1)
                    Y_hat = torch.argmax(Y_prob, dim=1)
                    Y_prob = Y_prob.squeeze()
                    loss = loss_fn(logits, label)
                    loss.backward()
                    optimizer.step()

            if Y_hat is not None:
                acc_logger.log(Y_hat, label)
                error = calculate_error(Y_hat, label)
                train_error += error
            loss_value = loss.item()
            train_loss += loss_value
    else:
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
            
            optimizer.zero_grad()

            # transformer 模式下，针对 ViLa_Diff_MIL 单独处理（返回diff_loss和cls_loss）
            if args.model_type == 'ViLa_Diff_MIL':
                if use_amp:
                    with autocast():
                        Y_prob, Y_hat, diff_loss, cls_loss = model(data_s, coord_s, data_l, coords_l, label)
                    # 联合训练: diff_loss + cls_loss
                    loss = diff_loss + cls_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    Y_prob, Y_hat, diff_loss, cls_loss = model(data_s, coord_s, data_l, coords_l, label)
                    # 联合训练: diff_loss + cls_loss
                    loss = diff_loss + cls_loss
                    loss.backward()
                    optimizer.step()

                # 记录分类损失（仅用于日志）
                train_cls_loss += cls_loss.item()

            else:
                if use_amp:
                    with autocast():
                        if args.model_type == 'ViLa_MIL_Mammoth_DynaMoE':
                            _, Y_hat, loss, aux_loss = model(
                                data_s, coord_s, data_l, coords_l, label, current_epoch=epoch
                            )
                        else:
                            _, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.model_type == 'ViLa_MIL_Mammoth_DynaMoE':
                        _, Y_hat, loss, aux_loss = model(
                            data_s, coord_s, data_l, coords_l, label, current_epoch=epoch
                        )
                    else:
                        _, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)
                    loss.backward()
                    optimizer.step()

            if Y_hat is not None:
                acc_logger.log(Y_hat, label)
                error = calculate_error(Y_hat, label)
                train_error += error
            loss_value = loss.item()
            train_loss += loss_value

    train_loss /= len(loader)
    train_error /= len(loader)
    if args.model_type == 'ViLa_Diff_MIL':
        train_cls_loss /= len(loader)
        print('Epoch: {}, train_diff_loss: {:.4f}, train_cls_loss: {:.4f}, train_error: {:.4f}'.format(
            epoch, train_loss, train_cls_loss, train_error))
    elif args.model_type == 'ViLa_MIL_Mammoth_DynaMoE':
        train_cls_loss /= len(loader)
        print('Epoch: {}, train_loss: {:.4f}, routing_aux_loss: {:.4f}, train_error: {:.4f}'.format(
            epoch, train_loss, train_cls_loss, train_error))
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_str = 'N/A' if acc is None else f"{acc:.4f}"
        print('class {}: acc {}, correct {}/{}'.format(i, acc_str, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        # 针对 ViLa_Diff_MIL，train_loss 表示扩散损失
        if args.model_type == 'ViLa_Diff_MIL':
            writer.add_scalar('train/diff_loss', train_loss, epoch)
            writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
        elif args.model_type == 'ViLa_MIL_Mammoth_DynaMoE':
            writer.add_scalar('train/routing_aux_loss', train_cls_loss, epoch)
            writer.add_scalar('train/loss', train_loss, epoch)
        else:
            writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, mode='transformer', model_type='abmil'):
    """
    验证函数
    Args:
        cur: 当前折数
        epoch: 当前轮数
        model: 模型
        loader: 验证数据加载器
        n_classes: 类别数量
        early_stopping: 早停对象（可选）
        writer: TensorBoard写入器（可选）
        loss_fn: 损失函数（可选）
        results_dir: 结果保存目录（可选）
        mode: 模式 ('transformer' 或 'mil')
    Returns:
        bool: 是否应该停止训练
    """
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    # 使用列表收集有效样本的概率和标签，避免因为跳过 batch
    # 导致预分配数组中残留默认值，从而干扰 AUC 计算
    all_probs = []   # 对于二分类存储正类概率，对于多分类存储全类概率向量
    all_labels = []  # 对应的标签

    all_pred = []
    all_label = []
    with torch.no_grad():
        if mode == 'mil':
            for batch_idx, (features, label) in enumerate(loader):
                if features is None or not isinstance(features, torch.Tensor):
                    print(f"Warning: Skipping invalid batch {batch_idx} in validation.")
                    continue
                features, label = features.to(device, non_blocking=True), label.to(device, non_blocking=True)
                if model_type == 'abmil':
                    Y_prob, Y_hat, A = model(features)
                    Y_prob = Y_prob.squeeze()  # to scalar for prob
                    # Y_hat keep as [1]
                    loss = 0.  # Placeholder
                elif model_type == 'dsmil':
                    _, prediction_bag, _, _ = model(features)
                    logits = prediction_bag.squeeze(0)
                    Y_prob = F.softmax(logits, dim=0)
                    Y_hat = torch.argmax(logits)
                    Y_hat = Y_hat.unsqueeze(0)
                    label_processed = label.long()
                    if label.numel() == 0:
                        label_processed = torch.tensor([0], dtype=torch.long)
                    loss = loss_fn(prediction_bag, label_processed).item()

                elif model_type == 'transmil':
                    results = model(data=features)
                    logits = results['logits'].squeeze(0)
                    Y_prob = results['Y_prob']
                    Y_hat = results['Y_hat']
                    Y_hat = Y_hat.unsqueeze(0)
                    label_processed = label.long()
                    if label.numel() == 0:
                        label_processed = torch.tensor([0], dtype=torch.long)
                    loss = loss_fn(results['logits'], label_processed).item()

                elif model_type == 'clam_sb' or model_type == 'clam_mb':
                    logits, Y_prob, Y_hat, A_raw, results_dict = model(features)
                    Y_hat = Y_hat.squeeze(0)
                    loss = 0.  # Placeholder, CLAM may need proper loss calculation

                elif model_type == 'maxpooling' or model_type == 'meanpooling':
                    logits, Y_prob, Y_hat, A = model(features)
                    Y_prob = Y_prob.squeeze()
                    loss = 0.  # Placeholder

                elif model_type == 'TOP_MIL':
                    # TOP-MIL: MIL_CLIP 返回 (logits, A)，此处计算概率和预测
                    logits, A = model(features)
                    Y_prob = F.softmax(logits, dim=1)
                    Y_hat = torch.argmax(Y_prob, dim=1)
                    Y_prob = Y_prob.squeeze()
                    loss = 0.

                elif model_type == 'ViLa_MIL_Mammoth_DynaMoE':
                    Y_prob, Y_hat, loss, aux_loss = model(
                        data_s, coord_s, data_l, coords_l, label
                    )


                if Y_hat is not None:
                    acc_logger.log(Y_hat, label)
                    error = calculate_error(Y_hat, label)
                    val_error += error
                    all_pred.append(Y_hat.cpu())
                    all_label.append(label.cpu().unsqueeze(0))

                # 统一收集二分类的“正类概率”
                prob_np = Y_prob.detach().cpu().numpy()
                if n_classes == 2:
                    # 各模型输出形状不同：
                    # - 标量 Sigmoid 输出 (ABMIL)
                    # - 形如 [2] 或 [1,2] 的 softmax 输出
                    if prob_np.ndim == 0:
                        pos_prob = float(prob_np)
                    else:
                        flat = prob_np.reshape(-1)
                        # 默认取最后一个元素作为正类概率
                        pos_prob = float(flat[-1])
                    all_probs.append(pos_prob)
                else:
                    all_probs.append(prob_np.reshape(-1))
                all_labels.append(int(label.item()))
                val_loss += loss
        else:
            for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
                data_s, coord_s, data_l, coords_l, label = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                      data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True), \
                                                                      label.to(device, non_blocking=True)
                Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

                if Y_hat is not None:
                    acc_logger.log(Y_hat, label)
                    error = calculate_error(Y_hat, label)
                    val_error += error
                    all_pred.append(Y_hat.cpu())
                    all_label.append(label.cpu())

                prob_np = Y_prob.detach().cpu().numpy()
                if n_classes == 2:
                    flat = prob_np.reshape(-1)
                    # transformer 模型一般输出 [1, 2]，这里取第 2 个元素作为正类概率
                    pos_prob = float(flat[-1]) if flat.size > 1 else float(flat[0])
                    all_probs.append(pos_prob)
                else:
                    all_probs.append(prob_np.reshape(-1))
                all_labels.append(int(label.item()))
                val_loss += loss.item()

    val_error /= len(loader)
    val_loss /= len(loader)
    # 处理 F1：将 Tensor 列表合并为 numpy
    if len(all_label) > 0:
        all_label_np = torch.cat(all_label, dim=0).numpy().astype(np.int64)
        all_pred_np = torch.cat(all_pred, dim=0).numpy().astype(np.int64)
        val_f1 = f1_score(all_label_np, all_pred_np, average='macro')
    else:
        val_f1 = 0.0

    # 处理 AUC：使用仅包含有效样本的 all_labels/all_probs
    all_labels_np = np.array(all_labels)
    if n_classes == 2:
        all_probs_np = np.array(all_probs)
        # 需要同时包含两个类别才能计算 AUC
        if np.unique(all_labels_np).size > 1:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        else:
            auc = 0.5
    else:
        if len(all_probs) > 0:
            all_probs_np = np.vstack(all_probs)
            if np.unique(all_labels_np).size > 1:
                auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr')
            else:
                auc = 0.5
        else:
            auc = 0.5
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(val_loss, val_error, auc, val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_str = 'N/A' if acc is None else f"{acc:.4f}"
        print('class {}: acc {}, correct {}/{}'.format(i, acc_str, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(mode, model, loader, n_classes, model_type='abmil'):
    """
    结果汇总函数
    在测试集或验证集上评估模型并返回详细结果
    Args:
        mode: 模式（'transformer'等）
        model: 模型
        loader: 数据加载器
        n_classes: 类别数量
    Returns:
        patient_results: 每个样本的详细结果
        test_error: 测试错误率
        auc: AUC分数
        acc_logger: 准确率记录器
        test_f1: F1分数
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    all_pred = []
    all_label = []

    # 使用列表收集有效样本，避免因为跳过 batch 导致预分配数组中残留默认值
    all_probs = []   # 对于二分类存储 [p0, p1]，多分类存储全类概率向量
    all_labels = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

            if Y_hat is not None:
                acc_logger.log(Y_hat, label)
                error = calculate_error(Y_hat, label)
                test_error += error
                all_pred.append(Y_hat.cpu())
                all_label.append(label.cpu())
            prob_np = Y_prob.detach().cpu().numpy()
            prob_vec = prob_np.reshape(-1)

            if n_classes == 2:
                # 二分类：显式构造 [p0, p1]，便于后续使用第 2 维作为正类概率
                if prob_vec.size == 1:
                    p1 = float(prob_vec[0])
                else:
                    p1 = float(prob_vec[-1])
                probs = np.array([1.0 - p1, p1], dtype=float)
            else:
                probs = prob_vec

            all_probs.append(probs)
            all_labels.append(int(label.item()))

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        # 计算 transformer 模式下的整体指标
        if len(all_pred) > 0:
            test_error /= len(loader)
            # 将 Tensor 列表合并并转为 numpy int64
            all_label_np = torch.cat(all_label, dim=0).numpy().astype(np.int64)
            all_pred_np = torch.cat(all_pred, dim=0).numpy().astype(np.int64)
            test_f1 = f1_score(all_label_np, all_pred_np, average='macro')
        else:
            # 若没有有效预测，则错误率和F1置为0
            test_error = 0.0
            test_f1 = 0.0

        # 计算 AUC（使用仅包含有效样本的列表）
        if len(all_probs) > 0:
            all_probs_np = np.vstack(all_probs)
            all_labels_np = np.array(all_labels)

            if n_classes == 2:
                # 需要两个类别都出现才能计算 AUC
                if np.unique(all_labels_np).size > 1:
                    auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
                else:
                    auc = 0.5
            else:
                if np.unique(all_labels_np).size > 1:
                    auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr')
                else:
                    auc = 0.5
        else:
            auc = 0.5

        return patient_results, test_error, auc, acc_logger, test_f1

    elif mode == 'mil':
        for batch_idx, (features, label) in enumerate(loader):
            if features is None or not isinstance(features, torch.Tensor):
                print(f"Warning: Skipping invalid batch {batch_idx} in summary.")
                continue
            features, label = features.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                Y_hat = None
                Y_prob = None
                if model_type == 'abmil':
                    Y_prob, Y_hat, A = model(features)
                elif model_type == 'dsmil':
                    _, prediction_bag, _, _ = model(features)
                    logits = prediction_bag.squeeze(0)
                    Y_prob = F.softmax(logits, dim=0)
                    Y_hat = torch.argmax(Y_prob)
                    Y_hat = Y_hat.unsqueeze(0)

                elif model_type == 'transmil':
                    results = model(data=features)
                    Y_prob = results['Y_prob']
                    Y_hat = results['Y_hat']
                    Y_hat = Y_hat.unsqueeze(0)

                elif model_type == 'clam_sb' or model_type == 'clam_mb':
                    logits, Y_prob, Y_hat, A_raw, results_dict = model(features)
                    Y_hat = Y_hat.squeeze(0)

                elif model_type == 'maxpooling' or model_type == 'meanpooling':
                    logits, Y_prob, Y_hat, A = model(features)
                    Y_prob = Y_prob.squeeze()

                elif model_type == 'mil':
                    Y_prob, Y_hat, A = model(features)
                    Y_prob = Y_prob.squeeze()

                elif model_type == 'TOP_MIL':
                    # TOP-MIL: MIL_CLIP 返回 (logits, A)，在此计算概率和预测
                    logits, A = model(features)
                    Y_prob = F.softmax(logits, dim=1)
                    Y_hat = torch.argmax(Y_prob, dim=1)

                elif model_type == 'ViLa_MIL_Mammoth_DynaMoE':
                    Y_prob, Y_hat, loss, aux_loss = model(
                        features, label
                    )

            # 统一 Y_prob 形状并构造概率向量
            probs = None
            if Y_prob is not None:
                prob_np = Y_prob.detach().cpu().numpy()
                prob_vec = prob_np.reshape(-1)
                if n_classes == 2:
                    if prob_vec.size == 1:
                        p1 = float(prob_vec[0])
                    else:
                        p1 = float(prob_vec[-1])
                    probs = np.array([1.0 - p1, p1], dtype=float)
                else:
                    probs = prob_vec

            if Y_hat is not None:
                acc_logger.log(Y_hat, label)
                error = calculate_error(Y_hat, label)
                test_error += error
                all_pred.append(Y_hat.cpu())
                all_label.append(label.cpu().unsqueeze(0))

            if probs is not None:
                all_probs.append(probs)
                all_labels.append(int(label.item()))

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        # 归一化错误率
        if len(loader) > 0:
            test_error /= len(loader)
        else:
            test_error = 0.0

        # 转换为 numpy 计算 F1
        if len(all_label) > 0:
            all_label_np = torch.cat(all_label, dim=0).numpy().astype(np.int64)
            all_pred_np = torch.cat(all_pred, dim=0).numpy().astype(np.int64)
            test_f1 = f1_score(all_label_np, all_pred_np, average='macro')
        else:
            test_f1 = 0.0

        # 计算 AUC
        if len(all_probs) > 0:
            all_probs_np = np.vstack(all_probs)
            all_labels_np = np.array(all_labels)

            if n_classes == 2:
                if np.unique(all_labels_np).size > 1:
                    auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
                else:
                    auc = 0.5
            else:
                aucs = []
                binary_labels = label_binarize(all_labels_np, classes=[i for i in range(n_classes)])
                for class_idx in range(n_classes):
                    if class_idx in all_labels_np:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs_np[:, class_idx])
                        aucs.append(calc_auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))

                auc = np.nanmean(np.array(aucs))
        else:
            auc = 0.5

        return patient_results, test_error, auc, acc_logger, test_f1

