import torch
import numpy as np
import torch.nn as nn
# 从 torch.utils.data 导入各种数据加载器、采样器和权重采样器
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import math
from itertools import islice
import collections

# 全局设备设置：优先使用 CUDA (GPU)，否则使用 CPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""
	子集顺序采样器：
	按照给定的索引列表，顺序地、无放回地采样元素。
	用于测试集或验证集，确保每次运行的顺序固定。
	
	Arguments:
		indices (sequence): 索引序列
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		# 返回索引列表的迭代器
		return iter(self.indices)

	def __len__(self):
		# 返回索引列表的长度
		return len(self.indices)

def collate_MIL(batch):
	
	# 过滤掉所有 item[0] 不是 Tensor 的样本
	# item[0] 是特征，item[1] 是标签
	valid_batch = [item for item in batch if torch.is_tensor(item[0])] 
	
	# 如果批次中没有有效的 Tensor，返回 None，让 train_loop 跳过
	if not valid_batch:
		return None, None 

	# 对有效样本执行拼接
    # img: 所有实例特征在第 0 维拼接
	img = torch.cat([item[0] for item in valid_batch], dim = 0)
    
    # label: Slide 标签在第 0 维拼接（形成 [Actual_Batch_Size] 形状）
	label = torch.LongTensor([item[1] for item in valid_batch])
    
    # 保持原有的返回格式（img 是拼接后的实例特征，label 是 Slide 标签）
	return [img, label]



def collate_tranformer(batch):
	"""
	适用于 ViLa-MIL 或双尺度 Transformer 模型的自定义数据整理函数。
	处理四个输入（小尺度特征/坐标，大尺度特征/坐标）和一个标签。
	
	Args:
		batch (list): 包含 (img_s, coord_s, img_l, coord_l, label) 元组的列表。
		
	Returns:
		list: [img_s, coord_s, img_l, coord_l, label]
	"""
	# 小尺度图像特征（img_s）拼接
	img_s = torch.cat([item[0] for item in batch], dim = 0)
	# 小尺度坐标（coord_s）拼接
	coord_s = torch.cat([item[1] for item in batch], dim = 0)
	# 大尺度图像特征（img_l）拼接
	img_l = torch.cat([item[2] for item in batch], dim = 0)
	# 大尺度坐标（coord_l）拼接
	coord_l = torch.cat([item[3] for item in batch], dim = 0)
	# 标签（label）转换为 LongTensor
	label = torch.LongTensor([item[4] for item in batch])
	return [img_s, coord_s, img_l, coord_l, label]


def get_simple_loader(dataset, batch_size=1, num_workers=1, mode='clam'):
	"""
	获取一个简单的顺序 DataLoader。
	通常用于特征提取或不需要复杂采样的场景。
	
	Args:
		dataset (Dataset): PyTorch Dataset 对象。
		batch_size (int): 批次大小 (通常为 1，因为一个 Slide 是一个批次)。
		num_workers (int): 数据加载器的工作进程数。
		mode (str): 指定使用哪种 collate 函数 ('clam' 或 'transformer')。
	"""
	if(mode == 'transformer'):
		collate = collate_tranformer
	else:
		collate = collate_MIL # 默认为 collate_MIL

	# CUDA 相关的参数配置
	# Windows 下多进程 DataLoader 很容易出现内存峰值过高（prefetch + 多 worker 同时加载大 H5）
	# 默认将 worker 数降到 0；如需加速可在 Linux 环境再增大。
	if device.type == "cuda":
		import os
		effective_workers = 0 if os.name == 'nt' else num_workers
		kwargs = {
			'num_workers': effective_workers,
			'pin_memory': False,
		}
		if effective_workers > 0:
			kwargs.update({'prefetch_factor': 1, 'persistent_workers': False})
	else:
		kwargs = {}
	
	# 创建 DataLoader，使用顺序采样器 (SequentialSampler)
	loader = DataLoader(dataset, 
						batch_size=batch_size, 
						sampler = sampler.SequentialSampler(dataset), 
						collate_fn = collate_MIL if mode != 'transformer' else collate_tranformer, # 确保 collate_fn 被正确赋值
						**kwargs)
	return loader


def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='clam'):
	"""
	获取用于训练、验证或测试的数据加载器。
	支持随机采样、顺序采样和加权随机采样。
	
	Args:
		split_dataset (Dataset): 划分后的数据集（Train/Val/Test）。
		training (bool): 是否为训练模式 (使用随机或加权随机采样)。
		testing (bool): 是否为测试模式 (使用 SubsetSequentialSampler)。
		weighted (bool): 训练模式下是否使用加权随机采样来平衡类别。
		mode (str): 指定使用哪种 collate 函数 ('clam' 或 'transformer')。
	"""
	if(mode == 'transformer'):
		collate = collate_tranformer
	else:
		collate = collate_MIL # 默认为 collate_MIL

	# CUDA 相关的参数配置
	# Windows 下默认使用 num_workers=0 以避免 DataLoader worker 内存峰值导致的 MemoryError
	if device.type == "cuda":
		import os
		effective_workers = 0 if os.name == 'nt' else 4
		kwargs = {'num_workers': effective_workers, 'pin_memory': False}
		if effective_workers > 0:
			kwargs.update({'prefetch_factor': 1, 'persistent_workers': False})
	else:
		kwargs = {}
	
	if not testing:
		if training:
			if weighted:
				# 训练模式 + 加权：用于处理类别不平衡
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, 
									batch_size=1, 
									sampler = WeightedRandomSampler(weights, len(weights)), 
									collate_fn = collate, 
									**kwargs)
			else:
				# 训练模式 + 随机：随机打乱顺序
				loader = DataLoader(split_dataset, 
									batch_size=1, 
									sampler = RandomSampler(split_dataset), 
									collate_fn = collate, 
									**kwargs)
		else:
			# 验证模式：使用顺序采样
			loader = DataLoader(split_dataset, 
								batch_size=1, 
								sampler = SequentialSampler(split_dataset), 
								collate_fn = collate, 
								**kwargs)
	
	else:
		# 测试模式：随机选择 10% 的样本进行测试（❗注意：这段逻辑似乎有误，实际应该是选择所有测试样本）
		# 原始代码逻辑: 随机选择数据集长度的 10% 索引作为测试集。
		ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.1), replace = False)
		loader = DataLoader(split_dataset, 
							batch_size=1, 
							sampler = SubsetSequentialSampler(ids), # 使用自定义的顺序采样器
							collate_fn = collate, 
							**kwargs)

	return loader

def get_optim(model, args):
	"""
	根据参数选择并初始化优化器。
	
	Args:
		model (nn.Module): 待优化的模型。
		args: 包含优化器类型 (args.opt)、学习率 (args.lr) 和权重衰减 (args.reg) 的参数对象。
	"""
	# 过滤掉不需要梯度的参数
	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	
	if args.opt == "adam":
		optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	"""
	打印模型结构和参数数量。
	"""
	num_params = 0
	num_params_train = 0
	print(net) # 打印模型结构
	
	for param in net.parameters():
		n = param.numel() # 参数数量
		num_params += n
		if param.requires_grad:
			num_params_train += n # 可训练参数数量
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	"""
	生成自定义的 K-Fold 交叉验证分割（训练集、验证集、测试集）。
	支持类别平衡采样、自定义测试集和标签比例抽样（半监督学习）。
	
	Args:
		cls_ids (list): 每个类别中所有样本索引的列表。
		val_num (list): 每个类别中验证集的样本数量。
		test_num (list): 每个类别中测试集的样本数量。
		samples (int): 总样本数。
		n_splits (int): 需要生成的分割次数（K-Fold中的 K）。
		seed (int): 随机种子。
		label_frac (float): 训练集中使用的标签样本比例（用于半监督学习）。
		custom_test_ids (list): 预先定义的测试集索引，如果提供则不进行测试集采样。
		
	Yields:
		tuple: (sampled_train_ids, all_val_ids, all_test_ids) 分别为训练集、验证集和测试集索引。
	"""
	indices = np.arange(samples).astype(int)
	
	# 如果存在自定义测试集，从总索引中排除它们
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # 如果测试集已预定义，直接使用
			all_test_ids.extend(custom_test_ids)

		# 遍历每个类别
		for c in range(len(val_num)):
			# 获取该类别中，尚未被排除（如自定义测试集）的所有索引
			possible_indices = np.intersect1d(cls_ids[c], indices)
			
			# 随机选择验证集样本
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) 
			all_val_ids.extend(val_ids)

			# 从剩余索引中排除验证集
			remaining_ids = np.setdiff1d(possible_indices, val_ids) 

			if custom_test_ids is None:  # 如果没有预定义测试集，则采样测试集
				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				
				# 从剩余索引中排除测试集
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				# 全监督模式：所有剩余样本作为训练集
				sampled_train_ids.extend(remaining_ids)
			
			else:
				# 半监督模式：按比例抽取训练集标签
				sample_num = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				# 抽取剩余样本的前 sample_num 个作为训练集
				sampled_train_ids.extend(remaining_ids[slice_ids])

		# 返回当前折的训练、验证和测试索引
		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	"""
	获取迭代器中的第 n 个元素，或消费到第 n 个元素。
	"""
	if n is None:
		# 如果 n 为 None，消费所有元素，但只保留最近的 n=0 个元素
		return collections.deque(iterator, maxlen=0)
	else:
		# 获取第 n 个元素（0-indexed, islice 从 n 开始，next 取第一个）
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	"""
	计算分类错误率（1 - 准确率）。
	
	Args:
		Y_hat (Tensor): 预测标签。
		Y (Tensor): 真实标签。
	"""
	# 比较预测和真实标签，转换为浮点数，求均值（即准确率），然后用 1 减去
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	"""
	为类别不平衡的数据集计算样本权重（逆类频率）。
	用于 WeightedRandomSampler 以平衡训练过程。
	
	Args:
		dataset: 包含 slide_cls_ids 属性（每个类别样本索引）的数据集对象。
	"""
	N = float(len(dataset)) # 总样本数                                         
	
	# 计算每个类别的权重：总样本数 / 该类别样本数 (即逆类频率)
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	
	weight = [0] * int(N) # 初始化所有样本的权重列表                                       
	
	# 遍历所有样本，为其分配对应的类别权重
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx) # 获取样本标签                      
		weight[idx] = weight_per_class[y] # 分配权重                                 

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	"""
	初始化模型的权重。
	
	Args:
		module (nn.Module): 模型模块或层。
	"""
	for m in module.modules():
		if isinstance(m, nn.Linear):
			# 线性层使用 Xavier (Glorot) 初始化
			nn.init.xavier_normal_(m.weight)
			# 偏置项初始化为 0
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			# Batch Normalization 层，权重初始化为 1
			nn.init.constant_(m.weight, 1)
			# 偏置项初始化为 0
			nn.init.constant_(m.bias, 0)