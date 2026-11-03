"""
批处理工具模块
用于初始化和管理WSI处理的DataFrame
"""
import pandas as pd
import numpy as np
	
def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
	use_heatmap_args=False, save_patches=False):
	"""
	初始化一个pandas DataFrame，描述要处理的切片列表
	
	Args:
		slides (DataFrame or array-like): 
			包含切片ID列表的数组结构，如果是DataFrame，这些ID假设存储在'slide_id'列下
		seg_params (dict): 分割参数字典
		filter_params (dict): 过滤参数字典
		vis_params (dict): 可视化参数字典
		patch_params (dict): 补丁提取参数字典
		use_heatmap_args (bool): 是否包含热图参数（如ROI坐标）
		save_patches (bool): 是否保存补丁
		
	Returns:
		pandas.DataFrame: 初始化的DataFrame
	"""

	total = len(slides)
	# 提取切片ID
	if isinstance(slides, pd.DataFrame):
		slide_ids = slides.slide_id.values
	else:
		slide_ids = slides
	# 初始化默认字典
	default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)}

	# 如果使用热图参数，初始化空标签
	if use_heatmap_args:
		default_df_dict.update({'label': np.full((total), -1)})
	
	# 更新默认参数字典
	default_df_dict.update({
		'status': np.full((total), 'tbp'),  # 'tbp' = to be processed
		# 分割参数
		'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
		'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
		'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
		'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
		'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
		'keep_ids': np.full((total), seg_params['keep_ids']),
		'exclude_ids': np.full((total), seg_params['exclude_ids']),
		
		# 过滤参数
		'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
		'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
		'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

		# 可视化参数
		'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
		'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

		# 补丁提取参数
		'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
		'contour_fn': np.full((total), patch_params['contour_fn'])
		})

	# 如果保存补丁，添加黑白阈值参数
	if save_patches:
		default_df_dict.update({
			'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
			'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

	# 如果使用热图参数，初始化空的x,y坐标
	if use_heatmap_args:
		default_df_dict.update({'x1': np.empty((total)).fill(np.NaN), 
			'x2': np.empty((total)).fill(np.NaN), 
			'y1': np.empty((total)).fill(np.NaN), 
			'y2': np.empty((total)).fill(np.NaN)})

	# 如果输入是DataFrame，合并默认值
	if isinstance(slides, pd.DataFrame):
		temp_copy = pd.DataFrame(default_df_dict)  # 包含默认参数的临时DataFrame
		# 查找提供的DataFrame中的键
		# 如果存在，用默认值填充空字段；否则，将默认值作为新列插入
		for key in default_df_dict.keys(): 
			if key in slides.columns:
				mask = slides[key].isna()
				slides.loc[mask, key] = temp_copy.loc[mask, key]
			else:
				slides.insert(len(slides.columns), key, default_df_dict[key])
	else:
		# 如果输入不是DataFrame，创建新的DataFrame
		slides = pd.DataFrame(default_df_dict)
	
	return slides