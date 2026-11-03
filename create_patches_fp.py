"""
从全片扫描图像（WSI）中提取补丁（patches）的脚本
主要功能包括：组织分割、补丁提取和拼接可视化
(用CLAM)
运行命令：python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
具体：python create_patches_fp.py --source H:/dataset/SYSUFH_ESCC_Dataset/wsi --save_dir H:/dataset/SYSUFH_ESCC_Dataset --patch_size 1024 --step_size 1024 --patch
"""

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
import os
import numpy as np
import time
import argparse
import pandas as pd

def stitching(file_path, wsi_object, downscale = 64):
	"""
	将提取的补丁坐标拼接成热力图可视化
	
	参数:
		file_path: 补丁坐标文件路径
		wsi_object: WSI对象
		downscale: 下采样倍数，默认64
	
	返回:
		heatmap: 拼接后的热力图图像
		total_time: 拼接耗时（秒）
	"""
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(255,255,255), alpha=-1, draw_grid=True)
	total_time = time.time() - start

	return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
	"""
	对WSI进行组织分割，识别组织区域
	
	参数:
		WSI_object: WSI对象
		seg_params: 分割参数字典
		filter_params: 过滤参数字典
	
	返回:
		WSI_object: 处理后的WSI对象
		seg_time_elapsed: 分割耗时（秒）
	"""
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	"""
	从WSI中提取补丁（patches）
	
	参数:
		WSI_object: WSI对象
		**kwargs: 补丁提取的其他参数
	
	返回:
		file_path: 保存补丁坐标的文件路径
		patch_time_elapsed: 补丁提取耗时（秒）
	"""
	### Start Patch Timer
	start_time = time.time()

	# Patch
	magnification = WSI_object.wsi.properties['aperio.AppMag']
	kwargs['mag'] = str(magnification)
	file_path = WSI_object.process_contours(**kwargs)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, only_mask_save_dir, stitch_save_dir,
				  slide_name_file,
				  patch_size = 256, step_size = 256,
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
								'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False,
				  seg = False, save_mask = True,
				  stitch= False,
				  patch = False, auto_skip = True, process_list = None, uuid_name_file = None):
	"""
	主要的处理函数：对WSI进行分割和补丁提取
	
	参数:
		source: 原始WSI图像文件夹路径
		save_dir: 保存处理结果的目录
		patch_save_dir: 补丁保存目录
		mask_save_dir: 掩码保存目录
		only_mask_save_dir: 仅掩码保存目录
		stitch_save_dir: 拼接图像保存目录
		slide_name_file: 幻灯片名称文件
		patch_size: 补丁大小，默认256
		step_size: 补丁提取步长，默认256
		seg_params: 分割参数字典
		filter_params: 过滤参数字典
		vis_params: 可视化参数字典
		patch_params: 补丁提取参数字典
		patch_level: 补丁提取的层级
		use_default_params: 是否使用默认参数
		seg: 是否执行分割
		save_mask: 是否保存掩码
		stitch: 是否执行拼接
		patch: 是否提取补丁
		auto_skip: 是否自动跳过已处理的文件
		process_list: 处理列表文件路径
		uuid_name_file: UUID和名称映射文件路径
	
	返回:
		seg_times: 平均分割时间
		patch_times: 平均补丁提取时间
	"""

	# 读取UUID和名称映射文件
	all_data = np.array(pd.read_excel(uuid_name_file, engine='openpyxl', header=None))
	slides = []
	id_names = {}
	for data in all_data:
		slides.append(data[1])
		id_names[data[1]] = data[0]

	# 过滤出实际存在的幻灯片文件
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, id_names[str(slide)], slide))]
	
	# 初始化处理数据框
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	# 筛选需要处理的幻灯片
	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	# 检查是否为旧版CSV文件格式，如果是则启用兼容支持
	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
						  'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
						  'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
						  'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
						  'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	# 初始化时间统计变量
	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	# 遍历处理每个幻灯片
	for i in range(total):
		# 保存处理列表到CSV文件
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))

		# 标记当前幻灯片为已处理
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		# 如果启用自动跳过且补丁文件已存在，则跳过
		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# 初始化WSI对象
		full_path = os.path.join(source, id_names[slide], slide)
		WSI_object = WholeSlideImage(full_path)

		# 根据是否使用默认参数来设置当前参数
		if use_default_params:
			# 使用默认参数
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
		else:
			# 从数据框中读取每个幻灯片的特定参数
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}

			# 设置可视化参数
			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			# 设置过滤参数（处理旧版格式的面积参数）
			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			# 设置分割参数
			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			# 设置补丁提取参数
			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		# 自动确定可视化层级（如果未指定）
		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		# 自动确定分割层级（如果未指定）
		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		# 处理保留的组织ID列表
		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		# 处理排除的组织ID列表
		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		# 检查分割层级尺寸是否过大
		w, h = WSI_object.level_dim[current_seg_params['seg_level']]
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		# 更新数据框中的层级信息
		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		# 执行组织分割
		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

		# 保存掩码图像
		if save_mask:
			mask, only_mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id +'.jpg')
			mask.save(mask_path)
			only_mask_path = os.path.join(only_mask_save_dir, slide_id +'.png')
			only_mask.save(only_mask_path)

		# 提取补丁
		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)

		# 拼接补丁坐标生成热力图
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id +'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id +'.jpg')
				# print(stitch_path)
				heatmap.save(stitch_path)

		# 打印处理时间
		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		# 累计处理时间
		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	# 计算平均处理时间
	seg_times /= total
	patch_times /= total
	stitch_times /= total

	# 保存最终的处理列表
	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))

	return seg_times, patch_times

# 命令行参数解析
parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true',
					help='是否提取补丁')
parser.add_argument('--seg', default=False, action='store_true',
					help='是否执行分割')
parser.add_argument('--stitch', default=False, action='store_true',
					help='是否执行拼接')
parser.add_argument('--no_auto_skip', default=False, action='store_false',
					help='是否自动跳过已处理的文件')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

parser.add_argument('--slide_name_file', type=str,
					help='a file stored all slides name needed in this project')
parser.add_argument('--uuid_name_file', type=str,
					help='a file stored all slides info')

if __name__ == '__main__':
	args = parser.parse_args()

	# 设置保存目录路径
	patch_save_dir = os.path.join(args.save_dir, 'patches_' + str(args.patch_size))
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'graph_' + str(args.patch_size))
	only_mask_save_dir = os.path.join(args.save_dir, 'only_masks')

	# 处理列表文件路径
	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)
	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)

	# 组织目录字典
	directories = {'source': args.source,
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir,
				   'mask_save_dir' : mask_save_dir,
				   'only_mask_save_dir' : only_mask_save_dir,
				   'stitch_save_dir': stitch_save_dir}

	# 创建必要的目录
	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	# 设置默认参数
	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	# 如果指定了预设文件，则从预设文件加载参数
	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]

	# 组织参数字典
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
				  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	# 执行分割和补丁提取
	seg_times, patch_times = seg_and_patch(**directories, **parameters,
										   slide_name_file=args.slide_name_file,
										   patch_size = args.patch_size, step_size=args.step_size,
										   seg = args.seg,  use_default_params=False, save_mask = True,
										   stitch= args.stitch,
										   patch_level=args.patch_level, patch = args.patch,
										   process_list = process_list, auto_skip=args.no_auto_skip, uuid_name_file=args.uuid_name_file)
