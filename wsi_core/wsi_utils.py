"""
WSI工具函数模块
提供补丁过滤、HDF5操作、坐标生成、热图可视化等功能
"""
import h5py
import numpy as np
import os
from wsi_core.util_classes import Mosaic_Canvas
from PIL import Image
import math
import cv2

def isWhitePatch(patch, satThresh=5):
    """
    检查补丁是否为白色（低饱和度）
    
    Args:
        patch: 补丁图像（RGB格式）
        satThresh: 饱和度阈值
        
    Returns:
        bool: True表示是白色补丁
    """
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    """
    检查补丁是否为黑色（低亮度）
    
    Args:
        patch: 补丁图像（RGB格式）
        rgbThresh: RGB阈值
        
    Returns:
        bool: True表示是黑色补丁
    """
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    """
    检查补丁是否为黑色（基于像素百分比）
    
    Args:
        patch: 补丁图像（PIL Image）
        rgbThresh: RGB阈值
        percentage: 黑色像素百分比阈值
        
    Returns:
        bool: True表示是黑色补丁
    """
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    """
    检查补丁是否为白色（基于像素百分比）
    
    Args:
        patch: 补丁图像（PIL Image）
        rgbThresh: RGB阈值
        percentage: 白色像素百分比阈值
        
    Returns:
        bool: True表示是白色补丁
    """
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def coord_generator(x_start, x_end, x_step, y_start, y_end, y_step, args_dict=None):
    """
    坐标生成器
    生成网格中的所有坐标点
    
    Args:
        x_start: x起始坐标
        x_end: x结束坐标
        x_step: x步长
        y_start: y起始坐标
        y_end: y结束坐标
        y_step: y步长
        args_dict: 参数字典（可选）
        
    Yields:
        tuple or dict: 坐标或包含坐标的参数字典
    """
    for x in range(x_start, x_end, x_step):
        for y in range(y_start, y_end, y_step):
            if args_dict is not None:
                process_dict = args_dict.copy()
                process_dict.update({'pt':(x,y)})
                yield process_dict
            else:
                yield (x,y)

def savePatchIter_bag_hdf5(patch):
    """
    将补丁迭代保存到HDF5文件
    
    Args:
        patch (dict): 补丁信息字典，包含坐标、图像等
    """
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path= tuple(patch.values())
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "a")

    # 扩展数据集并添加图像
    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    # 如果存在坐标数据集，也添加坐标
    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x,y)

    file.close()

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """
    保存数据到HDF5文件
    
    Args:
        output_path (str): 输出文件路径
        asset_dict (dict): 数据字典
        attr_dict (dict): 属性字典（可选）
        mode (str): 文件模式（'a'追加，'w'写入）
        
    Returns:
        str: 输出文件路径
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            # 创建新数据集
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            # 添加属性
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            # 扩展现有数据集
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def initialize_hdf5_bag(first_patch, save_coord=False):
    """
    初始化HDF5文件（用于存储补丁包）
    
    Args:
        first_patch (dict): 第一个补丁的信息字典
        save_coord (bool): 是否保存坐标
        
    Returns:
        str: HDF5文件路径
    """
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())
    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # 初始化可调整大小的数据集
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:]  # 数据集可调整的最大维度（None表示无限制）
    dset = file.create_dataset('imgs', 
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    # 设置属性
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    # 如果保存坐标，创建坐标数据集
    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x,y)

    file.close()
    return file_path

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    """
    从分数窗口内采样索引
    
    Args:
        scores: 分数数组
        k: 采样数量
        start: 窗口起始位置（百分位数或值）
        end: 窗口结束位置（百分位数或值）
        convert_to_percentile: 是否转换为百分位数
        seed: 随机种子
        
    Returns:
        numpy.ndarray or int: 采样的索引数组，如果没有找到则返回-1
    """
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    # 找到分数窗口内的索引
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1 
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def top_k(scores, k, invert=False):
    """
    获取top-k索引
    
    Args:
        scores: 分数数组
        k: 前k个
        invert: 是否反转（获取最低的k个）
        
    Returns:
        numpy.ndarray: top-k索引
    """
    if invert:
        top_k_ids=scores.argsort()[:k]  # 最低k个
    else:
        top_k_ids=scores.argsort()[::-1][:k]  # 最高k个
    return top_k_ids

def to_percentiles(scores):
    """
    将分数转换为百分位数
    
    Args:
        scores: 分数数组
        
    Returns:
        numpy.ndarray: 百分位数数组
    """
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    """
    筛选指定边界框内的坐标和分数
    
    Args:
        scores: 分数数组
        coords: 坐标数组
        top_left: 左上角坐标
        bot_right: 右下角坐标
        
    Returns:
        tuple: (筛选后的分数, 筛选后的坐标)
    """
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    # 创建掩码：坐标在边界框内
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):
    """
    采样感兴趣区域（ROI）
    
    Args:
        scores: 分数数组
        coords: 坐标数组
        k: 采样数量
        mode: 采样模式（'range_sample', 'topk', 'reverse_topk'）
        seed: 随机种子
        score_start: 分数窗口起始位置
        score_end: 分数窗口结束位置
        top_left: 左上角坐标（可选）
        bot_right: 右下角坐标（可选）
        
    Returns:
        dict: 包含采样坐标和分数的字典
    """
    if len(scores.shape) == 2:
        scores = scores.flatten()

    # 转换为百分位数
    scores = to_percentiles(scores)
    # 如果指定了边界框，筛选坐标
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    # 根据模式采样
    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)  # 最高k个
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)  # 最低k个
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset

def DrawGrid(img, coord, shape, thickness=5, color=(0,0,0,255)):
    """
    在图像上绘制网格线
    
    Args:
        img: 图像数组
        coord: 坐标
        shape: 形状
        thickness: 线条粗细
        color: 颜色
        
    Returns:
        numpy.ndarray: 绘制后的图像
    """
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    """
    将补丁绘制到画布上（拼接）
    
    Args:
        canvas: 画布数组
        patch_dset: HDF5补丁数据集
        coords: 坐标数组
        patch_size: 补丁大小
        indices: 索引数组（可选）
        verbose: 详细程度
        draw_grid: 是否绘制网格
        
    Returns:
        PIL.Image: 拼接后的图像
    """
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    # 遍历所有补丁
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)  # 调整补丁大小
        coord = coords[patch_id]
        # 获取画布裁剪区域形状
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        # 将补丁粘贴到画布
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, verbose=1, draw_grid=True):
    """
    从坐标从WSI读取补丁并绘制到画布上
    
    Args:
        canvas: 画布数组
        wsi_object: WSI对象
        coords: 坐标数组
        patch_size: 补丁大小（相对于第0层）
        vis_level: 可视化层级
        indices: 索引数组（可选）
        verbose: 详细程度
        draw_grid: 是否绘制网格
        
    Returns:
        PIL.Image: 拼接后的图像
    """
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
    
    # 计算降采样后的补丁大小
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))

    center_coords = []
    # 遍历所有补丁
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        coord = coords[patch_id]
        # 从WSI读取补丁
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))

        # 计算画布上的坐标
        coord = np.ceil(coord / downsamples).astype(np.int32)
        center_coords.append(str(coord[0]+int(patch_size[0]/2))+'_'+str(coord[1]+int(patch_size[1]/2)))
        # 获取画布裁剪区域形状
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        # 将补丁粘贴到画布
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]

        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    # draw center point of each patch
    # canvas = np.ones(canvas.shape).astype('uint8')*255  # 生成白板
    # for idx in range(len(center_coords)):
    #     x = int(center_coords[idx].split('_')[0])
    #     y = int(center_coords[idx].split('_')[1])
    #     adj_coords = []
    #     for i in [x-patch_size[0], x, x+patch_size[0]]:
    #         for j in [y-patch_size[1], y, y+patch_size[1]]:
    #             if(str(i)+'_'+str(j) in center_coords):
    #                 adj_coords.append((i, j))

        # for end_coord in adj_coords:
        #     cv2.line(canvas, (x,y), end_coord, (0,0,255), 10, 4)  # thickness: 5600:10, 7200:15

    # for idx in range(len(center_coords)):
    #     x = int(center_coords[idx].split('_')[0])
    #     y = int(center_coords[idx].split('_')[1])
    #     cv2.circle(canvas, (x,y), 30, (252, 215, 103), -1)  # 8000:(0,255,0)/50,  7200:(9, 248, 232)/40, 5600:(252, 215, 103)/30

    return Image.fromarray(canvas)


def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    """
    从HDF5文件拼接补丁
    
    Args:
        hdf5_file_path (str): HDF5文件路径
        downscale (int): 降采样比例
        draw_grid (bool): 是否绘制网格
        bg_color (tuple): 背景颜色
        alpha (float): 透明度
        
    Returns:
        PIL.Image: 拼接后的图像
    """
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['imgs']
    coords = file['coords'][:]
    # 获取原始尺寸
    if 'downsampled_level_dim' in dset.attrs.keys():
        w, h = dset.attrs['downsampled_level_dim']
    else:
        w, h = dset.attrs['level_dim']
    print('original size: {} x {}'.format(w, h))
    # 计算降采样后的尺寸
    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(dset)))
    img_shape = dset[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    # 检查图像大小限制
    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    # 创建画布
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    # 绘制补丁
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap

def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    """
    从坐标文件拼接补丁（从WSI读取）
    
    Args:
        hdf5_file_path (str): HDF5坐标文件路径
        wsi_object: WSI对象
        downscale (int): 降采样比例
        draw_grid (bool): 是否绘制网格
        bg_color (tuple): 背景颜色
        alpha (float): 透明度
        
    Returns:
        PIL.Image: 拼接后的图像
    """
    wsi = wsi_object.getOpenSlide()
    vis_level = wsi.get_best_level_for_downsample(downscale)
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]
    w, h = wsi.level_dimensions[0]

    print('start stitching {}'.format(dset.attrs['name']))
    print('original size: {} x {}'.format(w, h))

    # 获取可视化层级的尺寸
    w, h = wsi.level_dimensions[vis_level]

    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(coords)))
    
    # 获取补丁参数
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
    # 计算参考补丁大小（第0层）
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print('ref patch size: {}x{}'.format(patch_size, patch_size))

    # 检查图像大小限制
    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    # 创建画布
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    # 绘制补丁
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap

def SamplePatches(coords_file_path, save_file_path, wsi_object, 
    patch_level=0, custom_downsample=1, patch_size=256, sample_num=100, seed=1, stitch=True, verbose=1, mode='w'):
    """
    从坐标文件采样补丁
    
    Args:
        coords_file_path (str): 坐标文件路径
        save_file_path (str): 保存文件路径
        wsi_object: WSI对象
        patch_level (int): 补丁层级
        custom_downsample (int): 自定义降采样
        patch_size (int): 补丁大小
        sample_num (int): 采样数量
        seed (int): 随机种子
        stitch (bool): 是否拼接成马赛克
        verbose (int): 详细程度
        mode (str): 文件模式
        
    Returns:
        tuple: (canvas, 总坐标数, 采样数)
    """
    file = h5py.File(coords_file_path, 'r')
    dset = file['coords']
    coords = dset[:]

    h5_patch_size = dset.attrs['patch_size']
    h5_patch_level = dset.attrs['patch_level']
    
    if verbose>0:
        print('in .h5 file: total number of patches: {}'.format(len(coords)))
        print('in .h5 file: patch size: {}x{} patch level: {}'.format(h5_patch_size, h5_patch_size, h5_patch_level))

    # 使用文件中的参数（如果未指定）
    if patch_level < 0:
        patch_level = h5_patch_level

    if patch_size < 0:
        patch_size = h5_patch_size

    # 随机采样索引
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(coords)), min(len(coords), sample_num), replace=False)

    target_patch_size = np.array([patch_size, patch_size])
    
    # 应用自定义降采样
    if custom_downsample > 1:
        target_patch_size = (np.array([patch_size, patch_size]) / custom_downsample).astype(np.int32)
        
    # 创建马赛克画布（如果启用）
    if stitch:
        canvas = Mosaic_Canvas(patch_size=target_patch_size[0], n=sample_num, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1)
    else:
        canvas = None
    
    # 遍历采样的补丁
    for idx in indices:
        coord = coords[idx]
        # 从WSI读取补丁
        patch = wsi_object.wsi.read_region(coord, patch_level, tuple([patch_size, patch_size])).convert('RGB')
        if custom_downsample > 1:
            patch = patch.resize(tuple(target_patch_size))

        # 如果启用拼接，粘贴到画布
        if stitch:
            canvas.paste_patch(patch)

        # 保存补丁到HDF5
        asset_dict = {'imgs': np.array(patch)[np.newaxis,...], 'coords': coord}
        save_hdf5(save_file_path, asset_dict, mode=mode)
        mode='a'  # 后续追加模式

    return canvas, len(coords), len(indices)