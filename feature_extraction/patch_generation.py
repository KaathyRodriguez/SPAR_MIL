"""
补丁生成脚本
从WSI的HDF5坐标文件中读取坐标，从原始WSI图像中提取补丁并保存为PNG图像
支持多线程并行处理

运行命令：python patch_generation.py 
"""
import os
import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 配置参数（需要根据实际情况修改）
slide_folder = 'F:/dataset/CAMELYON16/wsi'  # WSI图像文件夹路径
# 注意：如果您的WSI文件都在同一目录下，那么以下读取 Excel 文件的代码不再用于路径构建，
# 您可以根据需要选择保留或注释掉。
all_data = np.array(pd.read_excel('H:/dataset/CAMELYON16/label_x.xlsx', engine='openpyxl',  header=None))  # 所有数据的Excel文件路径
root_folder = 'H:/dataset/CAMELYON16/'  # 根文件夹路径
define_patch_size = 256  # 定义的补丁大小
patch_folder = 'H:/dataset/CAMELYON16' + '/patches_'+str(define_patch_size)+'/'  # 补丁坐标HDF5文件文件夹
save_folder = 'H:/dataset/CAMELYON16/patches_img_256'  # 补丁图像保存文件夹

# 根据补丁大小设置缩放比例和保存名称
if(define_patch_size == 2048):
    scale = 8
    save_name = '5x'
elif(define_patch_size == 1024):
    scale = 4  
    save_name = '10x'
elif(define_patch_size == 512):
    scale = 2
    save_name = '20x'
elif(define_patch_size == 256):
    scale = 1
    save_name = '40x'

# 创建根文件夹（如果不存在）
if(not os.path.exists(root_folder)):
    os.makedirs(root_folder)

# *** 已删除所有 UUID 映射逻辑 ***

def generate_patch(patch_file_name):
    """
    从HDF5坐标文件中读取坐标，从WSI中提取补丁并保存
    
    参数:
        patch_file_name: HDF5补丁坐标文件名 (e.g., 'slide_id.h5')
    """
    # 1. 构建HDF5坐标文件路径
    patch_path = os.path.join(patch_folder, patch_file_name)
    
    # 2. 确定SVS文件名
    svs_file_name = patch_file_name.replace('.h5', '.tif')# 需要修改的地方
    
    # 3. *** 核心修正：直接拼接 WSI 路径 ***
    # WSI 文件路径 = WSI 文件夹 + SVS 文件名
    slide_path = os.path.join(slide_folder, svs_file_name)

    # 检查 WSI 文件是否存在 (可选，但推荐)
    if not os.path.exists(slide_path):
        print(f"Warning: WSI file not found at {slide_path}. Skipping {patch_file_name}")
        return

    # 读取HDF5文件中的坐标信息
    try:
        f = h5py.File(patch_path, 'r')
        coords = f['coords']
        patch_level = coords.attrs['patch_level']  # 补丁提取的层级
        patch_size = coords.attrs['patch_size']  # 补丁大小
    except Exception as e:
        print(f"Error reading HDF5 file {patch_file_name}: {e}. Skipping.")
        return
    
    # 打开WSI图像
    try:
        slide = openslide.open_slide(slide_path)
    except Exception as e:
        print(f"Error opening WSI file {svs_file_name}: {e}. Skipping.")
        return

    try:
        # 获取放大倍数
        magnification = int(float(slide.properties.get('aperio.AppMag', 40)))
    except:
        magnification = 40  # 默认放大倍数
    
    # 设置保存路径
    save_path = os.path.join(save_folder, patch_file_name.replace('.h5', ''))
    
    # 检查是否已处理
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
        # 根据放大倍数计算调整后的补丁大小
        if(magnification == 40):
            resized_patch_size = int(patch_size/scale)
        elif(magnification == 20):
            resized_patch_size = int(patch_size/(scale/2))
        else:
             # 如果放大倍数不是 40x 或 20x，使用默认缩放
             resized_patch_size = int(patch_size/scale) 
        
        # 遍历所有坐标，提取并保存补丁
        for coord in tqdm(coords, desc=f"Processing {svs_file_name}"):
            # 确保坐标是整数类型
            coord = coord.astype(np.int64) 
            
            # 从WSI中读取补丁区域
            patch = slide.read_region(tuple(coord), int(patch_level), (int(patch_size), int(patch_size))).convert('RGB')
            # 调整补丁大小
            patch = patch.resize((resized_patch_size, resized_patch_size))
            # 生成补丁文件名（格式：x_y.png）
            patch_name = str(coord[0]) + '_' + str(coord[1]) + '.png'
            patch_save_path = os.path.join(save_path, patch_name)
            # 保存补丁图像
            patch.save(patch_save_path)
    else:
        print(patch_file_name + ': has been processed !')

# 使用线程池并行处理所有HDF5文件
pool = ThreadPoolExecutor(max_workers=16)
all_file_names = np.array(os.listdir(patch_folder))
for patch_file_name in all_file_names:
    # 仅处理 .h5 文件
    if patch_file_name.endswith('.h5'):
        pool.submit(generate_patch, patch_file_name)
pool.shutdown(wait=True)  # 等待所有任务完成