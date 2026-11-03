"""
补丁特征提取主脚本
从补丁图像中提取特征并保存为HDF5格式

运行命令：python patch_extraction.py --patches_path H:/dataset/CAMELYON16/patches_img_1024 --library_path E:/dataset/TCGA-BRCA/features_1024_clip_RN50 --model_name clip_RN50 --batch_size 64
"""
import os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from patch_extraction_utils import create_embeddings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

# 命令行参数解析
parser = argparse.ArgumentParser(description='Configurations for feature extraction')
parser.add_argument('--patches_path', type=str, help='补丁图像文件夹路径')
parser.add_argument('--library_path', type=str, help='特征保存目录路径')
parser.add_argument('--model_name', type=str, help='编码器模型名称')
parser.add_argument('--batch_size', type=int, help='批次大小')
args = parser.parse_args()

patches_path = args.patches_path
library_path = args.library_path
model_name = args.model_name
os.makedirs(library_path, exist_ok=True)  # 创建保存目录



# -----------------------------------------------------------
# 将所有启动多进程（如调用 create_embeddings）的执行代码放入主程序保护块
# -----------------------------------------------------------
if __name__ == '__main__':
    # 假设这是您原始的执行代码
    create_embeddings(patch_datasets=patches_path, embeddings_dir=library_path,
                      enc_name=model_name, dataset='CAMELYON16',batch_size=args.batch_size)
    
