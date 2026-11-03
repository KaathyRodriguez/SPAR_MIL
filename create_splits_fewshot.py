"""
创建few-shot学习的数据分割脚本
从原始数据分割中为每个类别随机选择N个样本作为训练集

运行命令：python create_splits_fewshot.py    需要修改配置参数N、data_folder、all_data(需要xlsx格式且label是数字格式)、save_folder
"""
import pandas as pd 
import numpy as np 
import os 

# 配置参数（需要根据实际情况修改）
N = 16  # few-shot的样本数量（每个类别）
data_folder = './splits/TCGA_BRCA/task_tcga_brca_subtyping_100'  # 原始数据分割文件夹路径
all_data = np.array(pd.read_excel('H:/dataset/TCGA-BRCA/label_x.xlsx', engine='openpyxl',  header=None))  # 所有数据的Excel文件路径
save_folder = './fewshot_splits_1/TCGA_BRCA/16-shot'  # 保存few-shot分割结果的文件夹路径

# 创建保存文件夹（如果不存在）
if(not os.path.exists(save_folder)):
    os.makedirs(save_folder)

# 处理5个不同的数据分割（5折交叉验证）
for j in range(5):
    # 设置文件路径
    orginal_data_split_path = data_folder + '/splits_'+str(j)+'.csv'  # 原始数据分割文件
    orginal_data_stastic_path = data_folder + '/splits_'+str(j)+'_descriptor.csv'  # 原始数据统计文件
    save_path = save_folder + '/splits_'+str(j)+'.csv'  # 保存few-shot分割的文件路径

    # 读取原始数据分割
    orginal_data_split = np.array(pd.read_csv(orginal_data_split_path))
    
    # 创建幻灯片名称到标签的映射字典
    slidename2label = {}
    for each_data in all_data:   
        slidename2label[each_data[1].rstrip('.svs')] = each_data[-1]   # 移除.svs后缀并映射到标签
    
    # 获取所有幻灯片的标签
    all_slide_label = []
    selected_train_slide = []
    for each_data in orginal_data_split:
        slide_label = slidename2label[each_data[1]]
        all_slide_label.append(slide_label)
    
    # 为每个唯一标签随机选择N个样本作为训练集
    unique_label = np.unique(all_slide_label)
    for each_label in unique_label:
        each_index = np.where(all_slide_label == each_label)[0]  # 找到该标签的所有索引
        selected_index = np.random.choice(each_index, size=N, replace=False)  # 随机选择N个样本
        for each_index in selected_index:
            selected_train_slide.append(orginal_data_split[each_index][1])  # 添加到选中的训练集

    # 更新数据分割：将选中的训练样本放在前面，其余设为NaN
    orginal_data_split[:, 1][0:len(selected_train_slide)] = selected_train_slide
    orginal_data_split[:, 1][len(selected_train_slide):-1] = np.nan

    # 读取统计信息以确定验证集和测试集的数量
    all_nums = np.array(pd.read_csv(orginal_data_stastic_path))
    val_num = np.sum(all_nums[:, 2])  # 验证集数量

    # 只保留训练集和验证集部分（根据验证集数量截取）
    new_data_split = orginal_data_split[:val_num]

    # 保存few-shot数据分割到CSV文件
    column_name = ['','train', 'val', 'test']
    csv = pd.DataFrame(columns=column_name, data = new_data_split)
    csv.to_csv(save_path, index=False)
