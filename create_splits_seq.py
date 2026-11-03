"""
创建序列化数据分割的脚本
用于全片扫描图像（WSI）分类任务的数据集划分
支持k折交叉验证和不同的标签比例

运行命令：python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 10
具体:python create_splits_seq.py --task task_pcr_vs_nonpcr --seed 1 --k 5
python create_splits_seq.py --task task_tcga_brca_subtyping --seed 1 --k 5
python create_splits_seq.py --task task_CAMELYON16_subtyping --seed 1 --k 5
"""
import os
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits
import argparse
import numpy as np

# 命令行参数解析
parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='标签使用比例 (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='随机种子 (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='分割数量，即k折交叉验证的k值 (default: 10)')
parser.add_argument('--task', type=str,
                    help='任务名称，如task_tcga_rcc_subtyping或task_tcga_lung_subtyping')
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='验证集比例 (default: 0.2)')
parser.add_argument('--test_frac', type=float, default= 0.2,
                    help='测试集比例 (default: 0.2)')
parser.add_argument('--dataset', type=str, default='CAMELYON16',
                    help='数据集名称')

args = parser.parse_args()

# 根据任务类型初始化数据集
if args.task == 'task_tcga_rcc_subtyping':
    # TCGA肾细胞癌亚型分类任务（3类）
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},  # 标签映射
                                                 patient_strat= True,  # 按患者分层
                                                 patient_voting='maj',  # 多数投票
                                                 ignore=[])

elif args.task == 'task_tcga_lung_subtyping':
    # TCGA肺癌亚型分类任务（2类）
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'LUAD':0, 'LUSC':1},  # 标签映射
                                                 patient_strat= True,  # 按患者分层
                                                 patient_voting='maj',  # 多数投票
                                                 ignore=[])

elif args.task == 'task_pcr_vs_nonpcr':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'H:/dataset/SYSUFH_ESCC_Dataset/label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'non-pCR':0, 'pCR':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_tcga_brca_subtyping':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'H:/dataset/TCGA-BRCA/label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'ILC':0, 'IDC':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_CAMELYON16_subtyping':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'H:/dataset/CAMELYON16/label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Normal':0, 'Tumor':1},
                            patient_strat=True,
                            ignore=[])

else:
    raise NotImplementedError

# 计算每个类别的样本数量
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
# 根据比例计算验证集和测试集的样本数量
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    # 确定要使用的标签比例列表
    if args.label_frac > 0:
        # 如果指定了具体比例，只使用该比例
        label_fracs = [args.label_frac]
    else:
        # 否则使用多个比例
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    # 对每个标签比例创建数据分割
    for lf in label_fracs:
        # 创建保存分割结果的目录
        split_dir = 'splits/'+ args.dataset +'/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        
        # 创建k个数据分割
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        
        # 对每个分割进行保存
        for i in range(args.k):
            dataset.set_splits()  # 设置当前分割
            descriptor_df = dataset.test_split_gen(return_descriptor=True)  # 生成描述符数据框
            splits = dataset.return_splits(from_id=True)  # 获取分割结果
            
            # 保存标准格式的分割文件
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            # 保存布尔格式的分割文件
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            # 保存描述符文件
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



