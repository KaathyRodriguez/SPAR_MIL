"""
补丁特征提取工具函数
支持多种预训练编码器（ResNet、CLIP、DINO、SimCLR等）从补丁图像中提取特征
"""
### Dependencies
# Base Dependencies
import os
# LinAlg / Stats / Plotting Dependencies
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
device = torch.device('cuda')
torch.multiprocessing.set_sharing_strategy('file_system')

# Model Architectures
from nn_encoder_arch.vision_transformer import vit_small
from nn_encoder_arch.resnet_trunc import resnet50_trunc_baseline


def eval_transforms_clip(pretrained=False):
    """
    为CLIP模型创建评估时的图像变换
    包括调整大小到224x224（CLIP的标准输入尺寸）
    
    参数:
        pretrained: 是否使用ImageNet预训练的归一化参数
    
    返回:
        图像变换组合
    """
    if pretrained:
        # ImageNet预训练的归一化参数
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        # 默认归一化参数
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Resize((224, 224)),  # CLIP需要224x224输入
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val

def eval_transforms(pretrained=False):
    """
    创建评估时的图像变换（通用版本，不调整大小）
    
    参数:
        pretrained: 是否使用ImageNet预训练的归一化参数
    
    返回:
        图像变换组合
    """
    if pretrained:
        # ImageNet预训练的归一化参数
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        # 默认归一化参数
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val


def torchvision_ssl_encoder(name: str, pretrained: bool = False, return_all_feature_maps: bool = False):
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()
    return pretrained_model


def save_embeddings(model, fname, dataloader, enc_name, overwrite=False):
    """
    提取特征并保存到HDF5文件
    
    参数:
        model: 特征提取模型
        fname: 保存文件名（不含扩展名）
        dataloader: 数据加载器
        enc_name: 编码器名称
        overwrite: 是否覆盖已存在的文件
    """
    # 如果文件已存在且不覆盖，则跳过
    if os.path.isfile('%s.h5' % fname) and (overwrite == False):
        return None

    embeddings, coords, file_names = [], [], []

    # 批量提取特征
    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            if(enc_name != 'clip_RN50' and enc_name != 'clip_ViTB32'):
                # 普通编码器直接调用模型
                embeddings.append(model(batch).detach().cpu().numpy().squeeze())
            else:
                # CLIP编码器需要使用encode_image方法
                embeddings.append(model.encode_image(batch).detach().cpu().numpy().squeeze())
            file_names.append(coord)

    # 从文件名中解析坐标
    for file_name in file_names:
        for coord in file_name:
            coord = coord.rstrip('.png').split('_')  # 文件名格式：x_y.png
            coords.append([int(coord[0]), int(coord[1])])

    print(fname)

    # 堆叠所有特征和坐标
    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    # 保存到HDF5文件
    f = h5py.File(fname+'.h5', 'w')
    f['features'] = embeddings
    f['coords'] = coords
    f.close()


def create_embeddings(embeddings_dir, enc_name, dataset, batch_size, save_patches=False,
                      patch_datasets='path/to/patch/datasets', assets_dir ='./ckpts/',
                      disentangle=-1, stage=-1):
    """
    创建补丁特征嵌入
    支持多种预训练编码器：ResNet、CLIP、DINO、SimCLR等
    
    参数:
        embeddings_dir: 特征保存目录
        enc_name: 编码器名称
        dataset: 数据集名称
        batch_size: 批次大小
        save_patches: 是否保存补丁（未使用）
        patch_datasets: 补丁数据集路径
        assets_dir: 预训练模型权重目录
        disentangle: 解耦参数（未使用）
        stage: DINO模型的中间层索引（-1表示使用最后一层）
    """
    print("Extracting Features for '%s' via '%s'" % (dataset, enc_name))
    
    # 根据编码器名称加载相应的模型
    if enc_name == 'resnet50_trunc':
        # 截断的ResNet50（只到layer3）
        model = resnet50_trunc_baseline(pretrained=True)
        eval_t = eval_transforms(pretrained=True)
    
    elif enc_name == 'clip_RN50':
        # CLIP ResNet50编码器
        import open_clip
        from open_clip import create_model_and_transforms 
        model_name = 'RN50' # open_clip 中 ResNet50 的规范名称
        pretrained_dataset = 'openai' # 使用 OpenAI 的预训练权重（RN50常用）
        
        # 修正解包：使用 model, _, _ 来接收三个返回值，并忽略两个 transform
        model, _, _ = create_model_and_transforms(
            model_name, 
            pretrained=pretrained_dataset, 
            device=device
        )
        # 模型加载后通常需要设置为评估模式
        model.eval() 
        eval_t = eval_transforms_clip(pretrained=True)
    
    elif enc_name == 'clip_ViTB32':
        # CLIP Vision Transformer B/32编码器
        import open_clip
        from open_clip import create_model_and_transforms
        
        # 查找模型和预处理函数
        model_name = 'ViT-B-32' # open_clip 中 ViT-B/32 的规范名称
        pretrained_dataset = 'laion2b_s34b_b79k' # 推荐的预训练数据集
    
        # 使用 open_clip 的正确加载函数
        model, _, _ = create_model_and_transforms(
            model_name, 
            pretrained=pretrained_dataset, 
            device=device
        )
        eval_t = eval_transforms_clip(pretrained=True) 

    elif enc_name == 'model_dino' or enc_name == 'dino_HIPT':
        # DINO Vision Transformer模型
        ckpt_path = os.path.join(assets_dir, enc_name+'.pth')
        assert os.path.isfile(ckpt_path)
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
        # 清理state_dict键名
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        eval_t = eval_transforms(pretrained=False)

    elif enc_name == 'model_simclr':
        # SimCLR自监督学习模型
        ckpt_path = os.path.join(assets_dir, enc_name+'.pt')
        assert os.path.isfile(ckpt_path)
        model = torchvision_ssl_encoder('resnet50', pretrained=True)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path), strict=False)
        eval_t = eval_transforms(pretrained=False)

    elif enc_name == 'model_simclr_histo_res18':
        # 组织病理学专用的SimCLR ResNet18模型
        ckpt_path = os.path.join(assets_dir, enc_name+'.ckpt')
        assert os.path.isfile(ckpt_path)

        model = torchvision.models.__dict__['resnet18'](pretrained=False)

        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state['state_dict']
        # 清理state_dict键名
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    else:
        pass

    # 将模型移到GPU并设置为评估模式
    model = model.to(device)
    if(enc_name != 'clip_ViTB32' and enc_name != 'clip_RN50'):
        model = torch.nn.DataParallel(model)  # CLIP模型不支持DataParallel
    model.eval()

    # DINO模型支持提取中间层特征
    if 'dino' in enc_name:
        _model = model
        if stage == -1:
            model = _model
        else:
            # 提取指定中间层的特征
            model = lambda x: torch.cat([x[:, 0] for x in _model.get_intermediate_layers(x, stage)], dim=-1)

    if stage != -1:
        _stage = '_s%d' % stage
    else:
        _stage = ''

    # 处理TCGA数据集
    if dataset == 'TCGA':
        # 遍历每个WSI的补丁文件夹
        for wsi_name in tqdm(os.listdir(patch_datasets)):
            dataset = PatchesDataset(os.path.join(patch_datasets, wsi_name), transform=eval_t)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            fname = os.path.join(embeddings_dir, wsi_name)
            if(not os.path.exists(fname)):
                save_embeddings(model, fname, dataloader, enc_name)
    elif dataset == 'SYSUFH_ESCC':
        # 遍历每个WSI的补丁文件夹（假设目录结构与TCGA相同）
        for wsi_name in tqdm(os.listdir(patch_datasets)):
            dataset = PatchesDataset(os.path.join(patch_datasets, wsi_name), transform=eval_t)
            # 保持多进程数据加载
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            fname = os.path.join(embeddings_dir, wsi_name)
            if(not os.path.exists(fname)):
                save_embeddings(model, fname, dataloader, enc_name)
    elif dataset == 'CAMELYON16':
        # 遍历每个WSI的补丁文件夹（假设目录结构与TCGA相同）
        for wsi_name in tqdm(os.listdir(patch_datasets)):
            dataset = PatchesDataset(os.path.join(patch_datasets, wsi_name), transform=eval_t)
            # 保持多进程数据加载
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            fname = os.path.join(embeddings_dir, wsi_name)
            if(not os.path.exists(fname)):
                save_embeddings(model, fname, dataloader, enc_name)
    


class PatchesDataset(Dataset):
    """
    补丁数据集类
    用于加载补丁图像文件（文件名格式：x_y.png，包含坐标信息）
    """
    def __init__(self, file_path, transform=None):
        """
        初始化补丁数据集
        
        参数:
            file_path: 补丁文件夹路径
            transform: 图像变换
        """
        file_names = os.listdir(file_path)
        imgs = []
        coords = []
        for file_name in file_names:
            imgs.append(os.path.join(file_path, file_name))
            coords.append(file_name)  # 文件名包含坐标信息（x_y.png格式）
        self.imgs = imgs
        self.coords = coords
        self.transform = transform

    def __getitem__(self, index):
        """
        获取单个补丁样本
        
        参数:
            index: 样本索引
        
        返回:
            img: 补丁图像（已应用变换）
            coord: 坐标文件名
        """
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')  # 转换为RGB格式
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        """返回数据集大小"""
        return len(self.imgs)
        
