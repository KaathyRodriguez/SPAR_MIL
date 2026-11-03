import pickle
import h5py
import os

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE' # 设置环境变量，禁用 HDF5 文件锁定

def save_pkl(filename, save_object):
    """
    保存对象到 pickle 文件
    """
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename):
    """
    从 pickle 文件加载对象
    """
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """
    保存数据到 HDF5 文件
    Args:
        output_path: 文件路径
        asset_dict: 要保存的数据字典 (key: 数据集名称, val: 数据数组)
        attr_dict: 数据集属性字典（可选）
        mode: 文件模式 ('a' - 追加/创建, 'w' - 覆盖/创建)
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            # 如果数据集不存在，创建新数据集
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            # 创建可调整大小的数据集
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val # 设置属性
        else:
            # 如果数据集已存在，追加数据
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0) # 调整大小
            dset[-data_shape[0]:] = val # 写入新数据
    file.close()
    return output_path