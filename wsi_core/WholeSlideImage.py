"""
全视野数字切片图像（Whole Slide Image, WSI）处理模块
提供WSI的加载、组织分割、补丁提取、可视化等功能
"""
import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
import math
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl

# 设置PIL图像最大像素数限制，防止内存溢出
Image.MAX_IMAGE_PIXELS = 933120000

class WholeSlideImage(object):
    """
    全视野数字切片图像处理类
    用于处理WSI文件的加载、组织分割、补丁提取和可视化
    """
    def __init__(self, path):
        """
        初始化WSI对象
        
        Args:
            path (str): WSI文件的完整路径
        """
        # 从路径中提取文件名（不含扩展名）
        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        # 使用OpenSlide打开WSI文件
        self.wsi = openslide.open_slide(path)
        # 获取各层级的降采样比例
        self.level_downsamples = self._assertLevelDownsamples()
        # 获取各层级的尺寸
        self.level_dim = self.wsi.level_dimensions

        # 组织轮廓（用于分割）
        self.contours_tissue = None
        # 肿瘤轮廓（用于标注）
        self.contours_tumor = None
        # HDF5文件路径（用于存储补丁）
        self.hdf5_file = None

    def getOpenSlide(self):
        """
        获取OpenSlide对象
        
        Returns:
            openslide.OpenSlide: OpenSlide对象
        """
        return self.wsi

    def initXML(self, xml_path):
        """
        从XML文件初始化肿瘤轮廓标注
        
        Args:
            xml_path (str): XML标注文件的路径
        """
        def _createContour(coord_list):
            """从坐标列表创建轮廓数组"""
            return np.array([[[int(float(coord.attributes['X'].value)),
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

        # 解析XML文件
        xmldoc = minidom.parse(xml_path)
        # 提取所有标注的坐标点
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        # 创建肿瘤轮廓列表
        self.contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
        # 按面积从大到小排序
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self,annot_path):
        """
        从文本文件初始化肿瘤轮廓标注
        
        Args:
            annot_path (str): 标注文本文件的路径
        """
        def _create_contours_from_dict(annot):
            """从字典格式的标注数据创建轮廓列表"""
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                # 处理多边形类型的标注
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour)
                # 处理其他类型的标注（如线段组）
                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        # 将线段组合并成轮廓
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour)

            return all_cnts

        # 读取并解析标注文件
        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
        # 创建肿瘤轮廓并排序
        self.contours_tumor  = _create_contours_from_dict(annot)
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        """
        从pickle文件加载分割结果
        
        Args:
            mask_file (str): 包含分割结果的pickle文件路径
        """
        import pickle
        # 加载分割结果
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']  # 组织中的空洞
        self.contours_tissue = asset_dict['tissue']  # 组织轮廓

    def saveSegmentation(self, mask_file):
        """
        将分割结果保存到pickle文件
        
        Args:
            mask_file (str): 保存分割结果的pickle文件路径
        """
        # 保存分割结果
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False,
                      filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """
        对组织进行分割
        通过HSV颜色空间 -> 中值滤波 -> 二值化阈值处理
        
        Args:
            seg_level (int): 用于分割的层级
            sthresh (int): 饱和度阈值下限
            sthresh_up (int): 饱和度阈值上限
            mthresh (int): 中值滤波核大小
            close (int): 形态学闭运算核大小
            use_otsu (bool): 是否使用Otsu自适应阈值
            filter_params (dict): 过滤参数，包含面积阈值等
            ref_patch_size (int): 参考补丁大小
            exclude_ids (list): 要排除的轮廓ID列表
            keep_ids (list): 要保留的轮廓ID列表
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
            根据面积过滤轮廓
            
            Args:
                contours: 轮廓列表
                hierarchy: 轮廓层级关系
                filter_params: 过滤参数字典
                
            Returns:
                foreground_contours: 过滤后的前景轮廓
                hole_contours: 对应的空洞轮廓
            """
            filtered = []

            # 找到前景轮廓的索引（父轮廓为-1的轮廓）
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []

            # 遍历前景轮廓索引
            for cont_idx in hierarchy_1:
                # 实际轮廓
                cont = contours[cont_idx]
                # 找到包含在此轮廓中的空洞索引（父轮廓的子轮廓）
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # 计算轮廓面积（包括空洞）
                a = cv2.contourArea(cont)
                # 计算每个空洞的面积
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # 实际的前景轮廓区域面积（减去空洞面积）
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                # 如果面积大于阈值，则保留
                if tuple((filter_params['a_t'],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)


            # 提取过滤后的前景轮廓
            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            # 处理每个前景轮廓对应的空洞
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                # 按面积从大到小排序
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # 只保留面积最大的max_n_holes个空洞
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []

                # 过滤空洞（面积大于阈值）
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        # 读取指定层级的图像
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        # 转换到HSV颜色空间
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 对饱和度通道应用中值滤波
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)

        # 二值化阈值处理
        if use_otsu:
            # 使用Otsu自适应阈值
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            # 使用固定阈值
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # 形态学闭运算（填充小孔洞）
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        # 计算缩放比例和调整过滤参数
        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        # 根据缩放比例调整面积阈值
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # 查找并过滤轮廓
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        # 过滤轮廓（过滤掉伪影）
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)

        # 将轮廓缩放回第0层级
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        # 根据keep_ids和exclude_ids过滤轮廓
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        # 应用过滤
        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0),
               line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
               number_contours=False, seg_display=True, annot_display=True):
        """
        可视化WSI，包括组织轮廓、空洞和肿瘤标注
        
        Args:
            vis_level (int): 可视化层级
            color (tuple): 组织轮廓颜色 (RGB)
            hole_color (tuple): 空洞颜色 (RGB)
            annot_color (tuple): 标注颜色 (RGB)
            line_thickness (int): 线条粗细
            max_size (int): 最大图像尺寸
            top_left (tuple): 左上角坐标
            bot_right (tuple): 右下角坐标
            custom_downsample (int): 自定义降采样比例
            view_slide_only (bool): 是否只显示切片（不显示轮廓）
            number_contours (bool): 是否在轮廓上显示编号
            seg_display (bool): 是否显示分割结果
            annot_display (bool): 是否显示标注
            
        Returns:
            tuple: (可视化图像, 掩码图像)
        """
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]  # 缩放比例

        # 计算区域大小
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]

        # 读取区域图像
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        only_mask = np.zeros(img.shape)  # 创建掩码图像

        if not view_slide_only:
            # 计算偏移量
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            # 根据缩放比例调整线条粗细
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            # 绘制组织轮廓
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    # 绘制轮廓（不编号）
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                    cv2.drawContours(only_mask, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else:  # 为每个轮廓添加编号
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        # 计算轮廓中心
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # 绘制轮廓并在中心添加文本
                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                # 绘制空洞
                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale),
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)
                    cv2.drawContours(only_mask, self.scaleContourDim(holes, scale),
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)

            # 绘制肿瘤标注
            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale),
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)

        # 转换为PIL图像
        img = Image.fromarray(img)
        only_mask = Image.fromarray(np.uint8(only_mask))

        # 应用自定义降采样
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        # 如果超过最大尺寸，则缩放
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))

        return img, only_mask


    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        """
        创建补丁并保存为HDF5格式
        
        Args:
            save_path (str): 保存路径
            patch_level (int): 补丁提取层级
            patch_size (int): 补丁大小
            step_size (int): 步长
            save_coord (bool): 是否保存坐标
            **kwargs: 其他参数
            
        Returns:
            str: HDF5文件路径
        """
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        # 遍历每个轮廓
        for idx, cont in enumerate(contours):
            # 获取补丁生成器
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)

            # 初始化HDF5文件（使用第一个补丁）
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # 空轮廓，跳过
                except StopIteration:
                    continue

                # 初始化HDF5文件
                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            # 保存所有补丁
            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file


    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
                           white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        """
        生成补丁的生成器函数
        
        Args:
            cont: 轮廓
            cont_idx: 轮廓索引
            patch_level: 补丁层级
            save_path: 保存路径
            patch_size: 补丁大小
            step_size: 步长
            custom_downsample: 自定义降采样
            white_black: 是否过滤黑白补丁
            white_thresh: 白色阈值
            black_thresh: 黑色阈值
            contour_fn: 轮廓检查函数类型
            use_padding: 是否使用填充
            
        Yields:
            dict: 补丁信息字典
        """
        # 获取轮廓的边界框
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        # 处理自定义降采样
        if custom_downsample > 1:
            assert custom_downsample == 2
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size,
                                                                                                       target_patch_size, target_patch_size))

        # 计算补丁降采样比例和参考补丁大小
        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])

        # 计算实际步长（考虑降采样）
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        # 选择轮廓检查函数
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        # 计算停止坐标
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            # 确保不超出图像边界
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        # 遍历所有可能的补丁位置
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                # 检查点是否在轮廓内且不在空洞中
                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]):
                    continue

                count+=1
                # 读取补丁
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                # 应用自定义降采样
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

                # 过滤黑白补丁
                if white_black:
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh):
                        continue

                # 构建补丁信息字典
                patch_info = {'x':x // (patch_downsample[0] * custom_downsample), 'y':y // (patch_downsample[1] * custom_downsample), 'cont_idx':cont_idx, 'patch_level':patch_level,
                              'downsample': self.level_downsamples[patch_level], 'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])//custom_downsample), 'level_dim': self.level_dim[patch_level],
                              'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}

                yield patch_info

        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        """
        检查点是否在空洞内
        
        Args:
            holes: 空洞列表
            pt: 点坐标
            patch_size: 补丁大小
            
        Returns:
            int: 1表示在空洞内，0表示不在
        """
        for hole in holes:
            # 检查补丁中心点是否在空洞内
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        """
        检查点是否在轮廓内且不在空洞中
        
        Args:
            cont_check_fn: 轮廓检查函数
            pt: 点坐标
            holes: 空洞列表
            patch_size: 补丁大小
            
        Returns:
            int: 1表示有效，0表示无效
        """
        if cont_check_fn(pt):
            if holes is not None:
                # 在轮廓内且不在空洞中
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        """
        缩放轮廓尺寸
        
        Args:
            contours: 轮廓列表
            scale: 缩放比例
            
        Returns:
            list: 缩放后的轮廓列表
        """
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        """
        缩放空洞尺寸
        
        Args:
            contours: 空洞轮廓列表
            scale: 缩放比例
            
        Returns:
            list: 缩放后的空洞列表
        """
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        """
        断言并计算各层级的降采样比例
        
        Returns:
            list: 各层级的降采样比例列表
        """
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]  # 第0层（最高分辨率）的尺寸

        # 计算每层的降采样比例
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            # 根据尺寸计算估计的降采样比例
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            # 如果估计值与实际值不同，使用估计值；否则使用实际值
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        """
        处理所有轮廓，提取补丁坐标并保存到HDF5
        
        Args:
            save_path (str): 保存路径
            patch_level (int): 补丁层级
            patch_size (int): 补丁大小
            step_size (int): 步长
            **kwargs: 其他参数
            
        Returns:
            str: HDF5文件路径
        """
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        # 每5%显示一次进度
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        # 遍历所有轮廓
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))

            # 处理单个轮廓
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs)
            if len(asset_dict) > 0:
                if init:
                    # 首次写入，创建文件
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    # 追加写入
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file


    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256,
                        contour_fn='four_pt', mag='40', use_padding=True, top_left=None, bot_right=None):
        """
        处理单个轮廓，提取有效补丁坐标
        
        Args:
            cont: 轮廓
            contour_holes: 轮廓对应的空洞
            patch_level: 补丁层级
            save_path: 保存路径
            patch_size: 补丁大小
            step_size: 步长
            contour_fn: 轮廓检查函数类型
            mag: 放大倍数（'20'或'40'）
            use_padding: 是否使用填充
            top_left: 左上角坐标（ROI）
            bot_right: 右下角坐标（ROI）
            
        Returns:
            tuple: (asset_dict, attr_dict) 包含坐标和属性的字典
        """
        # 获取轮廓边界框
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        # 根据放大倍数调整补丁大小
        if(mag == '20'):
            patch_size /= 2
            step_size /= 2

        # 计算补丁降采样和参考大小
        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])

        # 计算停止坐标
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        # 根据ROI调整边界
        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        # 检查调整后的边界是否有效
        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)

        # 选择轮廓检查函数
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        # 计算实际步长
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        # 生成所有候选坐标
        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        # 使用多进程处理坐标候选点
        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        # 并行检查所有候选坐标
        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        # 过滤掉None结果
        results = np.array([result for result in results if result is not None])

        print('Extracted {} coordinates'.format(len(results)))

        # 构建返回字典
        if len(results)>1:
            asset_dict = {'coords' :          results}

            attr = {'patch_size' :            patch_size,
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim':              self.level_dim[patch_level],
                    'name':                   self.name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        """
        处理单个坐标候选点（用于多进程）
        
        Args:
            coord: 坐标
            contour_holes: 空洞列表
            ref_patch_size: 参考补丁大小
            cont_check_fn: 轮廓检查函数
            
        Returns:
            coord or None: 如果坐标有效返回坐标，否则返回None
        """
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    def visHeatmap(self, scores, coords, vis_level=-1,
                   top_left=None, bot_right=None,
                   patch_size=(256, 256),
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
                   blur=False, overlap=0.0,
                   segment=True, use_holes=True,
                   convert_to_percentiles=False,
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        """
        可视化注意力热图
        
        Args:
            scores (numpy array): 注意力分数数组
            coords (numpy array): 对应的坐标数组 (n_patches x 2, 相对于第0层)
            vis_level (int): 可视化的WSI金字塔层级
            patch_size (tuple): 补丁尺寸 (相对于第0层)
            blank_canvas (bool): 是否使用空白画布绘制热图（vs. 使用原始切片）
            canvas_color (tuple): 画布颜色
            alpha (float): 将热图叠加到原始切片上的混合系数 [0, 1]
            blur (bool): 是否应用高斯模糊
            overlap (float): 相邻补丁之间的重叠百分比 [0, 1]（仅影响模糊半径）
            segment (bool): 是否使用组织分割轮廓（必须先调用self.segmentTissue）
            use_holes (bool): 是否也裁剪检测到的组织空洞（仅在segment==True时有效）
            convert_to_percentiles (bool): 是否将注意力分数转换为百分位数
            binarize (bool): 是否只显示大于阈值的补丁
            thresh (float): 二值化阈值
            max_size (int): 最大画布尺寸（超过则裁剪）
            custom_downsample (int): 额外降采样热图的因子
            cmap (str): matplotlib颜色映射名称
            
        Returns:
            PIL.Image: 热图图像
        """

        # 计算降采样和缩放比例
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]  # 从第0层到目标层的缩放比例

        # 将分数展平为一维数组
        if len(scores.shape) == 2:
            scores = scores.flatten()

        # 计算二值化阈值
        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)  # 自动阈值
            else:
                threshold =  thresh
        else:
            threshold = 0.0

        ##### 计算热图尺寸并过滤指定bbox区域外的坐标/分数 #####
        if top_left is not None and bot_right is not None:
            # 筛选ROI内的坐标和分数
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left  # 相对坐标
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            # 使用整个图像
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[vis_level]
            w, h = region_size

        # 缩放补丁大小和坐标到目标层级
        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        ###### 归一化过滤后的分数 ######
        if convert_to_percentiles:
            scores = to_percentiles(scores)  # 转换为百分位数

        scores /= 100  # 归一化到[0,1]

        ######## 计算原始注意力分数的热图（应用颜色映射前）
        # 通过在重叠区域累积分数来计算 ######

        # 热图叠加层：跟踪热图每个像素的注意力分数
        # 叠加计数器：跟踪每个像素累积注意力分数的次数
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)
        count = 0
        # 遍历所有补丁坐标
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0  # 二值化
                    count+=1
            else:
                score=0.0
            # 累积注意力分数
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # 累积计数器
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))

        # 获取注意力区域并平均累积的注意力分数
        zero_mask = counter == 0  # 未覆盖的区域

        # 计算平均分数
        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter
        # 应用高斯模糊
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)

        # 获取组织掩码（如果启用分割）
        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))

        # 创建画布
        if not blank_canvas:
            # 降采样原始图像作为画布
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # 使用空白画布
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255)))

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        # 获取颜色映射
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        # 为每个补丁应用颜色映射
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {}/{}'.format(idx, len(coords)))

            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                # 注意力块（原始注意力分数）
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]

                # 图像块（空白画布或原始图像）
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # 颜色块（将颜色映射应用到注意力块）
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # 组织掩码块
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                    # 只复制组织掩码部分的颜色块
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # 复制整个颜色块
                    img_block = color_block

                # 重写图像块
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()

        print('Done')
        del overlay

        # 对最终图像应用模糊（如果启用）
        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)

        # 混合热图和原始图像
        if alpha < 1.0:
            # 读取完整图像作为画布
            canvas = np.array(self.wsi.read_region((0,0), vis_level, self.wsi.level_dimensions[vis_level]).convert("RGB"))
            # 加权混合
            img = cv2.addWeighted(img, alpha, canvas, 1 - alpha, 0, canvas)

        # 转换为PIL图像
        img = Image.fromarray(img)
        w, h = img.size

        # 应用自定义降采样
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        # 如果超过最大尺寸，则缩放
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))

        return img


    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        """
        分块混合图像（用于大图像的高效混合）
        
        Args:
            img: 输入图像
            vis_level: 可视化层级
            top_left: 左上角坐标
            bot_right: 右下角坐标
            alpha: 混合系数
            blank_canvas: 是否使用空白画布
            block_size: 块大小
            
        Returns:
            numpy.ndarray: 混合后的图像
        """
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left  # 相对于(0,0)的偏移量
        # 遍历所有块
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                # 1. 通过偏移和缩放将WSI坐标转换为图像坐标
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                # 2. 计算混合块的结束点，注意不要超出图像边界
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue

                # 3. 获取混合块和大小
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)

                if not blank_canvas:
                    # 4. 读取实际WSI块作为画布块
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    # 4. 或创建空白画布块
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. 混合颜色块和画布块
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        """
        获取组织分割掩码
        
        Args:
            region_size: 区域尺寸
            scale: 缩放比例
            use_holes: 是否使用空洞
            offset: 偏移量
            
        Returns:
            numpy.ndarray: 组织掩码（布尔数组）
        """
        print('\ncomputing foreground tissue mask')
        # 创建掩码数组
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        # 缩放组织轮廓
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        # 计算偏移量
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        # 缩放空洞轮廓
        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        # 按面积从大到小排序
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        # 绘制组织轮廓
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            # 绘制空洞（如果启用）
            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)

        # 转换为布尔掩码
        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask




