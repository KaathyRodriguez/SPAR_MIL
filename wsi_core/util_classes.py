"""
工具类模块
包含马赛克画布和轮廓检查函数类
"""
import numpy as np
from PIL import Image
import cv2

class Mosaic_Canvas(object):
	"""
	马赛克画布类
	用于将多个补丁拼接成马赛克图像
	"""
	def __init__(self,patch_size=256, n=100, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1):
		"""
		初始化马赛克画布
		
		Args:
			patch_size (int): 原始补丁大小
			n (int): 补丁总数
			downscale (int): 降采样比例
			n_per_row (int): 每行补丁数
			bg_color (tuple): 背景颜色
			alpha (float): 透明度（<0表示不透明）
		"""
		self.patch_size = patch_size
		self.downscaled_patch_size = int(np.ceil(patch_size/downscale))  # 降采样后的补丁大小
		self.n_rows = int(np.ceil(n / n_per_row))  # 行数
		self.n_cols = n_per_row  # 列数
		w = self.n_cols * self.downscaled_patch_size
		h = self.n_rows * self.downscaled_patch_size
		# 创建画布
		if alpha < 0:
			canvas = Image.new(size=(w,h), mode="RGB", color=bg_color)
		else:
			canvas = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
		
		self.canvas = canvas
		self.dimensions = np.array([w, h])
		self.reset_coord()

	def reset_coord(self):
		"""重置坐标到起始位置"""
		self.coord = np.array([0, 0])

	def increment_coord(self):
		"""递增坐标到下一个补丁位置"""
		assert np.all(self.coord<=self.dimensions)
		# 如果当前行还有空间，向右移动
		if self.coord[0] + self.downscaled_patch_size <=self.dimensions[0] - self.downscaled_patch_size:
			self.coord[0]+=self.downscaled_patch_size
		else:
			# 否则换到下一行
			self.coord[0] = 0 
			self.coord[1]+=self.downscaled_patch_size
		

	def save(self, save_path, **kwargs):
		"""
		保存画布
		
		Args:
			save_path (str): 保存路径
			**kwargs: 其他保存参数
		"""
		self.canvas.save(save_path, **kwargs)

	def paste_patch(self, patch):
		"""
		粘贴补丁到画布
		
		Args:
			patch (PIL.Image): 要粘贴的补丁图像
		"""
		assert patch.size[0] == self.patch_size
		assert patch.size[1] == self.patch_size
		# 调整补丁大小并粘贴
		self.canvas.paste(patch.resize(tuple([self.downscaled_patch_size, self.downscaled_patch_size])), tuple(self.coord))
		self.increment_coord()

	def get_painting(self):
		"""获取画布图像"""
		return self.canvas

class Contour_Checking_fn(object):
	"""
	轮廓检查函数基类
	用于检查点是否在轮廓内
	"""
	def __call__(self, pt): 
		"""检查点是否在轮廓内（需要子类实现）"""
		raise NotImplementedError

class isInContourV1(Contour_Checking_fn):
	"""
	轮廓检查函数版本1
	检查点本身是否在轮廓内
	"""
	def __init__(self, contour):
		"""
		Args:
			contour: 轮廓
		"""
		self.cont = contour

	def __call__(self, pt): 
		"""
		检查点是否在轮廓内
		
		Args:
			pt: 点坐标
			
		Returns:
			int: 1表示在轮廓内，0表示不在
		"""
		return 1 if cv2.pointPolygonTest(self.cont, pt, False) >= 0 else 0

class isInContourV2(Contour_Checking_fn):
	"""
	轮廓检查函数版本2
	检查补丁中心点是否在轮廓内
	"""
	def __init__(self, contour, patch_size):
		"""
		Args:
			contour: 轮廓
			patch_size: 补丁大小
		"""
		self.cont = contour
		self.patch_size = patch_size

	def __call__(self, pt): 
		"""
		检查补丁中心点是否在轮廓内
		
		Args:
			pt: 补丁左上角坐标
			
		Returns:
			int: 1表示在轮廓内，0表示不在
		"""
		return 1 if cv2.pointPolygonTest(self.cont, (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2), False) >= 0 else 0

class isInContourV3_Easy(Contour_Checking_fn):
	"""
	轮廓检查函数版本3（简单版）
	检查4个点中是否有1个点在轮廓内即可通过测试
	"""
	def __init__(self, contour, patch_size, center_shift=0.5):
		"""
		Args:
			contour: 轮廓
			patch_size: 补丁大小
			center_shift: 中心偏移比例
		"""
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
		
	def __call__(self, pt): 
		"""
		检查4个点中是否有1个点在轮廓内
		
		Args:
			pt: 补丁左上角坐标
			
		Returns:
			int: 1表示通过，0表示不通过
		"""
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			# 计算4个检查点（中心周围的4个点）
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		# 只要有一个点在轮廓内就返回1
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) >= 0:
				return 1
		return 0

class isInContourV3_Hard(Contour_Checking_fn):
	"""
	轮廓检查函数版本3（困难版）
	检查4个点是否全部在轮廓内才能通过测试
	"""
	def __init__(self, contour, patch_size, center_shift=0.5):
		"""
		Args:
			contour: 轮廓
			patch_size: 补丁大小
			center_shift: 中心偏移比例
		"""
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
		
	def __call__(self, pt): 
		"""
		检查4个点是否全部在轮廓内
		
		Args:
			pt: 补丁左上角坐标
			
		Returns:
			int: 1表示全部在轮廓内，0表示有不在的
		"""
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			# 计算4个检查点（中心周围的4个点）
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		# 所有点都必须在轮廓内
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) < 0:
				return 0
		return 1



		