import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]  #确实是P3，P4，P5，P6，P7
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]  #strides:[8，16，32，64，128]
        if ratios is None:
            self.ratios = np.array([2.90, ])  #ratio: [2.9],所有尺度的链锚采用相同的1：2.9的比例扫描(接近行人的比例)
        if scales is None:
            self.scales = [np.array([38.,]), np.array([86.,]), np.array([112.,]), np.array([156.,]), np.array([328.,])]  #给出了5个特征级anchors的初始W和H:[38，38],[86，86],[112，112],[156，156],[328，328]，后续要进行实际行人的长宽比进行调整
        # print(self)
        # print('...')
    def forward(self, image_shape):
        
        image_shape = np.array(image_shape)   #image_shape:[512,512]
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]  #5个特征层级的H和W, P3：[64，64], P4：[32，32], P5：[16，16], P6：[8，8], P7：[4，4]

        # compute anchors over all pyramid levels: 计算5个特征层级的anchors
        all_anchors = np.zeros((0, 4)).astype(np.float32)  #初始化，维度4表示(x1,y1,x2,y2)

        for idx, p in enumerate(self.pyramid_levels):  #对5个特征级生成相应的anchors
            anchors         = generate_anchors(ratios=self.ratios, scales=self.scales[idx])  #生成某个特征图位于原点位置anchors，并将其转换为(x1, y1, x2, y2)的形式
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)  #生成某个特征图所有grid cell点上的anchors的坐标(x1,y1,x2,y2),shifted_anchors.shape:[特征图的长*特征图的宽,4(x1,y1,x2,y2)]，例如P3上为[4096，4]
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)  #5个特征级的所有anchors坐标,[4096,4](P3)+[1024,4](P4)+[256,4](P5)+[64,4](P6)+[16,4](P7) = [5456,4]

        all_anchors = np.expand_dims(all_anchors, axis=0)  #all_anchors.shape: [1,5456,4], np.expand_dims():用于扩展数组的形状，即在all_anchors的相应的axis轴上扩展维度

        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()

def generate_anchors(ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)  #num_anchors:1

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = np.tile(scales, (2, len(ratios))).T   #np.tile: 把scales沿着2行1列矩阵的各个方向进行复制. anchors[:, 2:]:[0.0，38，38]

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]  #areas:38*38=1444

    # correct for ratios,针对行人的长宽比(1:2.9)需要调整anchor的W和H，调整前后的面积相同. 特征图在原点坐标的anchor(x_ctr,y_ctr,w,h),并且x_ctr=y_ctr=0
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))  #np.repeat:用于将numpy数组重复，anchor的W: 22.314
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))   #anchor的H: 22.314(anchor.H)*2.9(ratio) = 64.711

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2), anchor坐标形式转化
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T  #np.tile(anchors[:, 2] * 0.5, (2, 1)): array([[11.157,11.157]]). anchors[:, 0::2]:第0、2列
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T  #anchors[:, 1::2]:第1、3列

    return anchors

def shift(shape, stride, anchors):  #特征图的shape*stride=原始图片的shape
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride  #构造一个表示x轴上的坐标的向量
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride  #构造一个表示y轴上的坐标的向量

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  #np.meshgrid:创建一个特征图P.shape[0]*特征图P.shape[1]的网格点坐标矩阵

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()    #shifts.shape:[4096,4].  np.vstack：垂直把数组堆叠起来, ravel()方法将数组维度拉成一维数组,例如shift_x.ravel().shape为64*64=4096

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]  #A:1，每个grid cell产生一个anchor，和yolov3一样
    K = shifts.shape[0]  #K:4096(P3,64*64),1024(P4,32*32),...,特征图上所有的anchor个数
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  #在笔记本上做出解释，难点
    all_anchors = all_anchors.reshape((K * A, 4))  #特征图所有grid cell点上的anchor的(x1,y1,x2,y2)，all_anchors.shape:[4096(特征图的长*宽),4(x1,y1,x2,y2)]

    return all_anchors

