import torch
import torch.nn as nn
import numpy as np
from DCT_extration import MultiSpectralAttentionLayer

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BA_module_resnet(nn.Module): # BA_module for the backbones of ResNet and ResNext
    def __init__(self, pre_channels, cur_channel, reduction=16):  #reduction
        super(BA_module_resnet, self).__init__()
        self.pre_fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
                nn.Linear(cur_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(cur_channel // reduction, cur_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.size()

        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion).view(b, cur_c, 1, 1)

        return att_weights


class BasicBlock(nn.Module):  #ResNet18和ResNet34
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  #就是Block
    expansion = 4  #扩展系数expansion表示的是单元输出与输入张量的通道数之比. 对于ResNet34，这个比是1,而对于ResNet50，这个比是4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=4, reduction=16):  #reduction
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups  # 重新计算输出层，base_width：width of bottleneck d
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False) #inplanes:输入通道数，planes:输出通道数
        # self.bn1 = nn.BatchNorm2d(planes*2)
        self.bn1 = nn.BatchNorm2d(width)  #width等于planes*2
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.ba = BA_module_resnet([width, width], 4*planes, reduction)  #调用了BA_module_resnet()函数，需要研究一下
        self.feature_extraction = nn.AdaptiveAvgPool2d(1)  # GAP(Global average pooled)

        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])  #resnet50_fca,并且输入图像随机裁剪为224*224. 第64层的特征图长*宽：56*56; 第128层的特征图长*宽：28*28；以此类推
        # c2wh = dict([(64, 128), (128, 64), (256, 32), (512, 16)])  #du: resnext50_fca由于输入图像随机裁剪为512*512
        self.feature_extraction1 = MultiSpectralAttentionLayer(width, c2wh[planes], c2wh[planes],  reduction=reduction,
                                                               freq_sel_method = 'top16')  #c2wh[planes=64]：56，freq_sel_method：频率分量选择方法
        self.feature_extraction2 = MultiSpectralAttentionLayer(width, c2wh[planes], c2wh[planes], reduction=reduction,
                                                               freq_sel_method='top16')
        self.feature_extraction3 = MultiSpectralAttentionLayer(4 * planes, c2wh[planes], c2wh[planes], reduction=reduction,
                                                               freq_sel_method='top16')

    def forward(self, x):
        residual = x

        out = self.conv1(x) #[bs*2,128,128,128] #
        out = self.bn1(out)  #[bs*2,128,128,128]
        out = self.relu(out)  #[bs*2,128,128,128]
        # F1 = self.feature_extraction(out)  #GAP, F11:[bs*2,128,1,1]
        # print(F11.shape)
        F1 = self.feature_extraction1(out)  #Fcanet, F1:[bs*2,128,1,1]

        out = self.conv2(out) #[bs*2,128,128,128]
        out = self.bn2(out)  #[bs*2,128,128,128]
        out = self.relu(out) #[bs*2,128,128,128]
        # F2 = self.feature_extraction(out)  #GAP, F22:[bs*2,128,1,1]
        # print(F22.shape)
        F2 = self.feature_extraction2(out) #Fcanet, F2:[bs*2,128,1,1]

        out = self.conv3(out)  #[bs*2,256,128,128]
        out = self.bn3(out)  #[bs*2,256,128,128]
        # F3 = self.feature_extraction(out) #GAP, F33:[bs*2,256,1,1]
        # print(F33.shape)
        F3 = self.feature_extraction3(out) ##Fcanet, F3:[bs*2,256,1,1]
        att = self.ba([F1, F2], F3)
        out = out * att


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()  #转成tensor
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()  #转成tensor
        else:
            self.std = std

    def forward(self, boxes, deltas):

        # print(boxes.shape)
        # print(deltas.shape)

        # print(deltas[0,0,:])
        # print(deltas[0,1,:])
        # print(deltas[0,2,:])
        # print(deltas[0,3,:])
        # print(deltas[0,4,:])
        # print(deltas[0,5,:])
        # print(deltas[0,6,:])
        # print(deltas[0,7,:])

        #boxes:相邻两帧的anchor坐标，transform (x1,y1,x2,y2) to x,y,w,h
        widths  = boxes[:, :, 2::4] - boxes[:, :, 0::4]
        heights = boxes[:, :, 3::4] - boxes[:, :, 1::4]
        ctr_x   = boxes[:, :, 0::4] + 0.5 * widths
        ctr_y   = boxes[:, :, 1::4] + 0.5 * heights
        #deltas：regression features(相邻两帧所有特征级上的边界框回归特征)
        dx = deltas[:, :, 0::4] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1::4] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2::4] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3::4] * self.std[3] + self.mean[3]

        # print(ctr_x.shape)
        # print(dx.shape)
        # print(widths.shape)
        # 预测框坐标(x,y,w,h)
        #du:相同特征级上，相邻2帧的anchor坐标是相同的(ctr_x,ctr_y,widths,heights)，由于regression中相邻两帧的特征不同，导致生成dx,dy,dw,dh也不同，因此相邻两帧的pred_bbox坐标也不同
        pred_ctr_x = ctr_x + dx * widths   #预测框的中心坐标x

        pred_ctr_y = ctr_y + dy * heights  #预测框的中心坐标y
        pred_w     = torch.exp(dw) * widths   #预测框的宽w
        pred_h     = torch.exp(dh) * heights  #预测框的高h
        #预测框坐标，transform (x,y,w,h) to (x1,y1,x2,y2)
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes_x1 = pred_boxes_x1[:, :, :, np.newaxis]  #np.newaxis:插入新维度
        pred_boxes_y1 = pred_boxes_y1[:, :, :, np.newaxis]
        pred_boxes_x2 = pred_boxes_x2[:, :, :, np.newaxis]
        pred_boxes_y2 = pred_boxes_y2[:, :, :, np.newaxis]

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=3).reshape(boxes.shape)  #pred_boxes:[1,43485,8], 其中8表示相邻2帧的(x1,y1,x2,y2)，torch.stack: 在dim=3(新的维度)将预测框的坐标(x1,y1,x2,y2)进行拼接

        #pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0::4] = torch.clamp(boxes[:, :, 0::4], min=0)
        boxes[:, :, 1::4] = torch.clamp(boxes[:, :, 1::4], min=0)

        boxes[:, :, 2::4] = torch.clamp(boxes[:, :, 2::4], max=width)
        boxes[:, :, 3::4] = torch.clamp(boxes[:, :, 3::4], max=height)
      
        return boxes


# import torch
# import torch.nn as nn
# import numpy as np

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

# class BBoxTransform(nn.Module):

#     def __init__(self, mean=None, std=None):
#         super(BBoxTransform, self).__init__()
#         if mean is None:
#             self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
#         else:
#             self.mean = mean
#         if std is None:
#             self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
#         else:
#             self.std = std

#     def forward(self, boxes, deltas):

#         widths  = boxes[:, :, 2] - boxes[:, :, 0]
#         heights = boxes[:, :, 3] - boxes[:, :, 1]
#         ctr_x   = boxes[:, :, 0] + 0.5 * widths
#         ctr_y   = boxes[:, :, 1] + 0.5 * heights

#         dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
#         dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
#         dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
#         dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

#         pred_ctr_x = ctr_x + dx * widths
#         pred_ctr_y = ctr_y + dy * heights
#         pred_w     = torch.exp(dw) * widths
#         pred_h     = torch.exp(dh) * heights

#         pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
#         pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
#         pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
#         pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

#         pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

#         return pred_boxes


# class ClipBoxes(nn.Module):

#     def __init__(self, width=None, height=None):
#         super(ClipBoxes, self).__init__()

#     def forward(self, boxes, img):

#         batch_size, num_channels, height, width = img.shape

#         boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
#         boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

#         boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
#         boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
#         return boxes
