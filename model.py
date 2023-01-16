import numpy as np
import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
from lib.nms import cython_soft_nms_wrapper


# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#resnext
model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):  #C3_size:512, C4_size:1024, C5_size:2048
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)  #M5
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)  #P5

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)   #P6:out_channel:256，W,H变为P5的1/2

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1) #P7:out_channel:256，W,H变为P6的1/2

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)   #M5
        P5_upsampled_x = self.P5_upsampled(P5_x)   #2x
        P5_x = self.P5_2(P5_x)   #P5
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x   #M4
        P4_upsampled_x = self.P4_upsampled(P4_x)  #2X
        P4_x = self.P4_2(P4_x)   #P4

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x   #M3
        P3_x = self.P3_2(P3_x)    #P3

        P6_x = self.P6(C5)    #P6

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)   #P7

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):  #成对边界框回归(Paired Boxes Regression)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，为每个目标返回一个边界框对
    def __init__(self, num_features_in, num_anchors=1, feature_size=256): #number_feature_in:512
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)  #3*3,卷积后的特征尺度不发生变化
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)  #3*3,卷积后的特征尺度不发生变化
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)   #3*3,卷积后的特征尺度不发生变化
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)    #3*3,卷积后的特征尺度不发生变化
        self.act4 = nn.ReLU()
        #成对回归模型中的最后一层num_anchors*8=8：用1个anchor预测相邻两帧中的同一目标对象bbox(x1,y1,x2,y2)的偏移量
        self.output = nn.Conv2d(feature_size, num_anchors*8, kernel_size=3, padding=1)  #输出最终结果,num_anchors*8 = 1*8
        # print(self)  #du add
        # print('...')

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors，我觉得应该是8*num_anchors(成对目标),成对边界框的(x1,y1,x2,y2)
        out = out.permute(0, 2, 3, 1)  #permute:将tensor的维度换位, out: (B,W,H,C=8)

        return out.contiguous().view(out.shape[0], -1, 8)  #contiguous: 即深拷贝，对out使用了.contiguous()后，改变后者的值，对out没有任何影响. view(B, W x H, C=8)

class ClassificationModel(nn.Module):  #目标分类分支(Object Classification branch)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，最后使用一个3*3的卷积加sigmoid激活函数预测置信度
    def __init__(self, num_features_in, num_anchors=1, num_classes=80, prior=0.01, feature_size=256):  #参数也和ID确认分支一样
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes   #self.num_classes: 1(行人)
        self.num_anchors = num_anchors   #self.num_anchors: 1
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1) #num_features_in: 512, feature_size:256
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)  #Conv2d(256，1，3，1)
        self.output_act = nn.Sigmoid()
        # print(self)
        # print('...')

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out


class ReidModel(nn.Module):  #ID确认分支(ID Verification branch)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，最后使用一个3*3的卷积加sigmoid激活函数预测置信度
    def __init__(self, num_features_in, num_anchors=1, num_classes=80, prior=0.01, feature_size=256):
        super(ReidModel, self).__init__()

        self.num_classes = num_classes  #self.num_classes: 1
        self.num_anchors = num_anchors  #self.num_anchors: 1
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1) #num_features_in: 512, feature_size:256
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
        # print(self)
        # print('...')

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out


class BAResNeXt(nn.Module):
    #width_per_groups不需要删除
    def __init__(self, num_classes, block, layers, zero_init_residual=False, groups=32,
                 width_per_group=4, replace_stride_with_dilation=None, norm_layer=None, reduction=16):   #以resnet50为例，layers:[3，4，6，3]，num_classed:1
        self.inplanes = 64
        super(BAResNeXt, self).__init__()

        if norm_layer is None:   #norm_layer代替了BatchNorm2d
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        self.dilation = 1
        self.reduction = reduction
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group  #resnext50_32x4d: width_per_group=4; resnext101_32x8d: width_per_group=8

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  #nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias)
        self.bn1 = norm_layer(64)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''
        #ba_resnet_fca使用expand
        self.expand = 1
        if 'Bottleneck' in str(block):
            self.expand = 4
        self.gate1 = self._make_gate(64*self.expand)
        self.gate2 = self._make_gate(128*self.expand)
        self.gate3 = self._make_gate(256*self.expand)
        self.gate4 = self._make_gate(512*self.expand)
        
        '''
        self.layer1 = self._make_layer(block, 64, layers[0], groups)    #dim=64, n_block = layers[0]=3
        self.layer2 = self._make_layer(block, 128, layers[1], groups, stride=2,
                                       dilate=replace_stride_with_dilation[0])  #dim=128, n_block = layers[1]=4
        self.layer3 = self._make_layer(block, 256, layers[2], groups, stride=2,
                                       dilate=replace_stride_with_dilation[1])  #dim=256, n_block = layers[2]=6
        self.layer4 = self._make_layer(block, 512, layers[3], groups, stride=2,
                                       dilate=replace_stride_with_dilation[2])  #dim=512, n_block = layers[3]=3

        if block == BasicBlock:   #这里开始有所区别，加了FPN(特征金字塔)
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:  #layers[1]:4(就是layer2的Bottleneck数), layers[2]:6(就是layer3的Bottleneck数),layers[3]:3(就是layer4的Bottleneck数)
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]  #[512, 1024, 2048]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])    #调用PyramidFeatures():特征金字塔(结合之前的resnet)
        self.num_classes = num_classes  #num_classes: 1

        self.regressionModel = RegressionModel(512)    #调用回归模型RegressionModel()，combined features的feature size：512，每个尺度(P3，P4，P5，P6，P7)上结合两个相邻帧的特征进行后续的预测
        #print(num_classes)
        self.classificationModel = ClassificationModel(512, num_classes=num_classes) #调用分类分支ClassificationModel()，num_classes=1(行人)
        self.reidModel = ReidModel(512, num_classes=num_classes)  #调用ID确认分支ReidModel()


        self.anchors = Anchors()  #调用anchors.py下的Anchors类，对于P3，P4，P5，P6，P7特征级，分别使用38，86，112，156，328尺度大小的anchor预测目标，且anchor的宽高比为1：2.9, stride分别为8，16，32，64，128
        self.regressBoxes = BBoxTransform()  #边界框BBox转换，输出的是预测框(pred_box)
        self.clipBoxes = ClipBoxes()  #？
        
        self.focalLoss = losses.FocalLoss()  #分类分支：FocalLoss，和回归损失，没有搞明白
        self.reidfocalLoss = losses.FocalLossReid()  #ID确认分支：FocalLossReid，没有搞明白
        #参数初始化
        '''
        for m in self.modules():   #手动去初始化的代码
            if isinstance(m, nn.Conv2d):  #使用isinstance来判断m属于什么类型
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  #m中的weight, bias都是variable(变量)，为了能学习参数以及反向传播
                m.bias.data.zero_()
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  #kaiming初始化
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:  # zero_init_residual =False
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
        

        #CTracker add
        prior = 0.01

        '''
        self.output_reg = nn.Conv2d(feature_size, num_anchors*8, kernel_size=3, padding=1)

        self.output_cls'''
        #分类模型参数(weight,bias)初始化
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        # ID确认模型参数(weight,bias)初始化
        self.reidModel.output.weight.data.fill_(0)
        self.reidModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        # 边界框回归模型参数(weight,bias)初始化
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()  #Freeze BatchNorm layers
    '''
        def _make_gate(self, planes):
        #previous block-gate
        inplanes = 2 * (planes // self.reduction)
        outplanes = planes // self.reduction
        Gate = nn.Sequential(
            nn.Linear(inplanes, outplanes, bias=False),
            nn.Sigmoid()
        )
        return Gate
    
    '''
    def _make_layer(self, block, planes, blocks, groups, stride=1, dilate=False):  #resnet有4个make layers, 一共16个blocks
        norm_layer = self._norm_layer  # 其实就是BatchNorm2d
        downsample = None
        previous_dilation = self.dilation  #previous_dilation:1
        if dilate:  #BA_resnet
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:  #block.expansion:4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            ) #得到downsample, residual = self.downsample, out += residual,planes * block.expansion(4):为了和Bottleneck最后一个卷积层的输入通道保持一致，可以进行残差操作

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups, base_width=self.base_width))    #调用Bottleneck(),只有第一个block才有dowmsample,且第2~4个layer的第1个block, 第2个卷积层才会出现stride = stride
        self.inplanes = planes * block.expansion   #layers第一次加载完Bottleneck后，self.inplanes就变成输入通道的4倍，后续几个Bottleneck的inplanes保持不变
        for i in range(1, blocks):  #加载完第1个Bottleneck, 加载剩下的Bottlenecks
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  #将BN层变为eval(),即不更新统计running_mean和running_val,但是优化、更新BN层的weight和bias的学习参数

    def forward(self, inputs, last_feat=None):
        if self.training:
            img_batch_1, annotations_1, img_batch_2, annotations_2 = inputs
            img_batch = torch.cat([img_batch_1, img_batch_2], 0)  #将相邻两个batch的图像在batch维度上拼接起来[8，3，512，512]，如果bs=1,则是相邻两帧图像进行拼接
            annotations = torch.cat([annotations_1, annotations_2], 0) #将相邻两个batch的GT在batch维度上拼接起来[8，38，6]，如果bs=1,则是相邻两帧GT进行拼接
        else:
            img_batch = inputs
            
        x = self.conv1(img_batch)  #resnet中layer1之前的操作，[B,C,H,W]=[bs*2,64,256,256]
        x = self.bn1(x)  #[bs*2,64,256,256]
        x = self.relu(x)  #[bs*2,64,256,256]
        x = self.maxpool(x)  #[bs*2,64,128,128]

        x1 = self.layer1(x)  #resnet的layer1, [bs*2,256,128,128]
        x2 = self.layer2(x1)   #resnet的layer2, [bs*2,512,64,64]
        x3 = self.layer3(x2) #resnet的layer3, [bs*2,1024,32,32]
        x4 = self.layer4(x3)  #resnet的layer4, [bs*2,2048,16,16]
        #feature[0].shape:[bs*2,256,64,64], feature[1].shape:[bs*2,256,32,32], feature[2].shape:[bs*2,256,16,16], feature[3].shape:[bs*2,256,8,8], feature[4].shape:[bs*2,256,4,4]
        features = self.fpn([x2, x3, x4])

        anchors = self.anchors(img_batch.shape[2:])  #将图片的H和W作为输入，得到一帧图像在5个特征级上的anchors：[1,5456,4], 一共5456个anchors

        if self.training:  #执行训练过程
            track_features = []  #就是Combined features, 即分别加载5个特征级上，相邻两个batch在通道C上拼接后的特征，例如track_feature[0].shape:[4,512,64,64]
            for ind, featmap in enumerate(features):
                featmap_t, featmap_t1 = torch.chunk(featmap, chunks = 2, dim = 0)  #featmap_t：第t帧的特征，featmap_t1:第t+1帧的特征. torch.chunk：对张量进行分块，chunks为分割的块数，dim表示沿着哪个轴分块
                track_features.append(torch.cat((featmap_t, featmap_t1), dim = 1)) #torch.cat((featmap_t, featmap_t1), dim = 1):通道C上将两帧的特征进行拼接

            reg_features = []  #加载边界框回归特征
            cls_features = []  #加载分类特征
            reid_features = []  #加载ID确认特征
            for ind, feature in enumerate(track_features):
                reid_mask = self.reidModel(feature)  #执行ID确认模型,得到ID确认模型的注意力映射

                reid_feat = reid_mask.permute(0, 2, 3, 1) #维度转换:(B,W,H,C)
                batch_size, width, height, _ = reid_feat.shape
                reid_feat = reid_feat.contiguous().view(batch_size, -1, self.num_classes)  #[bs,width*height,1],P3特征级为[4，64*64=4096，1]

                cls_mask = self.classificationModel(feature)  #执行分类模型，得到分类模型的注意力映射

                cls_feat = cls_mask.permute(0, 2, 3, 1) #维度转换:(B,W,H,C)
                cls_feat = cls_feat.contiguous().view(batch_size, -1, self.num_classes)  ##[bs,width*height,1]

                reg_in = feature * reid_mask * cls_mask  #联合注意力模型(JAM)，ID确认分支和目标分类分支预测的置信度得分用作注意力映射，这两个是互补的

                reg_feat = self.regressionModel(reg_in)  #reg_feat: [B, W x H, C=8]
                
                reg_features.append(reg_feat)    #加载5个特征级(P3,P4,P5,P6,P7)的边界框回归特征
                cls_features.append(cls_feat)    #加载5个特征级(P3,P4,P5,P6,P7)的目标分类特征
                reid_features.append(reid_feat)  #加载5个特征级(P3,P4,P5,P6,P7)的ID确认特征
            regression = torch.cat(reg_features, dim=1)  #将5个特征级的边界框回归特征进行拼接，regression.shape:[4,4096+1024+256+64+16=5456, 8]
            classification = torch.cat(cls_features, dim=1) #将5个特征级的目标分类特征进行拼接，classification.shape:[4,4096+1024+256+64+16=5456, 1]
            reid = torch.cat(reid_features, dim=1)  #将5个特征级的ID确认特征进行拼接，classification.shape:[4,4096+1024+256+64+16=5456, 1]

            return self.focalLoss(classification, regression, anchors, annotations_1, annotations_2), self.reidfocalLoss(reid, anchors, annotations_1, annotations_2)

        else:  #执行测试过程
            if last_feat is None:  #上一帧图像的features
                return torch.zeros(0), torch.zeros(0, 4), features  #idx=0返回
            track_features = [] #idx=1进入该代码
            for ind, featmap in enumerate(features):
                track_features.append(torch.cat((last_feat[ind], featmap), dim = 1))   #当前帧的特征(C=256)和上一帧的特征(C=256)在通道维度上进行拼接. track_features[0].shape:[1,512,136,240], track_features[1].shape:[1,512,68,120],track_features[2].shape:[1,512,34,60],track_features[3].shape:[1,512,17,30],track_features[4].shape:[1,512,9,15]


            reg_features = []  # 加载边界框回归特征
            cls_features = []  # 加载分类特征
            reid_features = []  # 加载ID确认特征
            for ind, feature in enumerate(track_features):
                reid_mask = self.reidModel(feature)  # 执行ID确认模型,得到ID确认模型的注意力映射

                # out is B x C x W x H, with C = n_classes + n_anchors
                reid_feat = reid_mask.permute(0, 2, 3, 1)  # 维度转换:(B,W,H,C)
                batch_size, width, height, _ = reid_feat.shape
                reid_feat = reid_feat.contiguous().view(batch_size, -1, self.num_classes)  # [1,width*height,1]

                cls_mask = self.classificationModel(feature)  # 执行分类模型，得到分类模型的注意力映射
                # out is B x C x W x H, with C = n_classes + n_anchors
                cls_feat = cls_mask.permute(0, 2, 3, 1)  # 维度转换:(1,W,H,C)
                cls_feat = cls_feat.contiguous().view(batch_size, -1, self.num_classes)  ##[1,width*height,1]

                reg_in = feature * reid_mask * cls_mask  # 联合注意力模型(JAM)，ID确认分支和目标分类分支预测的置信度得分用作注意力映射，这两个是互补的

                reg_feat = self.regressionModel(reg_in)  # reg_feat: [1, W x H, 8]
                
                reg_features.append(reg_feat)  # 加载5个特征级(P3,P4,P5,P6,P7)的边界框回归特征
                cls_features.append(cls_feat)  # 加载5个特征级(P3,P4,P5,P6,P7)的目标分类特征
                reid_features.append(reid_feat)  # 加载5个特征级(P3,P4,P5,P6,P7)的ID确认特征
            regression = torch.cat(reg_features, dim=1)  # 将5个特征级的边界框回归特征进行拼接，regression.shape:[4,32640+8160+2040+510+135=43485, 8]
            classification = torch.cat(cls_features, dim=1)  # 将5个特征级的目标分类特征进行拼接，classification.shape:[4,32640+8160+2040+510+135=43485, 1]
            reid_score = torch.cat(reid_features, dim=1)  # 将5个特征级的ID确认特征进行拼接，classification.shape:[4,32640+8160+2040+510+135=43485, 1]

            # anchors = np.concatenate((anchors, anchors), axis=1)
            anchors = torch.cat((anchors, anchors), dim=2)  #将相邻2帧图像在5个特征级上的anchors进行拼接：[1,43486,4+4=8], 一共43486个anchors，8表示2帧图像的anchor坐标(x1,y1,x2,y2)

            transformed_anchors = self.regressBoxes(anchors, regression)  ##transformed_anchors就是pred_boxes:[1,43485,8], 其中8表示相邻2帧的(x1,y1,x2,y2). 将输入的相邻两帧所有特征级上的anchor坐标和边界框回归特征，返回的是预测的bbox(重要)
            # transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.05)[0, :, 0]  #threshold:0.05, 背景都被滤除

            if scores_over_thresh.sum() == 0:  #MOT20-01 frame 2: 525
                # no boxes to NMS, just return
                return torch.zeros(0), torch.zeros(0, 4), features

            classification = classification[:, scores_over_thresh, :]  #[1，525，1]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]  #[1，525，8]，其中8表示相邻2帧的(x1,y1,x2,y2)，滤除bbox内是背景的预测框
            scores = scores[:, scores_over_thresh, :]  #[1，525，1]，前景的置信度得分
            reid_score = reid_score[:, scores_over_thresh, :]  #[1，525，1]，前后2帧预测的边界框为同一目标且都为前景的置信度得分
            #从5种不同尺度特征生成的边界框都使用soft-nms进行后处理,阈值th=0.7. 拼接后shape为[1，525，10],然后再[0，:,:]=[525,10],转为numpy形式
            final_bboxes = cython_soft_nms_wrapper(0.7, method='gaussian')(torch.cat([transformed_anchors[:, :, :].contiguous(), scores, reid_score], dim=2)[0, :, :].cpu().numpy())  #[158，10]

            return final_bboxes[:, -2], final_bboxes, features  #final_bboxes[:, -2]:前景的置信度得分


def resnext50_32x4d(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BAResNeXt(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)   #num_classes:1, 调用ResNet函数，创建ResNet模型, 4个layers的Bottleneck数量分别为：3，4，6，3
    if pretrained:  #pretrained = True
        # model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d'], model_dir='.'), strict=False)  #最上面给出了resnet50.pt的url地址下载指定的.pth文件. 深度学习框架提供的“Model Zoo”,有大量的大数据集上预训练的可供下载的模型
        checkpoint = torch.load('/home/du/PycharmProjects/CTracker-master/resnext50_32x4d-7cdf4587.pth') #torch.load(‘the/path/of/.pth’)  #如果已经下载好模型了，可以直接通过torch.load()导入
        model.load_state_dict(checkpoint, strict=False)  #将resnet50中的参数加载到新创建的model中，必须加上strict=False
        print(model)

    return model
#
# def resnext101_32x4d(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNeXt(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d'], model_dir='.'), strict=False)
#         checkpoint = torch.load('/home/du/PycharmProjects/CTracker-master/resnext101_32x4d-a5af3160.pth') #torch.load(‘the/path/of/.pth’)  #如果已经下载好模型了，可以直接通过torch.load()导入
#         model.load_state_dict(checkpoint, strict=False)  #将resnet50中的参数加载到新创建的model中，必须加上strict=False
#         print(model)
#     return model

# def resnext101_32x8d(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNeXt(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d'], model_dir='.'), strict=False)
#         checkpoint = torch.load('/home/du/PycharmProjects/CTracker-master/resnext101_32x8d-8ba56ff5.pth')  # torch.load(‘the/path/of/.pth’)  #如果已经下载好模型了，可以直接通过torch.load()导入
#         model.load_state_dict(checkpoint, strict=False)  # 将resnet50中的参数加载到新创建的model中，必须加上strict=False
#         print(model)
#     return model


# def resnet18(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
#     return model
#
#
# def resnet34(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
#     return model
#
#
# def resnet50(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)   #num_classes:1, 调用ResNet函数，创建ResNet模型, 4个layers的Bottleneck数量分别为：3，4，6，3
#     if pretrained:  #pretrained = True
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)  #最上面给出了resnet50.pt的url地址下载指定的.pth文件. 深度学习框架提供的“Model Zoo”,有大量的大数据集上预训练的可供下载的模型
#         checkpoint = torch.load('/home/du/PycharmProjects/CTracker-master/resnet50-19c8e357.pth') #torch.load(‘the/path/of/.pth’)  #如果已经下载好模型了，可以直接通过torch.load()导入
#         model.load_state_dict(checkpoint, strict=False)  #将resnet50中的参数加载到新创建的model中，必须加上strict=False
#         print(model)
#
#     return model
#
# def resnet101(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
#     return model
#
#
# def resnet152(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BAResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
#     return model
