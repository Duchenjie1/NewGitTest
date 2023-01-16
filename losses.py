import numpy as np
import torch
import torch.nn as nn
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  #

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])  #a[:, 2]:x2, a[:, 0]:x1,每一个anchor需要和所有GT做比较
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])  #a[:, 3]:y2, a[:, 1]:y2

    iw = torch.clamp(iw, min=0)  #iw约束最小值0，将没有交集的赋值为0
    ih = torch.clamp(ih, min=0)  #iw约束最小值0，将没有交集的赋值为0

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih  #并集

    ua = torch.clamp(ua, min=1e-8)  #约束并集

    intersection = iw * ih  #交集

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  #[5456,4]
        ## transform from (x1, y1, x2, y2) -> (x_ctr, y_ctr, w, h), anchor坐标形式转化
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            # get gt box: 获得真实目标框
            classification = classifications[j, :, :]  #获取每帧的目标分类特征
            regression = regressions[j, :, :]    #获取每帧的边界框回归特征

            bbox_annotation = annotations1[j, :, :]  #获取当前batch第一帧的GT
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            bbox_annotation_next = annotations2[j, :, :] #获取下一个batch第一帧的GT
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]

            # return zero loss if no gt boxes exist：如果没有真实目标框存在，则损失为0
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)  #将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量
            # IoU matching for latter anchors' label assign, IOU匹配后锚的标签分配, truth：真实，代表GT，prior:先验的意思，代表anchor
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) #IoU.shape:[5456,30] num_anchors x num_annotations，计算每个anchor和bbox_GT的IOU
            best_truth_overlap, best_truth_idx = torch.max(IoU, dim=1) #dim表示要消去的维度，这边维度1消去，找出5456个anchor分为与哪一个目标框(best_truth_idx,0~29)的IoU最大，因此best_truth_overlap.shape: num_anchors x 1

            best_prior_overlap, best_prior_idx = torch.max(IoU, dim=0) #找出30个目标框分别与哪一个anchor(best_prior_idx,0~5455)的IoU最大， num_annotations
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)  #按照指定的维度轴dim 根据index去对应位置，将原tensor用参数val值填充 # ensure best prior
            for j in range(best_prior_idx.size(0)):  #best_prior_idx.size(0)：30, best_truth_idx里面还记录第0，1，2，...,29个GT分别与哪一个的anchor的IOU最大
                best_truth_idx[best_prior_idx[j]] = j  #难点，例如best_truth_idx[5268]=0，表示第1个gt框与第5269个anchor的IOU最大

            # compute the label for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(best_truth_overlap, 0.4), :] = 0  #阈值为0.4, torch.lt:逐元素比较best_truth_overlap是否小于0.4, 背景

            positive_indices = torch.ge(best_truth_overlap, 0.5)  #前景(即目标行人)索引，anchor与gt的IOU大于阈值0.5,则存在前景目标进行匹配. torch.ge():逐元素比较best_truth_overlap是否大于等于0.5，若是大于则为True，若不是则为False，前景
            #那么best_truth_overlap在0.4~0.5区间的targets[]=-1
            num_positive_anchors = positive_indices.sum()  #前景anchors总数：196

            assigned_annotations = bbox_annotation[best_truth_idx, :]

            targets[positive_indices, :] = 0   #positive_indices有196个元素为True, positive_indices.sum():196
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1  #前景anchor的标签设置为1(一共196个), long()函数：将数字或字符串转换为一个长整型数. targets其实是一个矩阵，因此后面ssigned_annotations[positive_indices, 4].long()都是0，表示第1列

            # compute the loss for classification, 计算分类损失
            alpha_factor = torch.ones(targets.shape).cuda() * alpha  #alpha=0.25

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)  #torch.where()函数的作用是按照一定的规则合并两个tensor类型,满足torch.eq(targets, 1.)的位置，返回0.25，否则，返回0.75
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)  #torch.eq(targets, 1.).sum():196
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)  #gamma = 2.0

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))   #BCE loss

            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())  #torch.ne()：判断两个向量是否不相等

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))  #cls_loss.sum()必须要除以前景anchor的总数

            # compute the loss for regression，计算回归损失
            if positive_indices.sum() > 0:  #196
                assigned_annotations = assigned_annotations[positive_indices, :]   #assigned_annotations(已匹配的标注)：表示和正样本anchor匹配(IOU>0.5)的GT (重要)
                # assigned_annotations_next = assigned_annotations_next[positive_indices, :]
                assigned_ids = assigned_annotations[:, 5]  #已匹配标注的目标ID，可能存在一个目标ID同时被两个anchor匹配的情况(基本存在这种情况)
                assigned_annotations_next = torch.zeros_like(assigned_annotations)  #初始化下一帧匹配的标注
                reg_mask = torch.ones(assigned_annotations.shape[0], 8).cuda()
                # only learn regression for chained-anchors, whose id of target in two frames are the same
                for m in range(assigned_annotations_next.shape[0]):
                    assigned_id = assigned_annotations[m, 5]
                    match_flag = False
                    for n in range(bbox_annotation_next.shape[0]):
                        if bbox_annotation_next[n, 5] == assigned_id:  #在下一帧的gt中是否能找到与当前帧已匹配的目标ID
                            match_flag = True  #在下一帧的gt中能找到与当前帧已匹配的目标ID
                            assigned_annotations_next[m, :] = bbox_annotation_next[n, :]
                            break
                    if match_flag == False:
                        reg_mask[m, 4:] = 0  #在下一帧的gt中不能找到与当前帧已匹配的目标ID,则直接设置为0(可能存在这种情况，当下一帧目标消失在视野中时)
    
                anchor_widths_pi = anchor_widths[positive_indices]  #前景anchor的宽
                anchor_heights_pi = anchor_heights[positive_indices]  #前景anchor的高
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]  #前景anchor的中心坐标
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                # transform GT to x,y,w,h
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights
                #du: next frame匹配的标注，transform GT to x,y,w,h
                gt_widths_next  = assigned_annotations_next[:, 2] - assigned_annotations_next[:, 0]
                gt_heights_next = assigned_annotations_next[:, 3] - assigned_annotations_next[:, 1]
                gt_ctr_x_next   = assigned_annotations_next[:, 0] + 0.5 * gt_widths_next
                gt_ctr_y_next   = assigned_annotations_next[:, 1] + 0.5 * gt_heights_next

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)  #夹紧
                gt_heights = torch.clamp(gt_heights, min=1)  #夹紧

                gt_widths_next  = torch.clamp(gt_widths_next, min=1)  #夹紧
                gt_heights_next = torch.clamp(gt_heights_next, min=1)  #夹紧
                # compute regression targets
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi  #当前帧偏移dx
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi  #当前帧偏移dy
                targets_dw = torch.log(gt_widths / anchor_widths_pi)   #当前帧偏移dw
                targets_dh = torch.log(gt_heights / anchor_heights_pi)  #当前帧偏移dh

                targets_dx_next = (gt_ctr_x_next - anchor_ctr_x_pi) / anchor_widths_pi  #下一帧偏移dx
                targets_dy_next = (gt_ctr_y_next - anchor_ctr_y_pi) / anchor_heights_pi  #下一帧偏移dy
                targets_dw_next = torch.log(gt_widths_next / anchor_widths_pi)  #下一帧偏移dw
                targets_dh_next = torch.log(gt_heights_next / anchor_heights_pi)  #下一帧偏移dh


                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dx_next, targets_dy_next, targets_dw_next, targets_dh_next))  #链锚的偏移量:[8，196],沿着一个新维度对输入张量序列进行连接
                targets = targets.t()  #[196，8], .t()是.transpose()函数的简写版本

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]]).cuda()  #为什么要除以后面的系数？
                # compute losses
                regression_diff = torch.abs(targets - regression[positive_indices, :]) * reg_mask  #regression[positive_indices, :]:前景anchor的特征

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
class FocalLossReid(nn.Module):
    def forward(self, classifications, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []

        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]

            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]
            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)  #夹紧

            # compute the label for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            # IoU matching for latter anchors' label assign, current frame(当前帧)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            best_truth_overlap, best_truth_idx = torch.max(IoU, dim=1) # num_anchors x 1

            _, best_prior_idx = torch.max(IoU, dim=0) # num_annotations
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
            for j in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[j]] = j
 
            assigned_annotations = bbox_annotation[best_truth_idx, :]
            # IoU matching for latter anchors' label assign, next frame(下一帧)
            IoU_next = calc_iou(anchors[0, :, :], bbox_annotation_next[:, :4]) # 所有帧的anchors的[x1,y1,x2,y2]都相同，num_anchors x num_annotations
            best_truth_overlap_next, best_truth_idx_next = torch.max(IoU_next, dim=1) # num_anchors x 1

            _, best_prior_idx_next = torch.max(IoU_next, dim=0) # num_annotations
            best_truth_overlap_next.index_fill_(0, best_prior_idx_next, 2)  # ensure best prior
            for j in range(best_prior_idx_next.size(0)):
                best_truth_idx_next[best_prior_idx_next[j]] = j  #记录与每个gt最优匹配(IOU最大)的anchor,例如best_truth_idx[5268]=0，表示第5268个anchor与第0个gt框的IOU最大

            assigned_annotations_next = bbox_annotation_next[best_truth_idx_next, :]

            reid_pos_thres = 0.5  #author: 0.7

            # label for reid branch is 1 if and only if the following criterion is satisfied (前后两帧预测的边界框为同一目标且都为前景，则ID确认分支的真实标记设置为1)
            valid_samples = (torch.ge(best_truth_overlap, reid_pos_thres) & torch.ge(best_truth_overlap_next, 0.4)) | (torch.ge(best_truth_overlap_next, reid_pos_thres) & torch.ge(best_truth_overlap, 0.4))  #有效anchor实例:212个
            targets[valid_samples & torch.eq(assigned_annotations[:, 5], assigned_annotations_next[:, 5]), :] = 1  #situation1: 前后两帧预测的边界框为同一目标且都为前景，则ID确认分支的真实标记设置为1
            targets[valid_samples & torch.ne(assigned_annotations[:, 5], assigned_annotations_next[:, 5]), :] = 0  #situation2: 有4(212-208)种满足IOU大于阈值的情况，但前后两帧的边界框不属于同一目标(即前后两帧的ID不相等)，真实标记设置为0

            targets[torch.lt(best_truth_overlap, 0.4) | torch.lt(best_truth_overlap_next, 0.4), :] = 0  #situation3:背景的真实标记也设为0,5091个anchor属于该类情况
            targets[torch.lt(best_truth_overlap, reid_pos_thres) & torch.ge(best_truth_overlap, 0.4) & torch.lt(best_truth_overlap_next, reid_pos_thres) & torch.ge(best_truth_overlap_next, 0.4), :] = -1  #situation4: 有153种情况是大于等于0.4小于0.5，真实标签设为-1. torch.eq(targets, -1).sum()=153

            positive_indices = torch.ge(targets, 1)  #等价于torch.eq(targets, 1)，因为targets.max()=1
            
            num_positive_anchors = positive_indices.sum()  #208

            # compute losses
            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)  #满足torch.eq(targets, 1.)的位置，返回0.25，否则，返回0.75
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)  #估计是凑好的

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))  #和目标分类损失的计算基本一致

        return torch.stack(classification_losses).mean(dim=0, keepdim=True)

    