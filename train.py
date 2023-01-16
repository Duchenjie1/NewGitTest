import time
import os
random_seed = 20200804
os.environ['PYTHONHASHSEED'] = str(random_seed)
import copy
import argparse
import pdb
import collections
import sys
import random
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
from test import run_from_train
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, PhotometricDistort, RandomSampleCrop
from torch.utils.data import Dataset, DataLoader

# assert torch.__version__.split('.')[1] == '4'


print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

	parser = argparse.ArgumentParser(description='Simple training script for training a CTracker network.')

	parser.add_argument('--dataset', default='csv', type=str, help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--model_dir', default='./ctracker/', type=str, help='Path to save the model.')
	# parser.add_argument('--root_path', default='/dockerdata/home/changanwang/Dataset/Tracking/MOT17Det/', type=str, help='Path of the directory containing both label and images')
	parser.add_argument('--root_path', default='/home/du/PycharmProjects/CTracker-master/MOT20_ROOT/', type=str,
						help='Path of the directory containing both label and images')
	parser.add_argument('--csv_train', default='train_annots.csv', type=str, help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', default='train_labels.csv', type=str, help='Path to file containing class list (see readme)')
	
	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=60)

	parser = parser.parse_args(args)
	print(parser)  #输出上述设置的参数
	
	print(parser.model_dir)   # ./ctracker/
	if not os.path.exists(parser.model_dir):
	   os.makedirs(parser.model_dir)

	# Create the data loaders
	if parser.dataset == 'csv':
		if (parser.csv_train is None) or (parser.csv_train == ''):
			raise ValueError('Must provide --csv_train when training on COCO,')

		if (parser.csv_classes is None) or (parser.csv_classes == ''):
			raise ValueError('Must provide --csv_classes when training on COCO,')
        #transforms.Compose:transforms是pytorch的图像预处理包，例如裁剪、旋转等等，一般采用compose把多个步骤整合到一起. 调用dataloader.py中的CSVDataset类、RandomSampleCrop类:随机裁剪图像、PhotometricDistort:光度失真，Augmenter：将ndarrays转化为Tensors
		dataset_train = CSVDataset(parser.root_path, train_file=os.path.join(parser.root_path, parser.csv_train), class_list=os.path.join(parser.root_path, parser.csv_classes), \
			transform=transforms.Compose([RandomSampleCrop(), PhotometricDistort(), Augmenter(), Normalizer()]))#transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	# sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
	# ba_resnet50_fca:batch_size=4, ba_resnext50_32x4d_fca:batch_size=2, ba_resnext101_32x8d_fca:batch_size=1
	sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)   #divide into groups, one group = one batch,sampler.groups[0]=[3211,3212,3213,3214],sampler.groups[1]=[3215,3216,3217,3218],...
	dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)

	# Create the model
	if parser.depth == 50:
		retinanet = model.resnext50_32x4d(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnext101_32x4d(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 50, 101')

	# if parser.depth == 18:
	# 	retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 34:
	# 	retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 50:  #renet50
	# 	retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 101:
	# 	retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	# elif parser.depth == 152:
	# 	retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	# else:
	# 	raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()  #DataParallel:实现多GPU训练

	#get the number of models parameters
	print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in retinanet.parameters()])))


	retinanet.training = True

	# optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
	optimizer = optim.Adam(retinanet.parameters(), lr=5e-5)  #Adam优化器，计算高效、快速，适用于不稳定目标函数，自然的实现学习率的调整

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)  #学习率调整,optimizer:网络优化器； patience:容忍网络的性能不提升的次数，高于这个次数就降低学习率；verbose：每次更新向stdout输出一条信息

	loss_hist = collections.deque(maxlen=500)   #deque是一个双端队列，可以从两端append数据，如果需要随机访问，不建议用deque. deque的优势可以从两边append, appendleft数据，这是list不具备的

	retinanet.train()
	retinanet.module.freeze_bn()  #Freeze BatchNorm layers

	print('Num training images: {}'.format(len(dataset_train)))  #输出训练图片数目，MOT20为8931
	total_iter = 0
	for epoch_num in range(parser.epochs):

		retinanet.train()   #模型训练
		retinanet.module.freeze_bn()
		
		epoch_loss = []
		
		for iter_num, data in enumerate(dataloader_train):  #iter_num:迭代次数, data:{'img','annot','img_next','annot_next'}
			try:
				total_iter = total_iter + 1
				optimizer.zero_grad()  #梯度初始化为0，把loss关于weight的导数变成0

                #下面求分类分支损失、成对边界框回归损失和ID确认分支损失，data['img']:第t-1帧的输入图像, data['img_next']: 第t帧的图像，需要step into my code
				(classification_loss, regression_loss), reid_loss = retinanet([data['img'].cuda().float(), data['annot'], data['img_next'].cuda().float(), data['annot_next']])
			
				classification_loss = classification_loss.mean()  #多GPU训练训练，每个GPU返回一个loss，合到主GPU就是一个list，则需要loss.mean，下同
				regression_loss = regression_loss.mean()  #多GPU训练训练，取均值
				reid_loss = reid_loss.mean()  #多GPU训练训练，取均值

				# loss = classification_loss + regression_loss + track_classification_losses
				loss = classification_loss + regression_loss + reid_loss  #总的损失
				
				if bool(loss == 0):
					continue

				loss.backward() #反向传播求梯度

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)   #梯度裁剪：解决梯度消失或者爆炸问题，即设定阈值

				optimizer.step()  #优化器更新梯度参数

				loss_hist.append(float(loss))  #存储每次迭代的总损失
				epoch_loss.append(float(loss))
                #Running loss：np.mean(loss_hist),即历史损失的均值
				print('Epoch: {} | Iter: {} | Cls loss: {:1.5f} | Reid loss: {:1.5f} | Reg loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(reid_loss), float(regression_loss), np.mean(loss_hist)))
			except Exception as e:
				print(e)
				continue

		scheduler.step(np.mean(epoch_loss))	

	retinanet.eval()  #测试

	torch.save(retinanet, os.path.join(parser.model_dir, 'model_final.pt'))  #保存最终训练的模型文件
	run_from_train(parser.model_dir, parser.root_path)  #测试模型(不需要通过test.py也可以进行测试)

if __name__ == '__main__':
	main()
