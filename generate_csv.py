# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from matplotlib import rcParams
# config = {
#             "font.family": 'serif',
#             "font.size": 15,
#             "mathtext.fontset": 'stix',
#             "font.serif": ['SimSun'],
#          }
# rcParams.update(config)

# import matplotlib as mpl
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False


root_dir = 'MOT16_ROOT/train/'
root1_dir = 'train/'
result_csv = 'train_annots.csv'

fout = open(result_csv, 'w')

for data_name in os.listdir(root_dir):
	print(data_name)
	gt_path = os.path.join(root_dir, data_name, 'gt', 'gt.txt')
	print(gt_path)
	data_raw = np.loadtxt(gt_path, delimiter=',', dtype='float', usecols=(0,1,2,3,4,5,6,7,8))
	# np.lexsort()是对data_raw按照frame number从小到大进行排序，如果frame number相同，则按照ID从小到大进行排序
	data_sort = data_raw[np.lexsort(data_raw[:,::-1].T)]
	visible_raw = data_sort[:,8]
	# print(data_sort)
	# print(data_sort[-1, 0])
	img_num = data_sort[-1, 0]  #最后一行第1列数据

	# print(data_sort.shape[0])
	box_num = data_sort.shape[0]

	person_box_num = np.sum(data_sort[:,6] == 1)   #activate
	# print(person_box_num)

	for i in range(box_num):
		c = int(data_sort[i, 6])
		v = visible_raw[i]
		if c == 1 and v > 0.1:
			img_index = int(data_sort[i, 0])
			img_name = data_name + '/img1/' + str(img_index).zfill(6) + '.jpg'
			print(
					root1_dir + img_name + ', ' + str(int(data_sort[i, 1])) + ', ' + str(data_sort[i, 2]) + ', ' + str(
						data_sort[i, 3]) + ', ' + str(data_sort[i, 2] + data_sort[i, 4]) + ', ' + str(
						data_sort[i, 3] + data_sort[i, 5]) + ', person\n')
			fout.write(
				root1_dir + img_name + ', ' + str(int(data_sort[i, 1])) + ', ' + str(data_sort[i, 2]) + ', ' + str(
					data_sort[i, 3]) + ', ' + str(data_sort[i, 2] + data_sort[i, 4]) + ', ' + str(
					data_sort[i, 3] + data_sort[i, 5]) + ', person\n')

fout.close()

# fout = open(result_csv, 'w')
data_raw1 = pd.read_csv("train_annots.csv")
data = np.array(data_raw1)
id_num = max(data[:, 1])   #最大ID
flatten = [] #存各个ID的个数

for j in range(1, id_num+1):
	person_id_num = np.sum(data[:, 1] == j)  # ID为j的出现的帧数
	# print("id为"+str(j)+"的个数为:"+str(person_id_num))
	print("id为"+str(j)+"出现的帧数为:{}".format(str(person_id_num)) )
	flatten.append(person_id_num)
print(flatten)

#画id对应的帧数图(纵坐标是帧数flatten，横坐标是id)
# plt.figure(figsize=(10, 5), dpi=80)
# plt.figure()
figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)

# plt.xlim(0, 180)  #MOT17
# plt.ylim(0, 3000)   #MOT17

plt.xlim(0, 1250)  #MOT20
plt.ylim(0, 4500)   #MOT20

index = np.arange(1, id_num+1)
print(index)
plt.bar(index, flatten, 0.4, color="green")

font_label = {'family': 'SimHei',
         'weight': 'normal',
         'size': 23,
         }
font_legend = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 23,
         }
plt.tick_params(labelsize=23)
plt.xlabel("ID",font_legend)
plt.ylabel("帧数",font_label)
# plt.title("MOT20",font_legend)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.show()

'''

import matplotlib as plt
print(plt.get_cachedir())
'''
