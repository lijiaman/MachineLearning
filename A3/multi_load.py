import numpy as np 
import string
import random
import skimage
import skimage.io
import skimage.transform
import os
import sys
sys.path.append('/ais/gobi4/fashion/data/Category-Attribute-Prediction/')
img_path = '/ais/gobi4/fashion/scene/train/'
tmp = '/ais/gobi4/fashion/data/Category-Attribute-Prediction/'
#Build a dictionary for img path and category ID for the convenience of train,test,val split.

def load_image(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

def load_batchsize_images(batch_size=64):
	path_batch = []
	x_batch = []
	y_batch = []
	with open('train_multi_category.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(line[0])
                        arr = []
                        for i in xrange(24):
                            arr.append(float(line[i+1]))
    		        y_batch.append(arr)
    	
	f.close()
#        print("y_batch.shape")
#        print np.asarray(y_batch).shape
#        print y_batch[0]
#        print y_batch[3]
	for path in path_batch:
		x_batch.append(load_image(img_path+path))
	return np.asarray(x_batch), np.asarray(y_batch)
                
def load_val_images(batch_size=64):
	path_batch = []
	x_batch = []
	y_batch = []
	with open('val_multi_category.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(line[0])
                        arr = []
                        for i in xrange(24):
                            arr.append(float(line[i+1]))
    		        y_batch.append(arr)
    	
	f.close()
#        print("y_batch.shape")
#        print np.asarray(y_batch).shape
#        print y_batch[0]
#        print y_batch[3]
	for path in path_batch:
		x_batch.append(load_image(img_path+path))
	return np.asarray(x_batch), np.asarray(y_batch)

def build_annotation():
    trainfolder_path = '/ais/gobi4/fashion/scene/train/'
    with open('train_category.txt', 'w') as w_f:
        with open('train.txt', 'rb') as f:
            data = f.readlines()
            for line in data:
                line = line.split()
                img_path = trainfolder_path+"{:0>5d}".format(string.atoi(line[0], 10))+".jpg"
                w_f.write(img_path+'\t'+line[1]+'\n')
        f.close()
    w_f.close()

def split_val():
    with open('val_multi_category.txt', 'w') as w_f:
        with open('multilabel_truth.txt', 'rb') as f:
            lines = random.sample(f.readlines(), 2000)
            for line in lines:
                w_f.write(line)
        f.close()
    w_f.close()

def build_train():
    with open('train_multi_category.txt', 'w') as w_f:
        with open('multilabel_truth.txt', 'rb') as all_f:
            lines = all_f.readlines()
            for line in lines:
                ori_line = line
                line = line.split()
                exist = 0
                with open('val_multi_category.txt', 'rb') as f:
                    datas = f.readlines()
                    for data in datas:
                        data = data.split()
                        if data[0] == line[0]:
                            exist = 1
                           # print("exist")
                    if exist == 0:
                        w_f.write(ori_line)
                f.close()
        all_f.close()
    w_f.close()

#build_annotation()
#split_val()
#build_train()
load_batchsize_images()
