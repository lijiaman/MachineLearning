import numpy as np 
import string
import random
import skimage
import skimage.io
import skimage.transform
import os
from PIL import Image
#from imgaug import augmenters as iaa 
import sys
sys.path.append('/ais/gobi4/fashion/data/Category-Attribute-Prediction/')
img_path = '/ais/gobi4/fashion/data/Category-Attribute-Prediction/'
tmp = '/ais/gobi4/fashion/data/Category-Attribute-Prediction/'
#Build a dictionary for img path and category ID for the convenience of train,test,val split.
def load_image(path):
    img = skimage.io.imread(path)
   # pic = Image.open(path)
   # angle = random.randrange(0, 360)
   # r_pic = pic.rotate(angle)
   # img = np.array(r_pic.getdata()).reshape(r_pic.size[0], r_pic.size[1], 3)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
 #   img = tf.image.random_flip_left_right(img)
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
        y_index = []
	with open('part_train.txt') as f:
	#with open('/ais/gobi4/fashion/resample_train.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
    	
	f.close()
        category_num = 8
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for path in path_batch:
		x_batch.append(load_image(path))
	return np.asarray(x_batch), np.asarray(y_batch)
                
def load_val_images(batch_size, label):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
	#with open('/ais/gobi4/fashion/resample_val.txt') as f:
	with open('c'+str(label)+'_val.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
    	
	f.close()
        category_num = 8
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for path in path_batch:
		x_batch.append(load_image(path))
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
    with open('val_category.txt', 'w') as w_f:
        with open('train_category.txt', 'rb') as f:
            lines = random.sample(f.readlines(), 2000)
            for line in lines:
                w_f.write(line)
        f.close()
    w_f.close()

def build_train():
    with open('part_train.txt', 'w') as w_f:
        with open('train_category.txt', 'rb') as all_f:
            lines = all_f.readlines()
            for line in lines:
                line = line.split()
                exist = 0
                with open('val_category.txt', 'rb') as f:
                    datas = f.readlines()
                    for data in datas:
                        data = data.split()
                        if data[0] == line[0]:
                            exist = 1
                           # print("exist")
                    if exist == 0:
                        w_f.write(line[0]+'\t'+line[1]+'\n')
                f.close()
        all_f.close()
    w_f.close()

#build_annotation()
#split_val()
#build_train()
