import numpy as np 
import string
import random
import skimage
import skimage.io
import skimage.transform
import os
import sys
import csv
sys.path.append('/ais/gobi4/fashion/data/Cross-domain-Retrieval/')
trainfolder_path = '/ais/gobi4/fashion/scene/train/'

def csv_to_txt(filepath):
    with open('train.txt', 'a') as w_f:
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for id, label, usage in reader:
                #print id, label, usage
                w_f.write(id+'\t'+label+'\t'+usage+'\n')
        csvfile.close()
    w_f.close()

def txt_add_zero(filepath):
    with open(filepath, 'a') as w_f:
        for i in xrange(2000):
            w_f.write(str(i+971)+'\t'+str(0)+'\n')

def txt_to_csv(filepath):
    in_txt = csv.reader(open(filepath, "rb"), delimiter='\t')
    csv_file = "test_result.csv"
    out_csv = csv.writer(open(csv_file, 'wb'))

    out_csv.writerows(in_txt)

def split_category(label):
    with open('c8_val.txt', 'w') as w_f:
        with open('val_category.txt') as f:
            lines = f.readlines()
            for line in lines:
                ori_line = line
                line = line.split()
                if line[1] == str(label):
                    w_f.write(ori_line)
        f.close()
    w_f.close()

def load_image(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    #resized_img = skimage.transform.resize(crop_img, (227, 227))
    resized_img = skimage.transform.resize(crop_img, (224, 224))

    return resized_img

def load_train_val_dataset():
    id_set = []
    img_set = []
    label_set = []
    with open('train.txt', 'rb') as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            id_set.append(line[0])
            label_set.append(string.atoi(line[1],10)-1)
            img_path = trainfolder_path+"{:0>5d}".format(string.atoi(line[0], 10))+".jpg"
            #print img_path
            img_set.append(load_image(img_path))
        
    f.close()
    return np.asarray(img_set, dtype=np.float32), np.asarray(label_set, dtype=np.int32)

def imgpath_to_label():
    folder_path = '/ais/gobi4/fashion/scene/public_test/val'
    img_paths = os.listdir(folder_path)
    with open('public_test.txt', 'a') as w_f:
        for img in img_paths:
            label = string.atoi(img.strip('.jpg'), 10)
            w_f.write(str(label)+'\t'+folder_path+'/'+img+'\n')
    w_f.close()

def private_img_to_label():
    folder_path = '/ais/gobi4/fashion/test_128'
    img_paths = os.listdir(folder_path)
    with open('public_test.txt', 'a') as w_f:
        for img in img_paths:
            label = string.atoi(img.strip('.jpg'), 10)+971
            w_f.write(str(label)+'\t'+folder_path+'/'+img+'\n')
    w_f.close()
            

def load_test_dataset():
    id_set = []
    img_set = []
    with open('public_test.txt', 'rb') as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            img_set.append(load_image(line[1]))
            id_set.append(line[0])
    f.close()
    return np.asarray(id_set, dtype=np.int32), np.asarray(img_set, dtype=np.float32)

def resample():
    with open('/ais/gobi4/fashion/resample_val.txt', 'w') as w_f:
        with open('val_category.txt', 'rb') as f:
            lines = f.readlines()
            for line in lines:
                ori_line = line
                line = line.split()
                if line[1] == '4':
                    for i in xrange(5):
                        w_f.write(ori_line)
                elif line[1] == '3':
                    for i in xrange(3):
                        w_f.write(ori_line)
                elif line[1] == '6':
                    for i in xrange(23):
                        w_f.write(ori_line)
                elif line[1] == '7':
                    for i in xrange(111):
                        w_f.write(ori_line)
                elif line[1] == '8':
                    for i in xrange(44):
                        w_f.write(ori_line)
                else:
                    w_f.write(ori_line)
        f.close()
    w_f.close()
#csv_to_txt('/ais/gobi4/fashion/411a3/train.csv')
#load_train_dataset(
#imgpath_to_label()
#txt_add_zero("prediction_result.txt")
txt_to_csv("prediction_result.txt")
#resample()
#private_img_to_label()
#split_category(8)
