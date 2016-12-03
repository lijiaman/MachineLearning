import numpy as np 
import string
import random
import skimage
import skimage.io
import skimage.transform
import os
import sys
import csv

def get_truth_file(dic):
	with open('multilabel_truth.txt', 'w') as w_f:
		with open('train_bonus.txt', 'rb') as f:
		    data = f.readlines()
		    for line in data:
			line = line.split()
			w_f.write(line[0]+'\t')
			cols = len(line)
			arr = np.zeros(24)
			for i in xrange(cols):
				for k in dic.keys():
					if k == line[i]:
						arr[dic[k]-1] = 1
			for a_i in xrange(24):
				w_f.write(str(arr[a_i])+'\t')
			w_f.write('\n')
                f.close()
        w_f.close()

def get_dict():
    with open('train_bonus.txt', 'rb') as f:
        lines = f.readlines()
        label_id = 0
        dic= {}
        for line in lines:
            line = line.split()
            cols = len(line)
            for i in xrange(cols):
                if (not(line[i] in dic.keys())) and i > 0:
                    dic[line[i]] = label_id + 1
                    label_id += 1
       # print dic
        print("num of classes:")#24
        print label_id
    f.close()
    return dic

#dic = get_dict()

#get_truth_file(dic)
