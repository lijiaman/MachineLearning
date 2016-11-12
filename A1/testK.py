import numpy as np 
from run_knn import run_knn
import utils
import matplotlib.pyplot as plt

k_choices = [1, 3, 5, 7, 9]

train_inputs, train_targets = utils.load_train()
valid_inputs, valid_targets = utils.load_valid()

k_acc = {}
x = []
y = []
for k in k_choices:
	predict = run_knn(k, train_inputs, train_targets, valid_inputs)

	accuracy = np.mean(predict == valid_targets)
	k_acc[k] = accuracy
	x.append(k)
	y.append(accuracy)

print k_acc
plt.scatter(x, y)
plt.xlabel('k')
plt.ylabel('Validation accuracy')
plt.show()

test_inputs, test_targets = utils.load_test()
k_best = 5
k_arr = [k_best-2, k_best, k_best+2]
for k_i in k_arr:
	pred = run_knn(k_best, train_inputs, train_targets, test_inputs)
	acc = np.mean(pred == test_targets)
	print "k values is:"
	print k_i
	print "Test Accuracy is :"
	print acc


