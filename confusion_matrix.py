from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print(__doc__)

import numpy as np
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


def compute_and_plot_confusion_matrix(pred_label, gt_label):
	assert(pred_label.shape==gt_label.shape)

	# Compute confusion matrix
	cm = confusion_matrix(gt_label, pred_label)
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('audio_validation_confusion_matrix.pdf', format='pdf', dpi=1200)

	return cm

def sort_confusion_matrix_diagonal(confu_matrix):
	diagonal_cm = confu_matrix.diagonal()
	#print(diagonal_cm)
	sorted_diagonal_cm = sorted(enumerate(diagonal_cm), key=lambda x: x[1])
	print(sorted_diagonal_cm)
	return sorted_diagonal_cm

def match_number_and_real_classname(sorted_diagonal_cm, match_list_file):
	print("start to match the number and real classnames")
	with open(match_list_file) as f:
		lines = f.readlines()

	real_class_name_list=[]
	class_num_list=[]
	real_class_name_dict={}
	for i in range(len(lines)):
		real_class_name = lines[i].split(',')[0]
		class_num = int(lines[i].split(',')[1])
		real_class_name_dict[class_num]=real_class_name

	audio_invalid_list = []
	for i in range(len(sorted_diagonal_cm)):
		sorted_class_name=real_class_name_dict[sorted_diagonal_cm[i][0]]
		if sorted_diagonal_cm[i][1]==0:
			audio_invalid_list.append(sorted_diagonal_cm[i][0])
		with open('audio_valid_sorted_classes.txt', 'a') as f_sort:
			print(sorted_diagonal_cm[i][0], sorted_class_name, sorted_diagonal_cm[i][1], file=f_sort)
		

	np.save("audio_invalid_categories.npy", np.asarray(audio_invalid_list))


def main():
	pred_label=np.load("audio_valid_pred_label.npy")
	gt_label=np.load("audio_valid_gt_label.npy")
	print(pred_label.shape)
	print(gt_label.shape)

	confu_matrix = compute_and_plot_confusion_matrix(pred_label, gt_label)

	sorted_diagonal_cm=sort_confusion_matrix_diagonal(confu_matrix)

	match_number_and_real_classname(sorted_diagonal_cm, "./data/moments_categories.txt")

main()