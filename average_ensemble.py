'''
Average ensemble:
Average the predicted class probability from each models
Currently, only ResNet50 and TRN are used
'''

import numpy as np
import os
import time
import torch
import fnmatch

def get_class(fn):
    return fn.split('_')[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(logits, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = 1
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_resnet50_logits(logits_path, label_dict):

	logits_list = []
	names_list = []
	label_list = []
	fn = os.listdir(logits_path)
	
	#Sort to ensure matching order
	fn_names= np.sort(fn)

	for file_name in fn_names:
		if file_name.split("_")[1].split(".")[0]=="logits":
			logits_tmp = np.load(os.path.join(logits_path, file_name))
			logits_list.append(logits_tmp)
			label_tmp = file_name.split("_")[0]
			label_list.append(np.repeat(np.asarray(label_dict[label_tmp]),100))

		if file_name.split("_")[1].split(".")[0]=="names":
			name_tmp = np.load(os.path.join(logits_path, file_name))
			names_list.append(name_tmp)

	all_logits = np.concatenate(np.asarray(logits_list), axis =0)
	all_labels = np.concatenate(np.asarray(label_list),axis=0)
	all_names  = np.concatenate(np.asarray(names_list), axis=0)

	sorted_index = np.argsort(all_names)
	all_logits = [all_logits[i] for i in sorted_index]
	all_labels = [all_labels[i] for i in sorted_index]
	all_names = [all_names[i] for i in sorted_index]
	print("all_names in ResNet: ", len(all_names))
	return all_logits, all_labels, all_names
	

def load_TRN_logits(label_dict):

	logits_path = "/home/lili/Video/TRN-pytorch/moments_validation_logits.npy"
	labels_path = "/home/lili/Video/TRN-pytorch/moments_validataion_names.npy"
	logits = np.load(logits_path)
	str_labels  = np.load(labels_path)

	label_list = []
	for i in range(str_labels.shape[0]):
		label_name = str_labels[i].split('/')[0]
		int_label = label_dict[label_name]
		label_list.append(int_label)

	labels = np.asarray(label_list)

	sorted_index = np.argsort(str_labels)
	
	logits = [logits[i] for i in sorted_index]
	labels = [labels[i] for i in sorted_index]
	str_labels = [str_labels[i] for i in sorted_index]

	return logits, labels, str_labels

def load_audio_logits(correct_dict):
	logits_path = "/home/lili/Video/moments_ensemble/audio_converted_logits.npy"
	labels_path = "/home/lili/Video/moments_ensemble/audio_converted_labels.npy"
	name_path = "/home/lili/Video/moments_ensemble/audio_converted_paths.npy"
	
	all_logits = np.load(logits_path)

	all_labels = np.load(labels_path)
	
	all_names = np.load(name_path)

	sorted_index = np.argsort(all_names)

	all_logits = [all_logits[i] for i in sorted_index]
	all_labels = [all_labels[i] for i in sorted_index]
	all_names = [all_names[i] for i in sorted_index]

	return all_logits, all_labels, all_names

def get_label_dict(label_list):

    with open(label_list) as f1:
       lines_labels = f1.readlines()

    label_name_dict = {}

    for i in range(len(lines_labels)):
    
        label_name = lines_labels[i].split('\n')[0]
        label_number = i
        label_name_dict[label_name]=i

    return label_name_dict

def main():

	proc_start_time = time.time()

	label_dict = get_label_dict("/home/lili/Video/TRN-pytorch/pretrain/moments_categories.txt")
	
	resnet_logits, resnet_labels, resnet_names = load_resnet50_logits("/home/lili/Desktop/moments_raw_valid_features", label_dict)

	TRN_logits, TRN_labels, TRN_names = load_TRN_logits(label_dict)

	audio_logits, audio_labels, audio_names = load_audio_logits(label_dict)

	top1 = AverageMeter()
	top5 = AverageMeter()

	total_num = 33900

	for i in range(total_num):
		assert(resnet_labels[i]==TRN_labels[i]==audio_labels[i])
		Resnet_per_video_logits_resnet = np.expand_dims(np.mean(resnet_logits[i],axis=0), axis=0)
		
		TRN_per_video_logits = TRN_logits[i]

		per_audio_logits = audio_logits[i]

		per_video_logits = 1/3*(TRN_per_video_logits+Resnet_per_video_logits_resnet+per_audio_logits)
	
		per_video_label = np.expand_dims(audio_labels[i], axis=0)
		per_video_logits = torch.from_numpy(per_video_logits)
		per_video_label  = torch.from_numpy(per_video_label)

		prec1, prec5 = accuracy(per_video_logits, per_video_label, topk=(1, 5))
		top1.update(prec1[0], 1)
		top5.update(prec5[0], 1)

		cnt_time = time.time() - proc_start_time
		print('video {} done, total {}/{}, average {:.3f} sec/video, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                  total_num ,
                                                                  float(cnt_time) / (i+1), top1.avg, top5.avg))



main()