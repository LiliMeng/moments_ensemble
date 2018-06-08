from __future__ import division
import numpy as np
import os

def clean_key(key):
    if key.endswith('\n'):
        key = key[:-1]
    return key

# Load Moumita's hash table
hash_table = np.load("/home/lili/Downloads/hash_table.npy")
hash_table = np.expand_dims(hash_table, 0)[0]
hash_table_r = dict((hash_table[key], key)for key in hash_table.keys())

# Load Lili's hash table, and create a reverse lookup table
f = open('/home/lili/Video/TRN-pytorch/pretrain/moments_categories.txt')
lines= f.readlines()
hash_lili = dict((clean_key(value), key) for (key, value) in enumerate(lines))
hash_lili_r = dict((value, clean_key(key)) for (value, key) in enumerate(lines))

# Conversion dict 
moumita2lili = dict((hash_table[key], hash_lili[key]) for key in hash_table)

# Get the attn labels
labels = np.load('/home/lili/Video/spatial_temporal_LSTM/att_result/att_all_valid_labels_13.npy').reshape(-1)
logits = np.load('/home/lili/Video/spatial_temporal_LSTM/att_result/att_all_valid_logits_13.npy').reshape(-1, 339)
names = np.load('/home/lili/Video/spatial_temporal_LSTM/att_result/att_all_valid_names_13.npy').reshape(-1)
assert labels.shape == names.shape

sorted_logits = np.zeros_like(logits)
for hkey in hash_table.keys():
    # print(hkey, hash_lili_r[hash_lili[hkey]])
    new_ind = hash_lili[hkey]
    print("%s %s -> %s %s" % (hkey, hash_table[hkey], new_ind, hash_lili_r[hash_lili[hkey]]))
    sorted_logits[:, new_ind] = logits[:, hash_table[hkey]]

# Convert all the labels from hash_table_r --> hash_lili
conv_labels = [moumita2lili[lab] for lab in labels]

# Sanity Check
true_labels = [hash_lili[n.split('/')[0]] for n in names] 
assert conv_labels == true_labels

#np.save('/home/lili/Video/moments_ensemble/data/attention_LSTM/moumita2lili.npy', moumita2lili)
np.save('/home/lili/Video/moments_ensemble/data/attention_LSTM/attn_logits_13.npy', sorted_logits)
np.save('/home/lili/Video/moments_ensemble/data/attention_LSTM/attn_labels_13.npy', conv_labels)