# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 00:54:55 2020

@author: Faust
"""

import numpy as np
import pandas as pd
import random
import math

count_words = 50  # count words
target_count = 5  # classes
size_dataset = 500 #size dataset

dataset = np.random.randint(2, size=[size_dataset, count_words])

noise1 = np.random.randint(2, size=[size_dataset])
noise2 = np.random.randint(2, size=[size_dataset])
random_index = np.random.randint(45, size=[10])

for j in random_index:
    for i in range(0, size_dataset):
        dataset[i, j] = dataset[i, j] or noise1[j] or noise2[j]

target = np.random.randint(target_count, size=size_dataset)

for idx_target in range(0, target_count):
    for i in np.where(target == idx_target):
        if (random_index[0] + idx_target) in random_index:
            j = random_index[0] + idx_target
        else:
            j = random_index[0] + idx_target + 1
        dataset[i, j] = 1
df = pd.DataFrame(data=dataset)
df['class'] = target

# sum words of doc 
#df.iloc[0, 0:count_words].sum()

# 
#df.iloc[:, 5].sum()

TF = np.zeros(count_words)


for i in range(0, target_count):
    cl_i = df.loc[df['class'] == i, df.columns != 'class'].values
    word_occurrences_in_class = np.zeros(count_words)
    # count words in doc
    for j in range(0, count_words):
        word_occurrences_in_class[j] = cl_i[:, j].sum()
    # count words in class 
    word_in_document_count = cl_i.sum()
    TF_c = np.asarray([w / word_in_document_count for w in word_occurrences_in_class])
    if i <= 0:
        TF = np.append([TF], [TF_c], axis=0)
        TF = np.delete(TF, (0), axis=0)
    else:
        TF = np.append(TF, [TF_c], axis=0)

w_d = [df.iloc[:, i].sum() for i in range(0, count_words)]
col_D = df.iloc[:, 0:count_words].sum().sum()
IDF = np.asarray([math.log(col_D / w) for w in w_d])
TF_IDF = np.zeros((target_count, count_words))
for i in range(0, target_count):
    TF_IDF[i] = np.asarray([TF[i, j] * IDF[j] for j in range(0, count_words)])
