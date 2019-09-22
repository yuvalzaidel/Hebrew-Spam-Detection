# -*- coding: utf-8 -*-
import model
from sys import argv
import codecs
import time

from test_driver import test_driver
from util import get_datasets

import sys
sys.path.insert(1, './whatsapp')
import whatsapp

send_whatsapp = True
block_all_models_if_k_big = False
split_num = 0.25
k_fold_number = 5
include_dependency_tree_feature = True
include_pos_feature = False
spam_max_length = None
non_spam_max_length = 6000
use_morphological = False
use_both = False


def go(model_index):
    global send_whatsapp
    global split_num
    global k_fold_number
    global include_dependency_tree_feature
    global include_pos_feature
    global spam_max_length
    global non_spam_max_length
    global block_all_models_if_k_big
    global use_morphological
    global use_both

    if not use_morphological:
        include_dependency_tree_feature = False
        include_pos_feature = False


    if k_fold_number < 1:
        print("K is invalid, must be 1 or bigger.")
        return

    if block_all_models_if_k_big and model_index == 0 and k_fold_number > 1:
        print("Can't train for all models at once with k_fold_num > 1 , because of memory problems.")
        print("Please change k_fold_number to be 1, or choose a single model (1-5)")
        return

    start_time = time.time()

    print()
    print("==========")
    print()

    datasets = get_datasets(spam_max_length, non_spam_max_length, use_both, use_morphological)
    print("Starting cross-validation, with all the datasets:")
    """
    for x1 in datasets:
        print (x1)
        for x2 in datasets[x1]:
            print(x2)
            for x3 in datasets[x1][x2]:
                print(x3, datasets[x1][x2][x3])
    """

    for dataset in datasets.keys():
        print ("Testing model %d (%s) for dataset '%s':"%(model_index, model.model_desc(model_index), dataset))
        test_driver(model.model_desc(model_index), model.Model, datasets[dataset], model_index, split=split_num, k_fold_num=k_fold_number,
                    include_dependency_tree_feature=include_dependency_tree_feature, include_pos_feature=include_pos_feature, dataset_name=dataset)
        print()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@           Finished dataset %s           @@@@@"%dataset)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print()


    end_time = time.time()
    elapsed = end_time - start_time
    print()
    print("All time: %d seconds."%elapsed)
    print()
    print("FINISHED")

    if send_whatsapp:
        # get rid of none, to send message to whatsapp
        if (spam_max_length is None):
            spam_max_length = "None"
        if (non_spam_max_length is None):
            non_spam_max_length = "None"

        # send message to whatsapp
        msg="""הסתיימה הרצה בשרת.
        k = %d
        datasets = %s
        model index = %d
        model name = %s
        
        dependency_trees_feature = %r
        pos_feature = %r
        spam_max_length = %s
        non_spam_max_length = %s
        
        Total time: %d seconds"""%(k_fold_number, str(list(datasets.keys())), model_index, model.model_desc(model_index),include_dependency_tree_feature, include_pos_feature,str(spam_max_length),str(non_spam_max_length),elapsed)

        whatsapp.send_msg(msg)




def print_usage():
    print()
    print("Usage:")
    print('python go.py <n> - run for the given model number (0,1,2,3,4,5)')
    print()

if len(argv) < 2:
    print_usage()
else:
    if argv[1] in ['0','1','2','3','4','5']:
        go(int(argv[1]))
    else:
        print_usage()
        