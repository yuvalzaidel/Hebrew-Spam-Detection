## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
#import pandas as pd  
from sklearn.model_selection import train_test_split 
import sklearn.metrics 
import general_func

from numpy import array
from sklearn.model_selection import KFold

import time

print_wrong_decisions = True

def test_driver(model_desc, classifier_class, data, model_index, split=0.25, k_fold_num = 1, include_dependency_tree_feature=False, include_pos_feature=False, dataset_name="NONE"):
    global print_wrong_decisions

    labels_data = dict(list(data["spam"].items()) + list(data["non_spam"].items()))

    start_time_for_everything = time.time()

    labels = ["non spam", "spam"]
    lambda_for_FalsePositive_list = [1, 99, 999]

    if k_fold_num <= 1:
        k_fold_num = 1
        print("No K-FOLD cross validation set, splitting data by %.2f"%split)
    else:
        print("K-FOLD cross validation: %d-fold" % k_fold_num)

    print("\nprocessing with model %d"%model_index)

    # de-clutters the data
    all_tokens = set(list(data['spam'].keys())) | set(list(data['non_spam'].keys()))

    baseline_accuracy = len(data['non_spam']) / len(all_tokens)

    # building X, y for and getting a train-test split
    X_non_spam = list(data['non_spam'].keys())
    X_spam = list(data['spam'].keys())

    y_non_spam = [0] * len(X_non_spam)
    y_spam = [1] * len(X_spam)

    X = X_non_spam + X_spam
    y = y_non_spam + y_spam

    assert len(X) == len(y)

    data = []
    # get a train-test split
    if k_fold_num > 1:
        # train-test split by K
        for i in range(len(X)):
            data.append((X[i], y[i]))
        kfold = KFold(k_fold_num, True, 1)
        kfold = kfold.split(data)
    else:
        # train-test split by 'split' parameter, if k wasn't provided
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)
        data.append([])
        data.append([])
        for i in range(len(X_train)):
            data[0].append((X_train[i], y_train[i]))
        for i in range(len(X_test)):
            data[1].append((X_test[i], y_test[i]))

        # making a supervised single kfold, that is fitting to the format
        kfold = [data]

    all_models = None
    wrong_classification = {}
    kfold_index = 0
    all_time_testing = 0
    first_loop_flag = True

    for train, test in kfold:
        kfold_index += 1
        # if k was provided, splitting it to X_train, y_train, X_test, y_test
        # if k wasn't provided, these lists are ready from before
        if(k_fold_num > 1):
            print("\nStarting training for iteration %d"%kfold_index)
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for pair_index in train:
                X_train.append(data[pair_index][0])
                y_train.append(data[pair_index][1])
            for pair_index in test:
                X_test.append(data[pair_index][0])
                y_test.append(data[pair_index][1])

        # initializing the class and training models
        classifier = classifier_class(include_dependency_tree_feature, include_pos_feature, model_index, labels_data)

        all_models = classifier.train(X_train, y_train)

        # test on the test split
        print("using model %d's vectorization function to vectorize test data ..."%model_index)

        # initialize all information once for every model
        if first_loop_flag:
            first_loop_flag = False
            count_all_scores = {}
            minimum_scores = {}
            count_all_WAcc = {}
            minimum_WAcc = {}
            count_all_TCR = {}
            minimum_TCR = {}
            above_1_count_TCR = {}
            for model in all_models.keys():
                count_all_scores[model] = 0
                minimum_scores[model] = 99999
                count_all_WAcc[model] = [0] * len(lambda_for_FalsePositive_list)
                minimum_WAcc[model] = [99999] * len(lambda_for_FalsePositive_list)
                count_all_TCR[model] = [0] * len(lambda_for_FalsePositive_list)
                minimum_TCR[model] = [99999] * len(lambda_for_FalsePositive_list)
                above_1_count_TCR[model] = [0] * len(lambda_for_FalsePositive_list)


        models_y_pred = {}
        start_time_testing = time.time()

        # Testing all the models, saving the predicts
        for model in all_models.keys():
            print("Testing model %s (%d/%d)...." %(model, len(models_y_pred.keys())+1, len(all_models.keys())))
            # progress bar init
            classifier.initialize_pbar(len(X_test))
            y_pred = all_models[model].predict(list(map(classifier.vectorize, X_test)))
            models_y_pred[model] = y_pred
            # progress bar close
            classifier.close_pbar()

        end_time_testing = time.time()
        elapsed = end_time_testing - start_time_testing
        all_time_testing += elapsed
        print()
        print("Testing for iteration %d/%d took %s seconds." %(kfold_index, k_fold_num, elapsed))


        print()
        print("Evaluation:")
        # printing current iteration scores for every model
        for model in models_y_pred.keys():
            print("---------------")
            print("Iteration %d , Model %s"%(kfold_index, model))
            y_pred = models_y_pred[model]

            # saving the mistaken messages
            if print_wrong_decisions:
                for i in range(len(y_pred)):
                    if(y_test[i] != y_pred[i]):
                        wrong_msg_classification = X_test[i], y_pred[i]
                        if wrong_msg_classification not in wrong_classification:
                            wrong_classification[wrong_msg_classification] = {}
                        if model not in wrong_classification[wrong_msg_classification]:
                            wrong_classification[wrong_msg_classification][model] = 0
                        wrong_classification[wrong_msg_classification][model] += 1

            # calculating and printing scores, and saving scores for summary
            accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
            margin_from_baseline = accuracy_score - baseline_accuracy
            accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)

            cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
            recall = cm[0][0] / (cm[0][0] + cm[0][1])
            precision = cm[1][1] / (cm[1][1] + cm[1][0])

            F1 = 2 * (precision * recall) / (precision + recall)
            count_all_scores[model] += F1
            minimum_scores[model] = min(minimum_scores[model], F1)

            print("baseline_accuracy: {:.3f}".format(baseline_accuracy))
            print("model %d's accuracy: %.3f" % (model_index, accuracy_score))
            print("positive margin from baseline: {:.3f}".format(margin_from_baseline))
            print("confusion matrix:")
            general_func.print_cm(cm, labels)
            print("Recall (sensitivity): %.4f"%recall)
            print("Precision: %.4f"%precision)
            print("F1 score: %.4f"%F1)

            non_spam_count = cm[0][0] + cm[1][0]
            spam_count = cm[0][1] + cm[1][1]
            for i in range(len(lambda_for_FalsePositive_list)):
                lambda_for_FalsePositive = lambda_for_FalsePositive_list[i]
                wacc = (lambda_for_FalsePositive * cm[0][0] + cm[1][1]) / (lambda_for_FalsePositive * non_spam_count + spam_count)
                count_all_WAcc[model][i] += wacc
                minimum_WAcc[model][i] = min(minimum_WAcc[model][i], wacc)
                print("WAcc (Weighted accuracy) (lambda = %d): %.4f" %(lambda_for_FalsePositive, wacc))

                tcr = spam_count / (lambda_for_FalsePositive * cm[1][0] + cm[0][1])
                count_all_TCR[model][i] += tcr
                minimum_TCR[model][i] = min(minimum_TCR[model][i], tcr)
                if(tcr > 1):
                    above_1_count_TCR[model][i] += 1
                print("TCR (Total Cost Ratio) (lambda = %d): %.4f" %(lambda_for_FalsePositive, tcr))

            print()
            print(sklearn.metrics.classification_report(y_test, y_pred, labels=[0, 1]))

        # time for testing the current iteration
        end_time_second = time.time()
        elapsed = end_time_second - end_time_testing
        all_time_testing += elapsed

    # printing the mistaken messages
    if print_wrong_decisions:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++  Start wrong sentences ++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        spam_wrong_class_printing = "0"

        # sort non-spam first , and spam after
        wrong_classification_sorted = [[],[]]
        for wrong_msg in wrong_classification.keys():
            wrong_classification_sorted[int(wrong_msg[1])] += [wrong_msg]
        wrong_classification_sorted = wrong_classification_sorted[0] + wrong_classification_sorted[1]

        for wrong_msg in wrong_classification_sorted:
            if (spam_wrong_class_printing == wrong_msg[1]):
                spam_wrong_class_printing = str(int(spam_wrong_class_printing)+1)
                print()
                print ("Wrong class: %d (%s)"%(wrong_msg[1], labels[int(wrong_msg[1])]))
            print("Message: (Wrong class: %d (%s))"%(wrong_msg[1], labels[int(wrong_msg[1])]))
            print(wrong_msg[0])
            for wrong_model in wrong_classification[wrong_msg]:
                print ("Model: %s | Wrong count: %d"%(wrong_model, wrong_classification[wrong_msg][wrong_model]))
            print("-")

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++  End wrong sentences ++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # summary of all iterations ### maybe we can summary for k=1 also
    if (k_fold_num > 1):
        for i in range(3):
            print("=============================")

        # we want to print a matrix that summarizes all the scores.
        # here we add to the matrix its headlines
        headlines = ["*","Ave F1"]
        headlines.append("Low F1")
        for i in range(len(lambda_for_FalsePositive_list)):
            headlines.append("Ave WAcc %d"%lambda_for_FalsePositive_list[i])
            headlines.append("Low WAcc %d" % lambda_for_FalsePositive_list[i])
            headlines.append("Ave TCR %d" % lambda_for_FalsePositive_list[i])
            headlines.append("Low TCR %d" % lambda_for_FalsePositive_list[i])

        #shortHeadlines = ["*"] + list(range(len(headlines)-1))
        shortHeadlines = [x[0]+"_"+x.replace(" ","")[-4:] for x in headlines]
        matrix = [
            shortHeadlines
        ]

        # print every model's scores and filling rows to the matrix
        lambda_count = len(lambda_for_FalsePositive_list)
        for model in all_models.keys():
            matrix_row = [model[:3]]
            averages = []
            lowests = []
            print("=============================")
            print("MODEL %s"%model)
            print("AVERAGES: K-FOLD cross-validation with a split of %d:"%k_fold_num)

            matrix_row.append(round(count_all_scores[model] / k_fold_num,4))
            print("F1 score: %.4f"%matrix_row[-1])

            for i in range(lambda_count):
                averages.append(round(count_all_WAcc[model][i] / k_fold_num,4))
                print("WAcc score (lambda = %d): %.4f"%(lambda_for_FalsePositive_list[i],averages[-1]))

                averages.append(round(count_all_TCR[model][i] / k_fold_num,4))
                print("TCR score (lambda = %d): %.4f"%(lambda_for_FalsePositive_list[i], averages[-1]))

            print("===")
            print("LOWEST: K-FOLD cross-validation with a split of %d:" % k_fold_num)

            matrix_row.append(round(minimum_scores[model],4))
            print("F1 score: %.4f" %matrix_row[-1])

            for i in range(lambda_count):
                lowests.append(round(minimum_WAcc[model][i],4))
                print("WAcc score (lambda = %d): %.4f" % (lambda_for_FalsePositive_list[i],lowests[-1] ))

                lowests.append(round(minimum_TCR[model][i],4))
                print("TCR score (lambda = %d): %.4f" % (lambda_for_FalsePositive_list[i], lowests[-1]))

            # sorting the row so that 'average' value will be next to the fitting 'lowest' value
            for i in range(len(averages)):
                matrix_row += [averages[i],lowests[i]]
            matrix.append(matrix_row)

        for i in range(3):
            print("++++++++++++++")

        # print the matrix in a pretty way
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print ('\n'.join(table))

        print()
        print("Count times TCR was > 1 :")
        for x in above_1_count_TCR:
            all_tcr = ""
            for i in range(len(lambda_for_FalsePositive_list)):
                if(all_tcr != ""):
                    all_tcr += " , "
                all_tcr += "%d : %d"%(lambda_for_FalsePositive_list[i],above_1_count_TCR[x][i])
            print("Model %s - %s"%(x,all_tcr))

        # headlines and models' names are too long and make the matrix not pretty, so we print it on the side and will combine them later
        print()
        print("&&& headlines (first row) &&&")
        for x in headlines:
            print(x)

        print()
        print("&&& models (first column) &&&")
        for x in all_models.keys():
            print(x)


    # Summary of parameters and time
    print("|||||||||||")
    print("All the testing took %d seconds !" % all_time_testing)

    end_time_for_everything = time.time()
    elapsed = end_time_for_everything - start_time_for_everything
    print("All the process took %d seconds !" % elapsed)
    print()
    print("""Parameters:
    model index: %d
    model name: %s
    dataset: %s
    messages count: %d
    spam count: %d
    non spam count: %d
    include_dependency_tree_feature: %r
    include_pos: %r
    k_fold_number = %d
    if k_fold_number is 1, split data = %.2f
    """%(model_index,model_desc,dataset_name,len(X_non_spam) + len(X_spam),len(X_spam),len(X_non_spam),include_dependency_tree_feature,include_pos_feature,k_fold_num,split))

