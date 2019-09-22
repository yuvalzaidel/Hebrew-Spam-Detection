# -*- coding: utf-8 -*-

import codecs
import general_func
from general_func import treeNode

spam_max_length = None
non_spam_max_length = None

def get_datasets(spam_max_length_param, non_spam_max_length_param, use_both=False, use_morphological=False):
    global spam_max_length
    global non_spam_max_length

    spam_max_length = spam_max_length_param
    non_spam_max_length = non_spam_max_length_param

    if not use_morphological:
        datasets = {
            "sms" : extract_dataset_from_file("sms")  ,
            "email": extract_dataset_from_file("email")
        }
    else:
        datasets = {
            "sms_morphological": extract_dataset_from_file("sms_morphological"),
            "email_morphological": extract_dataset_from_file("email_morphological")
        }

    if use_both:
        if "email" in datasets and "sms" in datasets:
            datasets["both"] = union_datasets(datasets["sms"], datasets["email"])
        if "email_morphological" in datasets and "sms_morphological" in datasets:
            datasets["both_morphological"] = union_datasets(datasets["sms_morphological"], datasets["email_morphological"])

    # if no data provided for dataset , remove it
    keys_to_remove = []
    for key in datasets:
        if len(datasets[key]["spam"]) == 0 or len(datasets[key]["non_spam"]) == 0:
            keys_to_remove += [key]
    if len(keys_to_remove) > 0:
        keys_to_remove += ["both"]

    if use_both:
        keys_to_remove += ["sms","email","sms_morphological","email_morphological"]

    for key in keys_to_remove:
        if key in datasets:
            del(datasets[key])

    print()
    return datasets

def union_datasets(dataset1, dataset2):
    return {
        "spam" : dict(list(dataset1['spam'].items()) + list(dataset2['spam'].items())),
        "non_spam" : dict(list(dataset1['non_spam'].items()) + list(dataset2['non_spam'].items()))
    }

def extract_dataset_from_file(filename):
    global spam_max_length
    global non_spam_max_length

    dataset = {
        "spam" : [],
        "non_spam" : []
    }

    actual_filename = filename

    spam_filename = "./datasets/%s/spam.txt"%actual_filename
    non_spam_filename = "./datasets/%s/non_spam.txt"%actual_filename

    ul_count = 0
    if "sms" in actual_filename or "email" in actual_filename:
        spam_lines = read_file(spam_filename)
        non_spam_lines = read_file(non_spam_filename)

        # build spam set
        for line in spam_lines:
            line_strip = line.strip()
            if(len(line_strip) != 0):
                if("_morphological" in actual_filename):
                    if(line_strip == "ul"):
                        ul_count += 1
                        continue
                dataset["spam"] += [line_strip]

        # build non spam set
        for line in non_spam_lines:
            line_strip = line.strip()
            if(len(line_strip) != 0):
                if ("_morphological" in actual_filename):
                    # yap may can't in certain
                    if (line_strip == "ul"):
                        ul_count += 1
                        continue
                dataset["non_spam"] += [line_strip]

    print("%s UL COUNT (sentences that were mot accepted by YAP): %d"%(filename,ul_count))

    # limit the datasets
    if spam_max_length is not None and spam_max_length > 0:
        dataset["spam"] = dataset["spam"][:spam_max_length]
    if non_spam_max_length is not None and non_spam_max_length > 0:
        dataset["non_spam"] = dataset["non_spam"][:non_spam_max_length]


    # separate morphological sentences from its labels data
    labels_data = {
        "spam" : {},
        "non_spam": {}
    }
    for msg_type in dataset: # spam / non_spam
        for i in range(len(dataset[msg_type])):
            sentence = dataset[msg_type][i]
            real_sentence = ""
            sentence_labels_data = []
            msg_pos = [""]
            msg_all_roots = []
            current_tree = {
                "root" : None,
                "labels_data" : {}
            }
            if "_morphological" in filename:
                i = 0
                for morphema in sentence.split():
                    label_data = {}
                    if morphema.count('~') >= 3 and "" not in morphema.split("~")[-4:]:
                        i += 1
                        morphema = morphema.split("~")
                        tag = morphema[-1]
                        tag_order = morphema[-2]
                        pos = morphema[-3]
                        word = "~".join(morphema[:-3])
                        label_data = {
                            "tag" : tag,
                            "header" : tag_order,
                            "pos" : pos,
                            "word" : word
                        }
                        real_sentence += word+" "

                        # Build dependency tree
                        if str(i) not in current_tree["labels_data"]:
                            current_node = treeNode()
                            current_tree["labels_data"][str(i)] = current_node
                        else:
                            current_node = current_tree["labels_data"][str(i)]
                        current_node.val(word + " " + tag)

                        if tag_order == "0":
                            current_tree["root"] = current_node
                        # if not root, add node to parent
                        else:
                            # if parent node does not exist yet - create it
                            if tag_order not in current_tree["labels_data"]:
                                header_node = treeNode()
                                current_tree["labels_data"][tag_order] = header_node
                            # if parent exists, grab it
                            else:
                                header_node = current_tree["labels_data"][tag_order]
                            # add child to parent

                            header_node.child(current_node)
                    else:
                        real_sentence += morphema + " "

                        if current_tree["root"] is not None:
                            msg_all_roots += [current_tree["root"]]
                            current_tree = {
                                "root": None,
                                "labels_data": {}
                            }
                            i = 0


                    # Get all part of speech
                    if len(label_data) > 0:
                        msg_pos[-1] += label_data["pos"] + " "
                    else:
                        msg_pos[-1] = msg_pos[-1].strip()
                        if msg_pos[-1] != "":
                            msg_pos += [""]

                # finish building pos
                if len(msg_pos) > 1:
                    if msg_pos[-1] == "":
                        msg_pos = msg_pos[:-1]
                    msg_pos[-1] = msg_pos[-1].strip()

                # last sentence not inserted, if exists
                if current_tree["root"] is not None:
                    msg_all_roots += [current_tree["root"]]

                sentence_labels_data.append(msg_all_roots)
                sentence_labels_data.append(msg_pos)

            else:
                real_sentence = dataset[msg_type][i]

            real_sentence = real_sentence.strip()

            labels_data[msg_type][real_sentence] = sentence_labels_data

    return labels_data

def read_file(input_file):
    file_lines = []
    f = codecs.open(input_file, "r", "utf-8")
    with f as input_file:
        for line in input_file:
            file_lines += [line]

    return file_lines

