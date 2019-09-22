# -*- coding: utf-8 -*-
import json
import copy
import re
import random
from email.header import decode_header
import urllib.parse
import codecs
import subprocess
import os
import glob

run_yap_script_full_path = "/home/guyy18/run_yap"
output_directory_full_path = "/home/guyy18/output"
enter_line_var = "ENTERNEWLINEENDVAR"
subject_var = "MAILSUBJECTENDVAR"

spam_max_length = None
non_spam_max_length = None

def main():
    get_datasets()

def get_datasets(spam_max_length_param=100000000, non_spam_max_length_param=100000000):
    global run_yap_script_full_path
    global output_directory_full_path
    global enter_line_var
    global subject_var
    global spam_max_length
    global non_spam_max_length

    spam_max_length = spam_max_length_param
    non_spam_max_length = non_spam_max_length_param

    datasets = {
        "sms" : extract_dataset_from_file("sms"),
        "email": extract_dataset_from_file("email")
    }

    files = glob.glob('/home/guyy18/output/*')
    for f in files:
        os.remove(f)

    chars_to_separate = ['.',':','?','!',',']
    bash_bad_chars = ['(',')','"', '<', '>', "'", "`", "׳", "&", "*", "|",'\\']

    for dataset in datasets:
        for data_type in datasets[dataset]:
            output_file = "%s/%s_%s"%(output_directory_full_path, dataset,data_type)
            counter = 1
            for msg in datasets[dataset][data_type]:

                for chr in chars_to_separate:
                    msg = msg.replace(chr, " %s "%chr)
                index = 3
                # replacing problematic chars with other chars (YAP still tag as NNP)
                for chr in bash_bad_chars:
                    msg = msg.replace(chr, "{"*index + "}"*index)
                    index += 1

                # cleaning double white space
                msg = clean_doubled_white_space(msg)

                print(counter,"/",len(datasets[dataset][data_type]))
                counter += 1

                output = []
                mailsubject_flag = False
                if subject_var in msg:
                    mailsubject_flag = True
                    msg = msg.split(subject_var)
                    msg = [msg[0]] + msg[1].split(enter_line_var)
                else:
                    msg = msg.split(enter_line_var)

                counter = 1
                for sentence in msg:
                    print("sentence %d) %s"%(counter,sentence))
                    output += [bash_cmd('%s %s ' % (run_yap_script_full_path, sentence)).decode('utf8')]
                    print('%s %s ' % (run_yap_script_full_path, sentence))
                    print(output[-1])
                    print('-')
                    counter += 1

                if mailsubject_flag:
                    if(len(msg) > 2):
                        msg = [msg[0]+" "+subject_var+" "+msg[1]] + msg[2:]
                        output = [output[0] + " "+subject_var+" " + output[1]] + output[2:]
                    else:
                        msg = [msg[0] + " "+subject_var+" " + msg[1]]
                        output = [output[0] + " "+subject_var+" " + output[1]]
                msg = enter_line_var.join(msg)
                output = enter_line_var.join(output)

                # retrieving the replaced problematic chars
                while index >= 3:
                    index -= 1
                    output = output.replace("{"*index + "}"*index , bash_bad_chars[index-3])
                    msg = msg.replace("{" * index + "}" * index, bash_bad_chars[index - 3])

                print("final output:")
                print(output)
                print("--------")
                write_hebrew_file(output_file, [output])

    return datasets

def clean_doubled_white_space(txt):
    new_txt = re.sub(' +', ' ', txt)
    return re.sub('\t+', '\t', new_txt)

def write_hebrew_file(output_filename, data):
    f = codecs.open(output_filename, "a", "utf-8")
    with f as output_file:
        for line in data:
            output_file.write(line)
            if "\r\n" not in line:
                output_file.write("\r\n\n")

def bash_cmd(cmd):
    bashCommand = cmd
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output

def union_datasets(dataset1, dataset2):
    return {
        "spam" : dataset1['spam'] + dataset2['spam'],
        "non_spam" : dataset1['non_spam'] + dataset2['non_spam']
    }

def extract_dataset_from_file(filename):
    global spam_max_length
    global non_spam_max_length

    dataset = {
        "spam" : [],
        "non_spam" : []
    }

    actual_filename = filename
    if  '_' in actual_filename:
        actual_filename = actual_filename.aplit("_")[0]

    spam_filename = "./datasets/%s/spam.txt"%actual_filename
    non_spam_filename = "./datasets/%s/non_spam.txt"%actual_filename

    if "sms" == actual_filename or "email" == actual_filename:
        spam_lines = read_file(spam_filename)
        non_spam_lines = read_file(non_spam_filename)

        # build spam set
        for line in spam_lines:
            line_strip = line.strip()
            if(len(line_strip) != 0):
                dataset["spam"] += [line_strip]

        # build non spam set
        for line in non_spam_lines:
            line_strip = line.strip()
            if(len(line_strip) != 0):
                dataset["non_spam"] += [line_strip]

    # activate yap for the messages
    if "_morphological" in filename:
        for i in range(dataset["spam"]):
            dataset["spam"][i] = yap.msgToYap(dataset["spam"][i])
        for i in range(dataset["non_spam"]):
            dataset["non_spam"][i] = yap.msgToYap(dataset["non_spam"][i])

    # limit the datasets
    if spam_max_length is not None and spam_max_length > 0:
        dataset["spam"] = dataset["spam"][:spam_max_length]
    if non_spam_max_length is not None and non_spam_max_length > 0:
        dataset["non_spam"] = dataset["non_spam"][:non_spam_max_length]

    return dataset

def read_file(input_file):
    file_lines = []
    f = codecs.open(input_file, "r", "utf-8")
    with f as input_file:
        for line in input_file:
            file_lines += [line]

    return file_lines


if __name__ == "__main__":
    main()