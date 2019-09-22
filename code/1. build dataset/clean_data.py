# -*- coding: utf-8 -*-
from sys import argv
import os
import codecs
import re
import random
import mailbox
import sys
sys.path.insert(1, '../3. classifier')
import general_func

use_separator2 = False
seperate_stage2_messages = False

input_dir = "./original_data"
output_dir = "./cleaned_data"
separator1 = "------------------------------------------------------------------------------------"
separator2 = "+++++++++++++++++++"
emojies_count = {}


def clean_data(filename, stage, analyze_func, identify_message_start):
    global input_dir
    global output_dir
    global seperate_stage2_messages

    if stage == 2:
        input_dir = './manual_clean'
        output_dir = './processed_data'


    file_lines = []

    input_file = input_dir + "/%s.txt" % filename
    all_msgs = False
    if os.path.exists(input_file):
        f = codecs.open(input_file, "r", "utf-8")
        with f as input_file:
            for line in input_file:
                file_lines += [line]
    else:
        input_file = input_dir + "/%s.mbox" % filename
        mbox = mailbox.mbox(input_file)
        for message in mbox:
            try:
                body = general_func.getBody(message)
                if body is None:
                    continue

                subject = general_func.getSubject(message)
            except:
                continue

            msg = subject + general_func.replace_var("mailsubject",None) + body

            file_lines += [msg]
        all_msgs = True


    if stage == 2:
        answer_data = stage2_processing(file_lines, filename, all_msgs)
        if seperate_stage2_messages:
            final_data = []
            for msg in answer_data:
                final_data += [msg]
                final_data += [""]
            answer_data = final_data
    else:
        answer_data = stage1_processing(file_lines, filename, analyze_func, identify_message_start, all_msgs)



    try:
        print("\nData saved successfully in '%s' !"%write_list_to_file(filename, answer_data))
    except Exception as e:
        print("\nCould not save output, because - %s"%e)

def stage2_processing(file_lines, filename, are_all_msgs):
    global separator1
    global separator2
    global emojies_count

    print("Processing %s messages ..." % filename)

    all_messages = []
    seperator2_counter = 0

    if not are_all_msgs:
        first_empty_line = True
        msg = ""
        for line in file_lines:
            for chr in line:
                if general_func.is_emoji(chr):
                    if chr not in emojies_count:
                        emojies_count[chr] = 0
                    emojies_count[chr] += 1
            if(line != "\r\n"):
                first_empty_line = True
            else:
                if first_empty_line:
                    first_empty_line = False
                    continue

            if separator2 in line:
                seperator2_counter += 1
                all_messages += [msg]
                msg = ""
            elif separator1 in line:
                all_messages += [msg]
                msg = ""
            else:
                if msg != "" and "\r\n" not in msg[-5:]:
                    msg += "\r\n"
                msg += line
        all_messages += [msg]
    else:
        all_messages = file_lines

    print("\n-Found %d seperator2 and transformed them to separator1."%seperator2_counter)

    print("\n-Before removing duplicates: %d messages"%len(all_messages))
    msg_dict = {}
    for msg in all_messages:
        if msg not in msg_dict:
            msg_dict[msg] = 0
        msg_dict[msg] += 1
    all_messages = msg_dict.keys()
    print("-After removing duplicates: %d messages" %len(all_messages))

    processed_data = []
    for msg in all_messages:
        msg = msg.replace("\n","")
        if("\r" == msg[-1]):
            msg=msg[:-1]
        msg=msg.replace("\r", general_func.replace_var("enter", msg))#[:-len(general_func.replace_var("enter", msg))]
        msg = general_func.replace_forbidden(msg)
        processed_data += [msg]

    processed_data = processed_data[:-1]
    random.shuffle(processed_data)

    return processed_data

def stage1_processing(file_lines, filename, analyze_func, identify_message_start, are_all_msgs):
    global separator1
    global separator2
    global use_separator2
    
    print("Cleaning %s messages ..."%filename)

    all_messages = []
    if not are_all_msgs:
        msg = None
        for line in file_lines:
            if identify_message_start(line):
                if msg is not None:
                    all_messages += [msg]
                msg = [line]
            elif msg is not None:
                msg += [line]
            else:
                print("Skip line because it is not contained in a message (%s)"%line)
        all_messages += [msg]
    else:
        all_messages = file_lines

    cleaned_data = []
    msg_counter = 0
    separator_counter = 0

    all_messages = [msg for msg in all_messages if general_func.is_hebrew(analyze_func(msg))]

    for msg in all_messages:
        analyzed = analyze_func(msg)
        msg_counter += 1
        if len(analyzed) > 0:

            if len(cleaned_data) > 0:
                cleaned_data += [separator1]
                separator_counter += 1

            for i in range(len(analyzed)):
                if i > 0:
                    if use_separator2:
                        cleaned_data[-1] = cleaned_data[-1][:-3]
                        cleaned_data += [separator2]
                    separator_counter += 1
                cleaned_data += [analyzed[i]]

    print("Found %d original messages. %d messages were found after cleaning."%(msg_counter, len(cleaned_data)-separator_counter))
    cleaned_data = [msg for msg in cleaned_data if general_func.is_hebrew(msg)]
    return cleaned_data

def write_list_to_file(filename, data, isFullPath = False):
    global output_dir
    if isFullPath:
        output_filename = filename
    else:
        output_filename = output_dir + '/' + filename + '.txt'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(output_filename):
        os.remove(output_filename)


    f = codecs.open(output_filename, "w", "utf-8")
    with f as output_file:
        for line in data:
            output_file.write(line)
            if "\r\n" not in line:
                output_file.write("\r\n")

    return output_filename

def analyze_whatsapp(msg):
    inside_msgs = []

    whatsapp_system_messages = [
        "‫‏יצרת את קבוצת",
        "צר/ה את קבוצת",
        "צירף/ה אותך‬",
        "כעת ההודעות הנשלחות בקבוצה זו מאובטחות עם הצפנה מקצה",
        "מחקת הודעה זו.",
        "הודעה זו נמחקה",
        "צירף/ה את",
        "הינך מנהל/ת קבוצה",
        "צירפת את",
        "החליף/ה את תמונת",
        "עזב/ה.‬",
        "קוד האבטחה של",
        "הנושא שונה ל",
        "הסיר/ה את",
        "החלפת את הנושא אל"
    ]

    one_message = None
    empty_lines_counter = 0
    for line in msg:
        whatsapp_sys_flag = False
        for whatsapp_sys in whatsapp_system_messages:
            if whatsapp_sys in line:
                whatsapp_sys_flag = True
                break
        if whatsapp_sys_flag:
            continue
        if not re.search("[[0-9]+.[0-9]+.[0-9]{4}. [0-9]+:[0-9]+:[0-9]+] .*: ", line):
            line = [line]
        else:
            line = re.split("[[0-9]+.[0-9]+.[0-9]{4}. [0-9]+:[0-9]+:[0-9]+] .*: ", line)
            if(len(line) > 1):
                line = line[1:]

        if len(line) == 1:
            line = line[0]
        else:
            line = ": ".join(str(v) for v in line)

        if line == "\n":
            empty_lines_counter += 1
            continue

        if one_message is None:
            one_message = line
            empty_lines_counter = 0
            continue

        if empty_lines_counter == 0:
            one_message += line
        elif empty_lines_counter == 1:
            one_message += "\r\n"
            inside_msgs += [one_message]
            one_message = line
            empty_lines_counter = 0
        else:
            one_message += '\r\n' * empty_lines_counter
            one_message += "\r\n" + line
            empty_lines_counter = 0

    if one_message is not None:
        inside_msgs += [one_message]

    return inside_msgs


def identify_message_start_whatsapp(msg):
    if re.search("[[0-9]+.[0-9]+.[0-9]{4}. [0-9]+:[0-9]+:[0-9]+] .*: ", msg): # == "] Guy Twito: " in msg or "] יובל זיידל: " in msg:
        return True
    return False

# function is not needed because it ment to be used at section 1 of the script, which is not needed for emails.
def analyze_email(msg):
    # msg = list with message's lines
    # inside_msgs = list with messages that were in the message
    # None for skip sentence
    # otherwise, return list of messages

    inside_msgs = []

    #clean data

    if (True):  ## skip conditions
        return inside_msgs

    #fill inside msgs

    return inside_msgs

def identify_message_start_email(msg):
    return True

def print_usage():
    print()
    print("Usage:")
    print('python analyze.py [w / w2 / e / e2] [2]')
    print('w - for whatsapp (spam)')
    print('w2 - for whatsapp2 (non-spam)')
    print('e - for email (spam)')
    print('e2 - for email2 (non-spam)')
    print()
    print("If you are writing '2' , it means you ran the command first without 2 and checked that the messages are separated correctly, and they are ready for stage 2 (NEEDED ONLY FOR WHATSAPP).")
    print()


if len(argv) < 2:
    print_usage()
else:
    stage = 1
    if len(argv) > 2:
        if argv[2] == '2':
            stage = 2

    if argv[1] in ['w','w2']:
        filename = 'whatsapp'
        if '2' in argv[1]:
            filename += '2'
        clean_data(filename, stage, analyze_whatsapp, identify_message_start_whatsapp)
    elif argv[1]  in ['e','e2']:

        if stage == 1:
            print_usage()
        else:
            filename = 'email'
            if '2' in argv[1]:
                filename += '2'
            clean_data(filename, stage, analyze_email, identify_message_start_email)
    else:
        print_usage()
