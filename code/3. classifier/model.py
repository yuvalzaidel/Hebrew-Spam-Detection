## general imports

import re
import math
import general_func
from general_func import treeNode

from tqdm import tqdm
import time

from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *

models_names=["All classifiers", "MultinomialNB", "LogisticRegression", "KNeighbors", "SVM", "AdaBoost"]
models = {
    "MultinomialNB" : MultinomialNB(),
    "LogisticRegression" : LogisticRegression(solver='liblinear', multi_class='auto'),
    "KNeighbors" : KNeighborsClassifier(n_neighbors=70),
    "SVM" : SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
    "AdaBoost" : AdaBoostClassifier(n_estimators=200, learning_rate=1)


}

# more classifiers that can be used
"""
        "oneVsRest" : OneVsRestClassifier(LogisticRegression()),
        "CalibratedCV": CalibratedClassifierCV(),
        "PassiveAggressive": PassiveAggressiveClassifier(),
        "BernoulliNB" : BernoulliNB(),
        "OneVsRestLogisticRegression" : OneVsRestClassifier(LogisticRegression()),
        "ExtraTrees" : ExtraTreesClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Bagging": BaggingClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "CalibratedCV": CalibratedClassifierCV(),
        "Dummy": DummyClassifier(),
        "PassiveAggressive": PassiveAggressiveClassifier(),
        "Ridge": RidgeClassifier(),
        "RidgeCV": RidgeClassifierCV(),
        "SGD": SGDClassifier()
        """


# get name of a model by index
def model_desc(model_index):
    global models_names
    global models

    return models_names[model_index]


class Model():

    def __init__(self, include_dependency_tree_feature, include_pos_feature, model_index, labels_data):
        global models_names
        global models
        self.include_dependency_tree_feature = include_dependency_tree_feature
        self.include_pos_feature = include_pos_feature

        # chose a specific model
        if model_index > 0:
            models = {
                models_names[model_index]: models[models_names[model_index]]
            }

        self.classifiers = models


        self.labels_data = labels_data
        all_pos = {}
        if include_pos_feature or include_dependency_tree_feature:
            for msg in labels_data:
                if len(labels_data[msg][1]) > 0:
                    msg_pos_sentences = labels_data[msg][1]
                    for sentence_pos in msg_pos_sentences:
                        for pos in sentence_pos.split():
                            all_pos[pos] = True
            self.all_pos_tags_types = list(all_pos.keys())



    # choose neighbours by messages count (the more neighbours effects the accuracy and training time)
    def change_neighbors(self, n):
        if "KNeighbors" in self.classifiers:
            self.classifiers["KNeighbors"] = KNeighborsClassifier(n_neighbors=n)

    # init progress bar
    def initialize_pbar(self, size):
        self.pbar = tqdm(total=size)

    # close progress bar
    def close_pbar(self):
        self.pbar.close()

    # build ngrams, by words and by letters
    def _add_to_ngrams(self, ngrams, max_ngram_len, token, words=True, min_ngram_len=1):

        words_ngrams = words
        if words_ngrams:
            token = token.split()

        # per ngram length
        for n in range(min_ngram_len, max_ngram_len + 1):
            # sliding window iterate the token to extract its ngrams
            for idx in range(len(token) - n + 1):
                temp_ngram = token[idx : idx + n]
                ngram = []
                # make the VARS similar (without numbers)
                # there are also "ENTERLINE" and "EMAIL" but they are all the same.
                for one_token in temp_ngram:
                    if "ENDVAR" in one_token:
                        if "LINK" in one_token:
                            one_token = "LINK"
                        elif "EMOJI" in one_token:
                            one_token = "EMOJI"
                        elif "PHONENUMBER" in one_token:
                            one_token = "PHONENUMBER"
                    ngram += [one_token]

                if words_ngrams:
                    ngram = " ".join(ngram)
                else:
                    ngram = "".join(ngram)

                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

        # return value is used for test only
        return ngrams

    # build ngrams for trees
    def _add_to_tree_ngrams(self, ngrams, root):
        tree_info = self.calculate_tree_info(root)
        tree_ngrams = tree_info["ngrams"]
        for ngram in tree_ngrams:
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

    # make a feature vector
    def vectorize(self, msg): #, ii=[0]):
        self.pbar.update(1)
        all_vectors = []


        # ngram occurences
        vec1 = [0] * len(self.ngrams_words)
        # ngram occurences as prefix
        vec2 = [0] * len(self.ngrams_letters)

        # ngrams for words
        for idx, ngram in enumerate(self.ngrams_words):
            if ngram in msg:
                if vec1[idx]:
                    vec1[idx] += 1
                else:
                    vec1[idx] = 1
        all_vectors += vec1

        # ngrams for letters
        for idx, ngram in enumerate(self.ngrams_letters):
            if ngram in msg:
                if vec2[idx]:
                    vec2[idx] += 1
                else:
                    vec2[idx] = 1
        all_vectors += vec2

        # bag of words
        vec3 = [0,0]
        for word in general_func.hebrew_only(msg).split():
            for i in [0,1]:
                if word in self.bag_of_words[i]:
                    vec3[i] += 1
        all_vectors += vec3

        # features for dependency trees or pos
        if self.labels_data[msg] is not None and (self.include_dependency_tree_feature or self.include_pos_feature):
            # Get all part of speech (pos)
            # pos features
            if self.include_pos_feature:
                msg_pos = self.labels_data[msg][1]
                count_pos = {}
                vec_ngrams_pos_trees = [0] * len(self.ngrams_pos_trees)

                for sentence_pos in msg_pos:

                    # count_pos
                    for one_pos in sentence_pos.split():
                        if one_pos not in count_pos:
                            count_pos[one_pos] = 0
                        count_pos[one_pos] += 1

                    # ngrams for pos
                    for idx, ngram in enumerate(self.ngrams_pos_trees):
                        if ngram in sentence_pos:
                            if vec_ngrams_pos_trees[idx]:
                                vec_ngrams_pos_trees[idx] += 1
                            else:
                                vec_ngrams_pos_trees[idx] = 1


                vec_pos_tags_count = []
                for pos_type in self.all_pos_tags_types:
                    if pos_type in count_pos:
                        vec_pos_tags_count.append(count_pos[pos_type])
                    else:
                        vec_pos_tags_count.append(0)

                all_vectors += vec_pos_tags_count
                all_vectors += vec_ngrams_pos_trees


            if self.include_dependency_tree_feature:
                # dependency trees features !!!
                different_tags = {}
                deepest_depth = 0
                most_childs = 0
                all_ngrams = {}

                all_roots = self.labels_data[msg][0]

                for tree_root in all_roots:
                    if tree_root is not None:
                        tree_calculated_info = self.calculate_tree_info(tree_root)
                        most_childs = max(most_childs, tree_calculated_info["most_childs"])
                        deepest_depth = max(deepest_depth, tree_calculated_info["depth"])
                        for tag in tree_calculated_info["all_tags"].keys():
                            different_tags[tag] = True

                        for ngram in tree_calculated_info["ngrams"]:
                            if ngram not in all_ngrams:
                                all_ngrams[ngram] = 0
                            all_ngrams[ngram] += tree_calculated_info["ngrams"][ngram]

                # ngrams for pos dependency trees
                vec_ngrams_dependency_trees = [0] * len(self.ngrams_dependency_trees)
                for idx, ngram in enumerate(self.ngrams_dependency_trees):
                    if ngram in all_ngrams:
                        vec_ngrams_dependency_trees[idx] = all_ngrams[ngram]

                all_vectors += [deepest_depth]
                all_vectors += [most_childs]
                all_vectors += [len(list(different_tags.keys()))]
                all_vectors += vec_ngrams_dependency_trees



        vec7 = [0]
        vec8 = [0]
        vec9 = [0,0,0]
        vec10 = [0] * 6
        vec11 = [0,0]
        vec12 = [0,0]
        vec13 = [0]
        vec14 = [0,0]
        vec15 = [0]
        vec16 = [0]
        vec17 = [0,0]

        msg_split = msg.split()
        word_index = -1
        english_count = 0
        hebrew_count = 0
        sentences_count = 1
        numbers_count = 0
        words_count = 0
        letters_count = 0
        was_word = False
        for word in msg_split:
            word_index += 1
            if "ENDVAR" in word:
                word = word.split("ENDVAR")[0]
                no_IG_vars = ["MAILSUBJECT", "ENTERNEWLINE", "EMAIL"]
                IG_vars = ["EMOJI","PHONENUMBER","LINK"]
                IG = None
                if word not in no_IG_vars:
                    for var in IG_vars:
                        if var in word:
                            IG = word.split(var)[1]
                            break



                if word in ["MAILSUBJECT" ,"ENTERNEWLINE"]:
                    if was_word and word == "ENTERNEWLINE":
                        sentences_count += 1
                    was_word = False
                else:
                    was_word = True
                if word == "EMAIL":
                    words_count += 1
                if word == "PHONENUMBER":
                    if(IG == "@"):
                        IG = 6
                    else:
                        IG = int(IG)
                    words_count += 1
                    vec10[IG-1] += 1

                if word == "LINK":
                    IG = int(IG)
                    words_count += 1
                    vec9[0] += 1
                    vec9[1] += IG
                    vec9[2] += word_index / len(msg_split)

                if word == "EMOJI":
                    vec14[0] += 1
                    letters_count += 1


            else:
                was_word = True

                for chr in word:
                    if chr.isalpha():
                        english_count += 1
                    elif chr.isdigit():
                        numbers_count += 1
                hebrew_count += len(general_func.hebrew_only(word))

                words_count += 1
                letters_count += len(word)
                vec11[0] = max(vec11[0], len(word))

                signs = general_func.get_signs(word)
                vec12[0] += len(signs)
                for s in signs:
                    if s in [',','.',':','!','?',"'",'"',"-"]:
                        vec13[0] += 1


        vec7[0] = words_count
        vec8[0] = letters_count
        if vec9[0] > 0:
            vec9[1] = vec9[1] / vec9[0]
            vec9[2] = vec9[2] / vec9[0]
        if letters_count > 0:
            vec12[1] = vec12[0] / letters_count
            vec14[1] = vec14[0] / letters_count
            vec17[1] = numbers_count / letters_count
        vec17[0] = numbers_count

        if english_count > 0:
            vec15[0] = hebrew_count / english_count
        vec16[0] = sentences_count

        # words count
        all_vectors += vec7
        # letters count - without links/phones/email
        all_vectors += vec8
        # links count, links average length , link average place in message
        all_vectors += vec9
        # phone number types
        all_vectors += vec10
        # longest word (ENDVARS not included)
        all_vectors += vec11
        # signs count word (ENDVARS not included) , signs percent from message
        all_vectors += vec12
        # punctuation count
        all_vectors += vec13
        # emoji count , emoji percent from message
        all_vectors += vec14
        # relation between hebrew and english letters' count
        all_vectors += vec15
        # sentences count (define stop words)
        all_vectors += vec16
        # number count , number percent from message
        all_vectors += vec17


        # terminator , if no hebrew inside - not spam
        if not general_func.is_hebrew(msg):
            return len(all_vectors) * [0]

        return all_vectors


    def train(self, msgs, y):
        start_time = time.time()
        max_ngram_len = 3
        self.max_ngram_len_trees = 3
        max_words_in_bag = 2500
        # minimum occurence for word in spam messages or non-spam messages to be a frequent word
        min_word_occurences = 5
        # recommended number of neighbours is sqrt of n, we multiply by 3.
        self.change_neighbors(3*int(math.sqrt(len(msgs))))

        # extract all ngrams from all corpus tokens
        self.ngrams_words = dict()
        self.ngrams_letters = dict()
        self.ngrams_pos_trees = dict()
        self.ngrams_dependency_trees = dict()

        self.counters = dict(
            links_length = [],
            links_count = [],
            emojies_count = [],
            emojies_original = [],
            enters = []
        )

        self.bag_of_words = [{},{}]



        counter = 0
        for msg in msgs:
            # count emojies, links, enterlines etc
            self.save_counts(msg)
            # make bag of words
            self.update_bag_of_words(y[counter], msg)
            # words ngrams
            self._add_to_ngrams(self.ngrams_words, max_ngram_len, msg, words=True)
            # letters ngrams
            self._add_to_ngrams(self.ngrams_letters, max_ngram_len, msg, words=False)

            if len(self.labels_data[msg]) > 0:
                if self.include_pos_feature:
                    msg_pos_sentences = self.labels_data[msg][1]
                    for sentence_pos in msg_pos_sentences:
                        self._add_to_ngrams(self.ngrams_pos_trees, max_ngram_len, sentence_pos, words=True)

                if self.include_dependency_tree_feature:
                    msg_all_roots = self.labels_data[msg][0]
                    for tree_root in msg_all_roots:
                        self._add_to_tree_ngrams(self.ngrams_dependency_trees, tree_root)

            counter += 1

        # use self.counts to make statistics for all
        self.make_statistics(y)
        # save only for the relevant words in spam and non-spam
        self.clean_bag_of_words(max_words_in_bag, min_word_occurences)

        print("Vectorizing ....")
        self.initialize_pbar(len(msgs))
        X = list(map(self.vectorize, msgs))
        self.close_pbar()

        assert len(X) == len(y)

        # fitting the trained models, counting time (may be a long action)
        models_counter = 1
        for model in self.classifiers.keys():
            print("Fitting model %s (%d/%d)...."%(model,models_counter,len(self.classifiers.keys())))
            models_counter += 1
            start_time_fitting = time.time()

            self.classifiers[model].fit(X, y)

            end_time_fitting = time.time()
            elapsed = end_time_fitting - start_time_fitting
            print("fitting took %s seconds."%elapsed)
        print("-")
        print("Finished training.")

        end_time = time.time()
        elapsed = end_time - start_time
        print("All the training took %d seconds !"%elapsed)

        return self.classifiers


    def update_bag_of_words(self, y_val, msg):
        msg = [x for x in general_func.hebrew_only(msg).split() if len(x)>1]
        for word in msg:
            if word not in self.bag_of_words[y_val]:
                self.bag_of_words[y_val][word] = 0
            self.bag_of_words[y_val][word] += 1

    def clean_bag_of_words(self, max_words_in_bag, min_word_occurences):
        for i in [0,1]:
            self.bag_of_words[i] = general_func.sort_dict(self.bag_of_words[i])
            self.bag_of_words[i] = [x for x in self.bag_of_words[i] if x[1] >= min_word_occurences]

            temp_dict = {}
            for x in self.bag_of_words[i]:
                temp_dict [x[0]] = x[1]
            self.bag_of_words[i] = temp_dict

        temp_dict = [{},{}]
        for i in [0, 1]:
            for x in self.bag_of_words[i]:
                if x not in self.bag_of_words[1-i]:
                    if(len(temp_dict[i].keys()) >= max_words_in_bag):
                        break
                    else:
                        temp_dict[i][x] = self.bag_of_words[i][x]

        self.bag_of_words = temp_dict


    # count wanted data
    def save_counts(self, msg):
        emojies = re.findall('EMOJI.ENDVAR' , msg)
        links = re.findall('LINK[0-9]+ENDVAR', msg)

        self.counters["enters"] += [msg.count("ENTERNEWLINE")]

        self.counters["links_length"] += [general_func.average([int(x.split('ENDVAR')[0].split("LINK")[1]) for x in links])]
        self.counters["links_count"] += [len(links)]
        self.counters["emojies_count"] += [len(emojies)]
        self.counters["emojies_original"] += [[x.split('ENDVAR')[0][-1] for x in emojies]]

    def make_statistics(self, y):
        self.statistics = dict(
            links_length=[[], []],
            links_count=[[], []],
            emojies_count=[[], []],
            emojies_original=[[], []],
            enters=[[], []]
        )

        for i in range(len(y)):
            self.statistics["links_length"][y[i]] += [self.counters["links_length"][i]]
            self.statistics["links_count"][y[i]] += [self.counters["links_count"][i]]
            self.statistics["emojies_count"][y[i]] += [self.counters["emojies_count"][i]]
            self.statistics["emojies_original"][y[i]] += self.counters["emojies_original"][i]
            self.statistics["enters"][y[i]] += [self.counters["enters"][i]]

        for i in [0, 1]:
            self.statistics["links_length"][i] = general_func.average(self.statistics["links_length"][i])
            self.statistics["links_count"][i] = general_func.average(self.statistics["links_count"][i])
            self.statistics["emojies_count"][i] = general_func.average(self.statistics["emojies_count"][i])

            count_emojies_types = {}
            for emoji in self.statistics["emojies_original"][i]:
                if emoji not in count_emojies_types:
                    count_emojies_types[emoji] = 0
                count_emojies_types[emoji] += 1
            self.statistics["emojies_original"][i] = general_func.sort_dict(count_emojies_types)

            self.statistics["enters"][i] = general_func.average(self.statistics["enters"][i])


    # extract wanted data from the dependency trees
    def calculate_tree_info(self, root):
        all_info = {
            "most_childs": len(root.child()),
            "depth": 0,
            "all_tags": {},
            "ngrams": {}
        }

        root_tag = root.val().split()[1]
        all_info["all_tags"][root_tag] = True
        all_info["ngrams"][root_tag] = 1

        if len(root.child()) == 0:
            all_info["depth"] = 1
        else:
            for child in root.child():
                child_all_info = self.calculate_tree_info(child)
                all_info["most_childs"] = max(all_info["most_childs"], child_all_info["most_childs"])
                all_info["depth"] = max(all_info["depth"], child_all_info["depth"]+1)

                for tag in child_all_info["all_tags"]:
                    all_info["all_tags"][tag] = True


                ## create ngrams to all_info["ngrams"]
                child_tag = child.val().split()[1]
                two_gram = root_tag + " " + child_tag
                three_grams = []
                if len(child.child()) != 0:
                    for grand_child in child.child():
                        grand_child_tag = grand_child.val().split()[1]
                        three_grams += [two_gram + " " + grand_child_tag]
                new_ngrams = [two_gram] + three_grams
                for ngram in new_ngrams:
                    if ngram not in all_info["ngrams"]:
                        all_info["ngrams"][ngram] = 0
                    all_info["ngrams"][ngram] += 1

                for ngram in child_all_info["ngrams"]:
                    if ngram not in all_info["ngrams"]:
                        all_info["ngrams"][ngram] = 0
                    all_info["ngrams"][ngram] += child_all_info["ngrams"][ngram]



        return all_info