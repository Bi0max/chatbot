import itertools
import os
import pickle

import nltk

from chatbot.config import *


def initial_preprocess_file(path, whitelist):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            i += 1
            words = [w.lower() for w in nltk.word_tokenize(line)]
            words = [w for w in words if in_white_list(w, whitelist)]
            data.append(words)
            if i % 10000 == 0:
                print(f"{i + 1} done.")
    return data


def get_all_words(tokenized_from, tokenized_to):
    return list(itertools.chain(*(tokenized_from + tokenized_to)))


def in_white_list(_word, whitelist):
    for char in _word:
        if char not in whitelist:
            return False

    return True


def main():
    tokenized_from = initial_preprocess_file(TRAIN_FROM_PATH, PREPROCESSING_PARAMS['whitelist'])
    tokenized_to = initial_preprocess_file(TRAIN_TO_PATH, PREPROCESSING_PARAMS['whitelist'])
    all_words = get_all_words(tokenized_from, tokenized_to)

    for fname, obj in zip(["tokenized_from", "tokenized_to", "all_words"], [tokenized_from, tokenized_to, all_words]):
        path = os.path.join(DATA_DIR, f"{fname}.pickle")
        pickle.dump(obj, open(path, "wb"), protocol=2)


if __name__ == '__main__':
    print("Start tokenizing...")
    main()
