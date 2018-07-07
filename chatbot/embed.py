import itertools
import os
import pickle
import sys
import urllib.request
import zipfile

import nltk
import numpy as np

from chatbot.config import *


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove(glove_file, data_dir):
    if not os.path.exists(glove_file):
        glove_zip = glove_file + ".zip"
        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def index_words(all_words, vocab_size, special_tokens):
    # get frequency distribution
    freq_dist = nltk.FreqDist(all_words)
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    words = special_tokens + [x[0] for x in vocab]
    index2word = {i: w for i, w in enumerate(words)}
    word2index = {w: i for i, w in enumerate(words)}
    return index2word, word2index, freq_dist


def make_embedding_matrix(word2index, word2vec_map, embedding_dim, pad=None):
    embedding_matrix = np.zeros((len(word2index), embedding_dim))
    for word, i in word2index.items():
        embedding_vector = word2vec_map.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # words not found in embedding index will be random (variant of Xavier initialization).
            # in the future they should be trainable
            sd = 1 / np.sqrt(embedding_dim)  # Standard deviation to use
            embedding_matrix[i] = np.random.normal(0, sd, embedding_dim)
        if pad is not None:
            # pad embedding is all zeros
            embedding_matrix[word2index[pad]] = 0
    return embedding_matrix


def main():
    print("Indexing words and creating embedding matrix...")
    download_glove(PREPROCESSING_PARAMS['glove_model'], DATA_DIR)
    # read GLOVE file
    glove_word2index, glove_index2word, word2vec_map = read_glove_vecs(
        PREPROCESSING_PARAMS['glove_model'])

    # read file with all words from dataset
    path = os.path.join(DATA_DIR, "all_words.pickle")
    all_words = pickle.load(open(path, "rb"))

    # create word index
    index2word, word2index, _ = index_words(
        all_words, PREPROCESSING_PARAMS['vocab_size'], PREPROCESSING_PARAMS['special_tokens'])
    # create embedding matrix for chosen words
    embedding_matrix = make_embedding_matrix(
        word2index, word2vec_map, PREPROCESSING_PARAMS['glove_embedding_size'], pad=PREPROCESSING_PARAMS['pad'])

    for name, obj in zip(["embedding_matrix", "index2word", "word2index"],
                          [embedding_matrix, index2word, word2index]):
        pickle.dump(obj, open(PATHS[name], "wb"), protocol=2)


if __name__ == '__main__':
    main()
